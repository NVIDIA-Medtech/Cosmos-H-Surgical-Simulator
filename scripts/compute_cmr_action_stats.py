# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Compute normalization statistics for CMR Versius conditioning tensor.

This script computes statistics (mean, std, min, max, q01, q99) for:
1. 44D conditioning tensor (30D actions + 14D state conditioning):
   ACTIONS (30D, with motion scaling applied for poses):
   - Left arm:  xyz_rel (3D) + rot6d_rel (6D) + gripper (1D) = 10D
   - Right arm: xyz_rel (3D) + rot6d_rel (6D) + gripper (1D) = 10D
   - Energy buttons: left (1D) + right (1D) = 2D
   - Thumbstick X: left (1D) + right (1D) = 2D (endoscope/instrument control)
   - Thumbstick Y: left (1D) + right (1D) = 2D (endoscope/instrument control)
   - Thumbstick Button: left (1D) + right (1D) = 2D (instrument straighten function)
   - Clutch Button: left (1D) + right (1D) = 2D (engage/disengage arm control)
   STATE CONDITIONING (14D, sampled at action timesteps, absolute values):
   - Haptic engaged: left (1D) + right (1D) = 2D (persistent engagement state)
   - Arm linked to haptic: left (1D) + right (1D) = 2D (which arm 0-3 is active)
   - Arm instrument type: arm_0 (1D) + arm_1 (1D) + arm_2 (1D) + arm_3 (1D) = 4D
   - Arm HUD color: arm_0 (1D) + arm_1 (1D) + arm_2 (1D) + arm_3 (1D) = 4D
   - Electrosurgery mode: left (1D) + right (1D) = 2D (CUT/COAG selection)
2. 16D state observations (xyz + quat for each arm, matching main pipeline's raw format):
   - Left arm:  xyz (3D) + quat_xyzw (4D) + gripper (1D) = 8D
   - Right arm: xyz (3D) + quat_xyzw (4D) + gripper (1D) = 8D

The hybrid-relative conversion uses engagement-aware delta re-integration
to correctly handle clutch scenarios in CMR Versius surgical robot data.

Motion scaling is applied to convert from hand-controller-space to instrument-space,
which affects the magnitude of translation and rotation deltas. This is critical
for computing stats that match the actual action magnitudes used during training.

CLUTCH-AWARE FILTERING (matching dataset.py):
This script applies the same load-time filtering as the main training pipeline
(cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/dataset.py).
This ensures that statistics are computed on EXACTLY the same set of valid samples
that will be used during training, avoiding distribution mismatch issues.

Filtering rules:
  Rule 1: Discard if armlinkedtohaptic_* changes within action horizon (arm swap)
  Rule 2: Discard if both arms completely disengaged for entire action horizon

Use --no-filter to disable filtering (not recommended).

PIPELINE CONSISTENCY:
To ensure stats match the main training pipeline exactly:
  - Action horizon: Default is 12 (matching num_frames in groot_configs.py)
  - Rotation format: Uses quat input, matching groot_configs.py's input_rotation_format="quat"
  - Reference pose: Uses action array (matching modality.json's state.left_pose -> action)
  - Delta indices: [0, 6, 12, ..., 66] with frame_stride=6 (12 timesteps)

The stats are saved to stats_cosmos.json in the dataset's meta directory.

RAW PARQUET FORMAT (from CMR Versius dataset):
The parquet files contain flat arrays, NOT named columns:
  - observation.state: float64[100] array
  - action: float64[100] array

IMPORTANT: modality.json is the AUTHORITATIVE source for field naming.
- modality.json uses: translation_scaling, rotation_scaling, left_pose, left_energy, etc.
- info.json uses: translationscaling, rotationscaling, x_left, energyBtn_left, etc.
- This script uses modality.json naming convention for output stats keys.

Index mappings (raw indices from info.json):
  ACTION array (indices 0-25, rest zero-padded):
    Left:  0-2=xyz, 3-6=quat_xyzw, 7=clutch, 8=energy, 9=thumbstickBtn, 10=pince, 11-12=thumbstick
    Right: 13-15=xyz, 16-19=quat_xyzw, 20=clutch, 21=energy, 22=thumbstickBtn, 23=pince, 24-25=thumbstick
  
  STATE array (indices 0-23, rest zero-padded):
    0-1=haptic_armengageable (left/right), 2-6=arm_0..4_color, 7-11=arm_0..4_instrtype
    12=translation_scaling, 13=rotation_scaling, 16=hapticengaged_left, 17=hapticengaged_right
    20=armlinkedtohaptic_left, 21=armlinkedtohaptic_right

Usage:
    # Single dataset:
    python compute_cmr_action_stats.py --dataset-path /path/to/lerobot/dataset

    # All datasets under a root directory (auto-discovers subdirectories):
    python compute_cmr_action_stats.py --dataset-path-root /CMR_Versius

Examples:
    python compute_cmr_action_stats.py --dataset-path /CMR_Versius/cholecystectomy_480p
    python compute_cmr_action_stats.py --dataset-path-root /CMR_Versius
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.transform.state_action import (
    apply_motion_scaling_to_hybrid_relative,
    convert_to_hybrid_relative_with_engagement,
)

# Maximum number of workers for parallel processing
MAX_WORKERS = 128


# ============================================================================
# INDEX MAPPINGS FROM CMR VERSIUS info.json
# ============================================================================
# Note: We use modality.json naming convention (with underscores):
#   - "translation_scaling" (modality.json) vs "translationscaling" (info.json)
#   - "rotation_scaling" (modality.json) vs "rotationscaling" (info.json)
# The indices are the same (12 and 13), only the key names differ.
# ============================================================================

# Action array indices (from 100D array)
# Source: info.json -> features.action.names
ACTION_IDX = {
    # Left Hand Controller (0-12)
    "x_left": 0,
    "y_left": 1,
    "z_left": 2,
    "quat_x_left": 3,
    "quat_y_left": 4,
    "quat_z_left": 5,
    "quat_w_left": 6,
    "clutchBtn_left": 7,
    "energyBtn_left": 8,
    "thumbstickBtn_left": 9,
    "pince_left": 10,
    "thumbstick_x_left": 11,
    "thumbstick_y_left": 12,
    # Right Hand Controller (13-25)
    "x_right": 13,
    "y_right": 14,
    "z_right": 15,
    "quat_x_right": 16,
    "quat_y_right": 17,
    "quat_z_right": 18,
    "quat_w_right": 19,
    "clutchBtn_right": 20,
    "energyBtn_right": 21,
    "thumbstickBtn_right": 22,
    "pince_right": 23,
    "thumbstick_x_right": 24,
    "thumbstick_y_right": 25,
}

# State array indices (from 100D array)
# Source: info.json -> features.observation.state.names
# Note: Using underscore naming (translation_scaling) to match modality.json
STATE_IDX = {
    "haptic_left_armengageable": 0,
    "haptic_right_armengageable": 1,
    "arm_0_color": 2,
    "arm_1_color": 3,
    "arm_2_color": 4,
    "arm_3_color": 5,
    "arm_4_color": 6,
    "arm_0_instrtype": 7,
    "arm_1_instrtype": 8,
    "arm_2_instrtype": 9,
    "arm_3_instrtype": 10,
    "arm_4_instrtype": 11,
    "translation_scaling": 12,
    "rotation_scaling": 13,
    "electroSurgeryMode_left": 14,
    "electroSurgeryMode_right": 15,
    "hapticengaged_left": 16,
    "hapticengaged_right": 17,
    "icgmode": 18,
    "icgenabled": 19,
    "armlinkedtohaptic_left": 20,
    "armlinkedtohaptic_right": 21,
    "instrtype_left": 22,
    "instrtype_right": 23,
}


# ============================================================================
# CLUTCH-AWARE FILTERING (matching dataset.py logic)
# ============================================================================
# These filtering rules ensure stats are computed on the same set of valid
# samples that will be used during training. Without this, the stats would
# include invalid samples that get filtered out at load time.
#
# Rule 1: Discard if armlinkedtohaptic_* changes within action horizon (arm swap)
#         - When the surgeon swaps which arm a controller is connected to mid-sequence,
#           the relative action computation becomes invalid.
#
# Rule 2: Discard if completely disengaged for entire action horizon
#         - When both hapticengaged_left and hapticengaged_right are False for the
#           entire action horizon, there's no useful training signal.
# ============================================================================


def filter_valid_samples(
    state_array: np.ndarray,
    action_horizon: int,
    frame_stride: int,
) -> tuple[list[int], dict[str, int]]:
    """
    Filter valid starting indices using CMR clutch-aware rules.
    
    This implements the same filtering logic as dataset.py::_filter_episode_cmr_clutch
    to ensure stats are computed on exactly the same samples used during training.
    
    Args:
        state_array: State array of shape (T, 100)
        action_horizon: Number of action steps
        frame_stride: Frame stride between action steps
    
    Returns:
        valid_indices: List of valid starting indices
        stats: Dict with per-rule filtering counts
    """
    T = len(state_array)
    
    # Extract relevant columns
    engaged_left = state_array[:, STATE_IDX["hapticengaged_left"]]
    engaged_right = state_array[:, STATE_IDX["hapticengaged_right"]]
    arm_linked_left = state_array[:, STATE_IDX["armlinkedtohaptic_left"]]
    arm_linked_right = state_array[:, STATE_IDX["armlinkedtohaptic_right"]]
    
    # Calculate max delta for action horizon
    max_delta = (action_horizon - 1) * frame_stride
    effective_length = max(0, T - max_delta)
    
    # Compute delta indices pattern (matching dataset.py)
    delta_indices_pattern = [i * frame_stride for i in range(action_horizon)]
    
    # Stats tracking
    stats = {
        "total_effective": effective_length,
        "rule1_arm_swap_left": 0,
        "rule1_arm_swap_right": 0,
        "rule2_fully_disengaged": 0,
        "out_of_bounds": 0,
    }
    
    valid_indices = []
    
    for base_idx in range(effective_length):
        # Get indices for entire action horizon
        horizon_indices = np.array([base_idx + delta for delta in delta_indices_pattern])
        
        # Safety check: ensure indices are within bounds
        if horizon_indices[-1] >= T:
            stats["out_of_bounds"] += 1
            continue
        
        # Rule 1: Discard if arm mapping changes within horizon
        if len(np.unique(arm_linked_left[horizon_indices])) > 1:
            stats["rule1_arm_swap_left"] += 1
            continue
        if len(np.unique(arm_linked_right[horizon_indices])) > 1:
            stats["rule1_arm_swap_right"] += 1
            continue
        
        # Rule 2: Discard if completely disengaged for entire horizon
        if not (engaged_left[horizon_indices] > 0.5).any() and not (engaged_right[horizon_indices] > 0.5).any():
            stats["rule2_fully_disengaged"] += 1
            continue
        
        valid_indices.append(base_idx)
    
    stats["valid"] = len(valid_indices)
    return valid_indices, stats


def extract_pose_quat(action_array: np.ndarray, side: str) -> np.ndarray:
    """
    Extract pose (xyz + quat_xyzw) from action array for one arm.
    
    This matches the main pipeline which passes raw quaternions to 
    convert_to_hybrid_relative_with_engagement() with input_rotation_format="quat".
    
    Args:
        action_array: Full action array of shape (T, 100) or (100,)
        side: "left" or "right"
    
    Returns:
        pose: Array of shape (T, 7) or (7,) with xyz (3D) + quat_xyzw (4D)
    """
    if action_array.ndim == 1:
        action_array = action_array.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False
    
    if side == "left":
        xyz = action_array[:, ACTION_IDX["x_left"]:ACTION_IDX["z_left"]+1]  # (T, 3)
        quat = action_array[:, ACTION_IDX["quat_x_left"]:ACTION_IDX["quat_w_left"]+1]  # (T, 4)
    else:  # right
        xyz = action_array[:, ACTION_IDX["x_right"]:ACTION_IDX["z_right"]+1]  # (T, 3)
        quat = action_array[:, ACTION_IDX["quat_x_right"]:ACTION_IDX["quat_w_right"]+1]  # (T, 4)
    
    # Keep as quaternion (xyzw format) to match main pipeline
    # The convert_to_hybrid_relative_with_engagement() will handle quat->rotation_matrix
    pose = np.concatenate([xyz, quat], axis=-1)  # (T, 7)
    
    if squeeze:
        pose = pose.squeeze(0)
    
    return pose


def compute_stats(data: np.ndarray) -> dict:
    """Compute statistics for a numpy array (for small arrays)."""
    return {
        "mean": np.mean(data, axis=0).tolist(),
        "std": np.std(data, axis=0).tolist(),
        "min": np.min(data, axis=0).tolist(),
        "max": np.max(data, axis=0).tolist(),
        "q01": np.quantile(data, 0.01, axis=0).tolist(),
        "q99": np.quantile(data, 0.99, axis=0).tolist(),
    }


class StreamingStats:
    """
    Memory-efficient streaming statistics computation using Welford's algorithm.
    
    This class computes mean, std, min, max without storing all data in memory.
    For quantiles, it uses reservoir sampling to maintain a representative sample.
    
    Suitable for datasets with hundreds of millions of samples.
    """
    
    def __init__(self, num_dims: int, reservoir_size: int = 5_000_000):
        """
        Initialize streaming stats tracker.
        
        Args:
            num_dims: Number of dimensions in the data
            reservoir_size: Number of samples to keep for quantile estimation
        """
        self.num_dims = num_dims
        self.reservoir_size = reservoir_size
        
        # Welford's algorithm state for mean/variance
        self.count = 0
        self.mean = np.zeros(num_dims, dtype=np.float64)
        self.M2 = np.zeros(num_dims, dtype=np.float64)  # Sum of squared differences
        
        # Running min/max
        self.min_vals = np.full(num_dims, np.inf, dtype=np.float64)
        self.max_vals = np.full(num_dims, -np.inf, dtype=np.float64)
        
        # Reservoir for quantile estimation
        self.reservoir = None
        self.reservoir_count = 0
    
    def update(self, batch: np.ndarray):
        """
        Update statistics with a batch of data.
        
        Args:
            batch: Array of shape (N, num_dims)
        """
        if batch.shape[0] == 0:
            return
        
        # Update min/max
        batch_min = np.min(batch, axis=0)
        batch_max = np.max(batch, axis=0)
        self.min_vals = np.minimum(self.min_vals, batch_min)
        self.max_vals = np.maximum(self.max_vals, batch_max)
        
        # Welford's online algorithm for mean and variance (batched)
        for row in batch:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            delta2 = row - self.mean
            self.M2 += delta * delta2
        
        # Reservoir sampling for quantiles
        self._update_reservoir(batch)
    
    def update_fast(self, batch: np.ndarray):
        """
        Fast batch update for mean/variance using numpy operations.
        
        This is faster than row-by-row Welford's but requires the batch to fit in memory.
        Uses Chan's parallel algorithm for combining batch statistics.
        
        Args:
            batch: Array of shape (N, num_dims)
        """
        if batch.shape[0] == 0:
            return
        
        n_batch = batch.shape[0]
        
        # Update min/max
        batch_min = np.min(batch, axis=0)
        batch_max = np.max(batch, axis=0)
        self.min_vals = np.minimum(self.min_vals, batch_min)
        self.max_vals = np.maximum(self.max_vals, batch_max)
        
        # Batch statistics
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0, ddof=0)  # Population variance
        batch_M2 = batch_var * n_batch
        
        # Chan's parallel algorithm to combine with running stats
        if self.count == 0:
            self.mean = batch_mean
            self.M2 = batch_M2
            self.count = n_batch
        else:
            n_total = self.count + n_batch
            delta = batch_mean - self.mean
            self.mean = (self.count * self.mean + n_batch * batch_mean) / n_total
            self.M2 = self.M2 + batch_M2 + delta**2 * self.count * n_batch / n_total
            self.count = n_total
        
        # Reservoir sampling for quantiles
        self._update_reservoir(batch)
    
    def _update_reservoir(self, batch: np.ndarray):
        """Update reservoir with new batch using vectorized Algorithm R."""
        n_batch = batch.shape[0]
        
        if self.reservoir is None:
            # First batch - initialize reservoir
            if n_batch <= self.reservoir_size:
                self.reservoir = batch.copy()
                self.reservoir_count = n_batch
            else:
                # Random sample from first batch
                indices = np.random.choice(n_batch, self.reservoir_size, replace=False)
                self.reservoir = batch[indices].copy()
                self.reservoir_count = n_batch
        elif len(self.reservoir) < self.reservoir_size:
            # Reservoir not full yet - append what we can
            space_left = self.reservoir_size - len(self.reservoir)
            if n_batch <= space_left:
                # Append entire batch
                self.reservoir = np.vstack([self.reservoir, batch])
            else:
                # Append partial, then switch to replacement mode
                self.reservoir = np.vstack([self.reservoir, batch[:space_left]])
                # Process remaining with replacement sampling
                remaining = batch[space_left:]
                self._vectorized_reservoir_update(remaining, self.reservoir_count + space_left)
            self.reservoir_count += n_batch
        else:
            # Reservoir is full - use vectorized replacement sampling
            self._vectorized_reservoir_update(batch, self.reservoir_count)
            self.reservoir_count += n_batch
    
    def _vectorized_reservoir_update(self, batch: np.ndarray, start_count: int):
        """
        Vectorized reservoir sampling for when reservoir is already full.
        
        Uses the fact that for item i (0-indexed in batch), the probability of
        inclusion is reservoir_size / (start_count + i + 1).
        
        For efficiency, we compute which items get selected and their target
        indices all at once using numpy operations.
        """
        n_batch = batch.shape[0]
        if n_batch == 0:
            return
        
        # For each item in batch, compute probability of selection
        # P(select item i) = reservoir_size / (start_count + i + 1)
        item_indices = np.arange(n_batch)
        total_seen = start_count + item_indices + 1
        
        # Generate random values for selection decision
        rand_vals = np.random.random(n_batch)
        selection_probs = self.reservoir_size / total_seen
        
        # Determine which items are selected
        selected_mask = rand_vals < selection_probs
        selected_items = batch[selected_mask]
        
        if len(selected_items) == 0:
            return

        # For selected items, determine which reservoir slot they replace
        # Each selected item uniformly replaces a random slot
        target_slots = np.random.randint(0, self.reservoir_size, size=len(selected_items))
        
        # Apply updates (last write wins for duplicate slots, which is fine)
        self.reservoir[target_slots] = selected_items
    
    def get_stats(self) -> dict:
        """
        Compute final statistics.
        
        Returns:
            Dictionary with mean, std, min, max, q01, q99
        """
        if self.count == 0:
            return {
                "mean": [0.0] * self.num_dims,
                "std": [0.0] * self.num_dims,
                "min": [0.0] * self.num_dims,
                "max": [0.0] * self.num_dims,
                "q01": [0.0] * self.num_dims,
                "q99": [0.0] * self.num_dims,
            }
        
        # Compute std from M2
        variance = self.M2 / self.count
        std = np.sqrt(variance)
        
        # Compute quantiles from reservoir
        if self.reservoir is not None and len(self.reservoir) > 0:
            q01 = np.quantile(self.reservoir, 0.01, axis=0)
            q99 = np.quantile(self.reservoir, 0.99, axis=0)
        else:
            q01 = self.min_vals
            q99 = self.max_vals
        
        return {
            "mean": self.mean.tolist(),
            "std": std.tolist(),
            "min": self.min_vals.tolist(),
            "max": self.max_vals.tolist(),
            "q01": q01.tolist(),
            "q99": q99.tolist(),
        }


def process_episode(
    df: pd.DataFrame,
    action_horizon: int = 12,
    frame_stride: int = 6,
    apply_filtering: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """
    Process a single episode and compute hybrid-relative actions with motion scaling.
    
    Uses RAW PARQUET FORMAT with flat arrays:
      - observation.state: float64[100] array
      - action: float64[100] array
    
    IMPORTANT: When apply_filtering=True, this applies the same clutch-aware filtering
    as the main data loading pipeline (dataset.py). This ensures stats are computed
    on exactly the same set of valid samples used during training.
    
    Args:
        df: DataFrame for one episode (from parquet file)
        action_horizon: Number of action steps (default 12, matching num_frames in main pipeline)
        frame_stride: Frame stride for subsampling (default 6 for 10fps from 60Hz)
        apply_filtering: If True, apply clutch-aware filtering (Rules 1 & 2)
    
    Returns:
        rel_actions: Array of shape (N, 22) with hybrid-relative actions (motion-scaled)
        states: Array of shape (N, state_dim) with state observations
        filter_stats: Dict with filtering statistics for this episode
    """
    T = len(df)
    
    # Stack flat arrays into (T, 100) arrays
    action_array = np.vstack(df["action"].values)  # (T, 100)
    state_array = np.vstack(df["observation.state"].values)  # (T, 100)
    
    # Apply clutch-aware filtering if requested
    if apply_filtering:
        valid_indices, filter_stats = filter_valid_samples(state_array, action_horizon, frame_stride)
    else:
        # Without filtering, use all valid starting indices
        max_delta = (action_horizon - 1) * frame_stride
        effective_length = max(0, T - max_delta)
        valid_indices = list(range(effective_length))
        filter_stats = {
            "total_effective": effective_length,
            "rule1_arm_swap_left": 0,
            "rule1_arm_swap_right": 0,
            "rule2_fully_disengaged": 0,
            "out_of_bounds": 0,
            "valid": effective_length,
        }
    
    if not valid_indices:
        return np.empty((0, 44)), np.empty((0, 16)), filter_stats
    
    # Extract poses (xyz + quat) from action array
    # Keep as quaternion to match main pipeline (groot_configs.py uses input_rotation_format="quat")
    # The convert_to_hybrid_relative_with_engagement() handles quat->rot6d internally
    left_pose = extract_pose_quat(action_array, "left")  # (T, 7)
    right_pose = extract_pose_quat(action_array, "right")  # (T, 7)
    
    # Extract gripper values
    left_gripper = action_array[:, ACTION_IDX["pince_left"]:ACTION_IDX["pince_left"]+1]  # (T, 1)
    right_gripper = action_array[:, ACTION_IDX["pince_right"]:ACTION_IDX["pince_right"]+1]  # (T, 1)
    
    # Extract energy button values
    left_energy = action_array[:, ACTION_IDX["energyBtn_left"]:ACTION_IDX["energyBtn_left"]+1]  # (T, 1)
    right_energy = action_array[:, ACTION_IDX["energyBtn_right"]:ACTION_IDX["energyBtn_right"]+1]  # (T, 1)
    
    # Extract thumbstick values (for endoscope control and instrument straighten function)
    thumbstick_x_left = action_array[:, ACTION_IDX["thumbstick_x_left"]:ACTION_IDX["thumbstick_x_left"]+1]  # (T, 1)
    thumbstick_x_right = action_array[:, ACTION_IDX["thumbstick_x_right"]:ACTION_IDX["thumbstick_x_right"]+1]  # (T, 1)
    thumbstick_y_left = action_array[:, ACTION_IDX["thumbstick_y_left"]:ACTION_IDX["thumbstick_y_left"]+1]  # (T, 1)
    thumbstick_y_right = action_array[:, ACTION_IDX["thumbstick_y_right"]:ACTION_IDX["thumbstick_y_right"]+1]  # (T, 1)
    thumbstickBtn_left = action_array[:, ACTION_IDX["thumbstickBtn_left"]:ACTION_IDX["thumbstickBtn_left"]+1]  # (T, 1)
    thumbstickBtn_right = action_array[:, ACTION_IDX["thumbstickBtn_right"]:ACTION_IDX["thumbstickBtn_right"]+1]  # (T, 1)
    
    # Extract clutch button values (for engage/disengage arm control)
    # NOTE: These are ABSOLUTE values (binary button press), NOT converted to deltas like pose
    clutchBtn_left = action_array[:, ACTION_IDX["clutchBtn_left"]:ACTION_IDX["clutchBtn_left"]+1]  # (T, 1)
    clutchBtn_right = action_array[:, ACTION_IDX["clutchBtn_right"]:ACTION_IDX["clutchBtn_right"]+1]  # (T, 1)
    
    # Extract engagement status from state array (for clutch-aware processing)
    engaged_left = state_array[:, STATE_IDX["hapticengaged_left"]]  # (T,)
    engaged_right = state_array[:, STATE_IDX["hapticengaged_right"]]  # (T,)
    
    # =====================================================
    # STATE CONDITIONING VARIABLES (12D total)
    # These are sampled at action delta_indices and passed through as absolute values
    # =====================================================
    # Haptic engaged state (persistent, unlike clutchBtn which is momentary) - 2D
    cond_hapticengaged_left = state_array[:, STATE_IDX["hapticengaged_left"]:STATE_IDX["hapticengaged_left"]+1]  # (T, 1)
    cond_hapticengaged_right = state_array[:, STATE_IDX["hapticengaged_right"]:STATE_IDX["hapticengaged_right"]+1]  # (T, 1)
    # Which physical arm (0-3) each controller is linked to - 2D
    cond_armlinkedtohaptic_left = state_array[:, STATE_IDX["armlinkedtohaptic_left"]:STATE_IDX["armlinkedtohaptic_left"]+1]  # (T, 1)
    cond_armlinkedtohaptic_right = state_array[:, STATE_IDX["armlinkedtohaptic_right"]:STATE_IDX["armlinkedtohaptic_right"]+1]  # (T, 1)
    # Instrument type for each arm (0-3) - 4D
    cond_arm_0_instrtype = state_array[:, STATE_IDX["arm_0_instrtype"]:STATE_IDX["arm_0_instrtype"]+1]  # (T, 1)
    cond_arm_1_instrtype = state_array[:, STATE_IDX["arm_1_instrtype"]:STATE_IDX["arm_1_instrtype"]+1]  # (T, 1)
    cond_arm_2_instrtype = state_array[:, STATE_IDX["arm_2_instrtype"]:STATE_IDX["arm_2_instrtype"]+1]  # (T, 1)
    cond_arm_3_instrtype = state_array[:, STATE_IDX["arm_3_instrtype"]:STATE_IDX["arm_3_instrtype"]+1]  # (T, 1)
    # HUD color assignment for each arm (0-3) - 4D
    cond_arm_0_color = state_array[:, STATE_IDX["arm_0_color"]:STATE_IDX["arm_0_color"]+1]  # (T, 1)
    cond_arm_1_color = state_array[:, STATE_IDX["arm_1_color"]:STATE_IDX["arm_1_color"]+1]  # (T, 1)
    cond_arm_2_color = state_array[:, STATE_IDX["arm_2_color"]:STATE_IDX["arm_2_color"]+1]  # (T, 1)
    cond_arm_3_color = state_array[:, STATE_IDX["arm_3_color"]:STATE_IDX["arm_3_color"]+1]  # (T, 1)
    # Electrosurgery mode (CUT/COAG) - 2D
    cond_electroSurgeryMode_left = state_array[:, STATE_IDX["electroSurgeryMode_left"]:STATE_IDX["electroSurgeryMode_left"]+1]  # (T, 1)
    cond_electroSurgeryMode_right = state_array[:, STATE_IDX["electroSurgeryMode_right"]:STATE_IDX["electroSurgeryMode_right"]+1]  # (T, 1)
    
    # Extract motion scaling factors from state array
    trans_scaling = state_array[:, STATE_IDX["translation_scaling"]]  # (T,)
    rot_scaling = state_array[:, STATE_IDX["rotation_scaling"]]  # (T,)

    all_rel_actions = []
    all_states = []

    # Process only valid starting points (after filtering)
    for t in valid_indices:
        # Get delta indices for this sample
        delta_indices = [t + i * frame_stride for i in range(action_horizon)]
        
        # Reference state (t=0) - pose from action array (in CMR, action IS the state)
        ref_left_pose = left_pose[t]  # (9,)
        ref_right_pose = right_pose[t]  # (9,)
        ref_engaged_left = bool(engaged_left[t] > 0.5)
        ref_engaged_right = bool(engaged_right[t] > 0.5)
        
        # Action horizon data
        action_left_pose = left_pose[delta_indices]  # (H, 9)
        action_right_pose = right_pose[delta_indices]  # (H, 9)
        action_left_gripper = left_gripper[delta_indices].copy()  # (H, 1)
        action_right_gripper = right_gripper[delta_indices].copy()  # (H, 1)
        action_left_energy = left_energy[delta_indices].copy()  # (H, 1)
        action_right_energy = right_energy[delta_indices].copy()  # (H, 1)
        action_thumbstick_x_left = thumbstick_x_left[delta_indices].copy()  # (H, 1)
        action_thumbstick_x_right = thumbstick_x_right[delta_indices].copy()  # (H, 1)
        action_thumbstick_y_left = thumbstick_y_left[delta_indices].copy()  # (H, 1)
        action_thumbstick_y_right = thumbstick_y_right[delta_indices].copy()  # (H, 1)
        action_thumbstickBtn_left = thumbstickBtn_left[delta_indices].copy()  # (H, 1)
        action_thumbstickBtn_right = thumbstickBtn_right[delta_indices].copy()  # (H, 1)
        # Clutch buttons: ABSOLUTE values (no transformation, just sampled at delta indices)
        action_clutchBtn_left = clutchBtn_left[delta_indices].copy()  # (H, 1) - absolute
        action_clutchBtn_right = clutchBtn_right[delta_indices].copy()  # (H, 1) - absolute
        
        # State conditioning: ABSOLUTE values sampled at action delta indices for MLP conditioning
        action_cond_hapticengaged_left = cond_hapticengaged_left[delta_indices].copy()  # (H, 1)
        action_cond_hapticengaged_right = cond_hapticengaged_right[delta_indices].copy()  # (H, 1)
        action_cond_armlinkedtohaptic_left = cond_armlinkedtohaptic_left[delta_indices].copy()  # (H, 1)
        action_cond_armlinkedtohaptic_right = cond_armlinkedtohaptic_right[delta_indices].copy()  # (H, 1)
        action_cond_arm_0_instrtype = cond_arm_0_instrtype[delta_indices].copy()  # (H, 1)
        action_cond_arm_1_instrtype = cond_arm_1_instrtype[delta_indices].copy()  # (H, 1)
        action_cond_arm_2_instrtype = cond_arm_2_instrtype[delta_indices].copy()  # (H, 1)
        action_cond_arm_3_instrtype = cond_arm_3_instrtype[delta_indices].copy()  # (H, 1)
        action_cond_arm_0_color = cond_arm_0_color[delta_indices].copy()  # (H, 1)
        action_cond_arm_1_color = cond_arm_1_color[delta_indices].copy()  # (H, 1)
        action_cond_arm_2_color = cond_arm_2_color[delta_indices].copy()  # (H, 1)
        action_cond_arm_3_color = cond_arm_3_color[delta_indices].copy()  # (H, 1)
        action_cond_electroSurgeryMode_left = cond_electroSurgeryMode_left[delta_indices].copy()  # (H, 1)
        action_cond_electroSurgeryMode_right = cond_electroSurgeryMode_right[delta_indices].copy()  # (H, 1)
        
        action_engaged_left = engaged_left[delta_indices]  # (H,)
        action_engaged_right = engaged_right[delta_indices]  # (H,)
        
        # Convert to hybrid-relative with engagement awareness
        # Use input_rotation_format="quat" to match main pipeline (groot_configs.py)
        rel_left_pose = convert_to_hybrid_relative_with_engagement(
            action_data=action_left_pose,
            eef_pose=ref_left_pose,
            engaged=action_engaged_left,
            input_rotation_format="quat",  # Match main pipeline
            reference_rotation_format="quat",  # Match main pipeline
            ref_engaged=ref_engaged_left,
        )  # (H, 9) - output is always xyz + rot6d
        
        rel_right_pose = convert_to_hybrid_relative_with_engagement(
            action_data=action_right_pose,
            eef_pose=ref_right_pose,
            engaged=action_engaged_right,
            input_rotation_format="quat",  # Match main pipeline
            reference_rotation_format="quat",  # Match main pipeline
            ref_engaged=ref_engaged_right,
        )  # (H, 9) - output is always xyz + rot6d
        
        # Apply motion scaling to convert from hand-controller-space to instrument-space
        # Motion scaling converts raw controller deltas to actual instrument motion
        # This is critical for computing stats that match the actual action magnitudes
        trans_scale = float(trans_scaling[t])
        rot_scale = float(rot_scaling[t])
        
        rel_left_pose = apply_motion_scaling_to_hybrid_relative(
            rel_left_pose, trans_scale, rot_scale
        )
        rel_right_pose = apply_motion_scaling_to_hybrid_relative(
            rel_right_pose, trans_scale, rot_scale
        )
        
        # Apply sample-and-hold for gripper during clutch
        for i in range(action_horizon):
            if not action_engaged_left[i] > 0.5 and i > 0:
                action_left_gripper[i] = action_left_gripper[i - 1]
            if not action_engaged_right[i] > 0.5 and i > 0:
                action_right_gripper[i] = action_right_gripper[i - 1]
        
        # Zero energy buttons during clutch for safety
        for i in range(action_horizon):
            if not action_engaged_left[i] > 0.5:
                action_left_energy[i] = 0.0
            if not action_engaged_right[i] > 0.5:
                action_right_energy[i] = 0.0
        
        # Concatenate to 44D conditioning: 30D actions + 14D state conditioning
        # === ACTIONS (30D) ===
        # DELTA (relative to reference):
        #   Left pose: xyz_rel(3) + rot6d_rel(6) = 9D
        #   Right pose: xyz_rel(3) + rot6d_rel(6) = 9D
        # ABSOLUTE (raw values, not deltas):
        #   Gripper: left(1) + right(1) = 2D (with sample-and-hold during clutch)
        #   Energy: left(1) + right(1) = 2D (zeroed during clutch)
        #   Thumbstick X: left(1) + right(1) = 2D (pass-through)
        #   Thumbstick Y: left(1) + right(1) = 2D (pass-through)
        #   Thumbstick Button: left(1) + right(1) = 2D (pass-through)
        #   Clutch Button: left(1) + right(1) = 2D (pass-through, binary button press)
        # === STATE CONDITIONING (14D) ===
        #   Haptic engaged: left(1) + right(1) = 2D
        #   Arm linked to haptic: left(1) + right(1) = 2D
        #   Arm instrument type: arm_0(1) + arm_1(1) + arm_2(1) + arm_3(1) = 4D
        #   Arm HUD color: arm_0(1) + arm_1(1) + arm_2(1) + arm_3(1) = 4D
        #   Electrosurgery mode: left(1) + right(1) = 2D
        rel_action = np.concatenate([
            # Actions (30D)
            rel_left_pose,  # (H, 9)
            action_left_gripper,  # (H, 1)
            rel_right_pose,  # (H, 9)
            action_right_gripper,  # (H, 1)
            action_left_energy,  # (H, 1)
            action_right_energy,  # (H, 1)
            action_thumbstick_x_left,  # (H, 1)
            action_thumbstick_x_right,  # (H, 1)
            action_thumbstick_y_left,  # (H, 1)
            action_thumbstick_y_right,  # (H, 1)
            action_thumbstickBtn_left,  # (H, 1)
            action_thumbstickBtn_right,  # (H, 1)
            action_clutchBtn_left,  # (H, 1)
            action_clutchBtn_right,  # (H, 1)
            # State conditioning (12D)
            action_cond_hapticengaged_left,  # (H, 1)
            action_cond_hapticengaged_right,  # (H, 1)
            action_cond_armlinkedtohaptic_left,  # (H, 1)
            action_cond_armlinkedtohaptic_right,  # (H, 1)
            action_cond_arm_0_instrtype,  # (H, 1)
            action_cond_arm_1_instrtype,  # (H, 1)
            action_cond_arm_2_instrtype,  # (H, 1)
            action_cond_arm_3_instrtype,  # (H, 1)
            action_cond_arm_0_color,  # (H, 1)
            action_cond_arm_1_color,  # (H, 1)
            action_cond_arm_2_color,  # (H, 1)
            action_cond_arm_3_color,  # (H, 1)
            action_cond_electroSurgeryMode_left,  # (H, 1)
            action_cond_electroSurgeryMode_right,  # (H, 1)
        ], axis=-1)  # (H, 44)
        
        all_rel_actions.append(rel_action)
        
        # State: reference poses and grippers (absolute values)
        # Uses 7D quat format to match main pipeline (state is NOT transformed by CMRVersiusRelativeActionTransform)
        state = np.concatenate([
            ref_left_pose,  # (7,) - xyz + quat_xyzw
            left_gripper[t].flatten(),  # (1,)
            ref_right_pose,  # (7,) - xyz + quat_xyzw
            right_gripper[t].flatten(),  # (1,)
        ])  # (16,)
        all_states.append(state)
    
    if all_rel_actions:
        # Stack all samples: (N, H, 44) -> flatten to (N*H, 44) for stats
        rel_actions = np.vstack([a.reshape(-1, 44) for a in all_rel_actions])
        states = np.vstack(all_states)
        return rel_actions, states, filter_stats
    else:
        return np.empty((0, 44)), np.empty((0, 16)), filter_stats


def process_parquet_file(
    parquet_path: Path,
    action_horizon: int,
    frame_stride: int,
    apply_filtering: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], str | None]:
    """
    Worker function to process a single parquet file.
    
    This function is designed to be called from ProcessPoolExecutor.
    It loads the parquet file, processes it, and returns the results.
    
    Args:
        parquet_path: Path to the parquet file
        action_horizon: Number of action steps
        frame_stride: Frame stride for subsampling
        apply_filtering: Whether to apply clutch-aware filtering
    
    Returns:
        Tuple of (rel_actions, states, filter_stats, warning_message)
        warning_message is None if no issues, otherwise contains the warning
    """
    try:
        df = pd.read_parquet(parquet_path)
        
        # Verify raw parquet format
        if "observation.state" not in df.columns or "action" not in df.columns:
            return np.empty((0, 44)), np.empty((0, 16)), {}, f"Missing 'observation.state' or 'action' in {parquet_path}"
        
        # Verify array shapes
        sample_state = df["observation.state"].iloc[0]
        sample_action = df["action"].iloc[0]
        if sample_state.shape != (100,) or sample_action.shape != (100,):
            return np.empty((0, 44)), np.empty((0, 16)), {}, f"Unexpected array shapes in {parquet_path}: state={sample_state.shape}, action={sample_action.shape}"
        
        rel_actions, states, filter_stats = process_episode(
            df, action_horizon, frame_stride, apply_filtering=apply_filtering
        )
        
        return rel_actions, states, filter_stats, None
        
    except Exception as e:
        return np.empty((0, 44)), np.empty((0, 16)), {}, f"Error processing {parquet_path}: {e}"


def _is_lerobot_dataset(path: Path) -> bool:
    """Check if a directory is a LeRobot dataset (has data/ and meta/ subdirectories with parquet files)."""
    return (path / "data").is_dir() and (path / "meta").is_dir() and any((path / "data").rglob("*.parquet"))


def _discover_datasets(root: Path) -> list[Path]:
    """Auto-discover LeRobot datasets under a root directory.
    
    Checks each immediate subdirectory of root for the LeRobot dataset structure
    (data/ and meta/ subdirectories with parquet files).
    
    Args:
        root: Root directory containing one or more LeRobot dataset directories
    
    Returns:
        Sorted list of discovered dataset paths
    """
    datasets = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and _is_lerobot_dataset(child):
            datasets.append(child)
    return datasets


# Component slices (used at the end to extract per-component stats from reservoir)
# Actions (30D) + State conditioning (14D) = 44D total
ACTION_SLICES = {
    # === ACTIONS (30D) ===
    "action.left_pose": (0, 9),
    "action.left_gripper": (9, 10),
    "action.right_pose": (10, 19),
    "action.right_gripper": (19, 20),
    "action.left_energy": (20, 21),
    "action.right_energy": (21, 22),
    "action.thumbstick_x_left": (22, 23),
    "action.thumbstick_x_right": (23, 24),
    "action.thumbstick_y_left": (24, 25),
    "action.thumbstick_y_right": (25, 26),
    "action.thumbstickBtn_left": (26, 27),
    "action.thumbstickBtn_right": (27, 28),
    "action.clutchBtn_left": (28, 29),
    "action.clutchBtn_right": (29, 30),
    # === STATE CONDITIONING (14D) ===
    "action.cond_hapticengaged_left": (30, 31),
    "action.cond_hapticengaged_right": (31, 32),
    "action.cond_armlinkedtohaptic_left": (32, 33),
    "action.cond_armlinkedtohaptic_right": (33, 34),
    "action.cond_arm_0_instrtype": (34, 35),
    "action.cond_arm_1_instrtype": (35, 36),
    "action.cond_arm_2_instrtype": (36, 37),
    "action.cond_arm_3_instrtype": (37, 38),
    "action.cond_arm_0_color": (38, 39),
    "action.cond_arm_1_color": (39, 40),
    "action.cond_arm_2_color": (40, 41),
    "action.cond_arm_3_color": (41, 42),
    "action.cond_electroSurgeryMode_left": (42, 43),
    "action.cond_electroSurgeryMode_right": (43, 44),
}

STATE_SLICES = {
    "state.left_pose": (0, 7),
    "state.left_gripper": (7, 8),
    "state.right_pose": (8, 15),
    "state.right_gripper": (15, 16),
}

ACTION_DIM = 44
STATE_DIM = 16
RESERVOIR_SIZE = 5_000_000  # 5M samples for quantile estimation (~1.4GB for actions)


def process_single_dataset(
    dataset_path: Path,
    action_horizon: int,
    frame_stride: int,
    no_filter: bool,
    num_workers: int,
    max_episodes: int | None = None,
    output_path: Path | None = None,
) -> bool:
    """
    Process a single LeRobot dataset and write stats_cosmos-44D.json.
    
    Args:
        dataset_path: Path to the LeRobot dataset
        action_horizon: Number of action steps
        frame_stride: Frame stride for subsampling
        no_filter: If True, disable clutch-aware filtering
        num_workers: Number of parallel workers
        max_episodes: Max episodes to process (for testing)
        output_path: Output path override (default: dataset_path/meta/stats_cosmos-44D.json)
    
    Returns:
        True if successful, False if no valid data found
    """
    parquet_files = sorted(dataset_path.glob("data/*/*.parquet"))

    if not parquet_files:
        print(f"  No parquet files found in {dataset_path / 'data'}, skipping.")
        return False

    if max_episodes:
        parquet_files = parquet_files[:max_episodes]

    # Print configuration header
    print("=" * 80)
    print(f"CMR VERSIUS ACTION STATISTICS — {dataset_path.name}")
    print("=" * 80)
    print(f"Dataset path: {dataset_path}")
    print(f"Number of episodes: {len(parquet_files)}")
    print(f"Action horizon: {action_horizon}")
    print(f"Frame stride: {frame_stride}")
    print(f"Motion scaling: ENABLED (using observation.state indices 12 and 13)")
    print(f"Clutch-aware filtering: {'DISABLED (--no-filter)' if no_filter else 'ENABLED'}")
    print(f"Parallel workers: {num_workers}")
    print("-" * 80)
    if not no_filter:
        print("Filtering rules (matching dataset.py for training consistency):")
        print("  Rule 1: Discard if armlinkedtohaptic changes within horizon (arm swap)")
        print("  Rule 2: Discard if both arms completely disengaged for entire horizon")
    print("-" * 80)
    print("Using RAW PARQUET FORMAT:")
    print("  - observation.state[100] array with index mappings")
    print("  - action[100] array with index mappings")
    print("=" * 80)

    warnings_list = []
    
    # Streaming statistics trackers (memory-efficient)
    action_stats_tracker = StreamingStats(ACTION_DIM, reservoir_size=RESERVOIR_SIZE)
    state_stats_tracker = StreamingStats(STATE_DIM, reservoir_size=RESERVOIR_SIZE)
    
    # Aggregate filtering statistics
    aggregate_filter_stats = {
        "total_effective": 0,
        "rule1_arm_swap_left": 0,
        "rule1_arm_swap_right": 0,
        "rule2_fully_disengaged": 0,
        "out_of_bounds": 0,
        "valid": 0,
    }
    episodes_fully_filtered = 0
    episodes_partially_filtered = 0
    episodes_unfiltered = 0
    episodes_with_errors = 0
    total_action_samples = 0
    total_state_samples = 0

    # Create partial function with fixed arguments
    process_fn = partial(
        process_parquet_file,
        action_horizon=action_horizon,
        frame_stride=frame_stride,
        apply_filtering=not no_filter,
    )

    # Process files in parallel
    print(f"\nProcessing {len(parquet_files)} parquet files with {num_workers} workers...")
    print(f"Using STREAMING STATISTICS (reservoir size: {RESERVOIR_SIZE:,} samples)")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_path = {executor.submit(process_fn, pf): pf for pf in parquet_files}
        
        # Collect results with progress bar
        for future in tqdm(as_completed(future_to_path), total=len(future_to_path), desc=f"Processing {dataset_path.name}"):
            rel_actions, states, filter_stats, warning = future.result()
            
            # Handle warnings/errors
            if warning is not None:
                warnings_list.append(warning)
                episodes_with_errors += 1
                continue
            
            # Aggregate filter stats
            for key in aggregate_filter_stats:
                aggregate_filter_stats[key] += filter_stats.get(key, 0)
            
            # Track episode-level filtering
            if filter_stats.get("valid", 0) == 0:
                episodes_fully_filtered += 1
            elif filter_stats.get("valid", 0) < filter_stats.get("total_effective", 0):
                episodes_partially_filtered += 1
            else:
                episodes_unfiltered += 1
            
            # Update streaming statistics (instead of accumulating in memory)
            if len(rel_actions) > 0:
                total_action_samples += len(rel_actions)
                total_state_samples += len(states)
                
                action_stats_tracker.update_fast(rel_actions)
                state_stats_tracker.update_fast(states)
    
    # Print warnings if any
    if warnings_list:
        print(f"\n⚠️  {len(warnings_list)} warnings during processing:")
        for w in warnings_list[:10]:
            print(f"  - {w}")
        if len(warnings_list) > 10:
            print(f"  ... and {len(warnings_list) - 10} more warnings")

    if total_action_samples == 0:
        print("No valid data found!")
        return False
    
    # Print filtering summary
    if not no_filter:
        total_filtered = aggregate_filter_stats["total_effective"] - aggregate_filter_stats["valid"]
        total_rule1 = aggregate_filter_stats["rule1_arm_swap_left"] + aggregate_filter_stats["rule1_arm_swap_right"]
        
        print("\n" + "=" * 80)
        print("CLUTCH-AWARE FILTERING RESULTS")
        print("=" * 80)
        print(f"SAMPLE STATISTICS:")
        print(f"  Original samples (effective): {aggregate_filter_stats['total_effective']:,}")
        print(f"  Valid samples after filtering: {aggregate_filter_stats['valid']:,}")
        print(f"  Total samples filtered: {total_filtered:,} ({100 * total_filtered / max(1, aggregate_filter_stats['total_effective']):.2f}%)")
        print("-" * 80)
        print(f"PER-RULE BREAKDOWN:")
        print(f"  Rule 1 (arm swap - left):     {aggregate_filter_stats['rule1_arm_swap_left']:,} samples")
        print(f"  Rule 1 (arm swap - right):    {aggregate_filter_stats['rule1_arm_swap_right']:,} samples")
        print(f"  Rule 1 (arm swap - total):    {total_rule1:,} samples ({100 * total_rule1 / max(1, aggregate_filter_stats['total_effective']):.2f}%)")
        print(f"  Rule 2 (fully disengaged):    {aggregate_filter_stats['rule2_fully_disengaged']:,} samples ({100 * aggregate_filter_stats['rule2_fully_disengaged'] / max(1, aggregate_filter_stats['total_effective']):.2f}%)")
        if aggregate_filter_stats["out_of_bounds"] > 0:
            print(f"  Out of bounds (edge cases):   {aggregate_filter_stats['out_of_bounds']:,} samples")
        print("-" * 80)
        print(f"EPISODE STATISTICS:")
        print(f"  Total episodes: {len(parquet_files)}")
        print(f"  Episodes successfully processed: {len(parquet_files) - episodes_with_errors}")
        if episodes_with_errors > 0:
            print(f"  Episodes with errors: {episodes_with_errors} ({100 * episodes_with_errors / max(1, len(parquet_files)):.1f}%)")
        print(f"  Episodes fully filtered (0 valid samples): {episodes_fully_filtered} ({100 * episodes_fully_filtered / max(1, len(parquet_files)):.1f}%)")
        print(f"  Episodes partially filtered: {episodes_partially_filtered} ({100 * episodes_partially_filtered / max(1, len(parquet_files)):.1f}%)")
        print(f"  Episodes unfiltered (all valid): {episodes_unfiltered} ({100 * episodes_unfiltered / max(1, len(parquet_files)):.1f}%)")
        print("=" * 80)
    
    print(f"\nTotal relative action samples: ({total_action_samples:,}, {ACTION_DIM})")
    print(f"Total state samples: ({total_state_samples:,}, {STATE_DIM})")

    # Get stats from streaming trackers (memory-efficient)
    print("\nComputing final statistics from streaming trackers...")
    action_stats = action_stats_tracker.get_stats()
    state_stats = state_stats_tracker.get_stats()
    
    # Build stats dictionary
    stats = {
        "action": action_stats,
        "state": state_stats,
    }
    
    # Compute per-component stats from the full tracker's data
    print("Computing per-component statistics...")
    
    # Action components
    for key, (start, end) in ACTION_SLICES.items():
        component_stats = {
            "mean": action_stats["mean"][start:end],
            "std": action_stats["std"][start:end],
            "min": action_stats["min"][start:end],
            "max": action_stats["max"][start:end],
        }
        if action_stats_tracker.reservoir is not None:
            reservoir_slice = action_stats_tracker.reservoir[:, start:end]
            component_stats["q01"] = np.quantile(reservoir_slice, 0.01, axis=0).tolist()
            component_stats["q99"] = np.quantile(reservoir_slice, 0.99, axis=0).tolist()
        else:
            component_stats["q01"] = component_stats["min"]
            component_stats["q99"] = component_stats["max"]
        stats[key] = component_stats
    
    # State components
    for key, (start, end) in STATE_SLICES.items():
        component_stats = {
            "mean": state_stats["mean"][start:end],
            "std": state_stats["std"][start:end],
            "min": state_stats["min"][start:end],
            "max": state_stats["max"][start:end],
        }
        if state_stats_tracker.reservoir is not None:
            reservoir_slice = state_stats_tracker.reservoir[:, start:end]
            component_stats["q01"] = np.quantile(reservoir_slice, 0.01, axis=0).tolist()
            component_stats["q99"] = np.quantile(reservoir_slice, 0.99, axis=0).tolist()
        else:
            component_stats["q01"] = component_stats["min"]
            component_stats["q99"] = component_stats["max"]
        stats[key] = component_stats

    # Save stats
    final_output_path = output_path if output_path else dataset_path / "meta" / "stats_cosmos-44D.json"
    final_output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(final_output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved stats to {final_output_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Conditioning Stats Summary (44D = 30D actions + 14D state conditioning):")
    print(f"{'='*60}")
    print(f"Total action samples: {total_action_samples:,}")
    print(f"\nLeft Pose (xyz_rel + rot6d_rel, indices 0-8):")
    print(f"  mean: {action_stats['mean'][0:3]} ... (xyz)")
    print(f"  std:  {action_stats['std'][0:3]} ... (xyz)")
    print(f"\nLeft Gripper (index 9):")
    print(f"  mean: {action_stats['mean'][9]:.4f}, std: {action_stats['std'][9]:.4f}")
    print(f"\nRight Pose (xyz_rel + rot6d_rel, indices 10-18):")
    print(f"  mean: {action_stats['mean'][10:13]} ... (xyz)")
    print(f"  std:  {action_stats['std'][10:13]} ... (xyz)")
    print(f"\nRight Gripper (index 19):")
    print(f"  mean: {action_stats['mean'][19]:.4f}, std: {action_stats['std'][19]:.4f}")
    print(f"\nEnergy Buttons (indices 20-21):")
    print(f"  left  - mean: {action_stats['mean'][20]:.4f}, std: {action_stats['std'][20]:.4f}")
    print(f"  right - mean: {action_stats['mean'][21]:.4f}, std: {action_stats['std'][21]:.4f}")
    print(f"\nThumbstick X (indices 22-23):")
    print(f"  left  - mean: {action_stats['mean'][22]:.4f}, std: {action_stats['std'][22]:.4f}")
    print(f"  right - mean: {action_stats['mean'][23]:.4f}, std: {action_stats['std'][23]:.4f}")
    print(f"\nThumbstick Y (indices 24-25):")
    print(f"  left  - mean: {action_stats['mean'][24]:.4f}, std: {action_stats['std'][24]:.4f}")
    print(f"  right - mean: {action_stats['mean'][25]:.4f}, std: {action_stats['std'][25]:.4f}")
    print(f"\nThumbstick Button (indices 26-27):")
    print(f"  left  - mean: {action_stats['mean'][26]:.4f}, std: {action_stats['std'][26]:.4f}")
    print(f"  right - mean: {action_stats['mean'][27]:.4f}, std: {action_stats['std'][27]:.4f}")
    
    print(f"\n{'='*60}")
    print("State Stats Summary (16D - xyz + quat for each arm):")
    print(f"{'='*60}")
    print(f"Total state samples: {total_state_samples:,}")
    
    # Final consistency note
    if not no_filter:
        print(f"\n{'='*60}")
        print("CONSISTENCY NOTE")
        print(f"{'='*60}")
        print("These statistics were computed with clutch-aware filtering ENABLED.")
        print("This matches the filtering in the training data loading pipeline")
        print("(dataset.py::_filter_episode_cmr_clutch), ensuring that:")
        print("  - Stats represent the SAME samples used during training")
        print("  - Normalization will be accurate for valid training data")
        print("  - No distribution mismatch between stats and actual data")
        print(f"{'='*60}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Compute normalization stats for CMR Versius hybrid-relative actions",
        epilog=(
            "Examples:\n"
            "  Single dataset:\n"
            "    python compute_cmr_action_stats.py --dataset-path /CMR_Versius/cholecystectomy_480p\n\n"
            "  All datasets under a root directory:\n"
            "    python compute_cmr_action_stats.py --dataset-path-root /CMR_Versius\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    path_group = parser.add_mutually_exclusive_group(required=True)
    path_group.add_argument("--dataset-path", type=str, help="Path to a single LeRobot dataset")
    path_group.add_argument(
        "--dataset-path-root", type=str,
        help="Root directory containing multiple LeRobot datasets (auto-discovers subdirectories with data/ and meta/)"
    )
    parser.add_argument("--output", type=str, default=None, help="Output path override (only for --dataset-path mode; default: dataset_path/meta/stats_cosmos-44D.json)")
    parser.add_argument("--action-horizon", type=int, default=12, help="Action horizon (default: 12, matching main pipeline's num_frames)")
    parser.add_argument("--frame-stride", type=int, default=6, help="Frame stride (default: 6 for 10fps from 60Hz)")
    parser.add_argument("--max-episodes", type=int, default=None, help="Max episodes to process per dataset (for testing)")
    parser.add_argument("--no-filter", action="store_true", help="Disable clutch-aware filtering (not recommended)")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of parallel workers (default: min(cpu_count, MAX_WORKERS))")
    args = parser.parse_args()
    
    # Determine number of workers
    if args.num_workers is None:
        num_workers = min(os.cpu_count() or 8, MAX_WORKERS)
    else:
        num_workers = args.num_workers

    # Resolve dataset paths
    if args.dataset_path:
        # Single dataset mode
        dataset_paths = [Path(args.dataset_path)]
        if not dataset_paths[0].exists():
            print(f"ERROR: Dataset path does not exist: {dataset_paths[0]}")
            return
    else:
        # Root directory mode — auto-discover datasets
        root = Path(args.dataset_path_root)
        if not root.exists():
            print(f"ERROR: Root directory does not exist: {root}")
            return
        
        # Check if root itself is a dataset
        if _is_lerobot_dataset(root):
            print(f"NOTE: {root} is itself a LeRobot dataset. Processing it directly.")
            dataset_paths = [root]
        else:
            dataset_paths = _discover_datasets(root)
        
        if not dataset_paths:
            print(f"ERROR: No LeRobot datasets found under {root}")
            print(f"  (Expected subdirectories with data/ and meta/ containing parquet files)")
            return
        
        if args.output:
            print("WARNING: --output is ignored in --dataset-path-root mode (each dataset gets its own stats file)")
        
        print("#" * 80)
        print(f"AUTO-DISCOVERED {len(dataset_paths)} DATASETS UNDER {root}:")
        for i, dp in enumerate(dataset_paths, 1):
            n_parquets = len(list(dp.glob("data/*/*.parquet")))
            print(f"  {i}. {dp.name} ({n_parquets} episodes)")
        print("#" * 80)

    # Process each dataset
    total_start = time.time()
    results = {}
    
    for i, dataset_path in enumerate(dataset_paths, 1):
        if len(dataset_paths) > 1:
            print(f"\n{'#' * 80}")
            print(f"# DATASET {i}/{len(dataset_paths)}: {dataset_path.name}")
            print(f"{'#' * 80}")
        
        output_path = Path(args.output) if (args.output and len(dataset_paths) == 1) else None
        
        success = process_single_dataset(
            dataset_path=dataset_path,
            action_horizon=args.action_horizon,
            frame_stride=args.frame_stride,
            no_filter=args.no_filter,
            num_workers=num_workers,
            max_episodes=args.max_episodes,
            output_path=output_path,
        )
        results[dataset_path.name] = "✅ OK" if success else "❌ FAILED (no valid data)"
    
    total_elapsed = time.time() - total_start
    
    # Print overall summary for multi-dataset mode
    if len(dataset_paths) > 1:
        print(f"\n{'#' * 80}")
        print(f"# ALL DATASETS COMPLETE — {total_elapsed:.1f}s total")
        print(f"{'#' * 80}")
        for name, status in results.items():
            print(f"  {name}: {status}")
        print(f"{'#' * 80}")


if __name__ == "__main__":
    main()

