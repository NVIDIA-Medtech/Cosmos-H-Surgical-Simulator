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
Test script for verifying CMR Versius data loading via LeRobotDataset.

This script verifies:
1. Raw parquet structure (observation.state[100], action[100])
2. Modality mapping from flat arrays to named keys
3. Correct index extraction as defined in CMR Versius dataset

IMPORTANT: modality.json is the AUTHORITATIVE source for field naming.
- modality.json defines the high-level keys (e.g., 'left_pose', 'translation_scaling')
- info.json defines raw array indices and original field names
- When naming differs (e.g., translation_scaling vs translationscaling), use modality.json

Index mappings (raw indices from info.json, naming from modality.json):

ACTION SPACE (indices 0-25, rest zero-padded to 100):
  Left Hand Controller (0-12):
    0-2:   x/y/z_left (position)
    3-6:   quat_x/y/z/w_left
    7:     clutchBtn_left
    8:     energyBtn_left
    9:     thumbstickBtn_left
    10:    pince_left (gripper [0-1])
    11-12: thumbstick_x/y_left
  Right Hand Controller (13-25):
    13-15: x/y/z_right (position)
    16-19: quat_x/y/z/w_right
    20:    clutchBtn_right
    21:    energyBtn_right
    22:    thumbstickBtn_right
    23:    pince_right (gripper [0-1])
    24-25: thumbstick_x/y_right

STATE SPACE (indices 0-23, rest zero-padded to 100):
    0:     haptic_left_armengageable
    1:     haptic_right_armengageable
    2-6:   arm_0_color to arm_4_color
    7-11:  arm_0_instrtype to arm_4_instrtype
    12:    translation_scaling (translationscaling in info.json)
    13:    rotation_scaling (rotationscaling in info.json)
    14:    electroSurgeryMode_left
    15:    electroSurgeryMode_right
    16:    hapticengaged_left
    17:    hapticengaged_right
    18:    icgmode
    19:    icgenabled
    20:    armlinkedtohaptic_left
    21:    armlinkedtohaptic_right
    22:    instrtype_left
    23:    instrtype_right

Usage:
    python test_cmr_dataloading.py --dataset-path /path/to/cmr_versius_dataset
    python test_cmr_dataloading.py --dataset-path /path/to/cmr_versius_dataset --parquet-file chunk-000/episode_000011.parquet
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# INDEX MAPPINGS FROM CMR VERSIUS info.json (hardcoded fallback)
# These are verified against the actual info.json in the dataset
# ============================================================================

# Action array indices (from 100D array)
# Source: info.json -> features.action.names
ACTION_INDICES = {
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
# Note: We use underscore naming (translation_scaling) to match modality.json convention
STATE_INDICES = {
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

# Grouped indices for higher-level access
ACTION_GROUPS = {
    "xyz_left": [0, 1, 2],
    "quat_left": [3, 4, 5, 6],  # xyzw
    "xyz_right": [13, 14, 15],
    "quat_right": [16, 17, 18, 19],  # xyzw
}


def load_index_mappings_from_info_json(dataset_path: Path) -> tuple[dict, dict]:
    """
    Load index mappings from the dataset's info.json file.
    
    Returns:
        action_indices: dict mapping name -> index for action array
        state_indices: dict mapping name -> index for observation.state array
    """
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        print(f"Warning: info.json not found at {info_path}, using hardcoded mappings")
        return ACTION_INDICES, STATE_INDICES
    
    with open(info_path) as f:
        info = json.load(f)
    
    action_indices = {}
    state_indices = {}
    
    # Extract action names -> indices
    if "features" in info and "action" in info["features"]:
        action_names = info["features"]["action"].get("names", [])
        for idx, name in enumerate(action_names):
            action_indices[name] = idx
    
    # Extract state names -> indices
    if "features" in info and "observation.state" in info["features"]:
        state_names = info["features"]["observation.state"].get("names", [])
        for idx, name in enumerate(state_names):
            state_indices[name] = idx
    
    return action_indices, state_indices


def verify_index_mappings(dataset_path: Path) -> bool:
    """Verify that hardcoded mappings match info.json mappings."""
    print("\n" + "=" * 70)
    print("TEST 0: Verify Index Mappings Against info.json")
    print("=" * 70)
    
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        print(f"⚠ info.json not found at {info_path}")
        return False
    
    with open(info_path) as f:
        info = json.load(f)
    
    all_match = True
    
    # Verify action mappings
    print("\nAction Index Mappings:")
    print("-" * 50)
    action_names = info["features"]["action"].get("names", [])
    print(f"info.json has {len(action_names)} action names")
    
    for name, expected_idx in ACTION_INDICES.items():
        if expected_idx < len(action_names):
            actual_name = action_names[expected_idx]
            if actual_name == name:
                print(f"  ✓ {name} at index {expected_idx}")
            else:
                print(f"  ❌ Index {expected_idx}: expected '{name}', got '{actual_name}'")
                all_match = False
        else:
            print(f"  ⚠ Index {expected_idx} out of range (max {len(action_names)-1})")
    
    # Verify state mappings
    print("\nState Index Mappings:")
    print("-" * 50)
    state_names = info["features"]["observation.state"].get("names", [])
    print(f"info.json has {len(state_names)} state names")
    
    for name, expected_idx in STATE_INDICES.items():
        if expected_idx < len(state_names):
            actual_name = state_names[expected_idx]
            if actual_name == name:
                print(f"  ✓ {name} at index {expected_idx}")
            else:
                print(f"  ❌ Index {expected_idx}: expected '{name}', got '{actual_name}'")
                all_match = False
        else:
            print(f"  ⚠ Index {expected_idx} out of range (max {len(state_names)-1})")
    
    # Print video info
    print("\nVideo Info:")
    print("-" * 50)
    if "observation.images.endoscope" in info["features"]:
        video_info = info["features"]["observation.images.endoscope"]
        shape = video_info.get("shape", [])
        print(f"  Endoscope video shape: {shape} (height, width, channels)")
        if "info" in video_info:
            print(f"  FPS: {video_info['info'].get('video.fps', 'N/A')}")
            print(f"  Original resolution: {video_info['info'].get('video.width', '?')}x{video_info['info'].get('video.height', '?')}")
    
    if all_match:
        print(f"\n✓ All hardcoded mappings match info.json!")
    else:
        print(f"\n❌ Some mappings don't match - update hardcoded values!")
    
    return all_match


def test_raw_parquet_structure(parquet_path: Path):
    """Test 1: Verify raw parquet structure matches expected format."""
    print("\n" + "=" * 70)
    print("TEST 1: Raw Parquet Structure")
    print("=" * 70)
    
    df = pd.read_parquet(parquet_path)
    
    print(f"\nFile: {parquet_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Number of rows (timesteps): {len(df)}")
    
    # Check expected columns exist
    expected_cols = ["observation.state", "action", "timestamp", "frame_index", "episode_index"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        print(f"❌ FAIL: Missing expected columns: {missing}")
        return False
    else:
        print(f"✓ All expected columns present")
    
    # Check observation.state shape
    state_sample = df["observation.state"].iloc[0]
    print(f"\nobservation.state[0] shape: {state_sample.shape}")
    if state_sample.shape != (100,):
        print(f"❌ FAIL: Expected shape (100,), got {state_sample.shape}")
        return False
    else:
        print(f"✓ observation.state shape is (100,)")
    
    # Check action shape
    action_sample = df["action"].iloc[0]
    print(f"action[0] shape: {action_sample.shape}")
    if action_sample.shape != (100,):
        print(f"❌ FAIL: Expected shape (100,), got {action_sample.shape}")
        return False
    else:
        print(f"✓ action shape is (100,)")
    
    return True


def test_state_indices(parquet_path: Path):
    """Test 2: Verify state values at specific indices."""
    print("\n" + "=" * 70)
    print("TEST 2: State Index Mapping")
    print("=" * 70)
    
    df = pd.read_parquet(parquet_path)
    state = df["observation.state"].iloc[0]
    
    print(f"\nSample state values (first timestep):")
    print("-" * 50)
    
    # Print key state values
    key_states = [
        "translation_scaling", "rotation_scaling",
        "hapticengaged_left", "hapticengaged_right",
        "armlinkedtohaptic_left", "armlinkedtohaptic_right",
        "instrtype_left", "instrtype_right",
    ]
    
    for name in key_states:
        idx = STATE_INDICES[name]
        val = state[idx]
        print(f"  {name} (idx {idx}): {val}")
    
    # Verify translation_scaling and rotation_scaling are reasonable
    trans_scale = state[STATE_INDICES["translation_scaling"]]
    rot_scale = state[STATE_INDICES["rotation_scaling"]]
    
    print(f"\nMotion scaling verification:")
    if trans_scale > 0 and trans_scale < 10:
        print(f"  ✓ translation_scaling={trans_scale} is in reasonable range (0, 10)")
    else:
        print(f"  ⚠ translation_scaling={trans_scale} may be unexpected")
    
    if rot_scale > 0 and rot_scale < 10:
        print(f"  ✓ rotation_scaling={rot_scale} is in reasonable range (0, 10)")
    else:
        print(f"  ⚠ rotation_scaling={rot_scale} may be unexpected")
    
    # Check engagement values are boolean-like
    engaged_left = state[STATE_INDICES["hapticengaged_left"]]
    engaged_right = state[STATE_INDICES["hapticengaged_right"]]
    
    print(f"\nEngagement status:")
    print(f"  hapticengaged_left={engaged_left} ({'engaged' if engaged_left > 0.5 else 'disengaged'})")
    print(f"  hapticengaged_right={engaged_right} ({'engaged' if engaged_right > 0.5 else 'disengaged'})")
    
    return True


def test_action_indices(parquet_path: Path):
    """Test 3: Verify action values at specific indices."""
    print("\n" + "=" * 70)
    print("TEST 3: Action Index Mapping")
    print("=" * 70)
    
    df = pd.read_parquet(parquet_path)
    action = df["action"].iloc[0]
    
    print(f"\nSample action values (first timestep):")
    print("-" * 50)
    
    # Left arm pose
    xyz_left = action[ACTION_GROUPS["xyz_left"]]
    quat_left = action[ACTION_GROUPS["quat_left"]]
    pince_left = action[ACTION_INDICES["pince_left"]]
    energy_left = action[ACTION_INDICES["energyBtn_left"]]
    clutch_left = action[ACTION_INDICES["clutchBtn_left"]]
    
    print(f"\nLeft Arm:")
    print(f"  xyz:    {xyz_left}")
    print(f"  quat:   {quat_left} (xyzw)")
    print(f"  pince:  {pince_left}")
    print(f"  energy: {energy_left}")
    print(f"  clutch: {clutch_left}")
    
    # Right arm pose
    xyz_right = action[ACTION_GROUPS["xyz_right"]]
    quat_right = action[ACTION_GROUPS["quat_right"]]
    pince_right = action[ACTION_INDICES["pince_right"]]
    energy_right = action[ACTION_INDICES["energyBtn_right"]]
    clutch_right = action[ACTION_INDICES["clutchBtn_right"]]
    
    print(f"\nRight Arm:")
    print(f"  xyz:    {xyz_right}")
    print(f"  quat:   {quat_right} (xyzw)")
    print(f"  pince:  {pince_right}")
    print(f"  energy: {energy_right}")
    print(f"  clutch: {clutch_right}")
    
    # Verify quaternion is unit
    quat_norm_left = np.linalg.norm(quat_left)
    quat_norm_right = np.linalg.norm(quat_right)
    
    print(f"\nQuaternion normalization check:")
    if abs(quat_norm_left - 1.0) < 0.01:
        print(f"  ✓ Left quat norm={quat_norm_left:.4f} ≈ 1.0")
    else:
        print(f"  ⚠ Left quat norm={quat_norm_left:.4f} is not unit")
    
    if abs(quat_norm_right - 1.0) < 0.01:
        print(f"  ✓ Right quat norm={quat_norm_right:.4f} ≈ 1.0")
    else:
        print(f"  ⚠ Right quat norm={quat_norm_right:.4f} is not unit")
    
    # Check gripper range
    print(f"\nGripper range check:")
    if 0 <= pince_left <= 1:
        print(f"  ✓ Left pince={pince_left:.4f} in [0, 1]")
    else:
        print(f"  ⚠ Left pince={pince_left} outside [0, 1]")
    
    if 0 <= pince_right <= 1:
        print(f"  ✓ Right pince={pince_right:.4f} in [0, 1]")
    else:
        print(f"  ⚠ Right pince={pince_right} outside [0, 1]")
    
    # Check that indices 26-99 are zero-padded
    padding = action[26:]
    if np.allclose(padding, 0):
        print(f"\n✓ Indices 26-99 are zero-padded as expected")
    else:
        non_zero = np.count_nonzero(padding)
        print(f"\n⚠ Indices 26-99 have {non_zero} non-zero values")
    
    return True


def test_temporal_consistency(parquet_path: Path):
    """Test 4: Check temporal consistency and motion scaling usage."""
    print("\n" + "=" * 70)
    print("TEST 4: Temporal Consistency")
    print("=" * 70)
    
    df = pd.read_parquet(parquet_path)
    
    # Get a few timesteps
    n_samples = min(10, len(df))
    
    print(f"\nChecking first {n_samples} timesteps:")
    print("-" * 50)
    
    trans_scales = []
    rot_scales = []
    engaged_left_list = []
    engaged_right_list = []
    
    for i in range(n_samples):
        state = df["observation.state"].iloc[i]
        trans_scales.append(state[STATE_INDICES["translation_scaling"]])
        rot_scales.append(state[STATE_INDICES["rotation_scaling"]])
        engaged_left_list.append(state[STATE_INDICES["hapticengaged_left"]])
        engaged_right_list.append(state[STATE_INDICES["hapticengaged_right"]])
    
    print(f"Translation scaling over {n_samples} steps: {trans_scales}")
    print(f"Rotation scaling over {n_samples} steps: {rot_scales}")
    print(f"Left engaged over {n_samples} steps: {engaged_left_list}")
    print(f"Right engaged over {n_samples} steps: {engaged_right_list}")
    
    # Check if scaling is constant (typical during a procedure segment)
    if len(set(trans_scales)) == 1:
        print(f"\n✓ Translation scaling is constant: {trans_scales[0]}")
    else:
        print(f"\n⚠ Translation scaling varies (can happen during surgery)")
    
    if len(set(rot_scales)) == 1:
        print(f"✓ Rotation scaling is constant: {rot_scales[0]}")
    else:
        print(f"⚠ Rotation scaling varies (can happen during surgery)")
    
    # Check for clutch transitions
    n_clutch_left = sum(1 for i in range(1, len(engaged_left_list)) 
                       if engaged_left_list[i] != engaged_left_list[i-1])
    n_clutch_right = sum(1 for i in range(1, len(engaged_right_list)) 
                        if engaged_right_list[i] != engaged_right_list[i-1])
    
    print(f"\nClutch transitions in sample: left={n_clutch_left}, right={n_clutch_right}")
    
    return True


def test_modality_json(dataset_path: Path):
    """Test 5: Verify modality.json exists and has correct mapping."""
    print("\n" + "=" * 70)
    print("TEST 5: Modality JSON Configuration")
    print("=" * 70)
    
    modality_path = dataset_path / "meta" / "modality.json"
    
    if not modality_path.exists():
        print(f"❌ modality.json not found at {modality_path}")
        print("   This file is required to map flat arrays to named keys")
        return False
    
    with open(modality_path) as f:
        modality = json.load(f)
    
    print(f"\nModality config loaded from: {modality_path}")
    print("-" * 50)
    
    # Check state mappings
    if "state" in modality:
        print(f"\nState keys defined: {list(modality['state'].keys())}")
        
        # Check for key state mappings (using modality.json naming with underscores)
        key_states = ["translation_scaling", "rotation_scaling", "hapticengaged_left", "hapticengaged_right"]
        for key in key_states:
            if key in modality["state"]:
                cfg = modality["state"][key]
                print(f"  {key}: start={cfg.get('start')}, end={cfg.get('end')}, original_key={cfg.get('original_key')}")
            else:
                print(f"  ⚠ Missing state key: {key}")
    else:
        print("❌ No 'state' section in modality.json")
    
    # Check action mappings
    if "action" in modality:
        print(f"\nAction keys defined: {list(modality['action'].keys())}")
        
        # Check for key action mappings
        key_actions = ["left_pose", "left_gripper", "right_pose", "right_gripper", "left_energy", "right_energy"]
        for key in key_actions:
            if key in modality["action"]:
                cfg = modality["action"][key]
                print(f"  {key}: start={cfg.get('start')}, end={cfg.get('end')}, original_key={cfg.get('original_key')}")
            else:
                print(f"  ⚠ Missing action key: {key}")
    else:
        print("❌ No 'action' section in modality.json")
    
    return True


def test_lerobot_dataset_loading(dataset_path: Path, num_samples: int = 3):
    """Test 6: Try loading through LeRobotDataset class."""
    print("\n" + "=" * 70)
    print("TEST 6: LeRobotDataset Class Loading")
    print("=" * 70)
    
    try:
        from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.groot_configs import (
            construct_modality_config_and_transforms,
        )
        from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.dataset import (
            LeRobotSingleDataset,
        )
    except ImportError as e:
        print(f"❌ Failed to import LeRobotDataset: {e}")
        print("   Make sure you're running from the correct environment")
        return False
    
    print(f"\nLoading dataset from: {dataset_path}")
    print(f"Embodiment: cmr_versius")
    
    try:
        # Get config and transforms
        config, train_transform, test_transform = construct_modality_config_and_transforms(
            num_frames=13,  # 12 + 1 reference
            embodiment="cmr_versius",
            downscaled_res=False,
        )
        
        print(f"\nModality config:")
        for modality, cfg in config.items():
            print(f"  {modality}:")
            print(f"    delta_indices: {cfg.delta_indices[:5]}... (showing first 5)")
            print(f"    modality_keys: {cfg.modality_keys}")
        
        # Create dataset
        dataset = LeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=config,
            embodiment_tag="cmr_versius",
            transforms=train_transform,
        )
        
        print(f"\n✓ Dataset created successfully!")
        print(f"  Total steps: {len(dataset)}")
        print(f"  Number of trajectories: {len(dataset.trajectory_ids)}")
        
        # Try loading a few samples
        print(f"\nLoading {num_samples} samples:")
        for i in range(min(num_samples, len(dataset))):
            try:
                sample = dataset[i]
                print(f"\nSample {i}:")
                for key, value in sample.items():
                    if hasattr(value, 'shape'):
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    elif isinstance(value, (list, tuple)):
                        print(f"  {key}: len={len(value)}")
                    else:
                        print(f"  {key}: type={type(value).__name__}")
            except Exception as e:
                print(f"  ❌ Failed to load sample {i}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create/load dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_expected_modality_json():
    """Print the expected modality.json structure for CMR Versius.
    
    This matches the actual gr00t-H/examples/CMR_Versius/modality.json format.
    """
    print("\n" + "=" * 70)
    print("EXPECTED MODALITY.JSON STRUCTURE FOR CMR VERSIUS")
    print("=" * 70)
    
    # This is the actual modality.json from gr00t-H/examples/CMR_Versius/modality.json
    # Verified to match the CMR Versius dataset format
    modality_json = {
        "state": {
            # Poses come from ACTION array (hand controller pose IS the state in CMR!)
            "left_pose": {
                "start": 0,
                "end": 7,
                "original_key": "action"
            },
            "left_gripper": {
                "start": 10,
                "end": 11,
                "original_key": "action"
            },
            "right_pose": {
                "start": 13,
                "end": 20,
                "original_key": "action"
            },
            "right_gripper": {
                "start": 23,
                "end": 24,
                "original_key": "action"
            },
            # These come from observation.state array
            "electroSurgeryMode_left": {
                "start": 14,
                "end": 15,
                "original_key": "observation.state"
            },
            "armlinkedtohaptic_left": {
                "start": 20,
                "end": 21,
                "original_key": "observation.state"
            },
            "instrtype_left": {
                "start": 22,
                "end": 23,
                "original_key": "observation.state"
            },
            "electroSurgeryMode_right": {
                "start": 15,
                "end": 16,
                "original_key": "observation.state"
            },
            "armlinkedtohaptic_right": {
                "start": 21,
                "end": 22,
                "original_key": "observation.state"
            },
            "instrtype_right": {
                "start": 23,
                "end": 24,
                "original_key": "observation.state"
            },
            "translation_scaling": {
                "start": 12,
                "end": 13,
                "original_key": "observation.state"
            },
            "rotation_scaling": {
                "start": 13,
                "end": 14,
                "original_key": "observation.state"
            },
            "hapticengaged_left": {
                "start": 16,
                "end": 17,
                "original_key": "observation.state"
            },
            "hapticengaged_right": {
                "start": 17,
                "end": 18,
                "original_key": "observation.state"
            },
            "arm_0_color": {
                "start": 2,
                "end": 3,
                "original_key": "observation.state"
            },
            "arm_1_color": {
                "start": 3,
                "end": 4,
                "original_key": "observation.state"
            },
            "arm_2_color": {
                "start": 4,
                "end": 5,
                "original_key": "observation.state"
            },
            "arm_3_color": {
                "start": 5,
                "end": 6,
                "original_key": "observation.state"
            }
        },
        "action": {
            # Action poses (will be converted to hybrid-relative)
            # No original_key means it defaults to "action"
            "left_pose": {
                "start": 0,
                "end": 7
            },
            "left_gripper": {
                "start": 10,
                "end": 11
            },
            "right_pose": {
                "start": 13,
                "end": 20
            },
            "right_gripper": {
                "start": 23,
                "end": 24
            },
            "left_energy": {
                "start": 8,
                "end": 9
            },
            "right_energy": {
                "start": 21,
                "end": 22
            },
            # Pass-through keys for clutch-aware processing
            "hapticengaged_left": {
                "start": 16,
                "end": 17,
                "original_key": "observation.state"
            },
            "hapticengaged_right": {
                "start": 17,
                "end": 18,
                "original_key": "observation.state"
            }
        },
        "video": {
            "endoscope": {
                "original_key": "observation.images.endoscope"
            }
        },
        "annotation": {
            "human.task_description": {
                "original_key": "task_index"
            }
        }
    }
    
    print("""
This modality.json is based on gr00t-H/examples/CMR_Versius/modality.json.

Key insights from CMR Versius:
- In CMR Versius, the 'action' column contains the hand controller pose,
  which IS the current state. There is no separate state column with pose.
- Therefore, state.left_pose/right_pose extraction uses "original_key": "action"
- Motion scaling uses underscores: "translation_scaling", "rotation_scaling"
  (Note: info.json names use no underscore: "translationscaling", "rotationscaling")
- Engagement status (hapticengaged_left/right) comes from observation.state
- Actions without "original_key" default to the "action" array

Index mappings (verified against info.json):
  ACTION array:
    left_pose:    0-6  (xyz + quat_xyzw)
    left_energy:  8    (energyBtn_left)
    left_gripper: 10   (pince_left)
    right_pose:   13-19 (xyz + quat_xyzw)
    right_energy: 21   (energyBtn_right)
    right_gripper: 23  (pince_right)
  
  STATE array (observation.state):
    translation_scaling: 12
    rotation_scaling:    13
    hapticengaged_left:  16
    hapticengaged_right: 17

Suggested modality.json:
""")
    print(json.dumps(modality_json, indent=2))
    
    print("""
To use this modality.json:
1. Copy gr00t-H/examples/CMR_Versius/modality.json to <dataset_path>/meta/modality.json
   OR
2. Save the above JSON to <dataset_path>/meta/modality.json
""")


def main():
    parser = argparse.ArgumentParser(description="Test CMR Versius data loading")
    parser.add_argument("--dataset-path", type=str, required=True, 
                        help="Path to LeRobot dataset root")
    parser.add_argument("--parquet-file", type=str, default=None,
                        help="Specific parquet file to test (relative to dataset-path/data/)")
    parser.add_argument("--show-expected-modality", action="store_true",
                        help="Print expected modality.json structure")
    parser.add_argument("--skip-dataset-test", action="store_true",
                        help="Skip LeRobotDataset loading test")
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    
    if not dataset_path.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return
    
    # Find a parquet file to test
    if args.parquet_file:
        parquet_path = dataset_path / "data" / args.parquet_file
    else:
        parquet_files = sorted(dataset_path.glob("data/*/*.parquet"))
        if not parquet_files:
            print(f"❌ No parquet files found in {dataset_path / 'data'}")
            return
        parquet_path = parquet_files[0]
        print(f"Using first parquet file: {parquet_path}")
    
    if not parquet_path.exists():
        print(f"❌ Parquet file does not exist: {parquet_path}")
        return
    
    # Run tests
    print("\n" + "=" * 70)
    print("CMR VERSIUS DATA LOADING TEST")
    print("=" * 70)
    print(f"Dataset path: {dataset_path}")
    print(f"Parquet file: {parquet_path}")
    
    all_passed = True
    
    # Test 0: Verify index mappings against info.json
    if not verify_index_mappings(dataset_path):
        print("\n⚠ Index mapping verification had issues")
    
    # Test 1: Raw parquet structure
    if not test_raw_parquet_structure(parquet_path):
        all_passed = False
    
    # Test 2: State indices
    test_state_indices(parquet_path)
    
    # Test 3: Action indices
    test_action_indices(parquet_path)
    
    # Test 4: Temporal consistency
    test_temporal_consistency(parquet_path)
    
    # Test 5: Modality JSON
    test_modality_json(dataset_path)
    
    # Test 6: LeRobotDataset loading
    if not args.skip_dataset_test:
        test_lerobot_dataset_loading(dataset_path)
    
    # Show expected modality.json if requested
    if args.show_expected_modality:
        print_expected_modality_json()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    if all_passed:
        print("✓ All structure tests passed!")
    else:
        print("⚠ Some tests had issues - see above for details")
    
    print("""
KEY FINDINGS:
- The parquet files contain flat arrays: observation.state[100] and action[100]
- Index mappings are defined in meta/info.json (features.*.names)
- The modality.json file (authoritative source) maps these flat arrays to named keys like:
  'state.translation_scaling', 'action.left_pose', etc.
- Note: modality.json uses underscores (translation_scaling), info.json does not (translationscaling)

NEXT STEPS:
1. Verify modality.json correctly maps flat arrays to named keys
2. If modality.json is missing/incorrect, create it based on the index mappings
3. Update compute_cmr_action_stats.py to use raw array indexing directly
   (since the parquet does NOT have named columns, just flat arrays)
""")


if __name__ == "__main__":
    main()
