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
Test script to analyze whether 6D rotation (rot6d) should be normalized.

BACKGROUND:
The 6D rotation representation (rot6d) consists of the first two columns of a 
rotation matrix, flattened. By definition, these values are bounded to [-1, 1]
since they are elements of an orthonormal matrix.

In the CMR Versius hybrid-relative action space:
- Translation delta (xyz): unbounded, varies based on motion
- Rotation delta (rot6d): bounded to [-1, 1] by definition
- Gripper (pince): bounded to [0, 1]
- Energy button: binary {0, 1}

QUESTION:
Should we normalize rot6d values using mean/std like other action dimensions?
Or should we skip normalization since they're already bounded?

This script:
1. Collects hybrid-relative rot6d values from the CMR dataset
2. Plots the raw (unnormalized) distribution - should be in [-1, 1]
3. Computes mean/std and plots normalized distribution
4. Compares the two to help decide on normalization strategy

Usage:
    python test_cmr_rot6d_distr.py --dataset-path /path/to/cmr_dataset
    python test_cmr_rot6d_distr.py --dataset-path /path/to/cmr_dataset --max-episodes 100
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import the rotation conversion functions
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.transform.state_action import (
    apply_motion_scaling_to_hybrid_relative,
    convert_to_hybrid_relative_with_engagement,
    quat_to_rotation_matrix,
    rotation_matrix_to_rot6d,
)


# ============================================================================
# INDEX MAPPINGS FROM CMR VERSIUS info.json
# ============================================================================

ACTION_IDX = {
    "x_left": 0, "y_left": 1, "z_left": 2,
    "quat_x_left": 3, "quat_y_left": 4, "quat_z_left": 5, "quat_w_left": 6,
    "clutchBtn_left": 7, "energyBtn_left": 8, "pince_left": 10,
    "x_right": 13, "y_right": 14, "z_right": 15,
    "quat_x_right": 16, "quat_y_right": 17, "quat_z_right": 18, "quat_w_right": 19,
    "clutchBtn_right": 20, "energyBtn_right": 21, "pince_right": 23,
}

STATE_IDX = {
    "translation_scaling": 12,
    "rotation_scaling": 13,
    "hapticengaged_left": 16,
    "hapticengaged_right": 17,
}


def quat_xyzw_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (xyzw format) to 6D rotation representation."""
    if quat.ndim == 1:
        quat = quat.reshape(1, 4)
        squeeze = True
    else:
        squeeze = False
    
    batch_shape = quat.shape[:-1]
    quat_flat = quat.reshape(-1, 4)
    
    rot6d_list = []
    for i in range(len(quat_flat)):
        rot_mat = quat_to_rotation_matrix(quat_flat[i])
        rot6d = rotation_matrix_to_rot6d(rot_mat)
        rot6d_list.append(rot6d)
    
    rot6d = np.stack(rot6d_list, axis=0)
    rot6d = rot6d.reshape(*batch_shape, 6)
    
    if squeeze:
        rot6d = rot6d.squeeze(0)
    
    return rot6d


def extract_pose_rot6d(action_array: np.ndarray, side: str) -> np.ndarray:
    """Extract pose (xyz + rot6d) from action array for one arm."""
    if action_array.ndim == 1:
        action_array = action_array.reshape(1, -1)
        squeeze = True
            else:
        squeeze = False
    
    if side == "left":
        xyz = action_array[:, ACTION_IDX["x_left"]:ACTION_IDX["z_left"]+1]
        quat = action_array[:, ACTION_IDX["quat_x_left"]:ACTION_IDX["quat_w_left"]+1]
            else:
        xyz = action_array[:, ACTION_IDX["x_right"]:ACTION_IDX["z_right"]+1]
        quat = action_array[:, ACTION_IDX["quat_x_right"]:ACTION_IDX["quat_w_right"]+1]
    
    rot6d = quat_xyzw_to_rot6d(quat)
    pose = np.concatenate([xyz, rot6d], axis=-1)
    
    if squeeze:
        pose = pose.squeeze(0)
    
    return pose


def collect_rot6d_samples(
    dataset_path: Path,
    max_episodes: int | None = None,
    action_horizon: int = 12,
    frame_stride: int = 6,
) -> dict:
    """
    Collect hybrid-relative rot6d samples from the dataset.
    
    Returns dict with:
        - raw_rot6d_left: (N, 6) raw rot6d values for left arm
        - raw_rot6d_right: (N, 6) raw rot6d values for right arm
        - rel_rot6d_left: (N, 6) hybrid-relative rot6d deltas for left arm
        - rel_rot6d_right: (N, 6) hybrid-relative rot6d deltas for right arm
        - xyz_left: (N, 3) translation deltas for left arm (for comparison)
        - xyz_right: (N, 3) translation deltas for right arm
    """
    parquet_files = sorted(dataset_path.glob("data/*/*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {dataset_path / 'data'}")
    
    if max_episodes:
        parquet_files = parquet_files[:max_episodes]
    
    print(f"Processing {len(parquet_files)} episodes...")
    
    # Accumulators
    raw_rot6d_left = []
    raw_rot6d_right = []
    rel_rot6d_left = []
    rel_rot6d_right = []
    rel_xyz_left = []
    rel_xyz_right = []
    
    for pf in tqdm(parquet_files, desc="Collecting rot6d samples"):
        df = pd.read_parquet(pf)
        T = len(df)
        
        if T < action_horizon * frame_stride:
            continue
        
        # Stack arrays
        action_array = np.vstack(df["action"].values)
        state_array = np.vstack(df["observation.state"].values)
        
        # Extract poses in rot6d format
        left_pose = extract_pose_rot6d(action_array, "left")
        right_pose = extract_pose_rot6d(action_array, "right")
        
        # Get engagement and scaling
        engaged_left = state_array[:, STATE_IDX["hapticengaged_left"]]
        engaged_right = state_array[:, STATE_IDX["hapticengaged_right"]]
        trans_scaling = state_array[:, STATE_IDX["translation_scaling"]]
        rot_scaling = state_array[:, STATE_IDX["rotation_scaling"]]
        
        # Collect raw rot6d values
        raw_rot6d_left.append(left_pose[:, 3:9])  # rot6d part
        raw_rot6d_right.append(right_pose[:, 3:9])
        
        # Compute hybrid-relative deltas
        max_delta = (action_horizon - 1) * frame_stride
        
        for t in range(0, T - max_delta, frame_stride * 2):  # Sample every 2 action horizons
            delta_indices = [t + i * frame_stride for i in range(action_horizon)]
            
            ref_left_pose = left_pose[t]
            ref_right_pose = right_pose[t]
            ref_engaged_left = bool(engaged_left[t] > 0.5)
            ref_engaged_right = bool(engaged_right[t] > 0.5)
            
            action_left_pose = left_pose[delta_indices]
            action_right_pose = right_pose[delta_indices]
            action_engaged_left = engaged_left[delta_indices]
            action_engaged_right = engaged_right[delta_indices]
            
            # Convert to hybrid-relative
            try:
                rel_left = convert_to_hybrid_relative_with_engagement(
                    action_data=action_left_pose,
                    eef_pose=ref_left_pose,
                    engaged=action_engaged_left,
                    input_rotation_format="rot6d",
                    reference_rotation_format="rot6d",
                    ref_engaged=ref_engaged_left,
                )
                rel_right = convert_to_hybrid_relative_with_engagement(
                    action_data=action_right_pose,
                    eef_pose=ref_right_pose,
                    engaged=action_engaged_right,
                    input_rotation_format="rot6d",
                    reference_rotation_format="rot6d",
                    ref_engaged=ref_engaged_right,
                )
                
                # Apply motion scaling
                trans_scale = float(trans_scaling[t])
                rot_scale = float(rot_scaling[t])
                
                rel_left = apply_motion_scaling_to_hybrid_relative(rel_left, trans_scale, rot_scale)
                rel_right = apply_motion_scaling_to_hybrid_relative(rel_right, trans_scale, rot_scale)
                
                # Extract xyz and rot6d separately
                rel_xyz_left.append(rel_left[:, 0:3])
                rel_xyz_right.append(rel_right[:, 0:3])
                rel_rot6d_left.append(rel_left[:, 3:9])
                rel_rot6d_right.append(rel_right[:, 3:9])
                
            except Exception as e:
                continue
    
    # Stack all samples
    return {
        "raw_rot6d_left": np.vstack(raw_rot6d_left),
        "raw_rot6d_right": np.vstack(raw_rot6d_right),
        "rel_rot6d_left": np.vstack(rel_rot6d_left),
        "rel_rot6d_right": np.vstack(rel_rot6d_right),
        "rel_xyz_left": np.vstack(rel_xyz_left),
        "rel_xyz_right": np.vstack(rel_xyz_right),
    }


def plot_distributions(samples: dict, output_path: Path):
    """
    Create distribution plots comparing normalized vs unnormalized rot6d.
    """
    # Combine left and right for analysis
    rel_rot6d = np.vstack([samples["rel_rot6d_left"], samples["rel_rot6d_right"]])
    rel_xyz = np.vstack([samples["rel_xyz_left"], samples["rel_xyz_right"]])
    raw_rot6d = np.vstack([samples["raw_rot6d_left"], samples["raw_rot6d_right"]])
    
    print(f"\n{'='*70}")
    print("ROT6D STATISTICS")
    print("="*70)
    print(f"\nTotal samples: {len(rel_rot6d)}")
    
    # Raw rot6d statistics
    print(f"\n--- RAW ROT6D (absolute, not delta) ---")
    print(f"Min:  {raw_rot6d.min(axis=0)}")
    print(f"Max:  {raw_rot6d.max(axis=0)}")
    print(f"Mean: {raw_rot6d.mean(axis=0)}")
    print(f"Std:  {raw_rot6d.std(axis=0)}")
    
    # Relative rot6d statistics
    print(f"\n--- HYBRID-RELATIVE ROT6D (delta, motion-scaled) ---")
    print(f"Min:  {rel_rot6d.min(axis=0)}")
    print(f"Max:  {rel_rot6d.max(axis=0)}")
    print(f"Mean: {rel_rot6d.mean(axis=0)}")
    print(f"Std:  {rel_rot6d.std(axis=0)}")
    
    # Check if values are in [-1, 1]
    in_range = (rel_rot6d >= -1).all() and (rel_rot6d <= 1).all()
    print(f"\nAll values in [-1, 1]: {in_range}")
    if not in_range:
        out_of_range = np.sum((rel_rot6d < -1) | (rel_rot6d > 1))
        total = rel_rot6d.size
        print(f"  Out of range: {out_of_range}/{total} ({100*out_of_range/total:.2f}%)")
        print(f"  Actual range: [{rel_rot6d.min():.4f}, {rel_rot6d.max():.4f}]")
    
    # Translation statistics for comparison
    print(f"\n--- HYBRID-RELATIVE XYZ (translation delta, motion-scaled) ---")
    print(f"Min:  {rel_xyz.min(axis=0)}")
    print(f"Max:  {rel_xyz.max(axis=0)}")
    print(f"Mean: {rel_xyz.mean(axis=0)}")
    print(f"Std:  {rel_xyz.std(axis=0)}")
    
    # Compute normalization stats for rot6d
    rot6d_mean = rel_rot6d.mean(axis=0)
    rot6d_std = rel_rot6d.std(axis=0)
    
    # Normalize rot6d
    normalized_rot6d = (rel_rot6d - rot6d_mean) / (rot6d_std + 1e-8)
    
    print(f"\n--- NORMALIZED ROT6D (mean/std) ---")
    print(f"Min:  {normalized_rot6d.min(axis=0)}")
    print(f"Max:  {normalized_rot6d.max(axis=0)}")
    print(f"Mean: {normalized_rot6d.mean(axis=0)}")
    print(f"Std:  {normalized_rot6d.std(axis=0)}")
    
    # =========================================================================
    # PLOT 1: Distribution of each rot6d component (unnormalized vs normalized)
    # =========================================================================
    fig, axes = plt.subplots(2, 6, figsize=(18, 8))
    fig.suptitle("6D Rotation Distribution: Unnormalized vs Normalized", fontsize=14)
    
    component_names = ["r00", "r10", "r20", "r01", "r11", "r21"]
    
    for i in range(6):
        # Unnormalized
        ax = axes[0, i]
        ax.hist(rel_rot6d[:, i], bins=100, density=True, alpha=0.7, color='blue')
        ax.axvline(x=-1, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=1, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='green', linestyle='--', alpha=0.5)
        ax.set_title(f"{component_names[i]} (raw)")
        ax.set_xlabel("Value")
        if i == 0:
            ax.set_ylabel("Density")
        
        # Normalized
        ax = axes[1, i]
        ax.hist(normalized_rot6d[:, i], bins=100, density=True, alpha=0.7, color='orange')
        ax.axvline(x=0, color='green', linestyle='--', alpha=0.5)
        ax.set_title(f"{component_names[i]} (normalized)")
        ax.set_xlabel("Value")
        if i == 0:
            ax.set_ylabel("Density")
    
    plt.tight_layout()
    plt.savefig(output_path / "rot6d_components_distribution.png", dpi=150)
    print(f"\nSaved: {output_path / 'rot6d_components_distribution.png'}")
    
    # =========================================================================
    # PLOT 2: Combined distribution of all rot6d values
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Unnormalized rot6d
    ax = axes[0]
    ax.hist(rel_rot6d.flatten(), bins=200, density=True, alpha=0.7, color='blue')
    ax.axvline(x=-1, color='red', linestyle='--', linewidth=2, label='[-1, 1] bounds')
    ax.axvline(x=1, color='red', linestyle='--', linewidth=2)
    ax.axvline(x=rel_rot6d.mean(), color='green', linestyle='-', linewidth=2, label=f'mean={rel_rot6d.mean():.4f}')
    ax.set_title("Hybrid-Relative Rot6D (Unnormalized)")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_xlim(-1.5, 1.5)
    
    # Normalized rot6d
    ax = axes[1]
    ax.hist(normalized_rot6d.flatten(), bins=200, density=True, alpha=0.7, color='orange')
    ax.axvline(x=0, color='green', linestyle='-', linewidth=2, label='mean=0')
    ax.set_title("Hybrid-Relative Rot6D (Normalized)")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    
    # XYZ for comparison
    ax = axes[2]
    ax.hist(rel_xyz.flatten(), bins=200, density=True, alpha=0.7, color='green')
    ax.axvline(x=rel_xyz.mean(), color='red', linestyle='-', linewidth=2, label=f'mean={rel_xyz.mean():.4f}')
    ax.set_title("Hybrid-Relative XYZ (Translation)")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / "rot6d_vs_xyz_distribution.png", dpi=150)
    print(f"Saved: {output_path / 'rot6d_vs_xyz_distribution.png'}")
    
    # =========================================================================
    # PLOT 3: Box plots comparing distributions
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Rot6d components
    ax = axes[0]
    
    # Define colors
    color_unnorm = '#3274A1'  # Blue
    color_norm = '#E1812C'    # Orange
    
    bp1 = ax.boxplot([rel_rot6d[:, i] for i in range(6)], 
                      positions=np.arange(6)*2, widths=0.6,
                      patch_artist=True, showfliers=False)  # Hide outliers for cleaner plot
    bp2 = ax.boxplot([normalized_rot6d[:, i] for i in range(6)], 
                      positions=np.arange(6)*2+0.7, widths=0.6,
                      patch_artist=True, showfliers=False)
    
    # Set colors for all elements of bp1 (unnormalized - blue)
    for element in ['boxes', 'whiskers', 'caps', 'medians']:
        for item in bp1[element]:
            if element == 'boxes':
                item.set_facecolor(color_unnorm)
                item.set_alpha(0.6)
            item.set_color(color_unnorm)
            item.set_linewidth(1.5)
    
    # Set colors for all elements of bp2 (normalized - orange)
    for element in ['boxes', 'whiskers', 'caps', 'medians']:
        for item in bp2[element]:
            if element == 'boxes':
                item.set_facecolor(color_norm)
                item.set_alpha(0.6)
            item.set_color(color_norm)
            item.set_linewidth(1.5)
    
    ax.set_xticks(np.arange(6)*2+0.35)
    ax.set_xticklabels(component_names)
    ax.axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='[-1, 1] bounds')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.5)
    ax.set_title("Rot6D Components: Unnormalized (blue) vs Normalized (orange)")
    ax.set_ylabel("Value")
    
    # Create custom legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_unnorm, alpha=0.6, label='Unnormalized'),
        Patch(facecolor=color_norm, alpha=0.6, label='Normalized'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # XYZ vs Rot6d magnitude
    ax = axes[1]
    xyz_mag = np.linalg.norm(rel_xyz, axis=1)
    rot6d_mag = np.linalg.norm(rel_rot6d, axis=1)
    ax.boxplot([xyz_mag, rot6d_mag], labels=['XYZ magnitude', 'Rot6D magnitude'])
    ax.set_title("Action Magnitude Comparison")
    ax.set_ylabel("L2 Norm")
    
    plt.tight_layout()
    plt.savefig(output_path / "rot6d_boxplots.png", dpi=150)
    print(f"Saved: {output_path / 'rot6d_boxplots.png'}")
    
    # =========================================================================
    # PLOT 4: Scatter plot of rot6d values
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Sample a subset for scatter plot
    n_scatter = min(10000, len(rel_rot6d))
    idx = np.random.choice(len(rel_rot6d), n_scatter, replace=False)
    
    # First two components
    ax = axes[0]
    ax.scatter(rel_rot6d[idx, 0], rel_rot6d[idx, 1], alpha=0.1, s=1)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--')
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_title("Rot6D: r00 vs r10 (should lie on unit sphere)")
    ax.set_xlabel("r00")
    ax.set_ylabel("r10")
    
    # r00 vs r20
    ax = axes[1]
    ax.scatter(rel_rot6d[idx, 0], rel_rot6d[idx, 2], alpha=0.1, s=1)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--')
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_title("Rot6D: r00 vs r20")
    ax.set_xlabel("r00")
    ax.set_ylabel("r20")
    
    plt.tight_layout()
    plt.savefig(output_path / "rot6d_scatter.png", dpi=150)
    print(f"Saved: {output_path / 'rot6d_scatter.png'}")
    
    plt.close('all')
    
    # =========================================================================
    # Analysis Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print("ANALYSIS SUMMARY")
    print("="*70)
    
    print("""
KEY OBSERVATIONS:

1. VALUE RANGE:
   - Rot6d values are derived from rotation matrix columns
   - By mathematical definition, individual elements are in [-1, 1]
   - However, HYBRID-RELATIVE rot6d may exceed [-1, 1] due to:
     * Motion scaling (rotation_scaling factor)
     * Delta computation between orientations
""")
    
    actual_in_range = (rel_rot6d >= -1).all() and (rel_rot6d <= 1).all()
    if actual_in_range:
        print("   ✓ All hybrid-relative rot6d values are in [-1, 1]")
        print("   → Normalization may not be necessary")
    else:
        pct_out = 100 * np.sum((rel_rot6d < -1) | (rel_rot6d > 1)) / rel_rot6d.size
        print(f"   ⚠ {pct_out:.2f}% of values are outside [-1, 1]")
        print(f"   → Consider normalization to handle outliers")
    
    print(f"""
2. DISTRIBUTION SHAPE:
   - Unnormalized rot6d: centered around {rel_rot6d.mean():.4f}, std={rel_rot6d.std():.4f}
   - After normalization: centered at 0, std=1
   
3. COMPARISON WITH XYZ:
   - XYZ (translation) range: [{rel_xyz.min():.4f}, {rel_xyz.max():.4f}]
   - XYZ mean: {rel_xyz.mean():.4f}, std: {rel_xyz.std():.4f}
   - Rot6d and XYZ have DIFFERENT scales
   
4. RECOMMENDATION:
""")
    
    xyz_std = rel_xyz.std()
    rot6d_std_mean = rel_rot6d.std()
    
    if actual_in_range and rot6d_std_mean < 0.5:
        print("""   SKIP normalization for rot6d because:
   - Values are already bounded to [-1, 1]
   - Distribution is already well-behaved
   - Normalizing would change the inherent meaning of rot6d
   
   Instead, consider:
   - Using 'min_max' normalization to ensure [-1, 1] → [-1, 1]
   - Or 'skip' normalization entirely for rot6d components""")
    else:
        print("""   APPLY normalization for rot6d because:
   - Some values exceed [-1, 1] bounds (due to delta computation)
   - Helps align rot6d scale with other normalized components
   
   However, be careful:
   - Normalized values may lose the [-1, 1] semantic meaning
   - During inference, need to denormalize correctly""")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze rot6d distribution to decide on normalization"
    )
    parser.add_argument("--dataset-path", type=str, required=True, 
                        help="Path to CMR Versius LeRobot dataset")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Output path for plots (default: dataset_path)")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Max episodes to process (for faster testing)")
    parser.add_argument("--action-horizon", type=int, default=12,
                        help="Action horizon (default: 12)")
    parser.add_argument("--frame-stride", type=int, default=6,
                        help="Frame stride (default: 6)")
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path) if args.output_path else dataset_path
    
    if not dataset_path.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("ROT6D NORMALIZATION ANALYSIS")
    print("="*70)
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_path}")
    
    # Collect samples
    samples = collect_rot6d_samples(
        dataset_path=dataset_path,
        max_episodes=args.max_episodes,
        action_horizon=args.action_horizon,
        frame_stride=args.frame_stride,
    )
    
    print(f"\nCollected samples:")
    for key, arr in samples.items():
        print(f"  {key}: {arr.shape}")
    
    # Plot and analyze
    plot_distributions(samples, output_path)
    
    # Save statistics to JSON
    stats = {
        "rel_rot6d": {
            "mean": samples["rel_rot6d_left"].mean(axis=0).tolist() + samples["rel_rot6d_right"].mean(axis=0).tolist(),
            "std": samples["rel_rot6d_left"].std(axis=0).tolist() + samples["rel_rot6d_right"].std(axis=0).tolist(),
            "min": samples["rel_rot6d_left"].min(axis=0).tolist() + samples["rel_rot6d_right"].min(axis=0).tolist(),
            "max": samples["rel_rot6d_left"].max(axis=0).tolist() + samples["rel_rot6d_right"].max(axis=0).tolist(),
        },
        "rel_xyz": {
            "mean": samples["rel_xyz_left"].mean(axis=0).tolist() + samples["rel_xyz_right"].mean(axis=0).tolist(),
            "std": samples["rel_xyz_left"].std(axis=0).tolist() + samples["rel_xyz_right"].std(axis=0).tolist(),
            "min": samples["rel_xyz_left"].min(axis=0).tolist() + samples["rel_xyz_right"].min(axis=0).tolist(),
            "max": samples["rel_xyz_left"].max(axis=0).tolist() + samples["rel_xyz_right"].max(axis=0).tolist(),
        },
    }
    
    with open(output_path / "rot6d_analysis_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved: {output_path / 'rot6d_analysis_stats.json'}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
