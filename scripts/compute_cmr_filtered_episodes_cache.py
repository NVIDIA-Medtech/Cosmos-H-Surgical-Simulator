#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Pre-compute CMR Versius clutch-aware filtering cache for training.

This script must be run BEFORE training to generate the filter cache files
that the training pipeline expects. Without these cache files, training will
fail to start.

The cache files are stored in each dataset's meta/ directory:
    {dataset_path}/meta/cmr_filter_cache_{split}_{hash}.json

The hash is based on the action_delta_indices configuration to ensure cache
invalidation when hyperparameters change.

Usage:
    # Compute cache for all CMR datasets with default settings (13 frames, train split)
    python scripts/compute_cmr_filtered_episodes_cache.py

    # Compute cache for specific dataset
    python scripts/compute_cmr_filtered_episodes_cache.py \
        --dataset-path /CMR_Versius/cholecystectomy_480p

    # Compute cache for test split
    python scripts/compute_cmr_filtered_episodes_cache.py --split test

    # Compute cache with custom frame count
    python scripts/compute_cmr_filtered_episodes_cache.py --num-frames 25

Example output:
    Computing filter cache for: /CMR_Versius/cholecystectomy_480p
    [CMR Filter] Cache will be saved to: .../cmr_filter_cache_train_abc123def456.json
    ...
    [CMR Filter] Cached filter results to: .../cmr_filter_cache_train_abc123def456.json
"""

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

# =============================================================================
# CMR Versius Clutch-Aware Filtering Constants
# =============================================================================
# These indices correspond to the raw observation.state array in CMR Versius parquet files.
# Used for load-time filtering to discard invalid training samples.
#
# From info.json -> features.observation.state.names:
#   Index 16: hapticengaged_left   - Whether left controller is actively controlling an arm
#   Index 17: hapticengaged_right  - Whether right controller is actively controlling an arm
#   Index 20: armlinkedtohaptic_left  - Which arm (0-3) the left controller maps to
#   Index 21: armlinkedtohaptic_right - Which arm (0-3) the right controller maps to
#
# References:
#   - gr00t-H/working_log/01_13_26/clutch_problem.md
#   - gr00t-H/gr00t/data/dataset/sharded_single_step_dataset.py::_filter_episode_fast
# =============================================================================

# These indices correspond to the raw observation.state array in CMR Versius parquet files.
CMR_RAW_INDEX_HAPTIC_ENGAGED_LEFT = 16
CMR_RAW_INDEX_HAPTIC_ENGAGED_RIGHT = 17
CMR_RAW_INDEX_ARM_LINKED_LEFT = 20
CMR_RAW_INDEX_ARM_LINKED_RIGHT = 21

CMR_MAX_FILTER_WORKERS = 64

# Default CMR dataset paths (matches data.py configuration)
DEFAULT_CMR_DATASET_PATHS = [
    "/CMR_Versius/cholecystectomy_480p",
    "/CMR_Versius/hysterectomy_480p",
    "/CMR_Versius/inguinal_hernia_480p",
    "/CMR_Versius/prostatectomy_480p",
]


def _filter_episode_cmr_clutch(
    episode_idx: int,
    dataset_path: Path,
    chunk_size: int,
    data_path_pattern: str,
    action_delta_indices: list[int],
    episode_length: int,
) -> tuple[int, list[int], dict[str, int]]:
    """Filter a single episode for CMR clutch-aware training sample validity.

    Uses PyArrow for fast column-only loading of observation.state to check clutch
    conditions. This is much faster than loading full DataFrame with video decoding.

    Filtering Rules:
      Rule 1: Discard if armlinkedtohaptic_* changes within action horizon (arm swap)
      Rule 2: Discard if completely disengaged for entire action horizon

    Args:
        episode_idx: Index of the episode to filter
        dataset_path: Path to the dataset root
        chunk_size: Number of episodes per chunk (for parquet path calculation)
        data_path_pattern: Pattern string for parquet file paths
        action_delta_indices: Delta indices from action modality config
        episode_length: Total length of the episode

    Returns:
        Tuple of (episode_idx, list of valid base_indices, stats_dict)
    """
    stats = {
        "rule1_arm_swap_left": 0,
        "rule1_arm_swap_right": 0,
        "rule2_fully_disengaged": 0,
        "out_of_bounds": 0,
    }

    chunk_idx = episode_idx // chunk_size
    parquet_path = dataset_path / data_path_pattern.format(
        episode_chunk=chunk_idx, episode_index=episode_idx
    )

    if not parquet_path.exists():
        return episode_idx, [], stats

    try:
        table = pq.read_table(parquet_path, columns=["observation.state"])
        state_data = table.column("observation.state").to_pylist()
    except Exception:
        return episode_idx, [], stats

    engaged_left = np.array(
        [s[CMR_RAW_INDEX_HAPTIC_ENGAGED_LEFT] for s in state_data], dtype=bool
    )
    engaged_right = np.array(
        [s[CMR_RAW_INDEX_HAPTIC_ENGAGED_RIGHT] for s in state_data], dtype=bool
    )
    arm_linked_left = np.array(
        [s[CMR_RAW_INDEX_ARM_LINKED_LEFT] for s in state_data], dtype=np.float32
    )
    arm_linked_right = np.array(
        [s[CMR_RAW_INDEX_ARM_LINKED_RIGHT] for s in state_data], dtype=np.float32
    )

    max_delta = max(action_delta_indices) if action_delta_indices else 0
    effective_length = max(0, episode_length - max_delta)

    valid_indices = []
    for base_idx in range(effective_length):
        horizon_indices = np.array([base_idx + delta for delta in action_delta_indices])

        if horizon_indices[-1] >= len(state_data):
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
        if not engaged_left[horizon_indices].any() and not engaged_right[horizon_indices].any():
            stats["rule2_fully_disengaged"] += 1
            continue

        valid_indices.append(base_idx)

    return episode_idx, valid_indices, stats


def filter_cmr_clutch_all_episodes(
    dataset_path: Path,
    trajectory_ids: np.ndarray,
    trajectory_lengths: np.ndarray,
    chunk_size: int,
    data_path_pattern: str,
    action_delta_indices: list[int],
    num_workers: int | None = None,
) -> dict[int, list[int]]:
    """Filter all episodes in parallel for CMR clutch-aware validity."""
    import os

    if num_workers is None:
        num_workers = min(os.cpu_count() or 8, CMR_MAX_FILTER_WORKERS)

    filter_fn = partial(
        _filter_episode_cmr_clutch,
        dataset_path=dataset_path,
        chunk_size=chunk_size,
        data_path_pattern=data_path_pattern,
        action_delta_indices=action_delta_indices,
    )

    max_delta = max(action_delta_indices) if action_delta_indices else 0
    print("=" * 80)
    print("CMR VERSIUS CLUTCH-AWARE FILTERING")
    print("=" * 80)
    print(f"  Dataset path: {dataset_path}")
    print(f"  Number of episodes: {len(trajectory_ids)}")
    print(f"  Total frames: {np.sum(trajectory_lengths):,}")
    print(f"  Action horizon (max delta): {max_delta}")
    print(f"  Action delta indices: {action_delta_indices[:5]}{'...' if len(action_delta_indices) > 5 else ''}")
    print(f"  Parallel workers: {num_workers}")
    print("-" * 80)
    print("Filtering rules applied:")
    print("  Rule 1: Discard if armlinkedtohaptic changes within horizon (arm swap)")
    print("  Rule 2: Discard if both arms completely disengaged for entire horizon")
    print("-" * 80)

    results = {}
    total_original = 0
    total_valid = 0
    episodes_fully_filtered = 0
    episodes_partially_filtered = 0
    episodes_unfiltered = 0

    aggregate_stats = {
        "rule1_arm_swap_left": 0,
        "rule1_arm_swap_right": 0,
        "rule2_fully_disengaged": 0,
        "out_of_bounds": 0,
    }

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(filter_fn, int(ep_idx), episode_length=int(ep_len)): (int(ep_idx), int(ep_len))
            for ep_idx, ep_len in zip(trajectory_ids, trajectory_lengths)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Filtering episodes"):
            ep_idx, valid_indices, ep_stats = future.result()
            _, ep_len = futures[future]

            effective_len = max(0, ep_len - max_delta)
            total_original += effective_len
            total_valid += len(valid_indices)
            results[ep_idx] = valid_indices

            for key in aggregate_stats:
                aggregate_stats[key] += ep_stats.get(key, 0)

            if len(valid_indices) == 0:
                episodes_fully_filtered += 1
            elif len(valid_indices) < effective_len:
                episodes_partially_filtered += 1
            else:
                episodes_unfiltered += 1

    total_filtered = total_original - total_valid
    total_rule1 = aggregate_stats["rule1_arm_swap_left"] + aggregate_stats["rule1_arm_swap_right"]

    print("=" * 80)
    print("CMR CLUTCH FILTERING RESULTS")
    print("=" * 80)
    print(f"SAMPLE STATISTICS:")
    print(f"  Original samples (effective): {total_original:,}")
    print(f"  Valid samples after filtering: {total_valid:,}")
    print(f"  Total samples filtered: {total_filtered:,} ({100 * total_filtered / max(1, total_original):.2f}%)")
    print("-" * 80)
    print(f"PER-RULE BREAKDOWN:")
    print(f"  Rule 1 (arm swap - left):     {aggregate_stats['rule1_arm_swap_left']:,} samples")
    print(f"  Rule 1 (arm swap - right):    {aggregate_stats['rule1_arm_swap_right']:,} samples")
    print(f"  Rule 1 (arm swap - total):    {total_rule1:,} samples ({100 * total_rule1 / max(1, total_original):.2f}%)")
    print(f"  Rule 2 (fully disengaged):    {aggregate_stats['rule2_fully_disengaged']:,} samples ({100 * aggregate_stats['rule2_fully_disengaged'] / max(1, total_original):.2f}%)")
    if aggregate_stats["out_of_bounds"] > 0:
        print(f"  Out of bounds (edge cases):   {aggregate_stats['out_of_bounds']:,} samples")
    print("-" * 80)
    print(f"EPISODE STATISTICS:")
    print(f"  Total episodes: {len(trajectory_ids)}")
    print(f"  Episodes fully filtered (0 valid samples): {episodes_fully_filtered} ({100 * episodes_fully_filtered / max(1, len(trajectory_ids)):.1f}%)")
    print(f"  Episodes partially filtered: {episodes_partially_filtered} ({100 * episodes_partially_filtered / max(1, len(trajectory_ids)):.1f}%)")
    print(f"  Episodes unfiltered (all valid): {episodes_unfiltered} ({100 * episodes_unfiltered / max(1, len(trajectory_ids)):.1f}%)")
    print("=" * 80)

    return results, {
        "total_original": total_original,
        "total_valid": total_valid,
        "aggregate_stats": aggregate_stats,
    }


def get_cache_path(dataset_path: Path, split: str, action_delta_indices: list[int]) -> Path:
    """Generate the cache file path with hash based on configuration."""
    dataset_name = dataset_path.name
    cache_key_data = f"{dataset_name}_{split}_{sorted(action_delta_indices)}"
    cache_hash = hashlib.md5(cache_key_data.encode()).hexdigest()[:12]
    cache_filename = f"cmr_filter_cache_{split}_{cache_hash}-44D.json"
    return dataset_path / "meta" / cache_filename


def compute_action_delta_indices(num_frames: int, timestep_interval: int = 6) -> list[int]:
    """Compute action delta indices matching groot_configs.py logic.
    
    Args:
        num_frames: Number of VIDEO frames (e.g., 13 = 1 context + 12 prediction)
        timestep_interval: Frame stride (default 6 for 10fps from 60Hz)
    
    Returns:
        List of delta indices for action sampling (num_frames - 1 entries)
        
    Note:
        This matches groot_configs.py which uses:
            num_action_frames = num_frames - 1  # 12 action timesteps for 13 video frames
            action_delta_indices = list(range(0, num_action_frames * timestep_interval, timestep_interval))
        
        The model expects num_actions to be divisible by temporal_compression_ratio (4).
        With num_frames=13: num_action_frames=12, and 12 % 4 = 0 ✓
    """
    # Action frames = video frames - 1 (no action for context frame)
    num_action_frames = num_frames - 1
    return [i * timestep_interval for i in range(num_action_frames)]


def load_dataset_metadata(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, int, str]:
    """Load dataset metadata needed for filtering."""
    # Load episodes.jsonl
    episode_path = dataset_path / "meta/episodes.jsonl"
    with open(episode_path, "r") as f:
        episode_metadata = [json.loads(line) for line in f]
    
    trajectory_ids = np.array([ep["episode_index"] for ep in episode_metadata])
    trajectory_lengths = np.array([ep["length"] for ep in episode_metadata])
    
    # Load info.json
    info_path = dataset_path / "meta/info.json"
    with open(info_path, "r") as f:
        info_meta = json.load(f)
    
    chunk_size = info_meta["chunks_size"]
    data_path_pattern = info_meta["data_path"]
    
    return trajectory_ids, trajectory_lengths, chunk_size, data_path_pattern


def compute_filter_cache(
    dataset_path: Path,
    split: str,
    num_frames: int,
    timestep_interval: int = 6,
    force: bool = False,
    num_workers: int | None = None,
) -> Path:
    """Compute and save the filter cache for a single dataset.
    
    Args:
        dataset_path: Path to the dataset root
        split: Data split ("train" or "test")
        num_frames: Number of frames for action horizon
        timestep_interval: Frame stride
        force: If True, recompute even if cache exists
        num_workers: Number of parallel workers (default: min(cpu_count, 64))
    
    Returns:
        Path to the saved cache file
    """
    dataset_path = Path(dataset_path)
    
    # Compute action delta indices
    action_delta_indices = compute_action_delta_indices(num_frames, timestep_interval)
    
    # Get cache path
    cache_path = get_cache_path(dataset_path, split, action_delta_indices)
    
    print(f"\n{'='*80}")
    print(f"Computing filter cache for: {dataset_path}")
    print(f"  Split: {split}")
    print(f"  Num video frames: {num_frames} (1 context + {num_frames - 1} prediction)")
    print(f"  Num action frames: {num_frames - 1}")
    print(f"  Timestep interval: {timestep_interval}")
    print(f"  Action delta indices: {action_delta_indices[:5]}... (len={len(action_delta_indices)})")
    print(f"  Cache path: {cache_path}")
    print(f"{'='*80}")
    
    # Check if cache already exists
    if cache_path.exists() and not force:
        print(f"[SKIP] Cache already exists: {cache_path}")
        print(f"       Use --force to recompute")
        return cache_path
    
    start_time = time.time()
    
    # Load dataset metadata
    trajectory_ids, trajectory_lengths, chunk_size, data_path_pattern = load_dataset_metadata(dataset_path)
    
    # Run filtering
    filter_results, filter_stats = filter_cmr_clutch_all_episodes(
        dataset_path=dataset_path,
        trajectory_ids=trajectory_ids,
        trajectory_lengths=trajectory_lengths,
        chunk_size=chunk_size,
        data_path_pattern=data_path_pattern,
        action_delta_indices=action_delta_indices,
        num_workers=num_workers,
    )
    
    # Build all_steps list
    all_steps = []
    for trajectory_id in trajectory_ids:
        valid_indices = filter_results.get(int(trajectory_id), [])
        for base_index in valid_indices:
            all_steps.append((int(trajectory_id), base_index))
    
    elapsed_time = time.time() - start_time
    
    # Prepare cache data
    cache_stats = {
        "raw_frames": int(np.sum(trajectory_lengths)),
        "effective_samples": filter_stats["total_original"],
        "valid_samples": len(all_steps),
        "data_reduction_pct": round(100 * (1 - len(all_steps) / max(1, filter_stats["total_original"])), 2),
        "compute_time_seconds": round(elapsed_time, 1),
        "filtering_stats": filter_stats["aggregate_stats"],
    }
    
    cache_data = {
        "action_delta_indices": action_delta_indices,
        "all_steps": all_steps,
        "stats": cache_stats,
        "dataset_name": dataset_path.name,
        "data_split": split,
        "num_frames": num_frames,
        "timestep_interval": timestep_interval,
    }
    
    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache_data, f)
    
    print(f"\n[SUCCESS] Cached filter results to: {cache_path}")
    print(f"          Valid samples: {len(all_steps):,}")
    print(f"          Compute time: {elapsed_time:.1f} seconds")
    
    return cache_path


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute CMR Versius clutch-aware filtering cache for training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to a single CMR dataset. If not provided, processes all default CMR datasets.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "both"],
        default="both",
        help="Data split to compute cache for (default: both)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=13,
        help="Number of VIDEO frames (1 context + N prediction). Action horizon = num_frames - 1. (default: 13 → 12 actions)",
    )
    parser.add_argument(
        "--timestep-interval",
        type=int,
        default=6,
        help="Frame stride / timestep interval (default: 6 for 10fps from 60Hz)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recompute even if cache already exists",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=f"Number of parallel workers (default: min(cpu_count, {CMR_MAX_FILTER_WORKERS}))",
    )
    
    args = parser.parse_args()
    
    # Determine which datasets to process
    if args.dataset_path:
        dataset_paths = [Path(args.dataset_path)]
    else:
        dataset_paths = [Path(p) for p in DEFAULT_CMR_DATASET_PATHS]
    
    # Determine which splits to process
    if args.split == "both":
        splits = ["train", "test"]
    else:
        splits = [args.split]
    
    print("=" * 80)
    print("CMR VERSIUS FILTER CACHE COMPUTATION")
    print("=" * 80)
    print(f"Datasets to process: {len(dataset_paths)}")
    for p in dataset_paths:
        print(f"  - {p}")
    print(f"Splits: {splits}")
    print(f"Num video frames: {args.num_frames} (1 context + {args.num_frames - 1} prediction)")
    print(f"Num action frames: {args.num_frames - 1} (matches model expectation)")
    print(f"Timestep interval: {args.timestep_interval}")
    print(f"Force recompute: {args.force}")
    print(f"Num workers: {args.num_workers or f'auto (max {CMR_MAX_FILTER_WORKERS})'}")
    print("=" * 80)
    
    # Process each dataset and split
    success_count = 0
    skip_count = 0
    error_count = 0
    cache_paths = []
    
    for dataset_path in dataset_paths:
        if not dataset_path.exists():
            print(f"\n[ERROR] Dataset path does not exist: {dataset_path}")
            error_count += 1
            continue
        
        for split in splits:
            try:
                cache_path = compute_filter_cache(
                    dataset_path=dataset_path,
                    split=split,
                    num_frames=args.num_frames,
                    timestep_interval=args.timestep_interval,
                    force=args.force,
                    num_workers=args.num_workers,
                )
                cache_paths.append(cache_path)
                if cache_path.exists():
                    success_count += 1
            except FileExistsError:
                skip_count += 1
            except Exception as e:
                print(f"\n[ERROR] Failed to compute cache for {dataset_path} ({split}): {e}")
                error_count += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Successfully computed: {success_count}")
    print(f"  Skipped (already exists): {skip_count}")
    print(f"  Errors: {error_count}")
    print("\nCache files:")
    for p in cache_paths:
        status = "✓" if p.exists() else "✗"
        print(f"  {status} {p}")
    print("=" * 80)
    
    if error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

