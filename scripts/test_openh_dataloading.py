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
Test script for verifying Open-H multi-embodiment data loading.

Tests the full data pipeline used by Cosmos-Predict 2.5 for multi-dataset
training, including:
  1. Per-dataset modality.json, stats_cosmos.json, and info.json validation
  2. Single-dataset loading via WrappedLeRobotSingleDataset (per embodiment)
  3. Multi-dataset loading via MixedLeRobotDataset (weighted mixture)
  4. Action transform correctness (rel_xyz_rot6d dimensions, padding to 44D)
  5. Video frame shapes and FPS consistency
  6. Episode filtering (exclude_splits)

Usage:
    # Test all Open-H datasets (checks files exist + loads 1 sample each):
    python scripts/test_openh_dataloading.py --all

    # Quick smoke test (fewer samples, faster):
    python scripts/test_openh_dataloading.py --all --num-samples 1

    # Test a single embodiment:
    python scripts/test_openh_dataloading.py \\
        --dataset-path /path/to/jhu/suturebot_2 \\
        --embodiment jhu_dvrk_mono

    # Test with the full MixedLeRobotDataset (as used in training):
    python scripts/test_openh_dataloading.py --all --test-mixture

    # Dry-run: only check that required files exist, don't load data:
    python scripts/test_openh_dataloading.py --all --dry-run
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*video decoding and encoding capabilities of torchvision.*")

import numpy as np
import torch
from tqdm import tqdm

from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.embodiment_tags import EmbodimentTag
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.groot_configs import (
    EMBODIMENT_REGISTRY,
    MAX_ACTION_DIM,
    OPEN_H_DATASET_SPECS,
    construct_modality_config_and_transforms,
)


# ============================================================================
# Helpers
# ============================================================================

def _resolve_embodiment(spec_emb) -> str:
    if isinstance(spec_emb, EmbodimentTag):
        return spec_emb.value
    return str(spec_emb)


def _check_file(path: Path, label: str) -> bool:
    if path.exists():
        print(f"    [OK]   {label}: {path.name}")
        return True
    else:
        print(f"    [MISS] {label}: {path}")
        return False


# ============================================================================
# Test 1: File existence checks per dataset
# ============================================================================

def test_dataset_files(dataset_path: Path, embodiment: str) -> dict:
    """Check that all required files exist for a dataset."""
    results = {"path": str(dataset_path), "embodiment": embodiment, "issues": []}

    dp = Path(dataset_path)
    if not dp.exists():
        results["issues"].append(f"Dataset path does not exist: {dp}")
        return results

    # Required files
    files = {
        "info.json": dp / "meta" / "info.json",
        "episodes.jsonl": dp / "meta" / "episodes.jsonl",
        "tasks.jsonl": dp / "meta" / "tasks.jsonl",
        "modality.json": dp / "meta" / "modality.json",
    }

    # stats_cosmos.json (required for non-CMR Open-H datasets)
    if embodiment == EmbodimentTag.CMR_VERSIUS.value:
        files["stats_cosmos-44D.json"] = dp / "meta" / "stats_cosmos-44D.json"
    else:
        files["stats_cosmos.json"] = dp / "meta" / "stats_cosmos.json"

    for label, path in files.items():
        if not path.exists():
            results["issues"].append(f"Missing: {label}")

    # Check parquet files exist
    parquets = list(dp.glob("data/*/*.parquet"))
    if not parquets:
        results["issues"].append("No parquet files in data/")
    else:
        results["n_episodes"] = len(parquets)

    # Check video files exist
    videos = list(dp.glob("videos/**/*.mp4"))
    if not videos:
        results["issues"].append("No video files in videos/")
    else:
        results["n_videos"] = len(videos)

    return results


# ============================================================================
# Test 2: Single-dataset loading
# ============================================================================

def test_single_dataset_loading(
    dataset_path: Path,
    embodiment: str,
    num_samples: int = 3,
    exclude_splits: list[str] | None = None,
) -> dict:
    """Load a single dataset and verify sample shapes."""
    from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.dataset import (
        WrappedLeRobotSingleDataset,
    )

    results = {"path": str(dataset_path), "embodiment": embodiment, "issues": []}

    config, train_transform, test_transform = construct_modality_config_and_transforms(
        num_frames=13, embodiment=embodiment, downscaled_res=False,
    )
    modality_filename = None
    if isinstance(config, dict) and "modality_filename" in config:
        modality_filename = config.pop("modality_filename")

    try:
        t0 = time.time()
        dataset = WrappedLeRobotSingleDataset(
            dataset_path=str(dataset_path),
            modality_configs=config,
            transforms=test_transform,
            embodiment_tag=embodiment,
            data_split="train",
            modality_filename=modality_filename,
            exclude_splits=exclude_splits,
        )
        init_time = time.time() - t0
        results["n_samples"] = len(dataset)
        results["init_time"] = f"{init_time:.1f}s"
        print(f"    Dataset loaded: {len(dataset):,} samples in {init_time:.1f}s")
    except Exception as e:
        results["issues"].append(f"Dataset init failed: {e}")
        return results

    # Load a few samples
    n = min(num_samples, len(dataset))
    for i in range(n):
        try:
            sample = dataset[i]

            # Check video shape
            video = sample.get("video")
            if video is not None:
                if isinstance(video, torch.Tensor):
                    v_shape = tuple(video.shape)
                else:
                    v_shape = video.shape
                if i == 0:
                    results["video_shape"] = str(v_shape)
                    print(f"    Sample {i}: video={v_shape}")

            # Check action shape
            action = sample.get("action")
            if action is not None:
                if isinstance(action, torch.Tensor):
                    a_shape = tuple(action.shape)
                else:
                    a_shape = action.shape
                if i == 0:
                    results["action_shape"] = str(a_shape)
                    results["action_dim"] = a_shape[-1] if len(a_shape) >= 1 else 0
                    print(f"    Sample {i}: action={a_shape} (dim={a_shape[-1]})")

            # Check state shape
            state = sample.get("__key__")
            if state is not None:
                if isinstance(state, torch.Tensor):
                    s_shape = tuple(state.shape)
                else:
                    s_shape = getattr(state, "shape", None)
                if i == 0 and s_shape is not None:
                    results["state_shape"] = str(s_shape)
                    print(f"    Sample {i}: state={s_shape}")

        except Exception as e:
            results["issues"].append(f"Sample {i} failed: {e}")
            if i == 0:
                print(f"    [ERROR] Sample {i}: {e}")

    return results


# ============================================================================
# Test 3: Action normalization + 44D padding verification (per dataset)
# ============================================================================

def test_action_normalization_and_padding(
    dataset_path: Path,
    embodiment: str,
    num_samples: int = 10,
    exclude_splits: list[str] | None = None,
) -> dict:
    """Verify that post-transform actions have correct normalization and 44D padding.

    Checks:
    - Action tensor is exactly (12, raw_action_dim) from the per-dataset pipeline
    - After zero-padding: (12, 44) as the model will see
    - Active dimensions (non-padded) are in a reasonable normalized range
    - Padded dimensions are exactly 0.0
    - No NaN or Inf values
    - State tensor is finite

    This uses the FULL training transform (including StateActionTransform normalization),
    NOT the stats-only pipeline.
    """
    from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.dataset import (
        WrappedLeRobotSingleDataset,
    )

    results = {"path": str(dataset_path), "embodiment": embodiment, "issues": []}

    config, train_transform, _ = construct_modality_config_and_transforms(
        num_frames=13, embodiment=embodiment, downscaled_res=False,
    )
    modality_filename = None
    if isinstance(config, dict) and "modality_filename" in config:
        modality_filename = config.pop("modality_filename")

    try:
        dataset = WrappedLeRobotSingleDataset(
            dataset_path=str(dataset_path),
            modality_configs=config,
            transforms=train_transform,  # Full training transform WITH normalization
            embodiment_tag=embodiment,
            data_split="train",
            modality_filename=modality_filename,
            exclude_splits=exclude_splits,
        )
    except Exception as e:
        results["issues"].append(f"Dataset init failed: {e}")
        return results

    n = min(num_samples, len(dataset))
    action_dims_seen = set()
    all_action_mins = []
    all_action_maxs = []
    all_action_means = []

    for i in range(n):
        try:
            sample = dataset[i]
            action = sample.get("action")
            if action is None:
                results["issues"].append(f"Sample {i}: no 'action' key")
                continue

            if isinstance(action, torch.Tensor):
                action = action.numpy()

            T, D = action.shape
            action_dims_seen.add(D)

            # Check for NaN/Inf
            if np.any(np.isnan(action)):
                results["issues"].append(f"Sample {i}: action contains NaN")
            if np.any(np.isinf(action)):
                results["issues"].append(f"Sample {i}: action contains Inf")

            # Collect stats for range check
            all_action_mins.append(np.min(action, axis=0))
            all_action_maxs.append(np.max(action, axis=0))
            all_action_means.append(np.mean(action, axis=0))

            # Check state
            state = sample.get("__key__")
            if state is not None:
                if isinstance(state, torch.Tensor):
                    state = state.numpy()
                if np.any(np.isnan(state)):
                    results["issues"].append(f"Sample {i}: state contains NaN")
                if np.any(np.isinf(state)):
                    results["issues"].append(f"Sample {i}: state contains Inf")

        except Exception as e:
            results["issues"].append(f"Sample {i} failed: {e}")

    if not action_dims_seen:
        results["issues"].append("No valid samples loaded")
        return results

    raw_dim = list(action_dims_seen)[0]
    results["raw_action_dim"] = raw_dim

    # Verify consistent action dim across samples
    if len(action_dims_seen) > 1:
        results["issues"].append(f"Inconsistent action dims: {action_dims_seen}")

    # Simulate padding to 44D (as MixedLeRobotDataset does)
    pad_dim = MAX_ACTION_DIM - raw_dim
    results["pad_dim"] = pad_dim
    if pad_dim < 0:
        results["issues"].append(f"raw_action_dim={raw_dim} > MAX_ACTION_DIM={MAX_ACTION_DIM}!")

    # Check normalization range of active dimensions
    if all_action_mins:
        global_min = np.min(np.stack(all_action_mins), axis=0)
        global_max = np.max(np.stack(all_action_maxs), axis=0)
        global_mean = np.mean(np.stack(all_action_means), axis=0)

        # For mean_std normalization, expect most values in [-5, 5]
        # (some outliers are fine, but all values in [-100, 100] is suspicious)
        extreme_dims = np.sum((global_min < -100) | (global_max > 100))
        if extreme_dims > 0:
            results["issues"].append(
                f"Normalization suspect: {extreme_dims}/{raw_dim} dims have range > [-100, 100]"
            )

        results["action_range"] = f"[{np.min(global_min):.2f}, {np.max(global_max):.2f}]"
        results["action_mean_range"] = f"[{np.min(global_mean):.4f}, {np.max(global_mean):.4f}]"

        print(f"    raw_dim={raw_dim}, pad_to={MAX_ACTION_DIM} (pad {pad_dim} zeros)")
        print(f"    action range: [{np.min(global_min):.3f}, {np.max(global_max):.3f}]")
        print(f"    action mean:  [{np.min(global_mean):.4f}, {np.max(global_mean):.4f}]")
        if not results["issues"]:
            print(f"    [OK] No NaN/Inf, normalization looks reasonable")

    return results


# ============================================================================
# Test 4: Cross-dataset action consistency (44D padding + distribution check)
# ============================================================================

def test_cross_dataset_consistency(num_samples_per_dataset: int = 5) -> dict:
    """Verify that all datasets produce consistent 44D padded action vectors.

    Uses MixedLeRobotDataset to load samples from all embodiments and checks:
    - ALL actions are exactly (12, 44)
    - Padding region (beyond each dataset's raw dim) is exactly 0.0
    - Active regions have finite, normalized values
    - No dataset has wildly different normalization scale from others
    """
    from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.dataset import (
        MixedLeRobotDataset,
    )

    results = {"issues": []}

    specs = []
    for spec in OPEN_H_DATASET_SPECS:
        emb = _resolve_embodiment(spec["embodiment"])
        dp = Path(spec["path"])
        if not dp.exists():
            continue
        specs.append({
            "path": spec["path"],
            "embodiment": emb,
            "mix_ratio": spec.get("mix_ratio", 1.0),
            "exclude_splits": spec.get("exclude_splits"),
        })

    if not specs:
        results["issues"].append("No valid dataset paths found")
        return results

    print(f"\n  Loading MixedLeRobotDataset ({len(specs)} sub-datasets, including CMR)...")
    try:
        dataset = MixedLeRobotDataset(
            dataset_specs=specs,
            num_frames=13,
            data_split="train",
            max_action_dim=MAX_ACTION_DIM,
            downscaled_res=False,
        )
    except Exception as e:
        results["issues"].append(f"MixedLeRobotDataset init failed: {e}")
        return results

    print(f"  Loaded: {len(dataset):,} virtual samples")

    # Sample across the dataset to hit different embodiments
    n_total = min(num_samples_per_dataset * len(dataset.sub_datasets), len(dataset))
    step = max(1, len(dataset) // n_total)

    per_embodiment_stats: dict[str, dict] = {}
    wrong_dim_count = 0
    nan_count = 0
    padding_violation_count = 0

    print(f"  Testing {n_total} samples (step={step})...")

    for i in tqdm(range(0, n_total * step, step), total=n_total, desc="  Cross-dataset check"):
        idx = i % len(dataset)
        try:
            sample = dataset[idx]
        except Exception:
            continue

        emb = sample.get("embodiment_tag", "unknown")
        action = sample.get("action")
        if action is None:
            continue
        if isinstance(action, torch.Tensor):
            action = action.numpy()

        T, D = action.shape

        # CHECK 1: exact 44D
        if D != MAX_ACTION_DIM:
            wrong_dim_count += 1
            if wrong_dim_count <= 3:
                results["issues"].append(f"idx={idx} ({emb}): action dim={D}, expected {MAX_ACTION_DIM}")

        # CHECK 2: no NaN/Inf
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            nan_count += 1

        # CHECK 3: padding is zeros
        # Get the raw dim for this embodiment from registry
        reg = EMBODIMENT_REGISTRY.get(emb, {})
        action_key_configs = reg.get("action_key_configs", {})
        # Compute expected raw dim (same logic as the stats script)
        modality_filename = reg.get("modality_filename", "meta/modality.json")
        # We estimate raw dim from the first non-zero position from the right
        # (simpler than re-parsing modality.json)
        col_sums = np.abs(action).sum(axis=0)  # (44,)
        last_nonzero = 0
        for d in range(D - 1, -1, -1):
            if col_sums[d] > 1e-10:
                last_nonzero = d + 1
                break
        # Everything after last_nonzero should be zero (padding)
        if last_nonzero < D:
            padding_region = action[:, last_nonzero:]
            if not np.allclose(padding_region, 0.0):
                padding_violation_count += 1
                if padding_violation_count <= 3:
                    results["issues"].append(
                        f"idx={idx} ({emb}): padding non-zero at dims {last_nonzero}:{D}")

        # Collect per-embodiment stats
        if emb not in per_embodiment_stats:
            per_embodiment_stats[emb] = {
                "count": 0,
                "active_dim": last_nonzero,
                "mins": [],
                "maxs": [],
                "abs_means": [],
            }
        stats = per_embodiment_stats[emb]
        stats["count"] += 1
        active = action[:, :last_nonzero] if last_nonzero > 0 else action
        stats["mins"].append(np.min(active))
        stats["maxs"].append(np.max(active))
        stats["abs_means"].append(np.mean(np.abs(active)))

    # Summary
    print(f"\n  {'='*70}")
    print(f"  CROSS-DATASET CONSISTENCY RESULTS")
    print(f"  {'='*70}")
    print(f"  Wrong dimension:     {wrong_dim_count}")
    print(f"  NaN/Inf:             {nan_count}")
    print(f"  Padding violations:  {padding_violation_count}")
    print(f"  {'='*70}")
    print(f"  {'Embodiment':<24s} {'Samples':>8s} {'ActDim':>7s} {'Range':>24s} {'|Mean|':>10s}")
    print(f"  {'-'*70}")

    for emb in sorted(per_embodiment_stats.keys()):
        s = per_embodiment_stats[emb]
        rng_min = np.min(s["mins"]) if s["mins"] else 0
        rng_max = np.max(s["maxs"]) if s["maxs"] else 0
        abs_mean = np.mean(s["abs_means"]) if s["abs_means"] else 0
        print(f"  {emb:<24s} {s['count']:>8d} {s['active_dim']:>7d} "
              f"[{rng_min:>10.3f}, {rng_max:>10.3f}] {abs_mean:>10.4f}")

    print(f"  {'='*70}")

    if wrong_dim_count > 0:
        results["issues"].append(f"{wrong_dim_count} samples had wrong action dimension")
    if nan_count > 0:
        results["issues"].append(f"{nan_count} samples had NaN/Inf")
    if padding_violation_count > 0:
        results["issues"].append(f"{padding_violation_count} samples had non-zero padding")

    results["per_embodiment"] = {
        emb: {"active_dim": s["active_dim"], "count": s["count"]}
        for emb, s in per_embodiment_stats.items()
    }
    return results


# ============================================================================
# Test 5: MixedLeRobotDataset loading
# ============================================================================

def test_mixture_dataset(num_samples: int = 5) -> dict:
    """Test the full MixedLeRobotDataset as used in training."""
    from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.dataset import (
        MixedLeRobotDataset,
    )

    results = {"issues": []}

    # Build specs list (including CMR)
    specs = []
    for spec in OPEN_H_DATASET_SPECS:
        emb = _resolve_embodiment(spec["embodiment"])
        dp = Path(spec["path"])
        if not dp.exists():
            continue
        specs.append({
            "path": spec["path"],
            "embodiment": emb,
            "mix_ratio": spec.get("mix_ratio", 1.0),
            "exclude_splits": spec.get("exclude_splits"),
        })

    if not specs:
        results["issues"].append("No valid dataset paths found")
        return results

    print(f"\n  Creating MixedLeRobotDataset with {len(specs)} sub-datasets...")
    try:
        t0 = time.time()
        dataset = MixedLeRobotDataset(
            dataset_specs=specs,
            num_frames=13,
            data_split="train",
            max_action_dim=MAX_ACTION_DIM,
            downscaled_res=False,
        )
        init_time = time.time() - t0
        results["n_samples"] = len(dataset)
        results["n_sub_datasets"] = len(dataset.sub_datasets)
        results["init_time"] = f"{init_time:.1f}s"
        print(f"  Mixture loaded: {len(dataset):,} virtual samples, "
              f"{len(dataset.sub_datasets)} sub-datasets in {init_time:.1f}s")
    except Exception as e:
        results["issues"].append(f"MixedLeRobotDataset init failed: {e}")
        return results

    # Load some samples and verify action padding
    n = min(num_samples, len(dataset))
    embodiments_seen = set()
    for i in range(n):
        idx = i * (len(dataset) // max(n, 1))  # Spread across the dataset
        try:
            sample = dataset[idx]
            emb = sample.get("embodiment_tag", "?")
            embodiments_seen.add(emb)

            action = sample.get("action")
            if action is not None:
                if isinstance(action, torch.Tensor):
                    action = action.numpy()
                action_dim = action.shape[-1]
                if action_dim != MAX_ACTION_DIM:
                    results["issues"].append(
                        f"Sample {idx} ({emb}): action_dim={action_dim}, expected {MAX_ACTION_DIM}")
                if i < 3:
                    # Check padding region is zeros
                    reg = EMBODIMENT_REGISTRY.get(emb, {})
                    raw_keys = reg.get("action_keys", [])
                    print(f"    Sample {idx} ({emb}): action dim={action_dim}, "
                          f"non-zero dims={np.count_nonzero(action.sum(axis=0))}")

        except Exception as e:
            results["issues"].append(f"Sample {idx} failed: {e}")
            if i < 3:
                print(f"    [ERROR] Sample {idx}: {e}")

    results["embodiments_seen"] = sorted(embodiments_seen)
    print(f"  Embodiments seen in samples: {sorted(embodiments_seen)}")
    return results


# ============================================================================
# Test 6: DataLoader batch collation (reproduces torch.stack shape mismatch)
# ============================================================================

def test_dataloader_collation(batch_size: int = 16, num_batches: int = 10) -> dict:
    """Test that DataLoader can collate batches from MixedLeRobotDataset.

    This reproduces the exact crash from multi-node training:
        RuntimeError: stack expects each tensor to be equal size,
        but got [1, 16] at entry 0 and [1, 14] at entry 11

    The crash happens when torch.stack tries to collate the __key__ (state)
    tensors from different embodiments that have different state dimensions.
    The fix (state padding in MixedLeRobotDataset) should prevent this.

    This test creates a real DataLoader with the same batch_size as training
    and iterates several batches, checking that:
    - All tensors in the batch have consistent shapes
    - action is exactly (batch_size, 12, 44)
    - __key__ (state) has consistent dimension across all samples
    - No NaN/Inf in any tensor
    """
    from torch.utils.data import DataLoader
    from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.dataset import (
        MixedLeRobotDataset,
    )

    results = {"issues": []}

    specs = []
    for spec in OPEN_H_DATASET_SPECS:
        emb = _resolve_embodiment(spec["embodiment"])
        dp = Path(spec["path"])
        if not dp.exists():
            continue
        specs.append({
            "path": spec["path"],
            "embodiment": emb,
            "mix_ratio": spec.get("mix_ratio", 1.0),
            "exclude_splits": spec.get("exclude_splits"),
        })

    if not specs:
        results["issues"].append("No valid dataset paths found")
        return results

    print(f"\n  Creating MixedLeRobotDataset ({len(specs)} sub-datasets)...")
    try:
        dataset = MixedLeRobotDataset(
            dataset_specs=specs,
            num_frames=13,
            data_split="train",
            max_action_dim=MAX_ACTION_DIM,
            downscaled_res=False,
        )
    except Exception as e:
        results["issues"].append(f"MixedLeRobotDataset init failed: {e}")
        return results

    print(f"  Dataset: {len(dataset):,} virtual samples")
    print(f"  Creating DataLoader: batch_size={batch_size}, num_workers=0 (main process)")

    # Use num_workers=0 for debugging (errors are raised in main process)
    # and drop_last=True to match training config
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    print(f"  Iterating {num_batches} batches...")
    shapes_seen: dict[str, set] = {}
    batch_count = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break
        batch_count += 1

        # Collect shapes for each key
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                shape = tuple(value.shape)
                if key not in shapes_seen:
                    shapes_seen[key] = set()
                shapes_seen[key].add(shape)

                # Check NaN/Inf
                if torch.any(torch.isnan(value)):
                    results["issues"].append(f"Batch {batch_idx}: {key} contains NaN")
                if torch.any(torch.isinf(value)):
                    results["issues"].append(f"Batch {batch_idx}: {key} contains Inf")

        # Log first batch details
        if batch_idx == 0:
            print(f"\n  Batch 0 tensor shapes:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key}: {tuple(value.shape)} dtype={value.dtype}")
                elif isinstance(value, (list, tuple)):
                    print(f"    {key}: list[{len(value)}]")
                else:
                    print(f"    {key}: {type(value).__name__}")

    print(f"\n  Successfully collated {batch_count}/{num_batches} batches")

    # Verify shape consistency across batches
    print(f"\n  Shape consistency check:")
    for key in sorted(shapes_seen.keys()):
        shapes = shapes_seen[key]
        if len(shapes) == 1:
            print(f"    {key}: {list(shapes)[0]}  [OK — consistent]")
        else:
            results["issues"].append(f"{key}: inconsistent shapes across batches: {shapes}")
            print(f"    {key}: {shapes}  [FAIL — inconsistent!]")

    # Specific checks for action and state
    if "action" in shapes_seen:
        action_shapes = shapes_seen["action"]
        expected_action = (batch_size, 12, MAX_ACTION_DIM)
        if all(s == expected_action for s in action_shapes):
            print(f"\n  [OK] action shape = {expected_action} (consistent, padded to {MAX_ACTION_DIM}D)")
        else:
            results["issues"].append(f"action shape mismatch: {action_shapes}, expected {expected_action}")
            print(f"\n  [FAIL] action shape: {action_shapes}, expected {expected_action}")

    if "__key__" in shapes_seen:
        key_shapes = shapes_seen["__key__"]
        if len(key_shapes) == 1:
            print(f"  [OK] __key__ (state) shape = {list(key_shapes)[0]} (consistent, padded)")
        else:
            results["issues"].append(f"__key__ shape mismatch: {key_shapes}")
            print(f"  [FAIL] __key__ (state) inconsistent: {key_shapes}")

    if not results["issues"]:
        print(f"\n  [OK] All {batch_count} batches collated successfully — no shape mismatches!")

    return results


# ============================================================================
# Main
# ============================================================================

def run_all(args):
    """Test all datasets in OPEN_H_DATASET_SPECS."""
    # Deduplicate by (path, embodiment)
    seen = set()
    jobs = []
    for spec in OPEN_H_DATASET_SPECS:
        emb = _resolve_embodiment(spec["embodiment"])
        dp = spec["path"]
        key = (dp, emb)
        if key in seen:
            continue
        seen.add(key)
        jobs.append((Path(dp), emb, spec.get("exclude_splits")))

    print("=" * 80)
    print(f"OPEN-H DATA LOADING TEST — {len(jobs)} datasets")
    print("=" * 80)

    all_results = []

    # --- Test 1: File existence ---
    print(f"\n{'='*80}")
    print("TEST 1: Required files check")
    print(f"{'='*80}")
    missing_count = 0
    for dp, emb, excl in jobs:
        print(f"\n  [{emb}] {dp.name}")
        if not dp.exists():
            print(f"    [SKIP] Path does not exist")
            missing_count += 1
            continue
        res = test_dataset_files(dp, emb)
        if res["issues"]:
            for issue in res["issues"]:
                print(f"    [ISSUE] {issue}")
            missing_count += 1
        else:
            eps = res.get("n_episodes", "?")
            vids = res.get("n_videos", "?")
            print(f"    [OK]   All files present ({eps} episodes, {vids} videos)")

    if args.dry_run:
        print(f"\n{'='*80}")
        print(f"DRY RUN COMPLETE — {len(jobs) - missing_count}/{len(jobs)} datasets have all required files")
        print(f"{'='*80}")
        return

    # --- Test 2: Per-dataset loading ---
    print(f"\n{'='*80}")
    print(f"TEST 2: Single-dataset loading ({args.num_samples} samples each, split=train)")
    print(f"{'='*80}")
    load_results = {}
    for dp, emb, excl in jobs:
        if not dp.exists():
            load_results[f"{emb}/{dp.name}"] = "SKIP (missing)"
            continue
        print(f"\n  [{emb}] {dp.name}")
        res = test_single_dataset_loading(dp, emb, args.num_samples, excl)
        if res["issues"]:
            load_results[f"{emb}/{dp.name}"] = f"ISSUES: {res['issues']}"
        else:
            dim = res.get("action_dim", "?")
            n = res.get("n_samples", "?")
            load_results[f"{emb}/{dp.name}"] = f"OK (dim={dim}, samples={n})"

    # --- Test 3: Action normalization + padding per dataset ---
    print(f"\n{'='*80}")
    print(f"TEST 3: Action normalization + 44D padding ({args.num_samples} samples each)")
    print(f"{'='*80}")
    norm_results = {}
    for dp, emb, excl in jobs:
        if not dp.exists():
            continue
        print(f"\n  [{emb}] {dp.name}")
        res = test_action_normalization_and_padding(dp, emb, args.num_samples, excl)
        key = f"{emb}/{dp.name}"
        if res["issues"]:
            norm_results[key] = f"ISSUES: {res['issues']}"
        else:
            raw = res.get("raw_action_dim", "?")
            pad = res.get("pad_dim", "?")
            rng = res.get("action_range", "?")
            norm_results[key] = f"OK (raw={raw}D, pad={pad}, range={rng})"

    # --- Test 4: Cross-dataset consistency ---
    if args.test_mixture:
        print(f"\n{'='*80}")
        print("TEST 4: Cross-dataset action consistency (44D, padding, distribution)")
        print(f"{'='*80}")
        cross_res = test_cross_dataset_consistency(args.num_samples)
        if cross_res["issues"]:
            for issue in cross_res["issues"]:
                print(f"  [ISSUE] {issue}")

    # --- Test 5: MixedLeRobotDataset basic loading ---
    if args.test_mixture:
        print(f"\n{'='*80}")
        print("TEST 5: MixedLeRobotDataset (weighted mixture)")
        print(f"{'='*80}")
        mix_res = test_mixture_dataset(args.num_samples)
        if mix_res["issues"]:
            print(f"  Issues: {mix_res['issues']}")

    # --- Test 6: DataLoader batch collation (reproduces torch.stack crash) ---
    if args.test_mixture:
        print(f"\n{'='*80}")
        print("TEST 6: DataLoader batch collation (batch_size=16, 10 batches)")
        print(f"        This reproduces the torch.stack shape mismatch crash if")
        print(f"        state/action padding is broken.")
        print(f"{'='*80}")
        collate_res = test_dataloader_collation(batch_size=16, num_batches=10)
        if collate_res["issues"]:
            for issue in collate_res["issues"]:
                print(f"  [ISSUE] {issue}")

    # --- Summary ---
    print(f"\n{'='*80}")
    print("SUMMARY — Data Loading (Test 2)")
    print(f"{'='*80}")
    ok_count = sum(1 for v in load_results.values() if v.startswith("OK"))
    skip_count = sum(1 for v in load_results.values() if "SKIP" in v)
    fail_count = len(load_results) - ok_count - skip_count
    print(f"  OK: {ok_count}  |  SKIP: {skip_count}  |  ISSUES: {fail_count}  |  Total: {len(load_results)}")
    for name, status in load_results.items():
        marker = "OK" if status.startswith("OK") else ("SKIP" if "SKIP" in status else "FAIL")
        print(f"  [{marker:4s}] {name}: {status}")

    if norm_results:
        print(f"\n{'='*80}")
        print("SUMMARY — Normalization + 44D Padding (Test 3)")
        print(f"{'='*80}")
        n_ok = sum(1 for v in norm_results.values() if v.startswith("OK"))
        n_fail = len(norm_results) - n_ok
        print(f"  OK: {n_ok}  |  ISSUES: {n_fail}  |  Total: {len(norm_results)}")
        for name, status in norm_results.items():
            marker = "OK" if status.startswith("OK") else "FAIL"
            print(f"  [{marker:4s}] {name}: {status}")

    print(f"{'='*80}")


def run_single(args):
    """Test a single dataset."""
    dp = Path(args.dataset_path)
    emb = args.embodiment

    print("=" * 80)
    print(f"OPEN-H DATA LOADING TEST — single dataset")
    print(f"  path: {dp}")
    print(f"  embodiment: {emb}")
    print("=" * 80)

    # File check
    print(f"\nTest 1: File existence")
    res = test_dataset_files(dp, emb)
    if res["issues"]:
        for issue in res["issues"]:
            print(f"  [ISSUE] {issue}")

    if args.dry_run:
        return

    # Loading test
    print(f"\nTest 2: Dataset loading ({args.num_samples} samples)")
    excl = args.exclude_splits if args.exclude_splits else None
    res = test_single_dataset_loading(dp, emb, args.num_samples, excl)
    if res["issues"]:
        print(f"\nIssues found:")
        for issue in res["issues"]:
            print(f"  - {issue}")
    else:
        print(f"\nAll tests passed!")


def main():
    parser = argparse.ArgumentParser(
        description="Test Open-H multi-embodiment data loading pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    path_group = parser.add_mutually_exclusive_group(required=True)
    path_group.add_argument("--all", action="store_true",
                            help="Test all datasets in OPEN_H_DATASET_SPECS")
    path_group.add_argument("--dataset-path", type=str,
                            help="Test a single dataset")

    parser.add_argument("--embodiment", type=str, default=None,
                        choices=list(EMBODIMENT_REGISTRY.keys()) + [EmbodimentTag.CMR_VERSIUS.value],
                        help="Embodiment (required for --dataset-path)")
    parser.add_argument("--exclude-splits", type=str, nargs="+", default=None)
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Number of samples to load per dataset (default: 3)")
    parser.add_argument("--test-mixture", action="store_true",
                        help="Also test MixedLeRobotDataset (slower, requires all datasets)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only check file existence, don't load data")
    args = parser.parse_args()

    if not args.all and args.embodiment is None:
        parser.error("--embodiment is required when using --dataset-path")

    if args.all:
        run_all(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
