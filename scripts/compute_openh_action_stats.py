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
Compute per-key normalization statistics for any Open-H embodiment.

Unlike compute_cmr_action_stats.py (which is CMR-specific with hardcoded raw
indices, clutch filtering, and motion scaling), this script is GENERIC: it
instantiates the real transform pipeline (GenericRelativeActionTransform) for
any embodiment registered in EMBODIMENT_REGISTRY and collects statistics on
the TRANSFORMED action/state output.

This guarantees that the statistics exactly match what the training pipeline
produces, regardless of the embodiment's delta conversion (rel_xyz_rot6d,
relative, delta, or absolute).

Output:
    meta/stats_cosmos.json — per-key statistics in the same format as
    stats_cosmos-44D.json (action.psm1_pose → {mean, std, min, max, q01, q99})

For CMR Versius, continue using compute_cmr_action_stats.py (it handles
the additional clutch filtering and motion scaling that are CMR-specific).

Usage:
    # All Open-H datasets at once (reads OPEN_H_DATASET_SPECS, includes exclude_splits):
    python compute_openh_action_stats.py --all

    # Quick test (10 samples per dataset):
    python compute_openh_action_stats.py --all --max-samples 10

    # Single dataset:
    python compute_openh_action_stats.py \\
        --dataset-path /path/to/lerobot/dataset \\
        --embodiment dvrk

    # Single dataset with episode filtering:
    python compute_openh_action_stats.py \\
        --dataset-path /path/to/stanford/Needle_Transfer \\
        --embodiment dvrk_stanford_real \\
        --exclude-splits fail bad_frames

    # All dVRK datasets under a root:
    python compute_openh_action_stats.py \\
        --dataset-path-root /path/to/jhu \\
        --embodiment jhu_dvrk_mono
"""

import argparse
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

# Suppress noisy torchvision video deprecation warnings
warnings.filterwarnings("ignore", message=".*video decoding and encoding capabilities of torchvision.*")

import numpy as np
import pandas as pd
from tqdm import tqdm

from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.embodiment_tags import EmbodimentTag
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.dataset import (
    LE_ROBOT_INFO_FILENAME,
    resolve_excluded_episode_indices,
)
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.schema import (
    LeRobotModalityMetadata,
    LeRobotStateActionMetadata,
)
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.groot_configs import (
    EMBODIMENT_REGISTRY,
    OPEN_H_DATASET_SPECS,
)
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.transform.state_action import (
    convert_to_hybrid_relative,
)

# Maximum parallel workers
MAX_WORKERS = 64


# ============================================================================
# Statistics helpers (streaming, memory-efficient)
# ============================================================================

class StreamingStats:
    """Memory-efficient streaming statistics using Welford's algorithm + reservoir sampling."""

    def __init__(self, num_dims: int, reservoir_size: int = 2_000_000):
        self.num_dims = num_dims
        self.reservoir_size = reservoir_size
        self.count = 0
        self.mean = np.zeros(num_dims, dtype=np.float64)
        self.M2 = np.zeros(num_dims, dtype=np.float64)
        self.min_vals = np.full(num_dims, np.inf, dtype=np.float64)
        self.max_vals = np.full(num_dims, -np.inf, dtype=np.float64)
        self.reservoir = None
        self.reservoir_count = 0

    def update(self, batch: np.ndarray):
        if batch.shape[0] == 0:
            return
        n = batch.shape[0]
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0, ddof=0)
        batch_M2 = batch_var * n
        self.min_vals = np.minimum(self.min_vals, np.min(batch, axis=0))
        self.max_vals = np.maximum(self.max_vals, np.max(batch, axis=0))
        if self.count == 0:
            self.mean = batch_mean
            self.M2 = batch_M2
            self.count = n
        else:
            n_total = self.count + n
            delta = batch_mean - self.mean
            self.mean = (self.count * self.mean + n * batch_mean) / n_total
            self.M2 = self.M2 + batch_M2 + delta ** 2 * self.count * n / n_total
            self.count = n_total
        # Reservoir sampling
        if self.reservoir is None:
            self.reservoir = batch[:self.reservoir_size].copy()
            self.reservoir_count = n
        elif len(self.reservoir) < self.reservoir_size:
            space = self.reservoir_size - len(self.reservoir)
            self.reservoir = np.vstack([self.reservoir, batch[:space]])
            self.reservoir_count += n
        else:
            # Algorithm R replacement
            for row in batch:
                self.reservoir_count += 1
                j = np.random.randint(0, self.reservoir_count)
                if j < self.reservoir_size:
                    self.reservoir[j] = row

    def get_stats(self) -> dict:
        if self.count == 0:
            z = [0.0] * self.num_dims
            return {"mean": z, "std": z, "min": z, "max": z, "q01": z, "q99": z}
        std = np.sqrt(self.M2 / self.count)
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


# ============================================================================
# Main processing
# ============================================================================

def _is_lerobot_dataset(path: Path) -> bool:
    return (path / "data").is_dir() and (path / "meta").is_dir() and any((path / "data").rglob("*.parquet"))


def _discover_datasets(root: Path) -> list[Path]:
    datasets = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and _is_lerobot_dataset(child):
            datasets.append(child)
    return datasets


def _process_episode_parquet(
    parquet_path: Path,
    modality_meta: dict,
    action_key_configs: dict,
    action_delta_indices: list[int],
    state_delta_indices: list[int],
    action_keys: list[str],
    state_keys: list[str],
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Worker function: process one parquet file (episode) without video decoding.

    Reads state/action arrays from parquet, applies delta conversion, concatenates
    per-key arrays, and returns the result for streaming stats collection.

    This runs in a subprocess via ProcessPoolExecutor for parallelism.

    Returns:
        (action_array, state_array, warning_or_None)
        action_array shape: (N_samples * T_action, action_dim)
        state_array shape:  (N_samples, state_dim)
    """
    try:
        df = pd.read_parquet(parquet_path)
        T = len(df)

        # Maximum delta for action horizon
        max_action_delta = max(action_delta_indices) if action_delta_indices else 0
        effective_length = max(0, T - max_action_delta)
        if effective_length == 0:
            return np.empty((0, 0)), np.empty((0, 0)), None

        # Extract per-key arrays from the flat parquet columns using modality metadata
        def extract_key_data(key: str, df: pd.DataFrame) -> np.ndarray:
            """Extract a named key's data from parquet using modality.json metadata."""
            modality, subkey = key.split(".", 1)
            meta = modality_meta.get(modality, {}).get(subkey)
            if meta is None:
                raise KeyError(f"{key} config not found")
            original_col = meta.get("original_key")
            if original_col is None:
                # Default: observation.state for state, action for action
                original_col = "observation.state" if modality == "state" else "action"
            start, end = meta["start"], meta["end"]
            col_data = np.stack(df[original_col].values)  # (T, D_flat)
            return col_data[:, start:end].astype(np.float32)  # (T, key_dim)

        # Process all valid starting indices
        all_action_rows = []
        all_state_rows = []
        n_skipped = 0

        for base_idx in range(effective_length):
            # --- State at t=0 ---
            state_parts = []
            for key in state_keys:
                s_idx = base_idx + state_delta_indices[0]  # delta_indices=[0]
                s_idx = max(0, min(s_idx, T - 1))
                key_data = extract_key_data(key, df)
                state_parts.append(key_data[s_idx])
            if state_parts:
                all_state_rows.append(np.concatenate(state_parts))

            # --- Action over horizon ---
            action_timestep_rows = []
            for a_delta in action_delta_indices:
                a_idx = base_idx + a_delta
                a_idx = max(0, min(a_idx, T - 1))
                action_parts = []
                for key in action_keys:
                    key_data = extract_key_data(key, df)
                    action_parts.append(key_data[a_idx])
                action_timestep_rows.append(np.concatenate(action_parts))
            action_horizon = np.stack(action_timestep_rows)  # (T_action, action_dim_raw)

            # --- Apply delta conversion per key ---
            # We need to apply the same GenericRelativeActionTransform logic
            # but directly on numpy arrays (no torch, no dataset wrapper).
            offset = 0
            converted_parts = []
            state_row = all_state_rows[-1] if all_state_rows else None
            skip_sample = False

            for key in action_keys:
                cfg = action_key_configs.get(key)
                key_data_raw = extract_key_data(key, df)
                raw_dim = key_data_raw.shape[1]

                # Extract this key's action horizon data
                key_action = action_horizon[:, offset:offset + raw_dim]

                if cfg is not None and cfg.rep == "rel_xyz_rot6d":
                    # Get reference state pose
                    ref_key = cfg.state_key
                    if ref_key and state_row is not None:
                        # Find the state key's slice in the concatenated state
                        s_off = 0
                        ref_pose = None
                        for sk in state_keys:
                            sk_data = extract_key_data(sk, df)
                            sk_dim = sk_data.shape[1]
                            if sk == ref_key:
                                ref_pose = state_row[s_off:s_off + sk_dim]
                                break
                            s_off += sk_dim

                        if ref_pose is not None:
                            # Guard: check for zero-norm quaternions (invalid data /
                            # padding at episode boundaries).  Scipy's Rotation.from_quat()
                            # crashes on [0,0,0,0].
                            if cfg.input_rotation_format == "quat":
                                quat_slice = key_action[:, 3:7]
                                norms = np.linalg.norm(quat_slice, axis=-1)
                                if np.any(norms < 1e-8):
                                    skip_sample = True
                                    break
                            if cfg.reference_rotation_format == "quat":
                                ref_quat = ref_pose[3:7] if len(ref_pose) >= 7 else ref_pose[3:]
                                if np.linalg.norm(ref_quat) < 1e-8:
                                    skip_sample = True
                                    break

                            key_action = convert_to_hybrid_relative(
                                action_data=key_action,
                                eef_pose=ref_pose,
                                input_rotation_format=cfg.input_rotation_format,
                                reference_rotation_format=cfg.reference_rotation_format,
                                input_quat_order=cfg.input_quat_order,
                                reference_quat_order=cfg.reference_quat_order,
                            )  # (T_action, 9)

                elif cfg is not None and cfg.rep == "relative":
                    # Joint-space subtraction
                    ref_key = cfg.state_key
                    if ref_key and state_row is not None:
                        s_off = 0
                        ref_val = None
                        for sk in state_keys:
                            sk_data = extract_key_data(sk, df)
                            sk_dim = sk_data.shape[1]
                            if sk == ref_key:
                                ref_val = state_row[s_off:s_off + sk_dim]
                                break
                            s_off += sk_dim
                        if ref_val is not None:
                            key_action = key_action - ref_val

                # delta / absolute: pass through unchanged
                converted_parts.append(key_action)
                offset += raw_dim

            if skip_sample:
                # Remove the state row we just added (it corresponds to this skipped sample)
                if all_state_rows:
                    all_state_rows.pop()
                n_skipped += 1
                continue

            converted_action = np.concatenate(converted_parts, axis=-1)  # (T_action, action_dim)
            all_action_rows.append(converted_action)

        if not all_action_rows:
            warn = None
            if n_skipped > 0:
                warn = f"{parquet_path.name}: all {effective_length} samples skipped ({n_skipped} had zero-norm quaternions)"
            return np.empty((0, 0)), np.empty((0, 0)), warn

        actions = np.concatenate(all_action_rows, axis=0)  # (N*T_action, action_dim)
        states = np.stack(all_state_rows) if all_state_rows else np.empty((0, 0))

        warn = None
        if n_skipped > 0:
            warn = (f"{parquet_path.name}: {n_skipped}/{effective_length} samples skipped "
                    f"(zero-norm quaternions), {len(all_action_rows)} valid")
        return actions, states, warn

    except Exception as e:
        import traceback
        return np.empty((0, 0)), np.empty((0, 0)), f"Error processing {parquet_path.name}: {type(e).__name__}: {e}"


def _load_modality_meta(dataset_path: Path, modality_filename: str) -> dict:
    """Load modality.json and return a simplified {modality: {subkey: {start, end, original_key}}} dict."""
    modality_path = dataset_path / modality_filename
    if not modality_path.exists():
        raise FileNotFoundError(f"Modality file not found: {modality_path}")

    with open(modality_path, "r") as f:
        raw = json.load(f)

    result: dict[str, dict] = {}
    for modality in ["state", "action"]:
        result[modality] = {}
        if modality not in raw:
            continue
        for subkey, meta in raw[modality].items():
            result[modality][subkey] = {
                "start": meta.get("start", 0),
                "end": meta.get("end", 1),
                "original_key": meta.get("original_key"),
            }
    return result


def process_single_dataset(
    dataset_path: Path,
    embodiment: str,
    num_frames: int,
    max_samples: int | None,
    output_filename: str,
    exclude_splits: list[str] | None = None,
    num_workers: int | None = None,
):
    """Process one dataset using parallel episode processing (no video decoding).

    Each parquet file (episode) is processed by a worker subprocess that:
    1. Reads state/action arrays from parquet (fast, no video)
    2. Applies the delta conversion (rel_xyz_rot6d, relative, etc.)
    3. Returns the transformed arrays for streaming stats collection

    This is ~100x faster than the sequential video-decoding approach.

    Args:
        dataset_path: Path to the LeRobot dataset.
        embodiment: Embodiment tag string.
        num_frames: Number of video frames (e.g. 13).
        max_samples: Max episodes to process (for testing). None = all.
        output_filename: Output filename in meta/ dir.
        exclude_splits: Split names from info.json to exclude.
        num_workers: Parallel workers (default: min(cpu_count, MAX_WORKERS)).
    """
    if num_workers is None:
        num_workers = min(os.cpu_count() or 8, MAX_WORKERS)

    print("=" * 80)
    print(f"COMPUTING OPEN-H ACTION STATS — {dataset_path.name}")
    print(f"  embodiment: {embodiment}")
    print(f"  num_frames: {num_frames}")
    print(f"  workers: {num_workers}")
    if exclude_splits:
        print(f"  exclude_splits: {exclude_splits}")
    print("=" * 80)

    reg = EMBODIMENT_REGISTRY.get(embodiment)
    if reg is None:
        raise ValueError(f"Unknown embodiment '{embodiment}'. Available: {list(EMBODIMENT_REGISTRY.keys())}")

    timestep_interval = reg["timestep_interval"]
    num_action_frames = num_frames - 1
    action_delta_indices = list(range(0, num_action_frames * timestep_interval, timestep_interval))
    state_delta_indices = [0]

    action_keys = reg["action_keys"]
    state_keys = reg["state_keys"]
    action_key_configs = reg.get("action_key_configs", {})
    modality_filename = reg.get("modality_filename", "meta/modality.json")

    # Load modality metadata (maps key names to parquet column indices)
    modality_meta = _load_modality_meta(dataset_path, modality_filename)

    # Discover parquet files (one per episode)
    parquet_files = sorted(dataset_path.glob("data/*/*.parquet"))
    if not parquet_files:
        print(f"  ERROR: No parquet files found in {dataset_path / 'data'}")
        return False

    # Apply exclude_splits filtering at episode level
    if exclude_splits:
        excluded_ids = resolve_excluded_episode_indices(dataset_path, exclude_splits)
        # Parse episode index from filename (e.g., episode_000042.parquet → 42)
        filtered = []
        for pf in parquet_files:
            try:
                ep_idx = int(pf.stem.split("_")[-1])
            except ValueError:
                filtered.append(pf)  # Can't parse → keep
                continue
            if ep_idx not in excluded_ids:
                filtered.append(pf)
        n_excluded = len(parquet_files) - len(filtered)
        print(f"  exclude_splits: removed {n_excluded} episodes, {len(filtered)} remaining")
        parquet_files = filtered

    if max_samples is not None:
        parquet_files = parquet_files[:max_samples]

    print(f"  Processing {len(parquet_files)} episodes with {num_workers} workers...")

    # Create partial function with fixed arguments for the worker
    worker_fn = partial(
        _process_episode_parquet,
        modality_meta=modality_meta,
        action_key_configs=action_key_configs,
        action_delta_indices=action_delta_indices,
        state_delta_indices=state_delta_indices,
        action_keys=action_keys,
        state_keys=state_keys,
    )

    # Process episodes in parallel
    action_tracker = None
    state_tracker = None
    total_action_samples = 0
    total_state_samples = 0
    episodes_with_warnings = 0
    episodes_empty = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(worker_fn, pf): pf for pf in parquet_files}

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Stats for {dataset_path.name}"):
            actions, states, warning = future.result()

            if warning:
                episodes_with_warnings += 1
                if episodes_with_warnings <= 10:
                    print(f"  [WARN] {warning}")

            if actions.size == 0:
                episodes_empty += 1
                continue

            # Lazy-init trackers based on first result's dimensions
            if action_tracker is None:
                action_dim = actions.shape[1]
                action_tracker = StreamingStats(action_dim)
                print(f"  Action dim (post-transform): {action_dim}")
            if state_tracker is None and states.size > 0:
                state_dim = states.shape[1]
                state_tracker = StreamingStats(state_dim)
                print(f"  State dim (post-transform):  {state_dim}")

            action_tracker.update(actions.astype(np.float64))
            total_action_samples += len(actions)

            if state_tracker is not None and states.size > 0:
                state_tracker.update(states.astype(np.float64))
                total_state_samples += len(states)

    if episodes_with_warnings > 10:
        print(f"  [WARN] {episodes_with_warnings} episodes had warnings (showing first 10)")
    if episodes_empty > 0:
        print(f"  [INFO] {episodes_empty}/{len(parquet_files)} episodes produced no valid samples "
              f"(zero-norm quaternions or empty episodes)")

    if action_tracker is None or action_tracker.count == 0:
        print(f"  ERROR: No valid samples found! ({episodes_empty} empty, {episodes_with_warnings} warnings)")
        return False

    valid_episodes = len(parquet_files) - episodes_empty
    print(f"\nValid episodes: {valid_episodes}/{len(parquet_files)}")
    print(f"Total action timesteps: {total_action_samples:,}")
    print(f"Total state samples:   {total_state_samples:,}")

    # ----------------------------------------------------------------
    # Build per-key stats by slicing the concatenated action vector
    # ----------------------------------------------------------------
    action_global = action_tracker.get_stats()
    stats: dict = {}

    # Determine per-key dimensions from modality metadata + action configs
    # For rel_xyz_rot6d keys, output is 9D (not raw 7D)
    action_key_dims: dict[str, tuple[int, int]] = {}
    offset = 0
    for key in action_keys:
        modality, subkey = key.split(".", 1)
        meta = modality_meta.get(modality, {}).get(subkey)
        if meta is None:
            continue
        raw_dim = meta["end"] - meta["start"]

        cfg = action_key_configs.get(key)
        if cfg is not None and cfg.rep == "rel_xyz_rot6d":
            out_dim = 9  # xyz(3) + rot6d(6)
        else:
            out_dim = raw_dim

        action_key_dims[key] = (offset, offset + out_dim)
        offset += out_dim

    state_key_dims: dict[str, tuple[int, int]] = {}
    offset = 0
    for key in state_keys:
        modality, subkey = key.split(".", 1)
        meta = modality_meta.get(modality, {}).get(subkey)
        if meta is None:
            continue
        dim = meta["end"] - meta["start"]
        state_key_dims[key] = (offset, offset + dim)
        offset += dim

    # Extract per-key stats
    for key, (s, e) in action_key_dims.items():
        key_stats = {
            "mean": action_global["mean"][s:e],
            "std": action_global["std"][s:e],
            "min": action_global["min"][s:e],
            "max": action_global["max"][s:e],
        }
        if action_tracker.reservoir is not None:
            res = action_tracker.reservoir[:, s:e]
            key_stats["q01"] = np.quantile(res, 0.01, axis=0).tolist()
            key_stats["q99"] = np.quantile(res, 0.99, axis=0).tolist()
        else:
            key_stats["q01"] = key_stats["min"]
            key_stats["q99"] = key_stats["max"]
        stats[key] = key_stats

    if state_tracker is not None:
        state_global = state_tracker.get_stats()
        for key, (s, e) in state_key_dims.items():
            key_stats = {
                "mean": state_global["mean"][s:e],
                "std": state_global["std"][s:e],
                "min": state_global["min"][s:e],
                "max": state_global["max"][s:e],
            }
            if state_tracker.reservoir is not None:
                res = state_tracker.reservoir[:, s:e]
                key_stats["q01"] = np.quantile(res, 0.01, axis=0).tolist()
                key_stats["q99"] = np.quantile(res, 0.99, axis=0).tolist()
            else:
                key_stats["q01"] = key_stats["min"]
                key_stats["q99"] = key_stats["max"]
            stats[key] = key_stats

    # Global concatenated stats for convenience
    stats["action"] = action_global
    if state_tracker is not None:
        stats["state"] = state_tracker.get_stats()

    # ----------------------------------------------------------------
    # Write to disk
    # ----------------------------------------------------------------
    out_path = dataset_path / "meta" / output_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved stats to {out_path}")
    print(f"Per-key action stats: {list(action_key_dims.keys())}")
    if state_key_dims:
        print(f"Per-key state stats:  {list(state_key_dims.keys())}")

    for key, (s, e) in action_key_dims.items():
        dim = e - s
        print(f"  {key} ({dim}D): mean_abs={np.mean(np.abs(action_global['mean'][s:e])):.6f}, "
              f"std_mean={np.mean(action_global['std'][s:e]):.6f}")

    return True


def _resolve_embodiment_string(spec_embodiment) -> str:
    """Normalise the 'embodiment' field from OPEN_H_DATASET_SPECS to a plain string."""
    if isinstance(spec_embodiment, EmbodimentTag):
        return spec_embodiment.value
    return str(spec_embodiment)


def run_all(args):
    """Process every dataset listed in OPEN_H_DATASET_SPECS (--all mode).

    Skips CMR Versius entries (they use compute_cmr_action_stats.py and
    stats_cosmos-44D.json instead).

    Each spec's ``exclude_splits`` is forwarded so that the same episodes
    excluded during training are also excluded from statistics computation.
    """
    # Deduplicate: multiple specs may share the same (path, embodiment).
    # Keep the first occurrence's exclude_splits.
    seen: set[tuple[str, str]] = set()
    jobs: list[tuple[Path, str, list[str] | None]] = []

    for spec in OPEN_H_DATASET_SPECS:
        emb = _resolve_embodiment_string(spec["embodiment"])
        dp = Path(spec["path"])

        # CMR has its own dedicated script → skip
        if emb == EmbodimentTag.CMR_VERSIUS.value:
            continue

        key = (str(dp), emb)
        if key in seen:
            continue
        seen.add(key)
        jobs.append((dp, emb, spec.get("exclude_splits", None)))

    if not jobs:
        print("No non-CMR datasets found in OPEN_H_DATASET_SPECS.")
        return

    print("#" * 80)
    print(f"OPEN-H BATCH MODE: {len(jobs)} dataset(s) to process")
    print("#" * 80)
    for i, (dp, emb, excl) in enumerate(jobs, 1):
        excl_str = f"  exclude={excl}" if excl else ""
        print(f"  [{i:2d}] [{emb:<22s}] {dp.name}{excl_str}")
    print("#" * 80)

    total_start = time.time()
    results: dict[str, str] = {}

    for i, (dp, emb, excl) in enumerate(jobs, 1):
        print(f"\n{'#' * 80}")
        print(f"# [{i}/{len(jobs)}] embodiment={emb}  path={dp.name}")
        if excl:
            print(f"#   exclude_splits={excl}")
        print(f"{'#' * 80}")

        if not dp.exists():
            print(f"  SKIPPED — path does not exist: {dp}")
            results[f"{emb}/{dp.name}"] = "SKIPPED (path missing)"
            continue

        try:
            ok = process_single_dataset(
                dataset_path=dp,
                embodiment=emb,
                num_frames=args.num_frames,
                max_samples=args.max_samples,
                output_filename=args.output_filename,
                exclude_splits=excl,
                num_workers=args.num_workers,
            )
            results[f"{emb}/{dp.name}"] = "OK" if ok else "FAILED"
        except Exception as e:
            print(f"  ERROR: {e}")
            results[f"{emb}/{dp.name}"] = f"ERROR ({e})"

    elapsed = time.time() - total_start
    print(f"\n{'#' * 80}")
    print(f"ALL DONE — {elapsed:.1f}s total, {len(results)} dataset(s)")
    print(f"{'#' * 80}")
    for name, status in results.items():
        print(f"  {name}: {status}")
    print(f"{'#' * 80}")


def run_single(args):
    """Process a single dataset or auto-discovered datasets (original mode)."""
    if args.dataset_path:
        dataset_paths = [Path(args.dataset_path)]
    else:
        root = Path(args.dataset_path_root)
        if _is_lerobot_dataset(root):
            dataset_paths = [root]
        else:
            dataset_paths = _discover_datasets(root)

    if not dataset_paths:
        print("ERROR: No datasets found!")
        return

    exclude_splits = args.exclude_splits if args.exclude_splits else None
    print(f"Found {len(dataset_paths)} dataset(s) for embodiment '{args.embodiment}'")
    if exclude_splits:
        print(f"  exclude_splits: {exclude_splits}")

    total_start = time.time()
    results = {}
    for i, dp in enumerate(dataset_paths, 1):
        if len(dataset_paths) > 1:
            print(f"\n{'#' * 80}")
            print(f"# DATASET {i}/{len(dataset_paths)}: {dp.name}")
            print(f"{'#' * 80}")

        ok = process_single_dataset(
            dataset_path=dp,
            embodiment=args.embodiment,
            num_frames=args.num_frames,
            max_samples=args.max_samples,
            output_filename=args.output_filename,
            exclude_splits=exclude_splits,
            num_workers=args.num_workers,
        )
        results[dp.name] = "OK" if ok else "FAILED"

    elapsed = time.time() - total_start
    if len(dataset_paths) > 1:
        print(f"\n{'#' * 80}")
        print(f"ALL DONE — {elapsed:.1f}s")
        for name, status in results.items():
            print(f"  {name}: {status}")
        print(f"{'#' * 80}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute normalization stats for Open-H embodiments (post-transform)",
        epilog=(
            "Modes:\n"
            "  --all                Process ALL datasets in OPEN_H_DATASET_SPECS\n"
            "                       (skips CMR Versius — use compute_cmr_action_stats.py).\n"
            "                       No --dataset-path or --embodiment needed.\n\n"
            "  --dataset-path + --embodiment\n"
            "                       Process a single dataset with a specific embodiment.\n\n"
            "  --dataset-path-root + --embodiment\n"
            "                       Auto-discover datasets under a root directory.\n\n"
            "Examples:\n"
            "  # All Open-H datasets at once:\n"
            "  python compute_openh_action_stats.py --all\n\n"
            "  # Quick test (10 samples per dataset):\n"
            "  python compute_openh_action_stats.py --all --max-samples 10\n\n"
            "  # Single dataset:\n"
            "  python compute_openh_action_stats.py \\\n"
            "      --dataset-path /path/to/suturebot_2 --embodiment dvrk\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- mutually exclusive: --all  vs  --dataset-path / --dataset-path-root ---
    path_group = parser.add_mutually_exclusive_group(required=True)
    path_group.add_argument(
        "--all", action="store_true",
        help="Process every dataset in OPEN_H_DATASET_SPECS (skips CMR Versius)",
    )
    path_group.add_argument("--dataset-path", type=str,
                            help="Path to a single LeRobot dataset")
    path_group.add_argument("--dataset-path-root", type=str,
                            help="Root directory containing multiple LeRobot datasets")

    parser.add_argument("--embodiment", type=str, default=None,
                        choices=list(EMBODIMENT_REGISTRY.keys()),
                        help="Embodiment tag (required for --dataset-path / --dataset-path-root; "
                             "ignored for --all)")
    parser.add_argument("--exclude-splits", type=str, nargs="+", default=None,
                        help="Split names from info.json to exclude (e.g., --exclude-splits fail bad_frames). "
                             "For --all mode, exclude_splits from OPEN_H_DATASET_SPECS are used automatically.")
    parser.add_argument("--num-frames", type=int, default=13,
                        help="Number of video frames (default: 13 = 1 context + 12 prediction)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max episodes per dataset (for quick testing)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help=f"Number of parallel workers (default: min(cpu_count, {MAX_WORKERS}))")
    parser.add_argument("--output-filename", type=str, default="stats_cosmos.json",
                        help="Output filename in meta/ dir (default: stats_cosmos.json)")
    args = parser.parse_args()

    # Validate: --dataset-path / --dataset-path-root require --embodiment
    if not args.all and args.embodiment is None:
        parser.error("--embodiment is required when using --dataset-path or --dataset-path-root")

    if args.all:
        run_all(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
