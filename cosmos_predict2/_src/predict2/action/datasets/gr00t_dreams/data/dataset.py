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

import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from random import randint

import imageio
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from einops import rearrange
from PIL import Image
from pydantic import BaseModel, ValidationError
from torch.utils.data import Dataset
from tqdm import tqdm

from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.embodiment_tags import EmbodimentTag
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.schema import (
    DatasetMetadata,
    DatasetStatisticalValues,
    LeRobotModalityMetadata,
    LeRobotStateActionMetadata,
)

# from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.transform. import ComposedModalityTransform
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.transform.base import ComposedModalityTransform
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.utils.video import (
    get_all_frames,
    get_frames_by_timestamps,
)

# from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.groot_configs import construct_modality_config_and_transforms

LE_ROBOT_DEFAULT_MODALITY_FILENAME = "meta/modality.json"
LE_ROBOT_CMR_MODALITY_FILENAME = "meta/modality-44D.json"
LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"
LE_ROBOT_INFO_FILENAME = "meta/info.json"


# =============================================================================
# Split-based episode filtering (ported from gr00t-H/gr00t/data/split_utils.py)
# =============================================================================

def _parse_split_ranges(split_spec: str | list[str]) -> list[tuple[int, int]]:
    """Parse split range specs like "0:499", "42", "0:99, 200:299" into (start, end) tuples.

    Supports colon ("0:499") and dash ("0-499") separators.
    Single indices like "3" become (3, 3).
    """
    raw_specs = split_spec if isinstance(split_spec, list) else [split_spec]
    ranges: list[tuple[int, int]] = []
    for raw in raw_specs:
        parts = [p for p in raw.replace(",", " ").split() if p]
        for part in parts:
            if ":" in part:
                s, e = part.split(":", 1)
            elif "-" in part:
                s, e = part.split("-", 1)
            else:
                s, e = part, part
            start, end = int(s), int(e)
            if end < start:
                raise ValueError(f"Invalid split range '{part}': end < start")
            ranges.append((start, end))
    return ranges


def resolve_excluded_episode_indices(
    dataset_path: Path,
    exclude_splits: list[str],
) -> set[int]:
    """Resolve episode indices that should be EXCLUDED based on info.json splits.

    Reads the ``splits`` dict from ``meta/info.json`` and unions the indices
    for every split name listed in *exclude_splits*.

    Args:
        dataset_path: Root of the LeRobot dataset.
        exclude_splits: Split names to exclude (e.g., ``["fail", "bad_frames"]``).

    Returns:
        Set of episode indices to exclude.  Empty set if info.json has no matching splits.
    """
    info_path = dataset_path / LE_ROBOT_INFO_FILENAME
    if not info_path.exists():
        print(f"[exclude_splits] WARNING: info.json not found at {info_path}, cannot filter")
        return set()

    with open(info_path, "r") as f:
        info = json.load(f)

    splits = info.get("splits", {}) or {}
    # Build a lowercased lookup
    split_map: dict[str, list[int]] = {}
    for name, spec in splits.items():
        ranges = _parse_split_ranges(spec)
        indices: list[int] = []
        for start, end in ranges:
            indices.extend(range(start, end + 1))
        split_map[name.lower()] = sorted(set(indices))

    excluded: set[int] = set()
    for split_name in exclude_splits:
        key = split_name.lower()
        if key in split_map:
            excluded |= set(split_map[key])
        else:
            print(f"[exclude_splits] WARNING: split '{split_name}' not found in {info_path}, ignoring")

    return excluded


LE_ROBOT_TASKS_FILENAME = "meta/tasks.jsonl"
LE_ROBOT_STATS_FILENAME = "meta/stats.json"
LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"


def _get_rank_prefix() -> str:
    """Get distributed rank prefix for logging.
    
    Tries multiple methods to get the rank:
    1. torch.distributed.get_rank() if initialized
    2. Environment variables (RANK, LOCAL_RANK, SLURM_PROCID)
    3. Falls back to empty string if not in distributed mode
    
    Returns:
        String like "[rank0] " or "" if rank unavailable
    """
    import os
    
    # Try torch.distributed first
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return f"[rank{dist.get_rank()}] "
    except Exception:
        pass
    
    # Try environment variables
    for env_var in ["RANK", "LOCAL_RANK", "SLURM_PROCID"]:
        rank = os.environ.get(env_var)
        if rank is not None:
            return f"[rank{rank}] "
    
    return ""


def _is_cmr_versius_data(dataset_path: Path, verbose: bool = True) -> bool:
    """Check if dataset is CMR Versius by looking for hapticengaged keys in modality.json.

    CMR Versius data has specific clutch-related fields that require special filtering.
    This function detects CMR data by checking for these signature fields.

    Args:
        dataset_path: Path to the dataset root directory
        verbose: If True, print detection results

    Returns:
        True if this is CMR Versius data, False otherwise
    """
    rank = _get_rank_prefix()
    try:
        # Try CMR-specific modality file first, then default
        modality_path = dataset_path / LE_ROBOT_CMR_MODALITY_FILENAME
        if not modality_path.exists():
            modality_path = dataset_path / LE_ROBOT_DEFAULT_MODALITY_FILENAME
        if not modality_path.exists():
            if verbose:
                print(f"{rank}[CMR Detection] modality.json not found at {dataset_path}, not CMR data")
            return False

        with open(modality_path, "r") as f:
            modality_config = json.load(f)

        # Check for CMR-specific keys in state or action modalities
        # hapticengaged_left/right are the definitive CMR clutch indicators
        state_keys = set(modality_config.get("state", {}).keys())
        action_keys = set(modality_config.get("action", {}).keys())
        all_keys = state_keys | action_keys

        has_haptic_left = "hapticengaged_left" in all_keys
        has_haptic_right = "hapticengaged_right" in all_keys
        is_cmr = has_haptic_left and has_haptic_right

        if verbose:
            if is_cmr:
                print(f"{rank}[CMR Detection] Found CMR Versius signature keys in modality.json:")
                print(f"{rank}[CMR Detection]   - hapticengaged_left: {has_haptic_left}")
                print(f"{rank}[CMR Detection]   - hapticengaged_right: {has_haptic_right}")
            else:
                print(f"{rank}[CMR Detection] Not CMR data (missing hapticengaged keys)")

        return is_cmr
    except Exception as e:
        if verbose:
            print(f"{rank}[CMR Detection] Error checking for CMR data: {e}")
        return False


def calculate_dataset_statistics(parquet_paths: list[Path]) -> dict:
    """Calculate the dataset statistics of all columns for a list of parquet files."""
    # Dataset statistics
    all_low_dim_data_list = []
    # Collect all the data
    for parquet_path in tqdm(
        sorted(list(parquet_paths)),
        desc="Collecting all parquet files...",
    ):
        # Load the parquet file
        parquet_data = pd.read_parquet(parquet_path)
        parquet_data = parquet_data
        all_low_dim_data_list.append(parquet_data)
    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    # Compute dataset statistics
    dataset_statistics = {}
    for le_modality in all_low_dim_data.columns:
        print(f"Computing statistics for {le_modality}...")
        np_data = np.vstack([np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]])
        dataset_statistics[le_modality] = {
            "mean": np.mean(np_data, axis=0).tolist(),
            "std": np.std(np_data, axis=0).tolist(),
            "min": np.min(np_data, axis=0).tolist(),
            "max": np.max(np_data, axis=0).tolist(),
            "q01": np.quantile(np_data, 0.01, axis=0).tolist(),
            "q99": np.quantile(np_data, 0.99, axis=0).tolist(),
        }
    return dataset_statistics


class ModalityConfig(BaseModel):
    """Configuration for a modality."""

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""


class LeRobotSingleDataset(Dataset):
    """
    Base dataset class for LeRobot that supports sharding.
    """

    def __init__(
        self,
        dataset_path: Path | str,
        modality_configs: dict[str, ModalityConfig],
        embodiment_tag: str | EmbodimentTag,
        video_backend: str = "decord",
        video_backend_kwargs: dict | None = None,
        transforms: ComposedModalityTransform | None = None,
        single_base_index: bool = False,
        modality_filename: str | None = None,
        exclude_splits: list[str] | None = None,
    ):
        """
        Initialize the dataset.

        Args:
            dataset_path (Path | str): The path to the dataset.
            modality_configs (dict[str, ModalityConfig]): The configuration for each modality. The keys are the modality names, and the values are the modality configurations.
                See `ModalityConfig` for more details.
            video_backend (str): Backend for video reading.
            video_backend_kwargs (dict): Keyword arguments for the video backend when initializing the video reader.
            transforms (ComposedModalityTransform): The transforms to apply to the dataset.
            embodiment_tag (EmbodimentTag): Overload the embodiment tag for the dataset. e.g. define it as "new_embodiment"
            modality_filename (str | None): Path to the modality metadata JSON file relative to
                dataset_path. If None, auto-detects: tries CMR-specific file first for CMR_VERSIUS,
                then falls back to the standard modality.json.
            exclude_splits (list[str] | None): Split names from info.json to exclude
                (e.g., ["fail", "bad_frames"]). Episodes in these splits are filtered out.
        """
        # first check if the path directory exists
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")

        self.modality_configs = modality_configs
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs if video_backend_kwargs is not None else {}
        self.transforms = transforms if transforms is not None else ComposedModalityTransform(transforms=[])

        self._dataset_path = Path(dataset_path)
        self._dataset_name = self._dataset_path.name

        # Resolve modality filename: explicit > CMR-specific > default
        if modality_filename is not None:
            self._modality_filename = modality_filename
        elif (self._dataset_path / LE_ROBOT_CMR_MODALITY_FILENAME).exists():
            self._modality_filename = LE_ROBOT_CMR_MODALITY_FILENAME
        else:
            self._modality_filename = LE_ROBOT_DEFAULT_MODALITY_FILENAME
        # Default data_split for base class (can be overridden by subclasses like WrappedLeRobotSingleDataset)
        if not hasattr(self, "data_split"):
            self.data_split = "full"
        if isinstance(embodiment_tag, EmbodimentTag):
            self.tag = embodiment_tag.value
        else:
            self.tag = embodiment_tag

        # Resolve excluded episode indices from info.json splits
        self._exclude_splits = exclude_splits
        if exclude_splits:
            self._excluded_episode_ids: set[int] = resolve_excluded_episode_indices(
                self._dataset_path, exclude_splits
            )
        else:
            self._excluded_episode_ids = set()

        self._metadata = self._get_metadata(EmbodimentTag(self.tag))
        self._trajectory_ids, self._trajectory_lengths = self._get_trajectories()
        self._all_steps = self._get_all_steps(single_base_index=single_base_index)
        self._modality_keys = self._get_modality_keys()
        self._delta_indices = self._get_delta_indices()
        self.set_transforms_metadata(self.metadata)
        self.set_epoch(0)

        print(f"Initialized dataset {self.dataset_name} with {embodiment_tag}")

        # LeRobot-specific config
        self._lerobot_modality_meta = self._get_lerobot_modality_meta()
        self._lerobot_info_meta = self._get_lerobot_info_meta()
        self._data_path_pattern = self._get_data_path_pattern()
        self._video_path_pattern = self._get_video_path_pattern()
        self._chunk_size = self._get_chunk_size()
        self._tasks = self._get_tasks()
        self.curr_traj_data = None
        self.curr_traj_id = None

        # Check if the dataset is valid
        self._check_integrity()

    @property
    def dataset_path(self) -> Path:
        """The path to the dataset that contains the METADATA_FILENAME file."""
        return self._dataset_path

    @property
    def metadata(self) -> DatasetMetadata:
        """The metadata for the dataset, loaded from metadata.json in the dataset directory"""
        return self._metadata

    @property
    def trajectory_ids(self) -> np.ndarray:
        """The trajectory IDs in the dataset, stored as a 1D numpy array of strings."""
        return self._trajectory_ids

    @property
    def trajectory_lengths(self) -> np.ndarray:
        """The trajectory lengths in the dataset, stored as a 1D numpy array of integers.
        The order of the lengths is the same as the order of the trajectory IDs.
        """
        return self._trajectory_lengths

    @property
    def all_steps(self) -> list[tuple[int, int]]:
        """The trajectory IDs and base indices for all steps in the dataset.
        Example:
            self.trajectory_ids: [0, 1, 2]
            self.trajectory_lengths: [3, 2, 4]
            return: [
                ("traj_0", 0), ("traj_0", 1), ("traj_0", 2),
                ("traj_1", 0), ("traj_1", 1),
                ("traj_2", 0), ("traj_2", 1), ("traj_2", 2), ("traj_2", 3)
            ]
        """
        return self._all_steps

    @property
    def modality_keys(self) -> dict:
        """The modality keys for the dataset. The keys are the modality names, and the values are the keys for each modality.

        Example: {
            "video": ["video.image_side_0", "video.image_side_1"],
            "state": ["state.eef_position", "state.eef_rotation"],
            "action": ["action.eef_position", "action.eef_rotation"],
            "language": ["language.human.task"],
            "timestamp": ["timestamp"],
            "reward": ["reward"],
        }
        """
        return self._modality_keys

    @property
    def delta_indices(self) -> dict[str, np.ndarray]:
        """The delta indices for the dataset. The keys are the modality.key, and the values are the delta indices for each modality.key."""
        return self._delta_indices

    @property
    def dataset_name(self) -> str:
        """The name of the dataset."""
        return self._dataset_name

    @property
    def lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_modality_meta

    @property
    def lerobot_info_meta(self) -> dict:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_info_meta

    @property
    def data_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._data_path_pattern

    @property
    def video_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._video_path_pattern

    @property
    def chunk_size(self) -> int:
        """The chunk size for the LeRobot dataset."""
        return self._chunk_size

    @property
    def tasks(self) -> pd.DataFrame:
        """The tasks for the dataset."""
        return self._tasks

    def _get_metadata(self, embodiment_tag: EmbodimentTag) -> DatasetMetadata:
        """Get the metadata for the dataset.

        Returns:
            dict: The metadata for the dataset.
        """

        # 1. Modality metadata
        modality_meta_path = self.dataset_path / self._modality_filename
        if not (modality_meta_path.exists()):
            modality_meta_path = Path(
                "/mnt/amlfs-03/shared/datasets/agibot-beta-converted-0512/agibotworld/modality.json"
            )
            print(
                "WARNING: Could not find modality.json in dataset path, falling back to /mnt/amlfs-03/shared/datasets/agibot-beta-converted-0512/agibotworld/modality.json"
            )
        assert modality_meta_path.exists(), f"Please provide a {self._modality_filename} file in {self.dataset_path}"

        # 1.1. State and action modalities
        simplified_modality_meta: dict[str, dict] = {}
        with open(modality_meta_path, "r") as f:
            le_modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        for modality in ["state", "action"]:
            simplified_modality_meta[modality] = {}
            le_state_action_meta: dict[str, LeRobotStateActionMetadata] = getattr(le_modality_meta, modality)
            for subkey in le_state_action_meta:
                state_action_dtype = np.dtype(le_state_action_meta[subkey].dtype)
                if np.issubdtype(state_action_dtype, np.floating):
                    continuous = True
                else:
                    continuous = False
                simplified_modality_meta[modality][subkey] = {
                    "absolute": le_state_action_meta[subkey].absolute,
                    "rotation_type": le_state_action_meta[subkey].rotation_type,
                    "shape": [le_state_action_meta[subkey].end - le_state_action_meta[subkey].start],
                    "continuous": continuous,
                }

        # 1.2. Video modalities
        le_info_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        assert le_info_path.exists(), f"Please provide a {LE_ROBOT_INFO_FILENAME} file in {self.dataset_path}"
        with open(le_info_path, "r") as f:
            le_info = json.load(f)
        simplified_modality_meta["video"] = {}
        # Top-level fps fallback (used when per-feature fps is unavailable)
        top_level_fps = le_info.get("fps", 30)

        for new_key in le_modality_meta.video:
            original_key = le_modality_meta.video[new_key].original_key
            if original_key is None:
                original_key = new_key

            # Find the feature entry — different LeRobot versions use different key patterns
            le_video_meta = None
            for candidate_key in [original_key, f"observation.images.{original_key}"]:
                if candidate_key in le_info.get("features", {}):
                    le_video_meta = le_info["features"][candidate_key]
                    break

            if le_video_meta is None:
                # Feature not in info.json — use sensible defaults
                print(f"WARNING: Video feature '{original_key}' not found in info.json features, "
                      f"using defaults (resolution will be determined at decode time)")
                simplified_modality_meta["video"][new_key] = {
                    "resolution": [512, 288],  # default, overridden at decode time
                    "channels": 3,
                    "fps": top_level_fps,
                }
                continue

            # Extract height/width — different LeRobot versions use different naming
            # Possible names: "height"/"width", "h"/"w", or positional [channels, height, width]
            names = le_video_meta.get("names", [])
            shape = le_video_meta.get("shape", [])

            height, width, channels = None, None, 3  # defaults

            # Try named lookup with multiple possible names
            for h_name in ["height", "h"]:
                if h_name in names:
                    height = shape[names.index(h_name)]
                    break
            for w_name in ["width", "w"]:
                if w_name in names:
                    width = shape[names.index(w_name)]
                    break
            for c_name in ["channel", "channels", "c"]:
                if c_name in names:
                    channels = shape[names.index(c_name)]
                    break

            # Fallback: positional inference from shape (common pattern: [C, H, W] or [H, W, C])
            if (height is None or width is None) and len(shape) >= 3:
                if shape[0] <= 4:  # CHW format (channels first)
                    channels, height, width = shape[0], shape[1], shape[2]
                else:  # HWC format
                    height, width, channels = shape[0], shape[1], shape[2]

            if height is None or width is None:
                height, width = 288, 512  # safe default

            # Extract fps — different LeRobot versions store it differently
            fps = top_level_fps  # default fallback
            for fps_source in [
                lambda: le_video_meta["video_info"]["video.fps"],
                lambda: le_video_meta["info"]["video.fps"],
                lambda: le_video_meta.get("fps", None),
            ]:
                try:
                    val = fps_source()
                    if val is not None:
                        fps = val
                        break
                except (KeyError, TypeError):
                    continue

            simplified_modality_meta["video"][new_key] = {
                "resolution": [width, height],
                "channels": channels,
                "fps": fps,
            }

        # 2. Dataset statistics
        # Priority order for stats file resolution:
        #   1. stats_cosmos-44D.json  (CMR Versius 44D conditioning stats)
        #   2. stats_cosmos.json      (generic Open-H post-transform stats)
        #   3. stats.json             (standard LeRobot raw column stats)
        #
        # stats_cosmos*.json files contain per-key statistics computed on the
        # TRANSFORMED action representation (e.g., 9D rot6d instead of 7D quat).
        # These are generated by:
        #   - CMR: scripts/compute_cmr_action_stats.py  → stats_cosmos-44D.json
        #   - Others: scripts/compute_openh_action_stats.py → stats_cosmos.json
        stats_path = self.dataset_path / LE_ROBOT_STATS_FILENAME
        if "agibot" in str(stats_path):
            print(
                "NOTE: Using standard action normalization at /mnt/amlfs-03/shared/datasets/agibot-beta-converted-0512/agibotworld/stats.json"
            )
            stats_path = Path("/mnt/amlfs-03/shared/datasets/agibot-beta-converted-0512/agibotworld/stats.json")
        elif embodiment_tag == EmbodimentTag.CMR_VERSIUS:
            # CMR Versius uses stats_cosmos-44D.json with hybrid-relative action statistics
            # (generated by scripts/compute_cmr_action_stats.py)
            cosmos_stats_path = self.dataset_path / "meta/stats_cosmos-44D.json"
            if cosmos_stats_path.exists():
                stats_path = cosmos_stats_path
                print(f"{_get_rank_prefix()}NOTE: CMR Versius using stats_cosmos-44D.json: {stats_path}")
            else:
                raise FileNotFoundError(
                    f"CMR Versius requires stats_cosmos-44D.json but not found at: {cosmos_stats_path}\n"
                    f"Run 'python scripts/compute_cmr_action_stats.py --dataset-path {self.dataset_path}' to generate it."
                )
        elif self.tag in self._get_open_h_tags():
            # Open-H embodiments REQUIRE stats_cosmos.json because
            # GenericRelativeActionTransform changes the action dimensionality
            # (e.g., 7D quat → 9D rot6d).  Falling back to raw stats.json
            # would apply wrong-dimensioned statistics and silently corrupt
            # the normalized actions.
            cosmos_stats_path = self.dataset_path / "meta/stats_cosmos.json"
            if cosmos_stats_path.exists():
                stats_path = cosmos_stats_path
                print(f"{_get_rank_prefix()}NOTE: Using post-transform stats: {stats_path}")
            else:
                raise FileNotFoundError(
                    f"\n{'='*80}\n"
                    f"MISSING POST-TRANSFORM STATISTICS\n"
                    f"{'='*80}\n"
                    f"Dataset:    {self.dataset_path}\n"
                    f"Embodiment: {self.tag}\n\n"
                    f"Open-H embodiments require stats_cosmos.json with statistics\n"
                    f"computed on the TRANSFORMED action representation (e.g., 9D rot6d\n"
                    f"instead of raw 7D quat).  The standard stats.json contains raw\n"
                    f"column statistics which are INCOMPATIBLE after delta conversion.\n\n"
                    f"To generate the required stats file, run:\n"
                    f"    python scripts/compute_openh_action_stats.py \\\n"
                    f"        --dataset-path {self.dataset_path} \\\n"
                    f"        --embodiment {self.tag}\n"
                    f"{'='*80}\n"
                )
        try:
            with open(stats_path, "r") as f:
                le_statistics = json.load(f)
            for stat in le_statistics.values():
                if isinstance(stat, int):
                    continue
                DatasetStatisticalValues.model_validate(stat)
        except (FileNotFoundError, ValidationError) as e:
            print(f"Failed to load dataset statistics: {e}")
            print(f"Calculating dataset statistics for {self.dataset_name}")
            # Get all parquet files in the dataset paths
            parquet_files = list((self.dataset_path).glob(LE_ROBOT_DATA_FILENAME))
            le_statistics = calculate_dataset_statistics(parquet_files)
        dataset_statistics = {}
        stats_log = {"per_key": [], "index_based": [], "skipped": []}
        for our_modality in ["state", "action"]:
            dataset_statistics[our_modality] = {}
            for subkey in simplified_modality_meta[our_modality]:
                full_key = f"{our_modality}.{subkey}"

                # For CMR Versius: stats_cosmos.json has per-key statistics already computed
                # (e.g., "action.left_pose", "state.right_gripper") because the transformed
                # action dimensions (9D rot6d) don't match modality.json indices (7D quat).
                # Use per-key stats directly if available.
                if full_key in le_statistics:
                    # Per-key statistics exist - use directly
                    dataset_statistics[our_modality][subkey] = {}
                    for stat_name, stat_value in le_statistics[full_key].items():
                        dataset_statistics[our_modality][subkey][stat_name] = (
                            stat_value if isinstance(stat_value, list) else [stat_value]
                        )
                    dim = len(dataset_statistics[our_modality][subkey].get("mean", []))
                    stats_log["per_key"].append(f"{full_key} ({dim}D)")
                else:
                    # Fall back to index-based extraction from global array
                    state_action_meta = le_modality_meta.get_key_meta(full_key)
                    assert isinstance(state_action_meta, LeRobotStateActionMetadata)
                    le_modality = state_action_meta.original_key

                    # For CMR Versius, some keys (e.g., hapticengaged_*, armlinkedtohaptic_*)
                    # are passthrough keys used internally but not in the final output.
                    # These reference 'observation.state' which doesn't exist in stats_cosmos.json.
                    # Skip these entirely - they don't need normalization statistics and
                    # should not be added to dataset_statistics (pydantic validation would fail).
                    if le_modality not in le_statistics:
                        stats_log["skipped"].append(f"{full_key} (ref: {le_modality})")
                        continue

                    dataset_statistics[our_modality][subkey] = {}
                    for stat_name in le_statistics[le_modality]:
                        indices = np.arange(
                            state_action_meta.start,
                            state_action_meta.end,
                        )
                        stat = np.array(le_statistics[le_modality][stat_name])
                        dataset_statistics[our_modality][subkey][stat_name] = stat[indices].tolist()
                    dim = state_action_meta.end - state_action_meta.start
                    stats_log["index_based"].append(f"{full_key} [{state_action_meta.start}:{state_action_meta.end}] from '{le_modality}' ({dim}D)")

        # Log statistics loading summary
        rank = _get_rank_prefix()
        print(f"{rank}[Stats Loading] Per-key stats ({len(stats_log['per_key'])}): {', '.join(stats_log['per_key'])}")
        if stats_log["index_based"]:
            print(f"{rank}[Stats Loading] Index-based extraction ({len(stats_log['index_based'])}): {', '.join(stats_log['index_based'])}")
        if stats_log["skipped"]:
            print(f"{rank}[Stats Loading] Skipped passthrough keys ({len(stats_log['skipped'])}): {', '.join(stats_log['skipped'])}")

        # 3. Full dataset metadata
        metadata = DatasetMetadata(
            statistics=dataset_statistics,  # type: ignore
            modalities=simplified_modality_meta,  # type: ignore
            embodiment_tag=embodiment_tag,
        )

        return metadata

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the trajectories in the dataset."""
        # Get trajectory lengths, IDs, and whitelist from dataset metadata
        episode_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
        with open(episode_path, "r") as f:
            episode_metadata = [json.loads(line) for line in f]
        trajectory_ids = []
        trajectory_lengths = []
        for episode in episode_metadata:
            trajectory_ids.append(episode["episode_index"])
            trajectory_lengths.append(episode["length"])
        return np.array(trajectory_ids), np.array(trajectory_lengths)

    def _get_all_steps(self, single_base_index=False) -> list[tuple[int, int]]:
        """Get the trajectory IDs and base indices for all steps in the dataset.

        For CMR Versius data, this method applies clutch-aware filtering to discard
        invalid training samples. The filtering removes samples where:

        Rule 1: armlinkedtohaptic_* changes within the action horizon (arm swap)
                - Invalid because the relative action computation assumes consistent
                  arm identity throughout the action horizon.

        Rule 2: Both hapticengaged_left and hapticengaged_right are False for
                the entire action horizon (fully disengaged)
                - No useful training signal when the surgeon is in menus or
                  repositioning without controlling any arm.

        Note: Rule 3 (masking disengaged timesteps within valid samples) is
        handled at transform-time by CMRVersiusRelativeActionTransform.

        References:
            - gr00t-H/working_log/01_13_26/clutch_problem.md
            - gr00t-H/gr00t/data/dataset/sharded_single_step_dataset.py

        Returns:
            list[tuple[int, int]]: A list of (trajectory_id, base_index) tuples.

        Example (without CMR filtering):
            self.trajectory_ids: [0, 1, 2]
            self.trajectory_lengths: [3, 2, 4]
            return: [
                (0, 0), (0, 1), (0, 2),
                (1, 0), (1, 1),
                (2, 0), (2, 1), (2, 2), (2, 3)
            ]
        """
        # Check if this is CMR Versius data that requires clutch-aware filtering
        is_cmr_data = _is_cmr_versius_data(self.dataset_path)
        rank = _get_rank_prefix()

        if is_cmr_data and not single_base_index:
            # Apply CMR clutch-aware filtering
            print(f"\n{rank}[CMR Detection] Dataset '{self.dataset_name}' identified as CMR Versius data")
            print(f"{rank}[CMR Detection] Clutch-aware filtering will be applied")
            return self._get_all_steps_cmr_filtered()
        else:
            # Standard step enumeration with optional split-based exclusion
            if is_cmr_data and single_base_index:
                print(f"\n{rank}[CMR Detection] CMR data detected but single_base_index=True, skipping filtering")

            excluded = self._excluded_episode_ids
            if excluded:
                print(f"{rank}[exclude_splits] Excluding {len(excluded)} episode(s) "
                      f"from splits {self._exclude_splits}: {sorted(excluded)[:20]}{'...' if len(excluded) > 20 else ''}")

            all_steps: list[tuple[int, int]] = []
            skipped_episodes = 0
            for trajectory_id, trajectory_length in zip(self.trajectory_ids, self.trajectory_lengths):
                if int(trajectory_id) in excluded:
                    skipped_episodes += 1
                    continue
                if single_base_index:
                    all_steps.append((trajectory_id, 0))
                else:
                    for base_index in range(trajectory_length):
                        all_steps.append((trajectory_id, base_index))

            total_eps = len(self.trajectory_ids)
            used_eps = total_eps - skipped_episodes
            print(f"{rank}[Step Enumeration] {len(all_steps):,} steps from {used_eps}/{total_eps} episodes"
                  f"{f' ({skipped_episodes} excluded by split filter)' if skipped_episodes else ''}")
            return all_steps

    def _get_all_steps_cmr_filtered(self) -> list[tuple[int, int]]:
        """Get all valid steps for CMR Versius data with clutch-aware filtering.

        This method loads PRE-COMPUTED filter cache from disk. The cache must be
        generated BEFORE training using:
            python scripts/compute_cmr_filtered_episodes_cache.py

        The cache file is stored in the dataset's meta/ directory with a hash based
        on action_delta_indices to ensure consistency when hyperparameters change.

        IMPORTANT: This method will FAIL if the cache file doesn't exist. This is
        intentional to avoid expensive computation during distributed training startup.

        Returns:
            list[tuple[int, int]]: Filtered list of (trajectory_id, base_index) tuples.

        Raises:
            FileNotFoundError: If the pre-computed cache file doesn't exist.
            ValueError: If the cache file is invalid or has mismatched action_delta_indices.
        """
        import hashlib
        import time

        rank = _get_rank_prefix()
        start_time = time.time()

        # Get action delta indices from modality config
        action_config = self.modality_configs.get("action")
        if action_config is None:
            # No action config, can't filter - return all steps
            print(f"{rank}[CMR Filter] WARNING: No action config found, skipping CMR clutch filtering")
            all_steps: list[tuple[int, int]] = []
            for trajectory_id, trajectory_length in zip(self.trajectory_ids, self.trajectory_lengths):
                for base_index in range(trajectory_length):
                    all_steps.append((trajectory_id, base_index))
            return all_steps

        action_delta_indices = action_config.delta_indices
        print(f"{rank}[CMR Filter] Using action delta indices: {action_delta_indices[:5]}{'...' if len(action_delta_indices) > 5 else ''} (len={len(action_delta_indices)})")

        # Generate cache key based on action_delta_indices (primary factor affecting filtering)
        # Include dataset name and split for uniqueness across different dataset configurations
        cache_key_data = f"{self._dataset_name}_{self.data_split}_{sorted(action_delta_indices)}"
        cache_hash = hashlib.md5(cache_key_data.encode()).hexdigest()[:12]
        cache_filename = f"cmr_filter_cache_{self.data_split}_{cache_hash}-44D.json"
        cache_path = self.dataset_path / "meta" / cache_filename

        # Load pre-computed cache (MUST exist - no fallback computation)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"\n{'='*80}\n"
                f"CMR VERSIUS FILTER CACHE NOT FOUND\n"
                f"{'='*80}\n"
                f"Expected cache file: {cache_path}\n\n"
                f"The CMR clutch-aware filter cache must be pre-computed BEFORE training.\n"
                f"This is required to avoid expensive computation during distributed training startup.\n\n"
                f"To generate the cache, run:\n"
                f"    python scripts/compute_cmr_filtered_episodes_cache.py \\\n"
                f"        --dataset-path {self.dataset_path} \\\n"
                f"        --split {self.data_split} \\\n"
                f"        --num-frames {len(action_delta_indices)}\n\n"
                f"Or to generate caches for all default CMR datasets:\n"
                f"    python scripts/compute_cmr_filtered_episodes_cache.py\n"
                f"{'='*80}\n"
            )

        print(f"{rank}[CMR Filter] Loading pre-computed cache from: {cache_path}")
        with open(cache_path, "r") as f:
            cache_data = json.load(f)

        # Verify cache is valid (same action_delta_indices)
        cached_indices = cache_data.get("action_delta_indices", [])
        if cached_indices != action_delta_indices:
            raise ValueError(
                f"\n{'='*80}\n"
                f"CMR VERSIUS FILTER CACHE INVALID\n"
                f"{'='*80}\n"
                f"Cache file: {cache_path}\n\n"
                f"The cached action_delta_indices don't match the current configuration:\n"
                f"  Cached:  {cached_indices[:5]}{'...' if len(cached_indices) > 5 else ''} (len={len(cached_indices)})\n"
                f"  Current: {action_delta_indices[:5]}{'...' if len(action_delta_indices) > 5 else ''} (len={len(action_delta_indices)})\n\n"
                f"This can happen if num_frames or timestep_interval changed.\n"
                f"Please regenerate the cache:\n"
                f"    python scripts/compute_cmr_filtered_episodes_cache.py \\\n"
                f"        --dataset-path {self.dataset_path} \\\n"
                f"        --split {self.data_split} \\\n"
                f"        --num-frames {len(action_delta_indices)} \\\n"
                f"        --force\n"
                f"{'='*80}\n"
            )

        # Extract all_steps from cache
        all_steps = [(int(ep), int(idx)) for ep, idx in cache_data["all_steps"]]
        cached_stats = cache_data.get("stats", {})

        elapsed_time = time.time() - start_time
        print(f"{rank}[CMR Filter] Loaded {len(all_steps):,} valid samples from cache in {elapsed_time:.2f}s")
        print(f"{rank}[CMR Filter] Cache stats: {cached_stats}")

        return all_steps

    def _get_modality_keys(self) -> dict:
        """Get the modality keys for the dataset.
        The keys are the modality names, and the values are the keys for each modality.
        See property `modality_keys` for the expected format.
        """
        modality_keys = defaultdict(list)
        for modality, config in self.modality_configs.items():
            modality_keys[modality] = config.modality_keys
        return modality_keys

    def _get_delta_indices(self) -> dict[str, np.ndarray]:
        """Restructure the delta indices to use modality.key as keys instead of just the modalities."""
        delta_indices: dict[str, np.ndarray] = {}
        for config in self.modality_configs.values():
            for key in config.modality_keys:
                delta_indices[key] = np.array(config.delta_indices)
        return delta_indices

    def _get_lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """Get the metadata for the LeRobot dataset."""
        modality_meta_path = self.dataset_path / self._modality_filename
        if not (modality_meta_path.exists()):
            modality_meta_path = Path(
                "/mnt/amlfs-03/shared/datasets/agibot-beta-converted-0512/agibotworld/modality.json"
            )
            print(
                "WARNING: Could not find modality.json in dataset path, falling back to /mnt/amlfs-03/shared/datasets/agibot-beta-converted-0512/agibotworld/modality.json"
            )
        assert modality_meta_path.exists(), f"Please provide a {self._modality_filename} file in {self.dataset_path}"
        with open(modality_meta_path, "r") as f:
            modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        return modality_meta

    def _get_lerobot_info_meta(self) -> dict:
        """Get the metadata for the LeRobot dataset."""
        info_meta_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        with open(info_meta_path, "r") as f:
            info_meta = json.load(f)
        return info_meta

    def _get_data_path_pattern(self) -> str:
        """Get the data path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["data_path"]

    def _get_video_path_pattern(self) -> str:
        """Get the video path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["video_path"]

    def _get_chunk_size(self) -> int:
        """Get the chunk size for the LeRobot dataset."""
        return self.lerobot_info_meta["chunks_size"]

    def _get_tasks(self) -> pd.DataFrame:
        """Get the tasks for the dataset."""
        tasks_path = self.dataset_path / LE_ROBOT_TASKS_FILENAME
        with open(tasks_path, "r") as f:
            tasks = [json.loads(line) for line in f]
        df = pd.DataFrame(tasks)
        return df.set_index("task_index")

    def _check_integrity(self):
        """Use the config to check if the keys are valid and detect silent data corruption."""
        ERROR_MSG_HEADER = f"Error occurred in initializing dataset {self.dataset_name}:\n"

        for modality_config in self.modality_configs.values():
            for key in modality_config.modality_keys:
                if key == "lapa_action" or key == "dream_actions":
                    continue  # no need for any metadata for lapa actions because it comes normalized
                # Check if the key is valid
                try:
                    self.lerobot_modality_meta.get_key_meta(key)
                except Exception as e:
                    raise ValueError(ERROR_MSG_HEADER + f"Unable to find key {key} in modality metadata:\n{e}")

    @staticmethod
    def _get_open_h_tags() -> frozenset[str]:
        """Return the set of Open-H embodiment tag strings that require stats_cosmos.json.

        Uses a lazy import to avoid a circular dependency (groot_configs.py
        imports ModalityConfig from this module).  The result is cached after
        the first call by Python's module import caching.
        """
        from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.groot_configs import (
            OPEN_H_EMBODIMENT_TAGS,
        )
        return OPEN_H_EMBODIMENT_TAGS

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        self.transforms.set_metadata(metadata)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        """Get the total number of data points in the dataset.

        Returns:
            int: the total number of data points in the dataset.
        """
        return len(self.all_steps)

    def __str__(self) -> str:
        """Get the description of the dataset."""
        return f"{self.dataset_name} ({len(self)} steps)"

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single step in a trajectory.

        Args:
            index (int): The index of the step to get.

        Returns:
            dict: The data for the step.
        """
        trajectory_id, base_index = self.all_steps[index]
        return self.transforms(self.get_step_data(trajectory_id, base_index))

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Get the RAW data for a single step in a trajectory. No transforms are applied.

        Args:
            trajectory_id (int): The name of the trajectory.
            base_index (int): The base step index in the trajectory.

        Returns:
            dict: The RAW data for the step.

        Example return:
            {
                "video": {
                    "video.image_side_0": [B, T, H, W, C],
                    "video.image_side_1": [B, T, H, W, C],
                },
                "state": {
                    "state.eef_position": [B, T, state_dim],
                    "state.eef_rotation": [B, T, state_dim],
                },
                "action": {
                    "action.eef_position": [B, T, action_dim],
                    "action.eef_rotation": [B, T, action_dim],
                },
            }
        """
        data = {}
        # Get the data for all modalities
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        return data

    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        """Get the data for a trajectory."""
        if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
            return self.curr_traj_data
        else:
            chunk_index = self.get_episode_chunk(trajectory_id)
            parquet_path = self.dataset_path / self.data_path_pattern.format(
                episode_chunk=chunk_index, episode_index=trajectory_id
            )
            assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
            return pd.read_parquet(parquet_path)

    def get_trajectory_index(self, trajectory_id: int) -> int:
        """Get the index of the trajectory in the dataset by the trajectory ID.
        This is useful when you need to get the trajectory length or sampling weight corresponding to the trajectory ID.

        Args:
            trajectory_id (str): The ID of the trajectory.

        Returns:
            int: The index of the trajectory in the dataset.
        """
        trajectory_indices = np.where(self.trajectory_ids == trajectory_id)[0]
        if len(trajectory_indices) != 1:
            raise ValueError(f"Error finding trajectory index for {trajectory_id}, found {trajectory_indices=}")
        return trajectory_indices[0]

    def get_episode_chunk(self, ep_index: int) -> int:
        """Get the chunk index for an episode index."""
        return ep_index // self.chunk_size

    def retrieve_data_and_pad(
        self,
        array: np.ndarray,
        step_indices: np.ndarray,
        max_length: int,
        padding_strategy: str = "first_last",
    ) -> np.ndarray:
        """Retrieve the data from the dataset and pad it if necessary.
        Args:
            array (np.ndarray): The array to retrieve the data from.
            step_indices (np.ndarray): The step indices to retrieve the data for.
            max_length (int): The maximum length of the data.
            padding_strategy (str): The padding strategy, either "first" or "last".
        """
        # Get the padding indices
        front_padding_indices = step_indices < 0
        end_padding_indices = step_indices >= max_length
        padding_positions = np.logical_or(front_padding_indices, end_padding_indices)
        # Retrieve the data with the non-padding indices
        # If there exists some padding, Given T step_indices, the shape of the retrieved data will be (T', ...) where T' < T
        raw_data = array[step_indices[~padding_positions]]
        assert isinstance(raw_data, np.ndarray), f"{type(raw_data)=}"
        # This is the shape of the output, (T, ...)
        if raw_data.ndim == 1:
            expected_shape = (len(step_indices),)
        else:
            expected_shape = (len(step_indices), *array.shape[1:])

        # Pad the data
        output = np.zeros(expected_shape)
        # Assign the non-padded data
        output[~padding_positions] = raw_data
        # If there exists some padding, pad the data
        if padding_positions.any():
            if padding_strategy == "first_last":
                # Use first / last step data to pad
                front_padding_data = array[0]
                end_padding_data = array[-1]
                output[front_padding_indices] = front_padding_data
                output[end_padding_indices] = end_padding_data
            elif padding_strategy == "zero":
                # Use zero padding
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    def get_video_path(self, trajectory_id: int, key: str) -> Path:
        chunk_index = self.get_episode_chunk(trajectory_id)
        original_key = self.lerobot_modality_meta.video[key].original_key
        if original_key is None:
            original_key = key
        video_filename = self.video_path_pattern.format(
            episode_chunk=chunk_index, episode_index=trajectory_id, video_key=original_key
        )
        if not (self.dataset_path / video_filename).exists():
            original_key = f"observation.images.{original_key}"
            video_filename = self.video_path_pattern.format(
                episode_chunk=chunk_index, episode_index=trajectory_id, video_key=original_key
            )
        return self.dataset_path / video_filename

    def get_video(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the video frames for a trajectory by a base index.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (str): The ID of the trajectory.
            key (str): The key of the video.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The video frames for the trajectory and frame indices. Shape: (T, H, W, C)
        """
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # print(f"{step_indices=}")
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        # if step_indices[-1] >= self.trajectory_lengths[trajectory_index]:
        #     step_indices -= (self.trajectory_lengths[trajectory_index] - step_indices[-1] + 1)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        video_path = self.get_video_path(trajectory_id, key)
        # Get the action/state timestamps for each frame in the video
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert "timestamp" in self.curr_traj_data.columns, f"No timestamp found in {trajectory_id=}"
        timestamp: np.ndarray = self.curr_traj_data["timestamp"].to_numpy()
        # Get the corresponding video timestamps from the step indices
        video_timestamp = timestamp[step_indices]

        try:
            return get_frames_by_timestamps(
                video_path.as_posix(),
                video_timestamp,
                video_backend=self.video_backend,
                video_backend_kwargs=self.video_backend_kwargs,
            )
        except Exception:
            self.video_backend = "torchvision_av"
            return get_frames_by_timestamps(
                video_path.as_posix(),
                video_timestamp,
                video_backend=self.video_backend,
                video_backend_kwargs=self.video_backend_kwargs,
            )

    def get_state_or_action(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the state or action data for a trajectory by a base index.
        If the step indices are out of range, pad with the data:
            if the data is stored in absolute format, pad with the first or last step data;
            otherwise, pad with zero.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The data for the trajectory and step indices.
        """
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        assert key.startswith(modality + "."), f"{key} must start with {modality + '.'}, got {key}"
        # Get the sub-key, e.g. state.joint_angles -> joint_angles
        key = key.replace(modality + ".", "")
        # Get the lerobot key
        le_state_or_action_cfg = getattr(self.lerobot_modality_meta, modality)
        le_key = le_state_or_action_cfg[key].original_key
        if le_key is None:
            le_key = key
        # Get the data array, shape: (T, D)
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert le_key in self.curr_traj_data.columns, f"No {le_key} found in {trajectory_id=}"
        data_array: np.ndarray = np.stack(self.curr_traj_data[le_key])  # type: ignore
        assert data_array.ndim == 2, f"Expected 2D array, got {data_array.shape} array"
        le_indices = np.arange(
            le_state_or_action_cfg[key].start,
            le_state_or_action_cfg[key].end,
        )
        data_array = data_array[:, le_indices]
        # Get the state or action configuration
        state_or_action_cfg = getattr(self.metadata.modalities, modality)[key]

        # Pad the data
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy="first_last" if state_or_action_cfg.absolute else "zero",
        )

    def get_language(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> list[str]:
        """Get the language annotation data for a trajectory by step indices.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the annotation.
            base_index (int): The base index of the trajectory.

        Returns:
            list[str]: The annotation data for the trajectory and step indices. If no matching data is found, return empty strings.
        """
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        # Get the end times corresponding to the closest indices
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)
        # Get the annotations
        task_indices: list[int] = []
        assert key.startswith("annotation."), f"Language key must start with 'annotation.', got {key}"
        subkey = key.replace("annotation.", "")
        annotation_meta = self.lerobot_modality_meta.annotation
        assert annotation_meta is not None, f"Annotation metadata is None for {subkey}"
        assert subkey in annotation_meta, (
            f"Annotation key {subkey} not found in metadata, available annotation keys: {annotation_meta.keys()}"
        )
        subkey_meta = annotation_meta[subkey]
        original_key = subkey_meta.original_key
        if original_key is None:
            original_key = key
        for i in range(len(step_indices)):
            task_indices.append(self.curr_traj_data[original_key][step_indices[i]].item())
        return self.tasks.loc[task_indices]["task"].tolist()

    def get_data_by_modality(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ):
        """Get the data corresponding to the modality for a trajectory by a base index.
        This method will call the corresponding helper method based on the modality.
        See the helper methods for more details.
        NOTE: For the language modality, the data is padded with empty strings if no matching data is found.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.
        """
        if modality == "video":
            return self.get_video(trajectory_id, key, base_index)
        elif modality == "state" or modality == "action":
            return self.get_state_or_action(trajectory_id, modality, key, base_index)
        elif modality == "language":
            return self.get_language(trajectory_id, key, base_index)
        else:
            raise ValueError(f"Invalid modality: {modality}")


class CachedLeRobotSingleDataset(LeRobotSingleDataset):
    def __init__(self, img_resize: tuple[int, int] | None = None, *args, **kwargs):
        """
        This class caches the video frames for each trajectory and key.
        It is recommended to use this class if the video frames need to be accessed multiple times.

        Args:
            resize_img (tuple[int, int], optional): The size to resize the video frames to reduce memory usage.
        """
        # Convert img_resize to tuple if it is not already
        if img_resize is not None and not isinstance(img_resize, tuple):
            img_resize = tuple(img_resize)
            assert len(img_resize) == 2, f"Expected tuple of length 2, got {img_resize}"
        self.img_resize = img_resize

        # Initialize img_resize attribute first to ensure it exists
        super().__init__(*args, **kwargs)
        cached_frames: dict[str, np.ndarray] = {}

        for key in self.modality_keys["video"]:
            all_frames = []
            key = key.replace("video.", "")
            for trajectory_id, trajectory_length in tqdm(
                zip(self.trajectory_ids, self.trajectory_lengths),
                total=len(self.trajectory_ids),
                desc=f"Caching {key} frames",
            ):
                video_path = self.get_video_path(trajectory_id, key)
                frames = get_all_frames(
                    video_path.as_posix(),
                    video_backend=self.video_backend,
                    video_backend_kwargs=self.video_backend_kwargs,
                    resize_size=img_resize,
                )
                assert frames.ndim == 4, f"Expected 4D array, got {frames.shape} array"
                assert frames.shape[3] == 3, f"Expected 3 channels, got {frames.shape[3]} channels"
                # assert (
                #     frames.shape[0] == trajectory_length
                # ), f"Expected {trajectory_length} frames, got {frames.shape[0]} frames"
                all_frames.append(frames)
            cached_frames[key] = np.concatenate(all_frames, axis=0)
            print(f"{key}: {cached_frames[key].shape}")
        self.cached_frames = cached_frames
        self.start_indices = np.cumsum(self.trajectory_lengths) - self.trajectory_lengths

    def get_video(self, trajectory_id: int, key: str, base_index: int) -> np.ndarray:
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        # Calculate the absolute indices
        absolute_indices = self.start_indices[trajectory_index] + step_indices
        return self.cached_frames[key][absolute_indices]

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Get the RAW data for a single step. No transforms are applied.

        Args:
            trajectory_id (str): The ID of the trajectory.
            base_index (int): The base index of the step.

        Returns:
            dict: The data for the step.
        """
        data = {}
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        # Get the data for all modalities
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        return data

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        if self.img_resize is not None:
            all_video_keys = [key for key in self.modality_keys["video"]]
            for key in metadata.modalities.video:
                if key in all_video_keys:
                    metadata.modalities.video[key].resolution = self.img_resize
        super().set_transforms_metadata(metadata)


class WrappedLeRobotSingleDataset(LeRobotSingleDataset):
    def __init__(
        self,
        *args,
        data_split="full",
        modality_filename: str | None = None,
        exclude_splits: list[str] | None = None,
        **kwargs,
    ):
        # Store data_split BEFORE calling super().__init__() because
        # _get_all_steps_cmr_filtered needs it for cache path generation
        self.data_split = data_split
        super().__init__(
            *args,
            modality_filename=modality_filename,
            exclude_splits=exclude_splits,
            **kwargs,
        )

        if data_split == "full":
            pass
        elif data_split == "train":
            self._all_steps = self._all_steps[: -len(self) // 20]
        elif data_split == "test":
            self._all_steps = self._all_steps[-len(self) // 20 :]

        print(f"Dataset is split into {data_split} data, with {len(self._all_steps)} steps.")

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the trajectories in the dataset."""
        # Get trajectory lengths, IDs, and whitelist from dataset metadata
        episode_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
        with open(episode_path, "r") as f:
            episode_metadata = [json.loads(line) for line in f]
        trajectory_ids = []
        trajectory_lengths = []
        for episode in episode_metadata:
            trajectory_ids.append(episode["episode_index"])
            trajectory_lengths.append(episode["length"])
        return np.array(trajectory_ids), np.array(trajectory_lengths)

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single step in a trajectory.

        Args:
            index (int): The index of the step to get.

        Returns:
            dict: The data for the step.
        """
        try:
            # pdb.set_trace()
            trajectory_id, base_index = self.all_steps[index]
            original_outputs = self.transforms(self.get_step_data(trajectory_id, base_index))

            # delta_actions = original_outputs["action"][1:] - original_outputs["action"][:-1]
            # delta_actions = original_outputs["action"][1:] - original_outputs["action"][[0]]
            # delta_actions /= torch.linspace(1, len(delta_actions), steps=len(delta_actions))[:, None]

            def printvideo(videos, filename):
                t_videos = rearrange(videos, "c f h w -> f h w c")
                t_videos = t_videos.detach().to(dtype=torch.uint8).cpu().contiguous().numpy()
                writer = imageio.get_writer(filename, fps=5)
                for frame in t_videos:
                    writer.append_data(frame)

            frames = torch.from_numpy(original_outputs["video"])
            # frames = torch.from_numpy(original_outputs["video"])
            frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
            frames = frames.squeeze(1).transpose(0, 1)
            # printvideo(frames, "example.mp4")

            text = ""
            if "annotation.human.coarse_action" in original_outputs:
                text = original_outputs["annotation.human.coarse_action"][0].split(":")[-1].strip()

            video_path = {
                key: self.get_video_path(trajectory_id, key.replace("video.", ""))
                for key in self.modality_keys["video"]
            }

            # State may be absent for embodiments without state modality keys
            state_tensor = original_outputs.get("state", torch.zeros(1))

            data = {
                "__key__": state_tensor,
                "action": original_outputs["action"],
                # "action": original_outputs["action"][:-1],
                # "action": delta_actions,
                # "action": torch.zeros_like(delta_actions),
                "video": frames,
                # "video_path": video_path,
                "ai_caption": "",
                "text": text,
                "t5_text_embeddings": torch.zeros(512, 1024, dtype=torch.bfloat16).cuda(),
                "t5_text_mask": torch.ones(512, dtype=torch.int64).cuda(),
                "fps": 4,
                "image_size": 256 * torch.ones(4).cuda(),
                "num_frames": 13,
                "padding_mask": torch.zeros(1, 256, 256).cuda(),
            }
            return data
        except Exception as e:
            trajectory_id, base_index = self.all_steps[index]
            print(f"[{self.tag}/{self.dataset_name}] Error at item {index} "
                  f"(episode={trajectory_id}, base_idx={base_index}): {e}")
            print("Retrying with a random index...")
            return self.__getitem__(randint(0, len(self) - 1))


class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        max_pixels=1920 * 1080,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
        repeat=1,
        args=None,
        dataset_path=None,
        data_split="train",
        embodiment=None,
        downscaled_res=False,
    ):
        if args is not None:
            # height = args.height
            # width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            dataset_path = args.dataset_path
            embodiment = args.embodiment
            downscaled_res = args.downscaled_res

        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        # self.height = height
        # self.width = width
        # self.height_division_factor = height_division_factor
        # self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat

        # from gr00t_dreams.data.dataset import WrappedLeRobotSingleDataset
        # from gr00t_dreams.groot_configs import construct_modality_config_and_transforms
        from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.groot_configs import (
            construct_modality_config_and_transforms,
        )

        self.dataset_path = []
        for p in dataset_path.split(","):
            if (Path(p) / "data").exists() and (Path(p) / "meta").exists() and (Path(p) / "videos").exists():
                self.dataset_path.append(p)
            else:
                # This is not a LeRobot dataset, assume it's a directory of LeRobot datasets
                for sub_p in Path(p).iterdir():
                    if sub_p.is_dir():
                        self.dataset_path.append(str(sub_p))

        self.lerobot_datasets = []
        for p in self.dataset_path:
            config, train_transform, test_transform = construct_modality_config_and_transforms(
                num_frames=num_frames, embodiment=embodiment, downscaled_res=downscaled_res
            )
            self.lerobot_datasets.append(
                WrappedLeRobotSingleDataset(
                    dataset_path=p,
                    modality_configs=config,
                    transforms=train_transform if data_split == "train" or data_split == "full" else test_transform,
                    embodiment_tag="gr1_unified" if "gr1" in embodiment else embodiment,
                    data_split=data_split,
                )
            )
        print(f"Loaded lerobot {data_split} dataset from {self.dataset_path} with {len(self)} samples.")

        # if height is not None and width is not None:
        #     print("Height and width are fixed. Setting `dynamic_resolution` to False.")
        #     self.dynamic_resolution = False
        # elif height is None and width is None:
        #     print("Height and width are none. Setting `dynamic_resolution` to True.")
        #     self.dynamic_resolution = True

    def __getitem__(self, data_id):
        data_id %= len(self)
        for dataset in self.lerobot_datasets:
            if data_id < len(dataset):
                break
            data_id -= len(dataset)
        lerobot_data = dataset[data_id]

        prompt = lerobot_data["text"]

        video = lerobot_data["video"]
        video_frames = []
        for i in range(video.shape[1]):
            frame = video[:, i, :, :]
            frame = Image.fromarray(frame.permute(1, 2, 0).numpy())
            video_frames.append(frame)
        if len(video_frames) != self.num_frames:
            print(
                f"Warning: Expected {self.num_frames} frames, but got {len(video_frames)} frames. Randomly sampling an item instead."
            )
            return self.__getitem__(randint(0, len(self) - 1))  # noqa: F821
        video_frames = np.stack([np.array(frame, dtype=np.uint8) for frame in video_frames])

        # Actions are now relative after CMRVersiusRelativeActionTransform in the pipeline
        # The transform converts 22D absolute actions to 22D relative actions
        actions = lerobot_data["action"]

        data = {
            "prompt": prompt,
            "video": torch.from_numpy(video_frames).permute(3, 0, 1, 2),
            # "action": torch.from_numpy(delta_actions),
            "action": actions,
            "ai_caption": "",
            "text": prompt,
            "t5_text_embeddings": torch.zeros(512, 1024, dtype=torch.bfloat16).cuda(),
            "t5_text_mask": torch.ones(512, dtype=torch.int64).cuda(),
            "fps": 4,
            "image_size": 256 * torch.ones(4).cuda(),
            "num_frames": 13,
            "padding_mask": torch.zeros(1, 256, 256).cuda(),
            "__key__": lerobot_data["__key__"],
        }
        # data = {
        #     "prompt": prompt,
        #     "video": video_frames,
        #     "action": delta_actions,
        # }
        return data

    def __len__(self):
        return sum([len(d) for d in self.lerobot_datasets]) * self.repeat


# =============================================================================
# Multi-Embodiment Mixture Dataset
# =============================================================================
# Analogous to gr00t-H's ShardedMixtureDataset + DatasetFactory, but for the
# Cosmos Predict2.5 video-generation pipeline.
#
# Each sub-dataset can have a different embodiment (and therefore different
# modality configs, transforms, action dimensions). Weighted sampling ensures
# that each embodiment contributes to training according to its mix_ratio.
# Action tensors are zero-padded to a common maximum dimension.
# =============================================================================


class MixedLeRobotDataset(torch.utils.data.Dataset):
    """Multi-embodiment mixture dataset with weighted sampling and action padding.

    This dataset combines multiple LeRobot sub-datasets with different embodiments
    into a single unified dataset suitable for Cosmos Predict2.5 training. It mirrors
    the gr00t-H DatasetFactory / ShardedMixtureDataset design but uses the simpler
    map-style Dataset interface with repeat-factor-based weighted sampling.

    Features:
        - Weighted sampling via per-dataset repeat factors derived from mix_ratio
        - Zero-padding of action tensors to a common ``max_action_dim``
        - Per-dataset modality configs, transforms, and statistics (auto-resolved)
        - Compatible with PyTorch DistributedSampler (deterministic indexing)

    Args:
        dataset_specs: List of dicts, each with keys:
            - ``path`` (str): Path to the LeRobot dataset directory.
            - ``embodiment`` (str): Embodiment tag string (must be in
              EMBODIMENT_REGISTRY or be one of the built-in embodiments).
            - ``mix_ratio`` (float, optional): Relative sampling weight. Default 1.0.
        num_frames: Number of video frames per sample (e.g. 13 = 1 context + 12 pred).
        data_split: One of ``"train"``, ``"test"``, ``"full"``.
        max_action_dim: All action tensors are zero-padded to this dimension.
            Default 44 (CMR Versius conditioning dimension).
        downscaled_res: If True, use 256x256 resolution for all videos.

    Example::

        specs = [
            {"path": "/data/cmr/chole", "embodiment": "cmr_versius", "mix_ratio": 1.0},
            {"path": "/data/dvrk/suturebot", "embodiment": "dvrk", "mix_ratio": 0.5},
        ]
        dataset = MixedLeRobotDataset(specs, num_frames=13, data_split="train")
    """

    def __init__(
        self,
        dataset_specs: list[dict],
        num_frames: int = 13,
        data_split: str = "train",
        max_action_dim: int = 44,
        downscaled_res: bool = False,
    ):
        from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.groot_configs import (
            construct_modality_config_and_transforms,
        )

        self.num_frames = num_frames
        self.max_action_dim = max_action_dim

        self.sub_datasets: list[WrappedLeRobotSingleDataset] = []
        self.mix_ratios: list[float] = []
        self.embodiment_tags: list[str] = []

        print("=" * 80)
        print("INITIALIZING MULTI-EMBODIMENT MIXTURE DATASET")
        print("=" * 80)

        for i, spec in enumerate(dataset_specs):
            path = spec["path"]
            # Accept both EmbodimentTag enum instances and plain strings
            raw_embodiment = spec["embodiment"]
            embodiment = raw_embodiment.value if isinstance(raw_embodiment, EmbodimentTag) else raw_embodiment
            mix_ratio = spec.get("mix_ratio", 1.0)

            print(f"\n[{i}] Loading: embodiment={embodiment}, mix_ratio={mix_ratio}")
            print(f"    path={path}")

            config, train_transform, test_transform = construct_modality_config_and_transforms(
                num_frames=num_frames, embodiment=embodiment, downscaled_res=downscaled_res,
            )

            # Extract modality_filename from config if present (set by registry)
            modality_filename = None
            if isinstance(config, dict) and "modality_filename" in config:
                modality_filename = config.pop("modality_filename")

            transform = train_transform if data_split in ("train", "full") else test_transform

            # Per-dataset episode filtering (e.g., exclude "fail", "bad_frames" splits)
            exclude_splits = spec.get("exclude_splits", None)

            dataset = WrappedLeRobotSingleDataset(
                dataset_path=path,
                modality_configs=config,
                transforms=transform,
                embodiment_tag=embodiment,
                data_split=data_split,
                modality_filename=modality_filename,
                exclude_splits=exclude_splits,
            )

            self.sub_datasets.append(dataset)
            self.mix_ratios.append(mix_ratio)
            self.embodiment_tags.append(embodiment)

            print(f"    => {len(dataset):,} samples loaded")

        # Determine max state dim across all sub-datasets for padding.
        # Different embodiments have different state dims (e.g., CMR=16D, UCB=14D).
        # The DataLoader collate (torch.stack) requires all tensors in a batch to
        # have the same shape, so we zero-pad the state to a common max dim.
        self._max_state_dim = 0
        for ds in self.sub_datasets:
            try:
                sample = ds[0]
                state = sample.get("__key__")
                if state is not None:
                    if isinstance(state, torch.Tensor):
                        self._max_state_dim = max(self._max_state_dim, state.shape[-1])
                    elif hasattr(state, "shape"):
                        self._max_state_dim = max(self._max_state_dim, state.shape[-1])
            except Exception:
                pass
        if self._max_state_dim > 0:
            print(f"Max state dim (for padding): {self._max_state_dim}")

        # Compute repeat factors from mix_ratios so that the proportion of
        # virtual samples from each dataset matches the desired distribution.
        #
        # For each dataset i:
        #   per_sample_weight_i = mix_ratio_i / len(dataset_i)
        # Normalize to the minimum weight to get integer repeat factors.
        self._compute_repeat_factors()

        # Print summary table
        self._print_summary()

    def _compute_repeat_factors(self):
        """Compute per-dataset integer repeat factors from mix_ratios."""
        per_sample_weights = []
        for ds, ratio in zip(self.sub_datasets, self.mix_ratios):
            ds_len = max(len(ds), 1)
            per_sample_weights.append(ratio / ds_len)

        min_weight = min(per_sample_weights)
        raw_factors = [w / min_weight for w in per_sample_weights]

        # Round and ensure at least 1
        self.repeat_factors = [max(1, round(f)) for f in raw_factors]

        # Compute virtual dataset sizes and cumulative offsets
        self.virtual_sizes = [len(ds) * rf for ds, rf in zip(self.sub_datasets, self.repeat_factors)]
        self._total_virtual_len = sum(self.virtual_sizes)
        self._cumulative_sizes = np.cumsum(self.virtual_sizes)

    def _print_summary(self):
        """Print a formatted summary of the mixture dataset."""
        print("\n" + "=" * 80)
        print("MIXTURE DATASET SUMMARY")
        print("=" * 80)
        header = f"{'Embodiment':<22} {'Path':<40} {'Real Len':>10} {'Mix Ratio':>10} {'Repeat':>7} {'Virtual':>10} {'% Total':>8}"
        print(header)
        print("-" * 80)
        for i, (ds, tag, ratio, rf, vs) in enumerate(zip(
            self.sub_datasets, self.embodiment_tags, self.mix_ratios,
            self.repeat_factors, self.virtual_sizes,
        )):
            path_short = Path(ds.dataset_path).name
            if len(path_short) > 38:
                path_short = "..." + path_short[-35:]
            pct = 100.0 * vs / max(self._total_virtual_len, 1)
            print(f"{tag:<22} {path_short:<40} {len(ds):>10,} {ratio:>10.3f} {rf:>7} {vs:>10,} {pct:>7.1f}%")
        print("-" * 80)
        print(f"{'Total':>22} {'':>40} {'':>10} {'':>10} {'':>7} {self._total_virtual_len:>10,} {'100.0':>7}%")
        print(f"Max action dim: {self.max_action_dim}")
        print(f"Num embodiments: {len(set(self.embodiment_tags))}")
        print("=" * 80 + "\n")

    def __len__(self) -> int:
        return self._total_virtual_len

    def __getitem__(self, idx: int) -> dict:
        """Get a sample with weighted sampling and action padding.

        The virtual index space is split into contiguous blocks per dataset,
        each repeated ``repeat_factors[i]`` times. Within each block, the
        real index is computed modularly.

        Returns:
            dict with keys: ``video``, ``action``, ``__key__``, ``text``,
            ``embodiment_tag``, ``t5_text_embeddings``, ``t5_text_mask``,
            ``fps``, ``image_size``, ``num_frames``, ``padding_mask``.
        """
        idx = idx % len(self)

        # Find which dataset this virtual index belongs to
        dataset_idx = int(np.searchsorted(self._cumulative_sizes, idx, side="right"))
        if dataset_idx > 0:
            local_idx = idx - int(self._cumulative_sizes[dataset_idx - 1])
        else:
            local_idx = idx

        # Map virtual local index to real sample index (modular)
        real_idx = local_idx % len(self.sub_datasets[dataset_idx])

        try:
            lerobot_data = self.sub_datasets[dataset_idx][real_idx]
        except Exception as e:
            print(f"[MixedLeRobotDataset] Error in sub-dataset {dataset_idx} "
                  f"({self.embodiment_tags[dataset_idx]}), idx={real_idx}: {e}")
            print("Retrying with a random index...")
            return self.__getitem__(randint(0, len(self) - 1))

        # --- Pad action to max_action_dim ---
        action = lerobot_data["action"]
        if isinstance(action, torch.Tensor):
            current_dim = action.shape[-1]
            if current_dim < self.max_action_dim:
                padding = torch.zeros(
                    *action.shape[:-1], self.max_action_dim - current_dim,
                    dtype=action.dtype, device=action.device,
                )
                action = torch.cat([action, padding], dim=-1)
        lerobot_data["action"] = action

        # --- Pad state (__key__) to max_state_dim ---
        # Different embodiments have different state dims (e.g., CMR=16D, UCB=14D).
        # Without padding, torch.stack in the DataLoader collate fails.
        state = lerobot_data.get("__key__")
        if state is not None and isinstance(state, torch.Tensor) and self._max_state_dim > 0:
            current_dim = state.shape[-1]
            if current_dim < self._max_state_dim:
                padding = torch.zeros(
                    *state.shape[:-1], self._max_state_dim - current_dim,
                    dtype=state.dtype, device=state.device,
                )
                state = torch.cat([state, padding], dim=-1)
            lerobot_data["__key__"] = state

        # --- Add embodiment metadata ---
        lerobot_data["embodiment_tag"] = self.embodiment_tags[dataset_idx]

        return lerobot_data

