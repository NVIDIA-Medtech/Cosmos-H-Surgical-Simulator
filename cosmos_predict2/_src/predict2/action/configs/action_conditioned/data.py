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

import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.action.datasets.dataset_local import Dataset_3D

try:
    from cosmos_predict2._src.predict2.action.configs.action_conditioned.experiment.gr00t_customized_gr1 import (
        register_gr00t_customized_gr1_data,
    )
except ImportError:
    register_gr00t_customized_gr1_data = None

# bridge dataset path
base_path = "datasets/bridge/"

train_annotation_path = os.path.join(base_path, "annotation/train")
val_annotation_path = os.path.join(base_path, "annotation/val")
test_annotation_path = os.path.join(base_path, "annotation/test")


# experiment for next-frame prediction
bridge_train_dataset = L(Dataset_3D)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=1,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[256, 320],
    val_start_frame_interval=1,
    mode="train",
)
bridge_val_dataset = L(Dataset_3D)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=1,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[256, 320],
    val_start_frame_interval=1,
    mode="val",
)

# experiment for action-sequence video prediction
bridge_13frame_480_640_train_dataset = L(Dataset_3D)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[480, 640],
    val_start_frame_interval=1,
    mode="train",
)
bridge_13frame_480_640_val_dataset = L(Dataset_3D)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[480, 640],
    val_start_frame_interval=1,
    mode="val",
)


# ------------------------------------------------------------


# create dataloader for each dataset
def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


def build_webdataset(webdataset_instance, **kwargs):
    """Helper function to build WebDataset from a WebDataset instance.

    WebDatasets need to call build_dataset() to get the actual iterable dataset
    that can be used with DataLoader.

    Args:
        webdataset_instance: An instantiated WebDataset object.
        **kwargs: Additional parameters to override on the webdataset instance
            before building. This allows experiment configs to override parameters
            like gripper_rescale_factor, num_action_per_chunk, etc.
    """
    # Apply any parameter overrides to the webdataset instance
    for key, value in kwargs.items():
        if hasattr(webdataset_instance, key):
            setattr(webdataset_instance, key, value)
    return webdataset_instance.build_dataset()


bridge_train_dataloader = L(DataLoader)(
    dataset=bridge_train_dataset,
    sampler=L(get_sampler)(dataset=bridge_train_dataset),
    batch_size=1,
    drop_last=True,
)
bridge_val_dataloader = L(DataLoader)(
    dataset=bridge_val_dataset,
    sampler=L(get_sampler)(dataset=bridge_val_dataset),
    batch_size=1,
    drop_last=True,
)

bridge_13frame_480_640_train_dataloader = L(DataLoader)(
    dataset=bridge_13frame_480_640_train_dataset,
    sampler=L(get_sampler)(dataset=bridge_13frame_480_640_train_dataset),
    batch_size=1,
    drop_last=True,
)
bridge_13frame_480_640_val_dataloader = L(DataLoader)(
    dataset=bridge_13frame_480_640_val_dataset,
    sampler=L(get_sampler)(dataset=bridge_13frame_480_640_val_dataset),
    batch_size=1,
    drop_last=True,
)


# ============================================================================
# CMR Versius dataset configuration
# ============================================================================
# CMR Versius surgical robot dataset in LeRobot format
# - 12 action frames (action horizon)
# - FRAME_STRIDE = 6 for 10fps from 60Hz original
# - Resolution: 960x720 (full) or 256x256 (downscaled)
# ============================================================================
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.dataset import LeRobotDataset

# Default path for CMR Versius dataset - can be overridden in experiment configs
base_path_cmr_versius = ("/CMR_Versius/cholecystectomy_480p,"
                         "/CMR_Versius/hysterectomy_480p,"
                         "/CMR_Versius/inguinal_hernia_480p,"
                         "/CMR_Versius/prostatectomy_480p")

cmr_versius_train_dataset = L(LeRobotDataset)(
    num_frames=13,  # 12 prediction frames + 1 reference frame
    time_division_factor=6,  # FRAME_STRIDE for 10fps from 60Hz
    time_division_remainder=1,
    max_pixels=960 * 720,
    data_file_keys=("video",),
    image_file_extension=("jpg", "jpeg", "png", "webp"),
    video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
    repeat=1,
    args=None,
    dataset_path=base_path_cmr_versius,
    data_split="train",
    embodiment="cmr_versius",
    downscaled_res=False,
)

cmr_versius_val_dataset = L(LeRobotDataset)(
    num_frames=13,
    time_division_factor=6,
    time_division_remainder=1,
    max_pixels=960 * 720,
    data_file_keys=("video",),
    image_file_extension=("jpg", "jpeg", "png", "webp"),
    video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
    repeat=1,
    args=None,
    dataset_path=base_path_cmr_versius,
    data_split="test",
    embodiment="cmr_versius",
    downscaled_res=False,
)

# Downscaled resolution variant (256x256)
cmr_versius_256_train_dataset = L(LeRobotDataset)(
    num_frames=13,
    time_division_factor=6,
    time_division_remainder=1,
    max_pixels=256 * 256,
    data_file_keys=("video",),
    image_file_extension=("jpg", "jpeg", "png", "webp"),
    video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
    repeat=1,
    args=None,
    dataset_path=base_path_cmr_versius,
    data_split="train",
    embodiment="cmr_versius",
    downscaled_res=True,
)

cmr_versius_256_val_dataset = L(LeRobotDataset)(
    num_frames=13,
    time_division_factor=6,
    time_division_remainder=1,
    max_pixels=256 * 256,
    data_file_keys=("video",),
    image_file_extension=("jpg", "jpeg", "png", "webp"),
    video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
    repeat=1,
    args=None,
    dataset_path=base_path_cmr_versius,
    data_split="test",
    embodiment="cmr_versius",
    downscaled_res=True,
)

cmr_versius_train_dataloader = L(DataLoader)(
    dataset=cmr_versius_train_dataset,
    sampler=L(get_sampler)(dataset=cmr_versius_train_dataset),
    batch_size=1,
    drop_last=True,
)

cmr_versius_val_dataloader = L(DataLoader)(
    dataset=cmr_versius_val_dataset,
    sampler=L(get_sampler)(dataset=cmr_versius_val_dataset),
    batch_size=1,
    drop_last=True,
)

cmr_versius_256_train_dataloader = L(DataLoader)(
    dataset=cmr_versius_256_train_dataset,
    sampler=L(get_sampler)(dataset=cmr_versius_256_train_dataset),
    batch_size=1,
    drop_last=True,
)

cmr_versius_256_val_dataloader = L(DataLoader)(
    dataset=cmr_versius_256_val_dataset,
    sampler=L(get_sampler)(dataset=cmr_versius_256_val_dataset),
    batch_size=1,
    drop_last=True,
)


# ============================================================================
# Open-H Multi-Embodiment Dataset Configuration
# ============================================================================
# OPEN_H_DATASET_SPECS is the single source of truth for all Open-H datasets.
# It lives in groot_configs.py alongside EMBODIMENT_REGISTRY so that every
# consumer (data.py, dataset.py stats check, compute_openh_action_stats.py)
# imports from one place.
# ============================================================================
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.dataset import MixedLeRobotDataset
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.groot_configs import (
    OPEN_H_DATASET_SPECS,
    MAX_ACTION_DIM,
)

open_h_multi_train_dataset = L(MixedLeRobotDataset)(
    dataset_specs=OPEN_H_DATASET_SPECS,
    num_frames=13,  # 1 context + 12 prediction frames
    data_split="train",
    max_action_dim=MAX_ACTION_DIM,
    downscaled_res=False,
)

open_h_multi_val_dataset = L(MixedLeRobotDataset)(
    dataset_specs=OPEN_H_DATASET_SPECS,
    num_frames=13,
    data_split="test",
    max_action_dim=MAX_ACTION_DIM,
    downscaled_res=False,
)

open_h_multi_train_dataloader = L(DataLoader)(
    dataset=open_h_multi_train_dataset,
    sampler=L(get_sampler)(dataset=open_h_multi_train_dataset),
    batch_size=1,
    drop_last=True,
)

open_h_multi_val_dataloader = L(DataLoader)(
    dataset=open_h_multi_val_dataset,
    sampler=L(get_sampler)(dataset=open_h_multi_val_dataset),
    batch_size=1,
    drop_last=True,
)


def register_training_and_val_data():
    cs = ConfigStore.instance()
    from cosmos_predict2._src.predict2.configs.common.mock_data import MOCK_DATA_INTERLEAVE_CONFIG

    # Always register mock dataloaders to satisfy defaults when not overridden
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="mock",
        node=MOCK_DATA_INTERLEAVE_CONFIG,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="mock",
        node=MOCK_DATA_INTERLEAVE_CONFIG,
    )

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="bridge_train",
        node=bridge_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="bridge_val",
        node=bridge_val_dataloader,
    )

    # 13 frame 480 640
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="bridge_13frame_480_640_train",
        node=bridge_13frame_480_640_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="bridge_13frame_480_640_val",
        node=bridge_13frame_480_640_val_dataloader,
    )

    # CMR Versius dataset (full resolution 960x720)
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="cmr_versius_train",
        node=cmr_versius_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="cmr_versius_val",
        node=cmr_versius_val_dataloader,
    )

    # CMR Versius dataset (downscaled 256x256)
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="cmr_versius_256_train",
        node=cmr_versius_256_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="cmr_versius_256_val",
        node=cmr_versius_256_val_dataloader,
    )

    # ============================================================================
    # Open-H Multi-Embodiment dataset (all embodiments, weighted mix)
    # ============================================================================
    # Mirrors gr00t-H exp62 weighting: CMR=50%, remaining 50% step-weighted.
    # Excludes TUM Ultrasound and SanoScience per project scope.
    # Action vectors are zero-padded to 44D (CMR conditioning dimension).
    # ============================================================================
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="open_h_multi_train",
        node=open_h_multi_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="open_h_multi_val",
        node=open_h_multi_val_dataloader,
    )

    # Register gr00t_customized_gr1 data
    if register_gr00t_customized_gr1_data is not None:
        register_gr00t_customized_gr1_data()
