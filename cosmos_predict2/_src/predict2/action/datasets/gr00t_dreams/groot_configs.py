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

from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.dataset import ModalityConfig
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.embodiment_tags import EmbodimentTag
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.transform.base import ComposedModalityTransform
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.transform.concat import ConcatTransform
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.transform.state_action import (
    ActionKeyConfig,
    CMRVersiusRelativeActionTransform,
    GenericRelativeActionTransform,
    RelativeActionTransform,
    StateActionToTensor,
    StateActionTransform,
)
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.transform.video import (
    VideoCrop,
    VideoResize,
    VideoToTensor,
)

# =============================================================================
# Open-H Multi-Embodiment Registry
# =============================================================================
# Each entry defines the dataset-specific parameters for a single embodiment.
# Action keys match the gr00t-H action space definitions. State keys are loaded
# for the model's __key__ field but are NOT appended to the action vector
# (except for CMR Versius which has its own CMR-specific path above).
#
# Video: Single camera per dataset (monocular) for video generation.
# Action: Raw parquet dimensions (no rot6d conversion; model learns raw format).
# State: Reference state at t=0 for context.
#
# The max unified action dimension is 44D (CMR Versius with state conditioning).
# All other datasets are zero-padded to 44D by MixedLeRobotDataset.
# =============================================================================


# Helper: create a standard dual-arm EEF + gripper action config (most common pattern)
def _dual_arm_eef_configs(
    pose1_key: str,
    grip1_key: str,
    pose2_key: str,
    grip2_key: str,
    state_pose1: str,
    state_pose2: str,
    input_rot: str = "quat",
    ref_rot: str = "quat",
    input_quat: str = "xyzw",
    ref_quat: str = "xyzw",
) -> dict[str, ActionKeyConfig]:
    return {
        pose1_key: ActionKeyConfig(
            rep="rel_xyz_rot6d",
            state_key=state_pose1,
            input_rotation_format=input_rot,
            reference_rotation_format=ref_rot,
            input_quat_order=input_quat,
            reference_quat_order=ref_quat,
        ),
        grip1_key: ActionKeyConfig(rep="absolute"),
        pose2_key: ActionKeyConfig(
            rep="rel_xyz_rot6d",
            state_key=state_pose2,
            input_rotation_format=input_rot,
            reference_rotation_format=ref_rot,
            input_quat_order=input_quat,
            reference_quat_order=ref_quat,
        ),
        grip2_key: ActionKeyConfig(rep="absolute"),
    }


EMBODIMENT_REGISTRY: dict[str, dict] = {
    # -----------------------------------------------------------------
    # dVRK JHU (stereo endoscope, 50Hz → stride 5 for 10fps)
    # Action: REL_XYZ_ROT6D for poses (quat → 9D), ABSOLUTE for grippers
    # -----------------------------------------------------------------
    "dvrk": {
        "timestep_interval": 5,
        "video_keys": ["video.endoscope_left"],
        "state_keys": [
            "state.psm1_pose",
            "state.psm1_gripper",
            "state.psm2_pose",
            "state.psm2_gripper",
        ],
        "action_keys": [
            "action.psm1_pose",
            "action.psm1_gripper",
            "action.psm2_pose",
            "action.psm2_gripper",
        ],
        "action_key_configs": _dual_arm_eef_configs(
            "action.psm1_pose",
            "action.psm1_gripper",
            "action.psm2_pose",
            "action.psm2_gripper",
            "state.psm1_pose",
            "state.psm2_pose",
        ),
        "video_width": 512,
        "video_height": 288,
        "modality_filename": "meta/modality.json",
        "normalization_mode": "mean_std",
    },
    # -----------------------------------------------------------------
    # dVRK JHU Monocular (left endoscope only, same action space as dvrk)
    # -----------------------------------------------------------------
    "jhu_dvrk_mono": {
        "timestep_interval": 5,
        "video_keys": ["video.endoscope_left"],
        "state_keys": [
            "state.psm1_pose",
            "state.psm1_gripper",
            "state.psm2_pose",
            "state.psm2_gripper",
        ],
        "action_keys": [
            "action.psm1_pose",
            "action.psm1_gripper",
            "action.psm2_pose",
            "action.psm2_gripper",
        ],
        "action_key_configs": _dual_arm_eef_configs(
            "action.psm1_pose",
            "action.psm1_gripper",
            "action.psm2_pose",
            "action.psm2_gripper",
            "state.psm1_pose",
            "state.psm2_pose",
        ),
        "video_width": 512,
        "video_height": 288,
        "modality_filename": "meta/modality.json",
        "normalization_mode": "mean_std",
    },
    # -----------------------------------------------------------------
    # dVRK UCB Debridement (~30Hz → stride 3)
    # modality.json video key: "camera_left" (not "endoscope")
    # Updated (gr00t-H 9e25db4): Cartesian EEF pose actions with REL_XYZ_ROT6D,
    # joint-angle state channels for normalization, and cartesian pose as
    # pass-through for the action reference frame (not normalized/concatenated).
    # -----------------------------------------------------------------
    "dvrk_ucb": {
        "timestep_interval": 3,
        "video_keys": ["video.camera_left"],
        "state_keys": [
            # Joint-angle + gripper state (normalized & concatenated as model input)
            "state.psm1_joints",
            "state.psm1_gripper",
            "state.psm2_joints",
            "state.psm2_gripper",
            # Cartesian pose (pass-through for REL_XYZ_ROT6D reference only)
            "state.psm1_pose",
            "state.psm2_pose",
        ],
        # Pass-through state keys: loaded and used by GenericRelativeActionTransform
        # as the reference frame for REL_XYZ_ROT6D conversion, but NOT normalized
        # or concatenated into the model's state input vector.
        "pass_through_state_keys": [
            "state.psm1_pose",
            "state.psm2_pose",
        ],
        "action_keys": [
            "action.psm1_pose",
            "action.psm1_gripper",
            "action.psm2_pose",
            "action.psm2_gripper",
        ],
        "action_key_configs": _dual_arm_eef_configs(
            "action.psm1_pose",
            "action.psm1_gripper",
            "action.psm2_pose",
            "action.psm2_gripper",
            "state.psm1_pose",
            "state.psm2_pose",
        ),
        "video_width": 512,
        "video_height": 288,
        "modality_filename": "meta/modality.json",
        "normalization_mode": "mean_std",
    },
    # -----------------------------------------------------------------
    # Hamlyn 30Hz (30Hz → stride 3, wxyz quaternions)
    # modality.json video key: "endoscope" (not "endoscope_left")
    # -----------------------------------------------------------------
    "hamlyn_30hz": {
        "timestep_interval": 3,
        "video_keys": ["video.endoscope"],
        "state_keys": [
            "state.left_arm_pose",
            "state.left_arm_gripper",
            "state.right_arm_pose",
            "state.right_arm_gripper",
        ],
        "action_keys": [
            "action.left_arm_pose",
            "action.left_arm_gripper",
            "action.right_arm_pose",
            "action.right_arm_gripper",
        ],
        "action_key_configs": _dual_arm_eef_configs(
            "action.left_arm_pose",
            "action.left_arm_gripper",
            "action.right_arm_pose",
            "action.right_arm_gripper",
            "state.left_arm_pose",
            "state.right_arm_pose",
            input_quat="wxyz",
            ref_quat="wxyz",
        ),
        "video_width": 512,
        "video_height": 288,
        "modality_filename": "meta/modality.json",
        "normalization_mode": "mean_std",
    },
    # -----------------------------------------------------------------
    # UCSD Surgical Learning (30Hz → stride 3, wxyz quaternions)
    # modality.json video key: "camera_left" (not "endoscope")
    # -----------------------------------------------------------------
    "dvrk_ucsd": {
        "timestep_interval": 3,
        "video_keys": ["video.camera_left"],
        "state_keys": [
            "state.psm_retraction_pose",
            "state.psm_retraction_gripper",
            "state.psm_cutter_pose",
            "state.psm_cutter_gripper",
        ],
        "action_keys": [
            "action.psm_retraction_pose",
            "action.psm_retraction_gripper",
            "action.psm_cutter_pose",
            "action.psm_cutter_gripper",
        ],
        "action_key_configs": _dual_arm_eef_configs(
            "action.psm_retraction_pose",
            "action.psm_retraction_gripper",
            "action.psm_cutter_pose",
            "action.psm_cutter_gripper",
            "state.psm_retraction_pose",
            "state.psm_cutter_pose",
            input_quat="wxyz",
            ref_quat="wxyz",
        ),
        "video_width": 512,
        "video_height": 288,
        "modality_filename": "meta/modality.json",
        "normalization_mode": "mean_std",
    },
    # -----------------------------------------------------------------
    # USTC Torin (24Hz → stride 2)
    # NOTE: The old knot_tying_all / needle_handover_all / needle_pickup_all
    # datasets have joint-space data (left_joints, right_joints) in modality.json,
    # NOT pose data (left_pose, right_pose). The newer quat_merged datasets
    # (exp86) have pose keys. Using joint keys for backward compatibility.
    # -----------------------------------------------------------------
    "ustc_torin": {
        "timestep_interval": 2,
        "video_keys": ["video.endoscope_left"],
        "state_keys": [
            "state.left_joints",
            "state.right_joints",
        ],
        "action_keys": [
            "action.left_joints",
            "action.right_joints",
        ],
        "action_key_configs": {
            "action.left_joints": ActionKeyConfig(rep="relative", state_key="state.left_joints"),
            "action.right_joints": ActionKeyConfig(rep="relative", state_key="state.right_joints"),
        },
        "video_width": 512,
        "video_height": 288,
        "modality_filename": "meta/modality.json",
        "normalization_mode": "mean_std",
    },
    # -----------------------------------------------------------------
    # Obuda dVRK (50Hz → stride 5)
    # -----------------------------------------------------------------
    "dvrk_obuda": {
        "timestep_interval": 5,
        "video_keys": ["video.endoscope_left"],
        "state_keys": [
            "state.psm1_pose",
            "state.psm1_gripper",
            "state.psm2_pose",
            "state.psm2_gripper",
        ],
        "action_keys": [
            "action.psm1_pose",
            "action.psm1_gripper",
            "action.psm2_pose",
            "action.psm2_gripper",
        ],
        "action_key_configs": _dual_arm_eef_configs(
            "action.psm1_pose",
            "action.psm1_gripper",
            "action.psm2_pose",
            "action.psm2_gripper",
            "state.psm1_pose",
            "state.psm2_pose",
        ),
        "video_width": 512,
        "video_height": 288,
        "modality_filename": "meta/modality.json",
        "normalization_mode": "mean_std",
    },
    # -----------------------------------------------------------------
    # Rob Surgical (4-arm, pose only, Euler rotation, ~30Hz → stride 3)
    # -----------------------------------------------------------------
    "rob_surgical": {
        "timestep_interval": 3,
        "video_keys": ["video.endoscope"],
        "state_keys": [
            "state.left_pose",
            "state.right_pose",
            "state.lap_pose",
            "state.aux_pose",
        ],
        "action_keys": [
            "action.left_pose",
            "action.right_pose",
            "action.lap_pose",
            "action.aux_pose",
        ],
        "action_key_configs": {
            "action.left_pose": ActionKeyConfig(
                rep="rel_xyz_rot6d",
                state_key="state.left_pose",
                input_rotation_format="euler",
                reference_rotation_format="euler",
            ),
            "action.right_pose": ActionKeyConfig(
                rep="rel_xyz_rot6d",
                state_key="state.right_pose",
                input_rotation_format="euler",
                reference_rotation_format="euler",
            ),
            "action.lap_pose": ActionKeyConfig(
                rep="rel_xyz_rot6d",
                state_key="state.lap_pose",
                input_rotation_format="euler",
                reference_rotation_format="euler",
            ),
            "action.aux_pose": ActionKeyConfig(
                rep="rel_xyz_rot6d",
                state_key="state.aux_pose",
                input_rotation_format="euler",
                reference_rotation_format="euler",
            ),
        },
        "video_width": 512,
        "video_height": 288,
        "modality_filename": "meta/modality.json",
        "normalization_mode": "mean_std",
    },
    # -----------------------------------------------------------------
    # Stanford Real dVRK (Euler rotation, 30Hz → stride 3)
    # -----------------------------------------------------------------
    "dvrk_stanford_real": {
        "timestep_interval": 3,
        "video_keys": ["video.endoscope_left"],
        "state_keys": [
            "state.psm1_pose",
            "state.psm1_gripper",
            "state.psm2_pose",
            "state.psm2_gripper",
        ],
        "action_keys": [
            "action.psm1_pose",
            "action.psm1_gripper",
            "action.psm2_pose",
            "action.psm2_gripper",
        ],
        "action_key_configs": _dual_arm_eef_configs(
            "action.psm1_pose",
            "action.psm1_gripper",
            "action.psm2_pose",
            "action.psm2_gripper",
            "state.psm1_pose",
            "state.psm2_pose",
            input_rot="euler",
            ref_rot="euler",
        ),
        "video_width": 512,
        "video_height": 288,
        "modality_filename": "meta/modality.json",
        "normalization_mode": "mean_std",
    },
    # -----------------------------------------------------------------
    # PolyU Simulated (single arm, ~30Hz → stride 3)
    # -----------------------------------------------------------------
    "polyu_sim": {
        "timestep_interval": 3,
        "video_keys": ["video.endoscope"],
        "state_keys": [
            "state.psm_cartesian_pose",
            "state.psm_gripper",
        ],
        "action_keys": [
            "action.psm_cartesian_pose",
            "action.psm_gripper",
        ],
        "action_key_configs": {
            "action.psm_cartesian_pose": ActionKeyConfig(rep="rel_xyz_rot6d", state_key="state.psm_cartesian_pose"),
            "action.psm_gripper": ActionKeyConfig(rep="absolute"),
        },
        "video_width": 512,
        "video_height": 288,
        "modality_filename": "meta/modality.json",
        "normalization_mode": "mean_std",
    },
    # -----------------------------------------------------------------
    # Moon Surgical (DELTA xyz only, ~30Hz → stride 3)
    # modality.json video key: "scope" (not "endoscope")
    # modality.json state keys: "right_arm_joints", "left_arm_joints" (not delta_xyz)
    # -----------------------------------------------------------------
    "moon": {
        "timestep_interval": 3,
        "video_keys": ["video.scope"],
        "state_keys": [
            "state.right_arm_joints",
            "state.left_arm_joints",
        ],
        "action_keys": [
            "action.right_arm_delta_xyz",
            "action.left_arm_delta_xyz",
        ],
        "action_key_configs": {
            "action.right_arm_delta_xyz": ActionKeyConfig(rep="delta"),
            "action.left_arm_delta_xyz": ActionKeyConfig(rep="delta"),
        },
        "video_width": 512,
        "video_height": 288,
        "modality_filename": "meta/modality.json",
        "normalization_mode": "mean_std",
    },
    # -----------------------------------------------------------------
    # JHU LSCR MIRACLE (15Hz → stride 1, explicit xyzw quaternions)
    # -----------------------------------------------------------------
    "jhu_lscr_miracle": {
        "timestep_interval": 1,
        "video_keys": ["video.endoscope_left"],
        "state_keys": [
            "state.psm1_pose",
            "state.psm1_gripper",
            "state.psm2_pose",
            "state.psm2_gripper",
        ],
        "action_keys": [
            "action.psm1_pose",
            "action.psm1_gripper",
            "action.psm2_pose",
            "action.psm2_gripper",
        ],
        "action_key_configs": _dual_arm_eef_configs(
            "action.psm1_pose",
            "action.psm1_gripper",
            "action.psm2_pose",
            "action.psm2_gripper",
            "state.psm1_pose",
            "state.psm2_pose",
            input_quat="xyzw",
            ref_quat="xyzw",
        ),
        "video_width": 512,
        "video_height": 288,
        "modality_filename": "meta/modality.json",
        "normalization_mode": "mean_std",
    },
    # -----------------------------------------------------------------
    # JHU LSCR SMARTS (10Hz → stride 1, explicit xyzw quaternions)
    # -----------------------------------------------------------------
    "jhu_lscr_smarts": {
        "timestep_interval": 1,
        "video_keys": ["video.endoscope_left"],
        "state_keys": [
            "state.psm1_pose",
            "state.psm1_gripper",
            "state.psm2_pose",
            "state.psm2_gripper",
        ],
        "action_keys": [
            "action.psm1_pose",
            "action.psm1_gripper",
            "action.psm2_pose",
            "action.psm2_gripper",
        ],
        "action_key_configs": _dual_arm_eef_configs(
            "action.psm1_pose",
            "action.psm1_gripper",
            "action.psm2_pose",
            "action.psm2_gripper",
            "state.psm1_pose",
            "state.psm2_pose",
            input_quat="xyzw",
            ref_quat="xyzw",
        ),
        "video_width": 512,
        "video_height": 288,
        "modality_filename": "meta/modality.json",
        "normalization_mode": "mean_std",
    },
    # -----------------------------------------------------------------
    # TUD TUNDRA (single arm UR5e, ~30Hz → stride 3)
    # -----------------------------------------------------------------
    "tud_tundra": {
        "timestep_interval": 3,
        "video_keys": ["video.laparoscope_left"],
        "state_keys": [
            "state.eef_pose",
            "state.gripper",
        ],
        "action_keys": [
            "action.eef_pose",
            "action.gripper",
        ],
        "action_key_configs": {
            "action.eef_pose": ActionKeyConfig(
                rep="rel_xyz_rot6d", state_key="state.eef_pose", reference_quat_order="xyzw"
            ),
            "action.gripper": ActionKeyConfig(rep="absolute"),
        },
        "video_width": 512,
        "video_height": 288,
        "modality_filename": "meta/modality.json",
        "normalization_mode": "mean_std",
    },
    # -----------------------------------------------------------------
    # Turin MITIC (dual arm pose only, no grippers, ~30Hz → stride 3)
    # -----------------------------------------------------------------
    "turin_mitic_ex_vivo": {
        "timestep_interval": 3,
        "video_keys": ["video.endoscope_left"],
        "state_keys": [
            "state.psm1_pose",
            "state.psm2_pose",
        ],
        "action_keys": [
            "action.psm1_pose",
            "action.psm2_pose",
        ],
        "action_key_configs": {
            "action.psm1_pose": ActionKeyConfig(rep="rel_xyz_rot6d", state_key="state.psm1_pose"),
            "action.psm2_pose": ActionKeyConfig(rep="rel_xyz_rot6d", state_key="state.psm2_pose"),
        },
        "video_width": 512,
        "video_height": 288,
        "modality_filename": "meta/modality.json",
        "normalization_mode": "mean_std",
    },
}

# Maximum unified action dimension across all embodiments.
# CMR Versius has 44D (30D actions + 14D state conditioning) = the largest.
# All other datasets are zero-padded to this dimension by MixedLeRobotDataset.
MAX_ACTION_DIM = 44


# =============================================================================
# Open-H Dataset Specifications  —  SINGLE SOURCE OF TRUTH
# =============================================================================
# This list defines EVERY Open-H dataset: path, embodiment tag, and mix_ratio.
# Everything else (the set of Open-H embodiment tags, the stats-file check in
# dataset.py, the Hydra dataloader configs in data.py) is DERIVED from here.
#
# Weighting strategy (adapted from gr00t-H exp62/exp86 for Cosmos monocular):
#   - CMR Versius: 50% of total training (mix_ratio sum = 4.0)
#   - Remaining 50%: step-weighted by frame count (mix_ratio sum = 4.0)
#   - Grand total: 8.0
#   - Excludes TUM Ultrasound and SanoScience Surgical Simulation
#   - All video is monocular (endoscope_left); no stereo duplicates
#
# Non-CMR normalization: frames_i / (total_non_cmr_frames / 4.0)
#   total_non_cmr_frames = 5,488,040  →  norm_factor = 1,372,010
#
#   | Group                  | Frames      | Mix Ratio | % of Total |
#   |------------------------|-------------|-----------|------------|
#   | CMR (4 proc)           | ~17M        | 4.000     | 50.0%      |
#   | dVRK JHU mono (9 ds)   | 2,764,038   | 2.015     | 25.2%      |
#   | Stanford Real (3 ds)   | 874,437     | 0.637     | 8.0%       |
#   | Hamlyn (6 tasks)       | 544,573     | 0.397     | 5.0%       |
#   | Turin MITIC            | 388,690     | 0.283     | 3.5%       |
#   | UCSD (2 datasets)      | 314,917     | 0.230     | 2.9%       |
#   | UCB                    | 221,950     | 0.162     | 2.0%       |
#   | USTC (3 tasks)         | 185,319     | 0.135     | 1.7%       |
#   | LSCR ARCADE (2 ds)     | 183,096     | 0.133     | 1.7%       |
#   | Moon                   | 12,020      | 0.009     | 0.1%       |
#   | **Total**              |             | **8.000** | **100%**   |
#
# Frame counts sourced from gr00t-H exp62/exp79/exp86 configs and info.json.
# Dataset paths below are cluster defaults; override in experiment configs.
# =============================================================================

# Base paths (override per cluster / experiment)
_OPEN_H_BASE = "/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/datasets/Open-H"
_JHU_BASE = "/lustre/fsw/portfolios/healthcareeng/users/lzbinden/cache/huggingface/lerobot/jhu"
_LSCR_BASE = f"{_OPEN_H_BASE}/Surgical/JHU/LSCR"
_STANFORD_BASE = (
    f"{_OPEN_H_BASE}/Surgical/Stanford/Collaborative Haptics and Robotics in Medicine Lab/Real Robot (dVRK)"
)

OPEN_H_DATASET_SPECS: list[dict] = [
    # ===== CMR Versius (50% of total, 4 procedures) =====
    # CMR endoscope is monocular by nature — no stereo/mono distinction.
    # mix_ratio: 1.0 each × 4 = 4.0 total (50%)
    {
        "path": f"{_OPEN_H_BASE}/cmr-surgical-60hz-fixed/cholecystectomy_360p",
        "embodiment": EmbodimentTag.CMR_VERSIUS,
        "mix_ratio": 1.0,
    },
    {
        "path": f"{_OPEN_H_BASE}/cmr-surgical-60hz-fixed/hysterectomy_360p",
        "embodiment": EmbodimentTag.CMR_VERSIUS,
        "mix_ratio": 1.0,
    },
    {
        "path": f"{_OPEN_H_BASE}/cmr-surgical-60hz-fixed/inguinal_hernia_360p",
        "embodiment": EmbodimentTag.CMR_VERSIUS,
        "mix_ratio": 1.0,
    },
    {
        "path": f"{_OPEN_H_BASE}/cmr-surgical-60hz-fixed/prostatectomy_360p",
        "embodiment": EmbodimentTag.CMR_VERSIUS,
        "mix_ratio": 1.0,
    },
    # ===== dVRK JHU — ALL monocular (endoscope_left only), subtotal: 2.015 =====
    {
        "path": f"{_JHU_BASE}/srth_porcine_chole_fix",
        "embodiment": EmbodimentTag.JHU_DVRK_MONO,
        "mix_ratio": 1.369,
    },  # 1,878,393 frames
    {
        "path": f"{_JHU_BASE}/electrocautery_tumor_resection_fix",
        "embodiment": EmbodimentTag.JHU_DVRK_MONO,
        "mix_ratio": 0.160,
    },  # 219,007 frames
    {
        "path": f"{_JHU_BASE}/suturebot_2",
        "embodiment": EmbodimentTag.JHU_DVRK_MONO,
        "mix_ratio": 0.178,
    },  # 243,701 frames
    {
        "path": f"{_JHU_BASE}/suturebot_3",
        "embodiment": EmbodimentTag.JHU_DVRK_MONO,
        "mix_ratio": 0.134,
    },  # 183,366 frames
    {
        "path": f"{_JHU_BASE}/suturebot_tissue_2",
        "embodiment": EmbodimentTag.JHU_DVRK_MONO,
        "mix_ratio": 0.077,
    },  # 105,356 frames
    {
        "path": f"{_JHU_BASE}/srt_needle_pickup+handover",
        "embodiment": EmbodimentTag.JHU_DVRK_MONO,
        "mix_ratio": 0.042,
    },  # 58,305 frames
    {
        "path": f"{_JHU_BASE}/suturing_Jesse_processed_fix",
        "embodiment": EmbodimentTag.JHU_DVRK_MONO,
        "mix_ratio": 0.024,
    },  # 32,764 frames
    {
        "path": f"{_JHU_BASE}/srt_tissue_lift",
        "embodiment": EmbodimentTag.JHU_DVRK_MONO,
        "mix_ratio": 0.020,
    },  # 27,487 frames
    {
        "path": f"{_JHU_BASE}/jesse_pickup_only",
        "embodiment": EmbodimentTag.JHU_DVRK_MONO,
        "mix_ratio": 0.011,
    },  # 15,659 frames
    # ===== LSCR ARCADE (monocular, same dVRK action space), subtotal: 0.133 =====
    # Uses JHU_DVRK_MONO since Cosmos is monocular (endoscope_left only).
    {
        "path": f"{_LSCR_BASE}/ARCADE/Cholecystectomy",
        "embodiment": EmbodimentTag.JHU_DVRK_MONO,
        "mix_ratio": 0.130,
    },  # 177,808 frames
    {
        "path": f"{_LSCR_BASE}/ARCADE/cautery",
        "embodiment": EmbodimentTag.JHU_DVRK_MONO,
        "mix_ratio": 0.004,
        "exclude_splits": ["missing_videos"],
    },  # 5,288 frames
    # ===== Stanford Real (Euler RPY), subtotal: 0.637 =====
    # Excludes failed and bad-frame episodes per gr00t-H exp86.
    {
        "path": f"{_STANFORD_BASE}/Needle Transfer",
        "embodiment": EmbodimentTag.DVRK_STANFORD_REAL,
        "mix_ratio": 0.229,
        "exclude_splits": ["fail", "bad_frames"],
    },  # 313,882 frames
    {
        "path": f"{_STANFORD_BASE}/Tissue Retraction",
        "embodiment": EmbodimentTag.DVRK_STANFORD_REAL,
        "mix_ratio": 0.213,
    },  # 291,700 frames
    {
        "path": f"{_STANFORD_BASE}/Peg Transfer",
        "embodiment": EmbodimentTag.DVRK_STANFORD_REAL,
        "mix_ratio": 0.196,
        "exclude_splits": ["fail"],
    },  # 268,855 frames
    # ===== Hamlyn 30Hz (endoscope_left only), subtotal: 0.397 =====
    {
        "path": f"{_OPEN_H_BASE}/Surgical/Hamlyn/Suturing-2",
        "embodiment": EmbodimentTag.HAMLYN_30HZ,
        "mix_ratio": 0.183,
    },  # 251,355 frames
    {
        "path": f"{_OPEN_H_BASE}/Surgical/Hamlyn/peg_transfer",
        "embodiment": EmbodimentTag.HAMLYN_30HZ,
        "mix_ratio": 0.074,
    },  # 102,187 frames
    {
        "path": f"{_OPEN_H_BASE}/Surgical/Hamlyn/Suturing-1",
        "embodiment": EmbodimentTag.HAMLYN_30HZ,
        "mix_ratio": 0.073,
    },  # 100,067 frames
    {
        "path": f"{_OPEN_H_BASE}/Surgical/Hamlyn/needle_grasp_and_handover",
        "embodiment": EmbodimentTag.HAMLYN_30HZ,
        "mix_ratio": 0.034,
    },  # 46,582 frames
    {
        "path": f"{_OPEN_H_BASE}/Surgical/Hamlyn/knot_tying",
        "embodiment": EmbodimentTag.HAMLYN_30HZ,
        "mix_ratio": 0.022,
    },  # 30,222 frames
    {
        "path": f"{_OPEN_H_BASE}/Surgical/Hamlyn/Tissue_Retraction",
        "embodiment": EmbodimentTag.HAMLYN_30HZ,
        "mix_ratio": 0.010,
    },  # 14,160 frames
    # ===== Turin MITIC ex vivo, subtotal: 0.283 =====
    # Excludes failed episodes per gr00t-H exp86.
    {
        "path": f"{_OPEN_H_BASE}/Surgical/Turin/mitic_lerobot_ex_vivo",
        "embodiment": EmbodimentTag.TURIN_MITIC_EX_VIVO,
        "mix_ratio": 0.283,
        "exclude_splits": ["failure"],
    },  # 388,690 frames
    # ===== UCSD, subtotal: 0.230 =====
    {
        "path": f"{_OPEN_H_BASE}/Surgical/UCSD/surgical_learning_dataset",
        "embodiment": EmbodimentTag.DVRK_UCSD,
        "mix_ratio": 0.210,
    },  # 288,604 frames
    {
        "path": f"{_OPEN_H_BASE}/Surgical/UCSD/surgical_learning_dataset2",
        "embodiment": EmbodimentTag.DVRK_UCSD,
        "mix_ratio": 0.019,
    },  # 26,313 frames
    # ===== UCB Debridement, subtotal: 0.162 =====
    {
        "path": f"{_OPEN_H_BASE}/Surgical/UCBerkeley/debridement_lerobot",
        "embodiment": EmbodimentTag.DVRK_UCB,
        "mix_ratio": 0.162,
    },  # 221,950 frames
    # ===== USTC Torin, subtotal: 0.135 =====
    {
        "path": f"{_OPEN_H_BASE}/Surgical/USTC/knot_tying_all",
        "embodiment": EmbodimentTag.USTC_TORIN,
        "mix_ratio": 0.081,
    },  # 110,652 frames
    {
        "path": f"{_OPEN_H_BASE}/Surgical/USTC/needle_handover_all",
        "embodiment": EmbodimentTag.USTC_TORIN,
        "mix_ratio": 0.013,
    },  # 17,495 frames
    {
        "path": f"{_OPEN_H_BASE}/Surgical/USTC/needle_pickup_all",
        "embodiment": EmbodimentTag.USTC_TORIN,
        "mix_ratio": 0.042,
    },  # 57,172 frames
    # ===== Moon Surgical, subtotal: 0.009 =====
    # Excludes episodes with missing scope videos per gr00t-H exp86.
    {
        "path": f"{_OPEN_H_BASE}/Surgical/Moon/moon",
        "embodiment": EmbodimentTag.MOON,
        "mix_ratio": 0.009,
        "exclude_splits": ["missing_scope_videos"],
    },  # 12,020 frames
]

# Derived: the set of all Open-H embodiment tag strings.
# Used by dataset.py to enforce stats_cosmos.json requirement.
# Includes both EMBODIMENT_REGISTRY keys (non-CMR) and all tags from the specs.
OPEN_H_EMBODIMENT_TAGS: frozenset[str] = frozenset(
    {
        (spec["embodiment"].value if isinstance(spec["embodiment"], EmbodimentTag) else spec["embodiment"])
        for spec in OPEN_H_DATASET_SPECS
    }
    | set(EMBODIMENT_REGISTRY.keys())
)


def _build_generic_config_and_transforms(
    num_frames: int,
    reg: dict,
    downscaled_res: bool = False,
) -> tuple[dict, ComposedModalityTransform, ComposedModalityTransform]:
    """Build modality config and transforms for a generic (non-CMR) Open-H embodiment.

    This creates the standard pipeline:
      Video: ToTensor → Crop (train only) → Resize
      State/Action: ToTensor → mean_std normalization → Concat

    Supports ``pass_through_state_keys`` for embodiments where certain state
    keys are needed as references for action transforms (e.g. REL_XYZ_ROT6D)
    but should NOT be normalized or concatenated into the model's state input.
    For example, dVRK UCB uses cartesian pose as the action reference frame
    while joint angles are the actual state input.

    Args:
        num_frames: Total number of video frames (1 context + N-1 prediction).
        reg: Registry entry dict from EMBODIMENT_REGISTRY.  May include:
            - ``pass_through_state_keys``: list of state keys that are loaded
              and converted to tensors (so the action transform can read them)
              but excluded from normalization and concatenation.
        downscaled_res: If True, use 256×256 resolution.

    Returns:
        Tuple of (modality_config_dict, train_transform, test_transform).
    """
    timestep_interval = reg["timestep_interval"]

    # Video: all num_frames frames
    video_delta_indices = list(range(0, num_frames * timestep_interval, timestep_interval))

    # Action: num_frames - 1 action timesteps (prediction frames only)
    num_action_frames = num_frames - 1
    action_delta_indices = list(range(0, num_action_frames * timestep_interval, timestep_interval))

    config = {
        "video": ModalityConfig(
            delta_indices=video_delta_indices,
            modality_keys=reg["video_keys"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=reg["state_keys"],
        ),
        "action": ModalityConfig(
            delta_indices=action_delta_indices,
            modality_keys=reg["action_keys"],
        ),
        # Extra metadata for dataset initialization
        "modality_filename": reg.get("modality_filename", "meta/modality.json"),
    }

    width = reg["video_width"] if not downscaled_res else 256
    height = reg["video_height"] if not downscaled_res else 256
    norm_mode = reg.get("normalization_mode", "mean_std")

    video_keys = reg["video_keys"]
    state_keys = reg["state_keys"]
    action_keys = reg["action_keys"]
    action_key_configs = reg.get("action_key_configs", {})

    # Pass-through state keys: loaded and converted to tensors (so the action
    # transform can read them as reference frames), but NOT normalized or
    # concatenated into the model's state input vector.
    # Example: dVRK UCB uses cartesian pose (psm1_pose, psm2_pose) as the
    # REL_XYZ_ROT6D reference frame, while joint angles are the actual state input.
    pass_through_state_keys = set(reg.get("pass_through_state_keys", []))

    # State keys that get normalized and concatenated (excluding pass-through)
    normalizable_state_keys = [k for k in state_keys if k not in pass_through_state_keys]

    # Build the relative action transform (runs BEFORE normalization)
    # This converts raw absolute actions to deltas using the state reference.
    # IMPORTANT: Must run before StateActionTransform because normalization
    # would corrupt the reference state (e.g., quaternions) needed for
    # the relative conversion.
    rel_action_transform = GenericRelativeActionTransform(
        apply_to=action_keys,
        action_key_configs=action_key_configs,
    )

    train_transform = ComposedModalityTransform(
        transforms=[
            VideoToTensor(apply_to=video_keys),
            VideoCrop(apply_to=video_keys, scale=0.95),
            VideoResize(apply_to=video_keys, height=height, width=width, interpolation="linear"),
            # Convert ALL state keys to tensors (including pass-through keys
            # needed by the action transform for reference frame lookups)
            StateActionToTensor(apply_to=state_keys),
            StateActionToTensor(apply_to=action_keys),
            # Delta action conversion BEFORE normalization
            # (pass-through state keys are read here as reference frames)
            rel_action_transform,
            # Normalization AFTER delta conversion
            # Only normalize non-pass-through state keys
            StateActionTransform(
                apply_to=normalizable_state_keys,
                normalization_modes={k: norm_mode for k in normalizable_state_keys},
            ),
            StateActionTransform(
                apply_to=action_keys,
                normalization_modes={k: norm_mode for k in action_keys},
            ),
            # Only concatenate non-pass-through state keys into model input
            ConcatTransform(
                video_concat_order=video_keys,
                state_concat_order=normalizable_state_keys,
                action_concat_order=action_keys,
            ),
        ]
    )

    test_transform = ComposedModalityTransform(
        transforms=[
            VideoToTensor(apply_to=video_keys),
            VideoResize(apply_to=video_keys, height=height, width=width, interpolation="linear"),
            # Convert ALL state keys to tensors (including pass-through)
            StateActionToTensor(apply_to=state_keys),
            StateActionToTensor(apply_to=action_keys),
            # Delta action conversion BEFORE normalization
            rel_action_transform,
            # Normalization AFTER delta conversion
            StateActionTransform(
                apply_to=normalizable_state_keys,
                normalization_modes={k: norm_mode for k in normalizable_state_keys},
            ),
            StateActionTransform(
                apply_to=action_keys,
                normalization_modes={k: norm_mode for k in action_keys},
            ),
            ConcatTransform(
                video_concat_order=video_keys,
                state_concat_order=normalizable_state_keys,
                action_concat_order=action_keys,
            ),
        ]
    )

    return config, train_transform, test_transform


def construct_modality_config_and_transforms(num_frames, embodiment, downscaled_res=False):
    if embodiment == "gr1":
        timestep_interval = 2
        delta_indices = list(range(0, num_frames * timestep_interval, timestep_interval))
        video_key = "video.ego_view_freq20" if not downscaled_res else "video.ego_view_bg_crop_pad_res256_freq20"
        config = {
            "video": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=[video_key],
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=[
                    "state.left_arm",
                    "state.right_arm",
                    "state.left_hand",
                    "state.right_hand",
                    "state.waist",
                ],
            ),
            "action": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=[
                    "action.left_arm",
                    "action.right_arm",
                    "action.left_hand",
                    "action.right_hand",
                    "action.waist",
                ],
            ),
        }
    elif embodiment == "gr1_video_only":
        timestep_interval = 1
        delta_indices = list(range(0, num_frames * timestep_interval, timestep_interval))
        config = {
            "video": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=["video.ego_view_bg_crop_pad_res256_freq20"],
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=[
                    "state.left_arm",
                    "state.right_arm",
                    "state.left_hand",
                    "state.right_hand",
                    "state.waist",
                ],
            ),
            "action": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=[
                    "action.left_arm",
                    "action.right_arm",
                    "action.left_hand",
                    "action.right_hand",
                    "action.waist",
                ],
            ),
            "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.coarse_action"]),
        }
    elif embodiment == "agibot":
        timestep_interval = 4
        delta_indices = list(range(0, num_frames * timestep_interval, timestep_interval))
        video_key = "video.top_head" if not downscaled_res else "video.top_head_pad_res256_freq10"
        config = {
            "video": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=[video_key],
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=[
                    "state.left_arm_joint_position",
                    "state.right_arm_joint_position",
                    "state.left_effector_position",
                    "state.right_effector_position",
                    "state.head_position",
                    "state.waist_position",
                ],
            ),
            "action": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=[
                    "action.left_arm_joint_position",
                    "action.right_arm_joint_position",
                    "action.left_effector_position",
                    "action.right_effector_position",
                    "action.head_position",
                    "action.waist_position",
                    "action.robot_velocity",
                ],
            ),
        }
    elif embodiment == "cmr_versius":
        # CMR Versius surgical robot configuration
        # Original data is 60Hz, using FRAME_STRIDE=6 for 10fps effective rate
        timestep_interval = 6

        # Video: 13 frames (1 context + 12 prediction)
        video_delta_indices = list(range(0, num_frames * timestep_interval, timestep_interval))

        # Action: 12 timesteps (only for the 12 prediction frames, not context)
        # The model expects num_actions to be divisible by temporal_compression_ratio (4)
        # 12 actions / 4 = 3 latent temporal positions, each getting action embedding
        # Note: action timesteps start from index 0 (same as video context frame) because
        # the action at t=0 represents the transition FROM frame 0 TO frame 1
        num_action_frames = num_frames - 1  # 12 action timesteps for 13 video frames
        action_delta_indices = list(range(0, num_action_frames * timestep_interval, timestep_interval))

        config = {
            "video": ModalityConfig(
                delta_indices=video_delta_indices,
                modality_keys=["video.endoscope"],
            ),
            "state": ModalityConfig(
                delta_indices=[0],  # Single reference state for hybrid-relative
                modality_keys=[
                    # State pose for reference (xyz + quat = 7D each arm)
                    "state.left_pose",
                    "state.left_gripper",
                    "state.right_pose",
                    "state.right_gripper",
                    # Engagement status for clutch-aware processing
                    "state.hapticengaged_left",
                    "state.hapticengaged_right",
                    # Motion scaling factors (pass-through for hybrid-relative conversion)
                    "state.translation_scaling",
                    "state.rotation_scaling",
                ],
            ),
            "action": ModalityConfig(
                delta_indices=action_delta_indices,
                modality_keys=[
                    # Left arm: pose (xyz + quat = 7D raw, converted to xyz + rot6d = 9D hybrid-relative)
                    "action.left_pose",
                    "action.left_gripper",
                    # Right arm: pose (xyz + quat = 7D raw, converted to xyz + rot6d = 9D hybrid-relative)
                    "action.right_pose",
                    "action.right_gripper",
                    # Energy buttons (binary)
                    "action.left_energy",
                    "action.right_energy",
                    # Thumbstick controls (for endoscope control and instrument straighten function)
                    "action.thumbstick_x_left",
                    "action.thumbstick_x_right",
                    "action.thumbstick_y_left",
                    "action.thumbstick_y_right",
                    "action.thumbstickBtn_left",
                    "action.thumbstickBtn_right",
                    # Clutch button inputs (engage/disengage arm control)
                    "action.clutchBtn_left",
                    "action.clutchBtn_right",
                    # Engagement status (pass-through for clutch-aware processing)
                    # Note: hapticengaged keys (without cond_ prefix) are used for clutch-aware processing but removed after
                    "action.hapticengaged_left",
                    "action.hapticengaged_right",
                    # =====================================================
                    # STATE CONDITIONING VARIABLES (sampled at action timesteps)
                    # These are from observation.state but sampled at action delta_indices
                    # for MLP conditioning. They're passed through as absolute values.
                    # =====================================================
                    # Haptic engagement state (persistent, unlike clutchBtn which is momentary)
                    "action.cond_hapticengaged_left",
                    "action.cond_hapticengaged_right",
                    # Which physical arm (0-3) each controller is linked to
                    "action.cond_armlinkedtohaptic_left",
                    "action.cond_armlinkedtohaptic_right",
                    # Instrument type for each arm (0-3)
                    "action.cond_arm_0_instrtype",
                    "action.cond_arm_1_instrtype",
                    "action.cond_arm_2_instrtype",
                    "action.cond_arm_3_instrtype",
                    # HUD color assignment for each arm (0-3)
                    "action.cond_arm_0_color",
                    "action.cond_arm_1_color",
                    "action.cond_arm_2_color",
                    "action.cond_arm_3_color",
                    # Electrosurgery mode (CUT/COAG) selected on each controller
                    "action.cond_electroSurgeryMode_left",
                    "action.cond_electroSurgeryMode_right",
                ],
            ),
        }

    # =========================================================================
    # SutureBot (JHU dVRK, pre-concatenated LeRobot format)
    # =========================================================================
    # SutureBot uses a single concatenated action key 'action.action' (20D)
    # with dual-arm dVRK data: [arm1: xyz(3)+rot6d(6)+gripper(1)] × 2.
    # Uses RelativeActionTransform for delta conversion (different from the
    # per-key GenericRelativeActionTransform used by EMBODIMENT_REGISTRY entries).
    # =========================================================================
    elif embodiment == "suturebot":
        timestep_interval = 3
        delta_indices = list(range(0, num_frames * timestep_interval, timestep_interval))
        config = {
            "video": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=["video.observation.images.main"],
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=["state.observation.state"],
            ),
            "action": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=["action.action"],
            ),
        }

    # =========================================================================
    # Registry-based embodiments (all non-CMR Open-H datasets)
    # =========================================================================
    if embodiment in EMBODIMENT_REGISTRY:
        return _build_generic_config_and_transforms(num_frames, EMBODIMENT_REGISTRY[embodiment], downscaled_res)

    video_modality, state_modality, action_modality = config["video"], config["state"], config["action"]
    if embodiment == "gr1" or embodiment == "gr1_video_only":
        width = 832 if not downscaled_res else 256
        height = 480 if not downscaled_res else 256
    elif embodiment == "agibot":
        width = 640 if not downscaled_res else 256
        height = 480 if not downscaled_res else 256
    elif embodiment == "cmr_versius":
        # CMR Versius endoscope video resolution (original: 1920x1080, 16:9 aspect ratio)
        #
        # IMPORTANT: Resolution must be divisible by 16 (8x VAE compression × 2 patch size)
        # Valid 16:9 options: 512x288, 768x432, 1024x576, 1280x720
        # Invalid: 384x216 (216/16=13.5), 320x180 (180/16=11.25)
        #
        # Using 512x288 for fast PoC training while maintaining 16:9 aspect ratio
        # For production: consider 768x432 or 1280x720 (matches Cosmos 720p pretrain)
        # cf. https://docs.google.com/presentation/d/1G0mqiQRBQohDAMjMG6hpLzPVCZi3KbJxHSi4LXMJl5A/edit?slide=id.g3b869a60288_1_50#slide=id.g3b869a60288_1_50
        width = 512 if not downscaled_res else 256
        height = 288 if not downscaled_res else 256
    elif embodiment == "suturebot":
        # SutureBot: same resolution as CMR Versius (512x288, 16:9)
        width = 512 if not downscaled_res else 256
        height = 288 if not downscaled_res else 256

    # Build embodiment-specific transforms
    if embodiment == "cmr_versius":
        # CMR Versius uses hybrid-relative actions with rot6d rotation format
        # Final conditioning: 44D = 30D actions + 14D state conditioning
        #   Actions (30D):
        #     - left(9D pose + 1D gripper) + right(9D pose + 1D gripper) = 20D
        #     - energy(2D) + thumbstick_x(2D) + thumbstick_y(2D) + thumbstickBtn(2D) + clutchBtn(2D) = 10D
        #   State conditioning (14D, sampled at action timesteps):
        #     - haptic_engaged(2D) + armlinkedtohaptic(2D) + instrtype(4D) + color(4D) + electroSurgeryMode(2D) = 14D
        # Note: hapticengaged keys (without cond_ prefix) are used for clutch-aware processing but removed after

        # Keys that get concatenated into final conditioning tensor (exclude pass-through keys)
        cmr_action_output_keys = [
            # === ACTIONS (30D) ===
            "action.left_pose",
            "action.left_gripper",
            "action.right_pose",
            "action.right_gripper",
            "action.left_energy",
            "action.right_energy",
            "action.thumbstick_x_left",
            "action.thumbstick_x_right",
            "action.thumbstick_y_left",
            "action.thumbstick_y_right",
            "action.thumbstickBtn_left",
            "action.thumbstickBtn_right",
            "action.clutchBtn_left",
            "action.clutchBtn_right",
            # === STATE CONDITIONING (12D) ===
            "action.cond_hapticengaged_left",
            "action.cond_hapticengaged_right",
            "action.cond_armlinkedtohaptic_left",
            "action.cond_armlinkedtohaptic_right",
            "action.cond_arm_0_instrtype",
            "action.cond_arm_1_instrtype",
            "action.cond_arm_2_instrtype",
            "action.cond_arm_3_instrtype",
            "action.cond_arm_0_color",
            "action.cond_arm_1_color",
            "action.cond_arm_2_color",
            "action.cond_arm_3_color",
            # Electrosurgery mode
            "action.cond_electroSurgeryMode_left",
            "action.cond_electroSurgeryMode_right",
        ]

        # Thumbstick, clutch button, and state conditioning keys (ABSOLUTE values, pass-through)
        # These are NOT converted to deltas - they pass through as raw absolute values.
        # - Thumbstick: endoscope/instrument control (continuous)
        # - ClutchBtn: engage/disengage arm control (binary button press)
        # - State conditioning: system state sampled at action timesteps for MLP conditioning
        cmr_passthrough_action_keys = [
            # Thumbstick controls
            "action.thumbstick_x_left",
            "action.thumbstick_x_right",
            "action.thumbstick_y_left",
            "action.thumbstick_y_right",
            "action.thumbstickBtn_left",
            "action.thumbstickBtn_right",
            # Clutch buttons
            "action.clutchBtn_left",
            "action.clutchBtn_right",
            # State conditioning (from observation.state, sampled at action timesteps)
            "action.cond_hapticengaged_left",
            "action.cond_hapticengaged_right",
            "action.cond_armlinkedtohaptic_left",
            "action.cond_armlinkedtohaptic_right",
            "action.cond_arm_0_instrtype",
            "action.cond_arm_1_instrtype",
            "action.cond_arm_2_instrtype",
            "action.cond_arm_3_instrtype",
            "action.cond_arm_0_color",
            "action.cond_arm_1_color",
            "action.cond_arm_2_color",
            "action.cond_arm_3_color",
            # Electrosurgery mode
            "action.cond_electroSurgeryMode_left",
            "action.cond_electroSurgeryMode_right",
        ]

        # Keys that don't need normalization (engagement status, scaling factors, pass-through)
        cmr_state_passthrough_keys = [
            "state.hapticengaged_left",
            "state.hapticengaged_right",
            "state.translation_scaling",
            "state.rotation_scaling",
        ]
        cmr_action_passthrough_keys = ["action.hapticengaged_left", "action.hapticengaged_right"]

        # State keys to include in final output (excluding pass-through only keys)
        cmr_state_output_keys = [k for k in state_modality.modality_keys if k not in cmr_state_passthrough_keys]

        # NOTE: Normalization uses stats_cosmos.json (not stats.json) for CMR Versius.
        # Run scripts/compute_cmr_action_stats.py to generate stats_cosmos.json with
        # correct statistics for the 9D hybrid-relative pose format.
        # Stats loading is handled in dataset.py which checks for stats_cosmos.json first.

        train_transform = ComposedModalityTransform(
            transforms=[
                VideoToTensor(apply_to=video_modality.modality_keys),
                VideoCrop(apply_to=video_modality.modality_keys, scale=0.95),
                VideoResize(apply_to=video_modality.modality_keys, height=height, width=width, interpolation="linear"),
                StateActionToTensor(apply_to=state_modality.modality_keys),
                StateActionToTensor(apply_to=action_modality.modality_keys),
                # CMR Versius relative action transform: converts absolute actions to hybrid-relative
                # IMPORTANT: Must run BEFORE state normalization because it uses raw state poses
                # (state.left_pose, state.right_pose) as reference for relative computation.
                # Normalizing quaternions would produce invalid rotation matrices!
                # CMR data stores poses as quaternions (7D: xyz + quat_xyzw), output is rot6d (9D)
                CMRVersiusRelativeActionTransform(
                    apply_to=["action.left_pose", "action.right_pose"],
                    pose_keys={
                        "action.left_pose": "state.left_pose",
                        "action.right_pose": "state.right_pose",
                    },
                    gripper_keys=["action.left_gripper", "action.right_gripper"],
                    energy_keys=["action.left_energy", "action.right_energy"],
                    thumbstick_keys=cmr_passthrough_action_keys,
                    engaged_left_key="state.hapticengaged_left",
                    engaged_right_key="state.hapticengaged_right",
                    input_rotation_format="quat",  # CMR data uses quaternions (xyzw)
                    reference_rotation_format="quat",  # State also uses quaternions (xyzw)
                    # Motion scaling (converts hand-controller-space to instrument-space):
                    translation_scaling_key="state.translation_scaling",
                    rotation_scaling_key="state.rotation_scaling",
                    # Remove passthrough keys after processing (not needed in final output)
                    action_passthrough_keys=cmr_action_passthrough_keys,
                    state_passthrough_keys=cmr_state_passthrough_keys,
                ),
                # State normalization (uses stats_cosmos.json) - AFTER CMRVersiusRelativeActionTransform
                # State poses remain as raw 7D (xyz + quat), normalized here for model input
                StateActionTransform(
                    apply_to=cmr_state_output_keys,
                    normalization_modes={key: "mean_std" for key in cmr_state_output_keys},
                ),
                # Action normalization (uses stats_cosmos.json with 9D hybrid-relative pose stats)
                StateActionTransform(
                    apply_to=cmr_action_output_keys,
                    normalization_modes={key: "mean_std" for key in cmr_action_output_keys},
                ),
                ConcatTransform(
                    video_concat_order=video_modality.modality_keys,
                    state_concat_order=cmr_state_output_keys,
                    action_concat_order=cmr_action_output_keys,
                ),
            ]
        )
        test_transform = ComposedModalityTransform(
            transforms=[
                VideoToTensor(apply_to=video_modality.modality_keys),
                VideoResize(apply_to=video_modality.modality_keys, height=height, width=width, interpolation="linear"),
                StateActionToTensor(apply_to=state_modality.modality_keys),
                StateActionToTensor(apply_to=action_modality.modality_keys),
                # CMR Versius relative action transform - BEFORE state normalization
                CMRVersiusRelativeActionTransform(
                    apply_to=["action.left_pose", "action.right_pose"],
                    pose_keys={
                        "action.left_pose": "state.left_pose",
                        "action.right_pose": "state.right_pose",
                    },
                    gripper_keys=["action.left_gripper", "action.right_gripper"],
                    energy_keys=["action.left_energy", "action.right_energy"],
                    thumbstick_keys=cmr_passthrough_action_keys,
                    engaged_left_key="state.hapticengaged_left",
                    engaged_right_key="state.hapticengaged_right",
                    input_rotation_format="quat",
                    reference_rotation_format="quat",
                    translation_scaling_key="state.translation_scaling",
                    rotation_scaling_key="state.rotation_scaling",
                    # Remove passthrough keys after processing (not needed in final output)
                    action_passthrough_keys=cmr_action_passthrough_keys,
                    state_passthrough_keys=cmr_state_passthrough_keys,
                ),
                # State normalization (uses stats_cosmos.json) - AFTER CMRVersiusRelativeActionTransform
                StateActionTransform(
                    apply_to=cmr_state_output_keys,
                    normalization_modes={key: "mean_std" for key in cmr_state_output_keys},
                ),
                # Action normalization (uses stats_cosmos.json with 9D hybrid-relative pose stats)
                StateActionTransform(
                    apply_to=cmr_action_output_keys,
                    normalization_modes={key: "mean_std" for key in cmr_action_output_keys},
                ),
                ConcatTransform(
                    video_concat_order=video_modality.modality_keys,
                    state_concat_order=cmr_state_output_keys,
                    action_concat_order=cmr_action_output_keys,
                ),
            ]
        )
    elif embodiment == "suturebot":
        # SutureBot uses pre-concatenated 20D actions with RelativeActionTransform.
        # Pipeline: ToTensor → RelativeAction (absolute→relative) → Normalize → Concat
        train_transform = ComposedModalityTransform(
            transforms=[
                VideoToTensor(apply_to=video_modality.modality_keys),
                VideoCrop(apply_to=video_modality.modality_keys, scale=0.95),
                VideoResize(apply_to=video_modality.modality_keys, height=height, width=width, interpolation="linear"),
                StateActionToTensor(apply_to=state_modality.modality_keys),
                StateActionTransform(
                    apply_to=state_modality.modality_keys,
                    normalization_modes={key: "mean_std" for key in state_modality.modality_keys},
                ),
                StateActionToTensor(apply_to=action_modality.modality_keys),
                RelativeActionTransform(apply_to=action_modality.modality_keys),
                StateActionTransform(
                    apply_to=action_modality.modality_keys,
                    normalization_modes={key: "mean_std" for key in action_modality.modality_keys},
                ),
                ConcatTransform(
                    video_concat_order=video_modality.modality_keys,
                    state_concat_order=state_modality.modality_keys,
                    action_concat_order=action_modality.modality_keys,
                ),
            ]
        )
        test_transform = ComposedModalityTransform(
            transforms=[
                VideoToTensor(apply_to=video_modality.modality_keys),
                VideoResize(apply_to=video_modality.modality_keys, height=height, width=width, interpolation="linear"),
                StateActionToTensor(apply_to=state_modality.modality_keys),
                StateActionTransform(
                    apply_to=state_modality.modality_keys,
                    normalization_modes={key: "mean_std" for key in state_modality.modality_keys},
                ),
                StateActionToTensor(apply_to=action_modality.modality_keys),
                RelativeActionTransform(apply_to=action_modality.modality_keys),
                StateActionTransform(
                    apply_to=action_modality.modality_keys,
                    normalization_modes={key: "mean_std" for key in action_modality.modality_keys},
                ),
                ConcatTransform(
                    video_concat_order=video_modality.modality_keys,
                    state_concat_order=state_modality.modality_keys,
                    action_concat_order=action_modality.modality_keys,
                ),
            ]
        )
    else:
        # Default transforms for gr1, agibot, etc.
        train_transform = ComposedModalityTransform(
            transforms=[
                VideoToTensor(apply_to=video_modality.modality_keys),
                VideoCrop(apply_to=video_modality.modality_keys, scale=0.95),
                VideoResize(apply_to=video_modality.modality_keys, height=height, width=width, interpolation="linear"),
                # VideoColorJitter(apply_to=video_modality.modality_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
                StateActionToTensor(apply_to=state_modality.modality_keys),
                StateActionTransform(
                    apply_to=state_modality.modality_keys,
                    normalization_modes={key: "min_max" for key in state_modality.modality_keys},
                ),
                StateActionToTensor(apply_to=action_modality.modality_keys),
                StateActionTransform(
                    apply_to=action_modality.modality_keys,
                    normalization_modes={key: "min_max" for key in action_modality.modality_keys},
                ),
                ConcatTransform(
                    video_concat_order=video_modality.modality_keys,
                    state_concat_order=state_modality.modality_keys,
                    action_concat_order=action_modality.modality_keys,
                ),
            ]
        )
        test_transform = ComposedModalityTransform(
            transforms=[
                VideoToTensor(apply_to=video_modality.modality_keys),
                VideoResize(apply_to=video_modality.modality_keys, height=height, width=width, interpolation="linear"),
                StateActionToTensor(apply_to=state_modality.modality_keys),
                StateActionTransform(
                    apply_to=state_modality.modality_keys,
                    normalization_modes={key: "min_max" for key in state_modality.modality_keys},
                ),
                StateActionToTensor(apply_to=action_modality.modality_keys),
                StateActionTransform(
                    apply_to=action_modality.modality_keys,
                    normalization_modes={key: "min_max" for key in action_modality.modality_keys},
                ),
                ConcatTransform(
                    video_concat_order=video_modality.modality_keys,
                    state_concat_order=state_modality.modality_keys,
                    action_concat_order=action_modality.modality_keys,
                ),
            ]
        )

    return config, train_transform, test_transform
