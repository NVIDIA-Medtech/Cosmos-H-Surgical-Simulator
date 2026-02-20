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

from enum import Enum


class EmbodimentTag(Enum):
    GR1 = "gr1"
    """
    The GR1 dataset.
    """

    GR1_unified = "gr1_unified"
    """
    The GR1 unified dataset.
    """

    FRANKA = "franka"
    """
    The FRANKA dataset.
    """

    SO100 = "so100"
    """
    The SO100 dataset.
    """

    ROBOCASA = "robocasa_panda_omron"
    """
    The ROBOCASA dataset.
    """

    NEW_EMBODIMENT = "new_embodiment"
    """
    Any new embodiment for finetuning.
    """

    AGIBOT = "agibot"

    # =========================================================================
    # Open-H Surgical Embodiment Tags
    # Matching gr00t-H/gr00t/data/embodiment_tags.py
    # =========================================================================

    CMR_VERSIUS = "cmr_versius"
    """
    The CMR Versius surgical robot dataset.
    Dual-arm surgical robot with:
    - Hybrid-relative action representation for EEF poses
    - 44D conditioning space: 30D actions + 14D state conditioning
    - Clutch-aware filtering and action zeroing
    """

    DVRK = "dvrk"
    """
    The da Vinci Research Kit (dVRK) surgical robot (JHU stereo).
    Dual-arm (PSM1/PSM2) with Cartesian EEF pose + gripper.
    Raw action: psm1_pose(7D) + psm1_gripper(1D) + psm2_pose(7D) + psm2_gripper(1D) = 16D.
    """

    JHU_DVRK_MONO = "jhu_dvrk_mono"
    """
    Monocular JHU dVRK surgical robot variant.
    Uses only the left endoscope video stream with the same dual-arm state/action
    representation as the standard dVRK embodiment.
    """

    DVRK_UCB = "dvrk_ucb"
    """
    The UCBerkeley dVRK debridement dataset.
    Dual-arm (PSM1/PSM2) with cartesian EEF control (16D).
    Uses REL_XYZ_ROT6D for pose actions with quaternion inputs.
    Joint-angle state channels for normalization; cartesian pose as
    pass-through for action reference frame.
    """

    HAMLYN_30HZ = "hamlyn_30hz"
    """
    Hamlyn Centre dVRK surgical robot dataset - 30Hz tasks.
    Dual-arm with Cartesian EEF pose + gripper.
    Raw action: left_arm_pose(7D) + left_arm_gripper(1D) + right_arm_pose(7D) + right_arm_gripper(1D) = 16D.
    """

    DVRK_UCSD = "dvrk_ucsd"
    """
    The UCSD surgical learning dataset.
    Dual-arm (retraction + cutter) with delta EEF pose actions.
    Raw action: psm_retraction_pose(7D) + psm_retraction_gripper(1D) + psm_cutter_pose(7D) + psm_cutter_gripper(1D) = 16D.
    """

    USTC_TORIN = "ustc_torin"
    """
    The USTC Torin surgical dataset.
    Dual-arm with Cartesian delta actions + energy channel.
    Raw action: left_pose(7D) + left_gripper(1D) + right_pose(7D) + right_gripper(1D) = 16D.
    """

    DVRK_OBUDA = "dvrk_obuda"
    """
    The Obuda University Open-H dVRK datasets.
    Dual-arm (PSM1/PSM2) with Cartesian EEF pose + gripper.
    Raw action: psm1_pose(7D) + psm1_gripper(1D) + psm2_pose(7D) + psm2_gripper(1D) = 16D.
    """

    ROB_SURGICAL = "rob_surgical"
    """
    The Rob Surgical (bitrack) dataset.
    Single endoscope video with 4-arm Cartesian EEF state/action.
    Raw action: left_pose(7D) + right_pose(7D) + lap_pose(7D) + aux_pose(7D) = 28D.
    """

    DVRK_STANFORD_REAL = "dvrk_stanford_real"
    """
    Stanford real-robot dVRK datasets (Needle Transfer, Tissue Retraction, Peg Transfer).
    Dual-arm (PSM1/PSM2) with absolute EEF pose actions in Euler RPY.
    Raw action: psm1_pose(7D) + psm1_gripper(1D) + psm2_pose(7D) + psm2_gripper(1D) = 16D.
    """

    POLYU_SIM = "polyu_sim"
    """
    The PolyU simulated surgical dataset.
    Single-arm surgical robot with cartesian pose + gripper.
    Raw action: psm_cartesian_pose(7D) + psm_gripper(1D) = 8D.
    """

    MOON = "moon"
    """
    The Moon Surgical Maestro assistant dataset.
    Dual-arm robot with delta translation actions only.
    Raw action: right_arm_delta_xyz(3D) + left_arm_delta_xyz(3D) = 6D.
    """

    JHU_LSCR_MIRACLE = "jhu_lscr_miracle"
    """
    JHU LSCR MIRACLE datasets.
    Dual-arm joint-angle control with relative actions.
    Raw action: psm1_pose(7D) + psm1_gripper(1D) + psm2_pose(7D) + psm2_gripper(1D) = 16D.
    """

    JHU_LSCR_SMARTS = "jhu_lscr_smarts"
    """
    JHU LSCR SMARTS offline datasets.
    Dual-arm joint-angle control with relative actions.
    Raw action: psm1_pose(7D) + psm1_gripper(1D) + psm2_pose(7D) + psm2_gripper(1D) = 16D.
    """

    TUD_TUNDRA = "tud_tundra"
    """
    The TUD TUNDRA UR5e surgical assistance dataset.
    Single-arm with Cartesian EEF pose + gripper.
    Raw action: eef_pose(7D) + gripper(1D) = 8D.
    """

    TURIN_MITIC_EX_VIVO = "turin_mitic_ex_vivo"
    """
    The Turin MITIC ex vivo surgical dataset.
    Dual-arm dVRK (PSM1/PSM2) with absolute EEF pose actions.
    Raw action: psm1_pose(7D) + psm2_pose(7D) = 14D (no grippers).
    """
