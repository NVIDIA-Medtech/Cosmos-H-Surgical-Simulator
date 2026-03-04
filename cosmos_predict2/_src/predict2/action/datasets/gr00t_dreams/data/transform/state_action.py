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

import functools
import random
from typing import Any, ClassVar

import numpy as np

# import pytorch3d.transforms as pt
import torch
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator
from scipy.spatial.transform import Rotation

from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.schema import (
    DatasetMetadata,
    RotationType,
    StateActionMetadata,
)
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.transform.base import (
    InvertibleModalityTransform,
    ModalityTransform,
)

# =============================================================================
# Rotation helper functions for CMR Versius
# =============================================================================
#
# These functions use the COLUMN convention for 6D rotation representation,
# matching the gr00t-H / CMR Versius standard:
#   - Input: [col1, col2] = [r00, r10, r20, r01, r11, r21] (first two COLUMNS flattened)
#   - Reference: Zhou et al., "On the Continuity of Rotation Representations in Neural Networks"
#   - Matching: gr00t-H/gr00t/data/state_action/pose.py
#
# =============================================================================


def rot6d_to_rotation_matrix(rot6d: np.ndarray) -> np.ndarray:
    """
    Convert 6D rotation representation (COLUMN convention) to rotation matrix.

    This is the gr00t-H / CMR Versius standard format where the 6D representation
    consists of the first two COLUMNS of the rotation matrix, flattened.

    Reference: Zhou et al., "On the Continuity of Rotation Representations in Neural Networks"
    Matching: gr00t-H/gr00t/data/state_action/pose.py::rot6d_to_rotation_matrix

    Args:
        rot6d: 6D rotation vector of shape (6,), representing first two columns
               of rotation matrix flattened as [r00, r10, r20, r01, r11, r21]

    Returns:
        3x3 rotation matrix
    """
    # Reshape to (2, 3) then transpose to get columns as (3, 2)
    rot6d = np.asarray(rot6d).reshape(2, 3).T

    # First two columns of the rotation matrix
    col1 = rot6d[:, 0]
    col2 = rot6d[:, 1]

    # Normalize first column
    col1 = col1 / np.linalg.norm(col1)

    # Gram-Schmidt orthogonalization for second column
    col2 = col2 - np.dot(col1, col2) * col1
    col2 = col2 / np.linalg.norm(col2)

    # Third column is cross product (ensures right-handed coordinate system)
    col3 = np.cross(col1, col2)

    # Construct rotation matrix by stacking columns
    return np.column_stack([col1, col2, col3])


def rot6ds_to_rotation_matrices(rot6ds: np.ndarray) -> np.ndarray:
    """
    Batch convert 6D rotation representations (COLUMN convention) to rotation matrices.

    Matching: gr00t-H/gr00t/data/state_action/pose.py::rot6ds_to_rotation_matrices

    Args:
        rot6ds: 6D rotation vectors of shape (N, 6)

    Returns:
        Rotation matrices of shape (N, 3, 3)
    """
    N = rot6ds.shape[0]

    # Reshape to (N, 2, 3) then transpose to get columns as (N, 3, 2)
    cols = rot6ds.reshape(N, 2, 3).transpose(0, 2, 1)

    col1 = cols[:, :, 0]  # (N, 3)
    col2 = cols[:, :, 1]  # (N, 3)

    # Gram-Schmidt orthogonalization (vectorized)
    col1_norm = np.linalg.norm(col1, axis=1, keepdims=True)
    col1 = col1 / np.maximum(col1_norm, 1e-8)

    # col2 = col2 - (col1 · col2) * col1
    dot = np.sum(col1 * col2, axis=1, keepdims=True)
    col2 = col2 - dot * col1
    col2_norm = np.linalg.norm(col2, axis=1, keepdims=True)
    col2 = col2 / np.maximum(col2_norm, 1e-8)

    # col3 = col1 × col2
    col3 = np.cross(col1, col2)

    # Stack columns to form rotation matrices: (N, 3, 3)
    return np.stack([col1, col2, col3], axis=2)


def rotation_matrix_to_rot6d(rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to 6D rotation representation (COLUMN convention).

    Matching: gr00t-H/gr00t/data/state_action/pose.py::rotation_matrix_to_rot6d

    Args:
        rotation_matrix: 3x3 rotation matrix

    Returns:
        6D rotation vector of shape (6,), representing first two columns
        flattened as [r00, r10, r20, r01, r11, r21]
    """
    return rotation_matrix[:, :2].T.flatten()


def rotation_matrices_to_rot6d(rotation_matrices: np.ndarray) -> np.ndarray:
    """
    Batch convert rotation matrices to 6D rotation representations (COLUMN convention).

    Matching: gr00t-H/gr00t/data/state_action/pose.py::rotation_matrices_to_rot6d

    Args:
        rotation_matrices: Rotation matrices of shape (N, 3, 3)

    Returns:
        6D rotation vectors of shape (N, 6)
    """
    return rotation_matrices[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)


def quats_to_rotation_matrices(quats: np.ndarray, order: str = "xyzw") -> np.ndarray:
    """
    Batch convert quaternions to rotation matrices.

    Matching: gr00t-H/gr00t/data/state_action/pose.py::quats_to_rotation_matrices

    Args:
        quats: Quaternions of shape (N, 4)
        order: Quaternion ordering - "xyzw" (scipy default) or "wxyz" (scalar-first)

    Returns:
        Rotation matrices of shape (N, 3, 3)
    """
    if order.lower() == "wxyz":
        # Convert from wxyz to xyzw: [w, x, y, z] -> [x, y, z, w]
        quats = quats[:, [1, 2, 3, 0]]
    return Rotation.from_quat(quats).as_matrix()


def quat_to_rotation_matrix(quat: np.ndarray, order: str = "xyzw") -> np.ndarray:
    """
    Convert quaternion to 3x3 rotation matrix.

    Matching: gr00t-H/gr00t/data/state_action/pose.py::quat_to_rotation_matrix

    Args:
        quat: Quaternion array of shape (4,)
        order: Quaternion ordering - "xyzw" (scipy default) or "wxyz" (scalar-first)

    Returns:
        3x3 rotation matrix
    """
    quat = np.asarray(quat)
    if order.lower() == "wxyz":
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])
    return Rotation.from_quat(quat).as_matrix()


def convert_to_hybrid_relative(
    action_data: np.ndarray,
    eef_pose: np.ndarray,
    input_rotation_format: str = "quat",
    reference_rotation_format: str = "quat",
    input_quat_order: str = "xyzw",
    reference_quat_order: str = "xyzw",
) -> np.ndarray:
    """
    Convert absolute action data to hybrid-relative representation.

    Matches gr00t-H's ``convert_to_rel_xyz_rot6d`` (gr00t/data/state_action/pose.py).

    Hybrid-relative means:
    - Translation: relative to reference EEF position (delta from reference)
    - Rotation: relative to reference orientation, expressed in 6D format

    Args:
        action_data: Absolute action data of shape (H, D) where:
            - H is the action horizon
            - D = 3 (xyz) + 4 (quat) = 7 for quat format
            - D = 3 (xyz) + 3 (euler) = 6 for euler format
            - D = 3 (xyz) + 6 (rot6d) = 9 for rot6d format
            - Does NOT include gripper; handle gripper separately
        eef_pose: Reference end-effector pose:
            - Shape (7,) for quat format: xyz + quaternion
            - Shape (6,) for euler format: xyz + euler (RPY)
            - Shape (9,) for rot6d format: xyz + rot6d
        input_rotation_format: Format of rotation in action_data
            ("quat" | "rot6d" | "euler")
        reference_rotation_format: Format of rotation in eef_pose
            ("quat" | "rot6d" | "euler")
        input_quat_order: Quaternion ordering in action_data
            ("xyzw" | "wxyz"). Only used when input_rotation_format="quat".
        reference_quat_order: Quaternion ordering in eef_pose
            ("xyzw" | "wxyz"). Only used when reference_rotation_format="quat".

    Returns:
        Hybrid-relative actions of shape (H, 9) with xyz (relative) + rot6d (relative)
    """
    H = action_data.shape[0]

    # Extract reference position and rotation
    ref_xyz = eef_pose[:3]

    # Handle zero-norm reference quaternion: use identity rotation.
    # This can happen at episode boundaries where the reference state is zero-padded.
    if reference_rotation_format == "quat":
        ref_quat = eef_pose[3:7]
        if np.linalg.norm(ref_quat) < 1e-8:
            ref_R = np.eye(3, dtype=np.float32)
        else:
            ref_R = quat_to_rotation_matrix(ref_quat, order=reference_quat_order)
    elif reference_rotation_format == "rot6d":
        ref_rot6d = eef_pose[3:9]
        ref_R = rot6d_to_rotation_matrix(ref_rot6d)
    elif reference_rotation_format == "euler":
        ref_euler = eef_pose[3:6]
        ref_R = Rotation.from_euler("xyz", ref_euler).as_matrix()
    else:
        raise ValueError(f"Unknown reference_rotation_format: {reference_rotation_format}")

    result = np.zeros((H, 9), dtype=np.float32)  # Always output xyz + rot6d

    # Translation: vectorized subtraction
    result[:, :3] = action_data[:, :3] - ref_xyz

    # Rotation: batch convert to rotation matrices
    if input_rotation_format == "quat":
        action_quats = action_data[:, 3:7]  # (H, 4)

        # Handle zero-norm quaternions gracefully: replace with identity rotation.
        # These occur at episode boundaries (zero-padded rows) in some datasets
        # (e.g., UCSD surgical_learning_dataset). Rather than crashing scipy's
        # Rotation.from_quat(), we detect them and output zero relative delta
        # (= "no movement") for those timesteps.
        quat_norms = np.linalg.norm(action_quats, axis=-1)  # (H,)
        zero_mask = quat_norms < 1e-8  # (H,) True for invalid timesteps

        if np.any(zero_mask):
            # Replace zero-norm quats with a valid unit quaternion (identity)
            # so scipy doesn't crash. The output for these timesteps will be
            # overwritten with identity rotation below.
            safe_quats = action_quats.copy()
            if input_quat_order == "wxyz":
                safe_quats[zero_mask] = [1.0, 0.0, 0.0, 0.0]  # wxyz identity
            else:
                safe_quats[zero_mask] = [0.0, 0.0, 0.0, 1.0]  # xyzw identity
            action_Rs = quats_to_rotation_matrices(safe_quats, order=input_quat_order)
        else:
            action_Rs = quats_to_rotation_matrices(action_quats, order=input_quat_order)
    elif input_rotation_format == "rot6d":
        action_rot6ds = action_data[:, 3:9]  # (H, 6)
        action_Rs = rot6ds_to_rotation_matrices(action_rot6ds)  # (H, 3, 3)
        zero_mask = None
    elif input_rotation_format == "euler":
        action_eulers = action_data[:, 3:6]  # (H, 3) RPY
        action_Rs = np.array([Rotation.from_euler("xyz", e).as_matrix() for e in action_eulers])
        zero_mask = None
    else:
        raise ValueError(f"Unknown input_rotation_format: {input_rotation_format}")

    # Relative rotation: R_ref^T @ R_action for all H matrices
    relative_Rs = np.einsum("ij,hjk->hik", ref_R.T, action_Rs)

    # Convert back to rot6d (batch)
    result[:, 3:9] = rotation_matrices_to_rot6d(relative_Rs)

    # For zero-norm quaternion timesteps: set output to identity (zero delta).
    # xyz_rel = 0 (no translation), rot6d_rel = identity [1,0,0,0,1,0] (no rotation).
    if input_rotation_format == "quat" and zero_mask is not None and np.any(zero_mask):
        result[zero_mask, :3] = 0.0  # zero translation delta
        result[zero_mask, 3:9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # rot6d identity

    return result


class RotationTransform:
    """Adapted from https://github.com/real-stanford/diffusion_policy/blob/548a52bbb105518058e27bf34dcf90bf6f73681a/diffusion_policy/model/common/rotation_transformer.py"""

    valid_reps = ["axis_angle", "euler_angles", "quaternion", "rotation_6d", "matrix"]

    def __init__(self, from_rep="axis_angle", to_rep="rotation_6d"):
        """
        Valid representations

        Always use matrix as intermediate representation.
        """
        if from_rep.startswith("euler_angles"):
            from_convention = from_rep.split("_")[-1]
            from_rep = "euler_angles"
            from_convention = from_convention.replace("r", "X").replace("p", "Y").replace("y", "Z")
        else:
            from_convention = None
        if to_rep.startswith("euler_angles"):
            to_convention = to_rep.split("_")[-1]
            to_rep = "euler_angles"
            to_convention = to_convention.replace("r", "X").replace("p", "Y").replace("y", "Z")
        else:
            to_convention = None
        assert from_rep != to_rep, f"from_rep and to_rep cannot be the same: {from_rep}"
        assert from_rep in self.valid_reps, f"Invalid from_rep: {from_rep}"
        assert to_rep in self.valid_reps, f"Invalid to_rep: {to_rep}"

        forward_funcs = list()
        inverse_funcs = list()

        if from_rep != "matrix":
            funcs = [getattr(pt, f"{from_rep}_to_matrix"), getattr(pt, f"matrix_to_{from_rep}")]  # noqa: F821
            if from_convention is not None:
                funcs = [functools.partial(func, convention=from_convention) for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != "matrix":
            funcs = [getattr(pt, f"matrix_to_{to_rep}"), getattr(pt, f"{to_rep}_to_matrix")]  # noqa: F821
            if to_convention is not None:
                funcs = [functools.partial(func, convention=to_convention) for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        inverse_funcs = inverse_funcs[::-1]

        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(x: torch.Tensor, funcs: list) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        for func in funcs:
            x = func(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"
        return self._apply_funcs(x, self.forward_funcs)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"
        return self._apply_funcs(x, self.inverse_funcs)


class Normalizer:
    valid_modes = ["q99", "mean_std", "min_max", "binary"]

    def __init__(self, mode: str, statistics: dict):
        self.mode = mode
        self.statistics = statistics
        for key, value in self.statistics.items():
            self.statistics[key] = torch.tensor(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"

        # Normalize the tensor
        if self.mode == "q99":
            # Range of q99 is [-1, 1]
            q01 = self.statistics["q01"].to(x.dtype)
            q99 = self.statistics["q99"].to(x.dtype)

            # In the case of q01 == q99, the normalization will be undefined
            # So we set the normalized values to the original values
            mask = q01 != q99
            normalized = torch.zeros_like(x)

            # Normalize the values where q01 != q99
            # Formula: 2 * (x - q01) / (q99 - q01) - 1
            normalized[..., mask] = (x[..., mask] - q01[..., mask]) / (q99[..., mask] - q01[..., mask])
            normalized[..., mask] = 2 * normalized[..., mask] - 1

            # Set the normalized values to the original values where q01 == q99
            normalized[..., ~mask] = x[..., ~mask].to(x.dtype)

            # Clip the normalized values to be between -1 and 1
            normalized = torch.clamp(normalized, -1, 1)

        elif self.mode == "mean_std":
            # Range of mean_std is not fixed, but can be positive or negative
            mean = self.statistics["mean"].to(x.dtype)
            std = self.statistics["std"].to(x.dtype)

            # In the case of std == 0, the normalization will be undefined
            # So we set the normalized values to the original values
            mask = std != 0
            normalized = torch.zeros_like(x)

            # Normalize the values where std != 0
            # Formula: (x - mean) / std
            normalized[..., mask] = (x[..., mask] - mean[..., mask]) / std[..., mask]

            # Set the normalized values to the original values where std == 0
            normalized[..., ~mask] = x[..., ~mask].to(x.dtype)

        elif self.mode == "min_max":
            # Range of min_max is [-1, 1]
            min = self.statistics["min"].to(x.dtype)
            max = self.statistics["max"].to(x.dtype)

            # In the case of min == max, the normalization will be undefined
            # So we set the normalized values to the original values
            mask = min != max
            normalized = torch.zeros_like(x)

            # Normalize the values where min != max
            # Formula: 2 * (x - min) / (max - min) - 1
            normalized[..., mask] = (x[..., mask] - min[..., mask]) / (max[..., mask] - min[..., mask])
            normalized[..., mask] = 2 * normalized[..., mask] - 1

            # Set the normalized values to the original values where min == max
            # normalized[..., ~mask] = x[..., ~mask].to(x.dtype)
            # Set the normalized values to 0 where min == max
            normalized[..., ~mask] = 0

        elif self.mode == "scale":
            # Range of scale is [0, 1]
            min = self.statistics["min"].to(x.dtype)
            max = self.statistics["max"].to(x.dtype)
            abs_max = torch.max(torch.abs(min), torch.abs(max))
            mask = abs_max != 0
            normalized = torch.zeros_like(x)
            normalized[..., mask] = x[..., mask] / abs_max[..., mask]
            normalized[..., ~mask] = 0

        elif self.mode == "binary":
            # Range of binary is [0, 1]
            normalized = (x > 0.5).to(x.dtype)
        else:
            raise ValueError(f"Invalid normalization mode: {self.mode}")

        return normalized

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"
        if self.mode == "q99":
            q01 = self.statistics["q01"].to(x.dtype)
            q99 = self.statistics["q99"].to(x.dtype)
            return (x + 1) / 2 * (q99 - q01) + q01
        elif self.mode == "mean_std":
            mean = self.statistics["mean"].to(x.dtype)
            std = self.statistics["std"].to(x.dtype)
            return x * std + mean
        elif self.mode == "min_max":
            min = self.statistics["min"].to(x.dtype)
            max = self.statistics["max"].to(x.dtype)
            return (x + 1) / 2 * (max - min) + min
        elif self.mode == "binary":
            return (x > 0.5).to(x.dtype)
        else:
            raise ValueError(f"Invalid normalization mode: {self.mode}")


class StateActionToTensor(InvertibleModalityTransform):
    """
    Transforms states and actions to tensors.
    """

    input_dtypes: dict[str, np.dtype] = Field(default_factory=dict, description="The input dtypes for each state key.")
    output_dtypes: dict[str, torch.dtype] = Field(
        default_factory=dict, description="The output dtypes for each state key."
    )

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to"}
        else:
            include = kwargs.pop("include", None)

        return super().model_dump(*args, include=include, **kwargs)

    @field_validator("input_dtypes", "output_dtypes", mode="before")
    def validate_dtypes(cls, v):
        for key, dtype in v.items():
            if isinstance(dtype, str):
                if dtype.startswith("torch."):
                    dtype_split = dtype.split(".")[-1]
                    v[key] = getattr(torch, dtype_split)
                elif dtype.startswith("np.") or dtype.startswith("numpy."):
                    dtype_split = dtype.split(".")[-1]
                    v[key] = np.dtype(dtype_split)
                else:
                    raise ValueError(f"Invalid dtype: {dtype}")
        return v

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            assert isinstance(value, np.ndarray), f"Unexpected input type: {type(value)}. Expected type: {np.ndarray}"
            data[key] = torch.from_numpy(value)
            if key in self.output_dtypes:
                data[key] = data[key].to(self.output_dtypes[key])
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            assert isinstance(value, torch.Tensor), (
                f"Unexpected input type: {type(value)}. Expected type: {torch.Tensor}"
            )
            data[key] = value.numpy()
            if key in self.input_dtypes:
                data[key] = data[key].astype(self.input_dtypes[key])
        return data


class StateActionTransform(InvertibleModalityTransform):
    """
    Class for state or action transform.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
        normalization_modes (dict[str, str]): The normalization modes for each state key.
            If a state key in apply_to is not present in the dictionary, it will not be normalized.
        target_rotations (dict[str, str]): The target representations for each state key.
            If a state key in apply_to is not present in the dictionary, it will not be rotated.
    """

    # Configurable attributes
    apply_to: list[str] = Field(..., description="The keys in the modality to load and transform.")
    normalization_modes: dict[str, str] = Field(
        default_factory=dict, description="The normalization modes for each state key."
    )
    target_rotations: dict[str, str] = Field(
        default_factory=dict, description="The target representations for each state key."
    )
    normalization_statistics: dict[str, dict] = Field(
        default_factory=dict, description="The statistics for each state key."
    )
    modality_metadata: dict[str, StateActionMetadata] = Field(
        default_factory=dict, description="The modality metadata for each state key."
    )

    # Model variables
    _rotation_transformers: dict[str, RotationTransform] = PrivateAttr(default_factory=dict)
    _normalizers: dict[str, Normalizer] = PrivateAttr(default_factory=dict)
    _input_dtypes: dict[str, np.dtype | torch.dtype] = PrivateAttr(default_factory=dict)

    # Model constants
    _DEFAULT_MIN_MAX_STATISTICS: ClassVar[dict] = {
        "rotation_6d": {
            "min": [-1, -1, -1, -1, -1, -1],
            "max": [1, 1, 1, 1, 1, 1],
        },
        "euler_angles": {
            "min": [-np.pi, -np.pi, -np.pi],
            "max": [np.pi, np.pi, np.pi],
        },
        "quaternion": {
            "min": [-1, -1, -1, -1],
            "max": [1, 1, 1, 1],
        },
        "axis_angle": {
            "min": [-np.pi, -np.pi, -np.pi],
            "max": [np.pi, np.pi, np.pi],
        },
    }

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to", "normalization_modes", "target_rotations"}
        else:
            include = kwargs.pop("include", None)

        return super().model_dump(*args, include=include, **kwargs)

    @field_validator("modality_metadata", mode="before")
    def validate_modality_metadata(cls, v):
        for modality_key, config in v.items():
            if isinstance(config, dict):
                config = StateActionMetadata.model_validate(config)
            else:
                assert isinstance(config, StateActionMetadata), f"Invalid source rotation config: {config}"
            v[modality_key] = config
        return v

    @model_validator(mode="after")
    def validate_normalization_statistics(self):
        for modality_key, normalization_statistics in self.normalization_statistics.items():
            if modality_key in self.normalization_modes:
                normalization_mode = self.normalization_modes[modality_key]
                if normalization_mode == "min_max":
                    assert "min" in normalization_statistics and "max" in normalization_statistics, (
                        f"Min and max statistics are required for min_max normalization, but got {normalization_statistics}"
                    )
                    assert len(normalization_statistics["min"]) == len(normalization_statistics["max"]), (
                        f"Min and max statistics must have the same length, but got {normalization_statistics['min']} and {normalization_statistics['max']}"
                    )
                elif normalization_mode == "mean_std":
                    assert "mean" in normalization_statistics and "std" in normalization_statistics, (
                        f"Mean and std statistics are required for mean_std normalization, but got {normalization_statistics}"
                    )
                    assert len(normalization_statistics["mean"]) == len(normalization_statistics["std"]), (
                        f"Mean and std statistics must have the same length, but got {normalization_statistics['mean']} and {normalization_statistics['std']}"
                    )
                elif normalization_mode == "q99":
                    assert "q01" in normalization_statistics and "q99" in normalization_statistics, (
                        f"q01 and q99 statistics are required for q99 normalization, but got {normalization_statistics}"
                    )
                    assert len(normalization_statistics["q01"]) == len(normalization_statistics["q99"]), (
                        f"q01 and q99 statistics must have the same length, but got {normalization_statistics['q01']} and {normalization_statistics['q99']}"
                    )
                elif normalization_mode == "binary":
                    assert len(normalization_statistics) == 1, (
                        f"Binary normalization should only have one value, but got {normalization_statistics}"
                    )
                    assert normalization_statistics[0] in [
                        0,
                        1,
                    ], f"Binary normalization should only have 0 or 1, but got {normalization_statistics[0]}"
                else:
                    raise ValueError(f"Invalid normalization mode: {normalization_mode}")
        return self

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        dataset_statistics = dataset_metadata.statistics
        modality_metadata = dataset_metadata.modalities

        # Check that all state keys specified in apply_to have their modality_metadata
        for key in self.apply_to:
            split_key = key.split(".", 1)
            assert len(split_key) == 2, "State keys should have at least two parts: 'modality.key'"
            if key not in self.modality_metadata:
                modality, state_key = split_key
                assert hasattr(modality_metadata, modality), f"{modality} config not found"
                assert state_key in getattr(modality_metadata, modality), f"{state_key} config not found"
                self.modality_metadata[key] = getattr(modality_metadata, modality)[state_key]

        # Check that all state keys specified in normalization_modes have their statistics in state_statistics
        for key in self.normalization_modes:
            split_key = key.split(".", 1)
            assert len(split_key) == 2, "State keys should have at least two parts: 'modality.key'"
            modality, state_key = split_key
            assert hasattr(dataset_statistics, modality), f"{modality} statistics not found"
            assert state_key in getattr(dataset_statistics, modality), f"{state_key} statistics not found"
            assert len(getattr(modality_metadata, modality)[state_key].shape) == 1, (
                f"{getattr(modality_metadata, modality)[state_key].shape=}"
            )
            self.normalization_statistics[key] = getattr(dataset_statistics, modality)[state_key].model_dump()

        # Initialize the rotation transformers
        for key in self.target_rotations:
            # Get the original representation of the state
            from_rep = self.modality_metadata[key].rotation_type
            assert from_rep is not None, f"Source rotation type not found for {key}"

            # Get the target representation of the state, will raise an error if the target representation is not valid
            to_rep = RotationType(self.target_rotations[key])

            # If the original representation is not the same as the target representation, initialize the rotation transformer
            if from_rep != to_rep:
                self._rotation_transformers[key] = RotationTransform(from_rep=from_rep.value, to_rep=to_rep.value)

        # Initialize the normalizers
        for key in self.normalization_modes:
            modality, state_key = key.split(".", 1)
            # If the state has a nontrivial rotation, we need to handle it more carefully
            # For absolute rotations, we need to convert them to the target representation and normalize them using min_max mode,
            # since we can infer the bounds by the representation
            # For relative rotations, we cannot normalize them as we don't know the bounds
            if key in self._rotation_transformers:
                # Case 1: Absolute rotation
                if self.modality_metadata[key].absolute:
                    # Check that the normalization mode is valid
                    assert self.normalization_modes[key] == "min_max", (
                        "Absolute rotations that are converted to other formats must be normalized using `min_max` mode"
                    )
                    rotation_type = RotationType(self.target_rotations[key]).value
                    # If the target representation is euler angles, we need to parse the convention
                    if rotation_type.startswith("euler_angles"):
                        rotation_type = "euler_angles"
                    # Get the statistics for the target representation
                    statistics = self._DEFAULT_MIN_MAX_STATISTICS[rotation_type]
                # Case 2: Relative rotation
                else:
                    raise ValueError(
                        f"Cannot normalize relative rotations: {key} that's converted to {self.target_rotations[key]}"
                    )
            # If the state is not continuous, we should not use normalization modes other than binary
            elif not self.modality_metadata[key].continuous and self.normalization_modes[key] != "binary":
                raise ValueError(f"{key} is not continuous, so it should be normalized using `binary` mode")
            # Initialize the normalizer
            else:
                statistics = self.normalization_statistics[key]
            self._normalizers[key] = Normalizer(mode=self.normalization_modes[key], statistics=statistics)

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                # We allow some keys to be missing in the data, and only process the keys that are present
                continue
            if key not in self._input_dtypes:
                input_dtype = data[key].dtype
                assert isinstance(input_dtype, torch.dtype), (
                    f"Unexpected input dtype: {input_dtype}. Expected type: {torch.dtype}"
                )
                self._input_dtypes[key] = input_dtype
            else:
                assert data[key].dtype == self._input_dtypes[key], (
                    f"All states corresponding to the same key must be of the same dtype, input dtype: {data[key].dtype}, expected dtype: {self._input_dtypes[key]}"
                )
            # Rotate the state
            state = data[key]
            if key in self._rotation_transformers:
                state = self._rotation_transformers[key].forward(state)
            # Normalize the state
            if key in self._normalizers:
                state = self._normalizers[key].forward(state)
            data[key] = state
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            state = data[key]
            assert isinstance(state, torch.Tensor), (
                f"Unexpected state type: {type(state)}. Expected type: {torch.Tensor}"
            )
            # Unnormalize the state
            if key in self._normalizers:
                state = self._normalizers[key].inverse(state)
            # Change the state back to its original representation
            if key in self._rotation_transformers:
                state = self._rotation_transformers[key].inverse(state)
            assert isinstance(state, torch.Tensor), (
                f"State should be tensor after unapplying transformations, but got {type(state)}"
            )
            # Only convert back to the original dtype if it's known, i.e. `apply` was called before
            # If not, we don't know the original dtype, so we don't convert
            if key in self._input_dtypes:
                original_dtype = self._input_dtypes[key]
                if isinstance(original_dtype, np.dtype):
                    state = state.numpy().astype(original_dtype)
                elif isinstance(original_dtype, torch.dtype):
                    state = state.to(original_dtype)
                else:
                    raise ValueError(f"Invalid input dtype: {original_dtype}")
            data[key] = state
        return data


class StateActionPerturbation(ModalityTransform):
    """
    Class for state or action perturbation.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
        std (float): Standard deviation of the noise to be added to the state or action.
    """

    # Configurable attributes
    std: float = Field(..., description="Standard deviation of the noise to be added to the state or action.")

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.training:
            # Don't perturb the data in eval mode
            return data
        if self.std < 0:
            # If the std is negative, we don't add any noise
            return data
        for key in self.apply_to:
            state = data[key]
            assert isinstance(state, torch.Tensor)
            transformed_data_min = torch.min(state)
            transformed_data_max = torch.max(state)
            noise = torch.randn_like(state) * self.std
            state += noise
            # Clip to the original range
            state = torch.clamp(state, transformed_data_min, transformed_data_max)
            data[key] = state
        return data


class StateActionDropout(ModalityTransform):
    """
    Class for state or action dropout.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
        dropout_prob (float): Probability of dropping out a state or action.
    """

    # Configurable attributes
    dropout_prob: float = Field(..., description="Probability of dropping out a state or action.")

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.training:
            # Don't drop out the data in eval mode
            return data
        if self.dropout_prob < 0:
            # If the dropout probability is negative, we don't drop out any states
            return data
        if self.dropout_prob > 1e-9 and random.random() < self.dropout_prob:
            for key in self.apply_to:
                state = data[key]
                assert isinstance(state, torch.Tensor)
                state = torch.zeros_like(state)
                data[key] = state
        return data


class StateActionSinCosTransform(ModalityTransform):
    """
    Class for state or action sin-cos transform.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
    """

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            state = data[key]
            assert isinstance(state, torch.Tensor)
            sin_state = torch.sin(state)
            cos_state = torch.cos(state)
            data[key] = torch.cat([sin_state, cos_state], dim=-1)
        return data


# =============================================================================
# CMR Versius hybrid-relative action conversion with engagement-awareness
# =============================================================================
# These functions match the gr00t-H implementation for CMR Versius surgical robot.


def convert_to_hybrid_relative_with_engagement(
    action_data: np.ndarray,
    eef_pose: np.ndarray,
    engaged: np.ndarray,
    input_rotation_format: str = "rot6d",
    reference_rotation_format: str = "rot6d",
    ref_engaged: bool = True,
) -> np.ndarray:
    """
    Compute hybrid-relative actions with engagement-aware delta re-integration.

    Instead of computing: action[t] = pose[t] - pose[ref]
    This function computes: action[t] = sum(delta[i] * engaged[i] for i in ref+1..t)

    This correctly handles clutch scenarios in CMR Versius surgical robot data:
    - Reference disengaged → later engaged (no phantom jump from repositioning)
    - Mid-horizon clutch events (disengaged deltas zeroed)
    - Repositioning during clutch-out (not counted as arm motion)

    Args:
        action_data: Absolute action data of shape (T, D) where D = 3 (xyz) + 6 (rot6d) = 9
        eef_pose: Reference end-effector pose of shape (9,) for rot6d format: xyz + rot6d
        engaged: Boolean engagement mask of shape (T,)
        input_rotation_format: Format of rotation in action_data ("rot6d" or "quat")
        reference_rotation_format: Format of rotation in eef_pose ("rot6d" or "quat")
        ref_engaged: Whether the reference frame (t=0 state) is engaged

    Returns:
        Hybrid-relative actions of shape (T, 9) with xyz (relative) + rot6d (relative)
    """
    T = action_data.shape[0]
    result = np.zeros((T, 9), dtype=np.float32)

    # Parse reference pose
    ref_xyz = eef_pose[:3]
    if reference_rotation_format == "quat":
        ref_R = quat_to_rotation_matrix(eef_pose[3:7], order="xyzw")
    elif reference_rotation_format == "rot6d":
        ref_R = rot6d_to_rotation_matrix(eef_pose[3:9])
    else:
        raise ValueError(f"Unknown reference_rotation_format: {reference_rotation_format}")

    # Batch convert all action rotations to matrices
    if input_rotation_format == "quat":
        action_Rs = quats_to_rotation_matrices(action_data[:, 3:7], order="xyzw")
    elif input_rotation_format == "rot6d":
        action_Rs = rot6ds_to_rotation_matrices(action_data[:, 3:9])
    else:
        raise ValueError(f"Unknown input_rotation_format: {input_rotation_format}")

    # Build validity mask
    engaged_bool = engaged.astype(bool)
    prev_engaged = np.concatenate([[ref_engaged], engaged_bool[:-1]])
    delta_valid = prev_engaged & engaged_bool

    # Compute translation deltas and cumulative sum
    action_xyz = action_data[:, :3]
    all_xyz = np.vstack([ref_xyz[np.newaxis, :], action_xyz])
    delta_xyz = np.diff(all_xyz, axis=0)
    masked_delta_xyz = delta_xyz * delta_valid[:, np.newaxis]
    result[:, :3] = np.cumsum(masked_delta_xyz, axis=0)

    # Compute rotation deltas
    prev_Rs = np.concatenate([ref_R[np.newaxis, :, :], action_Rs[:-1]], axis=0)
    delta_Rs = np.einsum("tji,tjk->tik", prev_Rs, action_Rs)

    # For invalid deltas, set delta_R to identity
    identity = np.eye(3, dtype=np.float32)
    delta_Rs = np.where(delta_valid[:, np.newaxis, np.newaxis], delta_Rs, identity)

    # Cumulative rotation product
    cumulative_R = np.eye(3, dtype=np.float32)
    cumulative_Rs = np.zeros((T, 3, 3), dtype=np.float32)
    for t in range(T):
        cumulative_R = cumulative_R @ delta_Rs[t]
        cumulative_Rs[t] = cumulative_R

    # Convert cumulative rotations to rot6d
    result[:, 3:9] = rotation_matrices_to_rot6d(cumulative_Rs)

    return result


def scale_rot6d_by_angle(rot6d: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Scale a rot6d representation by scaling its axis-angle magnitude.

    Args:
        rot6d: 6D rotation representation of shape (6,) or (N, 6)
        scale_factor: Factor to multiply the rotation angle by

    Returns:
        Scaled rot6d representation with same shape as input
    """
    from scipy.spatial.transform import Rotation

    single_input = rot6d.ndim == 1
    if single_input:
        rot6d = rot6d[np.newaxis, :]

    rot_matrices = rot6ds_to_rotation_matrices(rot6d)
    rotations = Rotation.from_matrix(rot_matrices)
    rotvecs = rotations.as_rotvec()

    angle_magnitudes = np.linalg.norm(rotvecs, axis=-1, keepdims=True)
    epsilon = 1e-8

    scaled_rotvecs = np.where(angle_magnitudes > epsilon, rotvecs * scale_factor, rotvecs)

    scaled_rotations = Rotation.from_rotvec(scaled_rotvecs)
    scaled_matrices = scaled_rotations.as_matrix()
    scaled_rot6d = rotation_matrices_to_rot6d(scaled_matrices)

    if single_input:
        scaled_rot6d = scaled_rot6d[0]

    return scaled_rot6d


def apply_motion_scaling_to_hybrid_relative(
    hybrid_rel_data: np.ndarray,
    translation_scaling: float,
    rotation_scaling: float,
) -> np.ndarray:
    """
    Apply motion scaling normalization to hybrid-relative actions.

    This converts from "hand-controller-space" to "instrument-space" by
    multiplying by the scaling factors.

    Args:
        hybrid_rel_data: Hybrid-relative actions of shape (H, 9) - xyz_rel + rot6d_rel
        translation_scaling: Translation scaling factor (e.g., 0.333, 0.5, 1.0)
        rotation_scaling: Rotation scaling factor (e.g., 1.0, 1.5, 2.0)

    Returns:
        Motion-scaled hybrid-relative actions of shape (H, 9) in instrument-space
    """
    result = np.zeros_like(hybrid_rel_data)
    result[:, :3] = hybrid_rel_data[:, :3] * translation_scaling
    rot6d = hybrid_rel_data[:, 3:9]
    result[:, 3:9] = scale_rot6d_by_angle(rot6d, rotation_scaling)
    return result


# Identity rotation in 6D format (first two columns of identity matrix flattened)
ROT6D_IDENTITY = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)


class CMRVersiusRelativeActionTransform(ModalityTransform):
    """
    Transform for converting CMR Versius absolute actions to hybrid-relative representation.

    This transform handles the complex conditioning space of the CMR Versius surgical robot:

    ACTIONS (30D):
    - Left arm: xyz (3) + rot6d (6) + gripper (1) = 10D
    - Right arm: xyz (3) + rot6d (6) + gripper (1) = 10D
    - Energy buttons: left (1) + right (1) = 2D
    - Thumbstick X: left (1) + right (1) = 2D (endoscope/instrument control)
    - Thumbstick Y: left (1) + right (1) = 2D (endoscope/instrument control)
    - Thumbstick Button: left (1) + right (1) = 2D (instrument straighten function)
    - Clutch Button: left (1) + right (1) = 2D (engage/disengage arm control)

    STATE CONDITIONING (14D, sampled at action timesteps):
    - Haptic engaged: left (1) + right (1) = 2D (persistent engagement state)
    - Arm linked to haptic: left (1) + right (1) = 2D (which arm 0-3 is active)
    - Arm instrument type: arm_0 (1) + arm_1 (1) + arm_2 (1) + arm_3 (1) = 4D
    - Arm HUD color: arm_0 (1) + arm_1 (1) + arm_2 (1) + arm_3 (1) = 4D
    - Electrosurgery mode: left (1) + right (1) = 2D (CUT/COAG selection)

    - Total: 44D conditioning space (30D actions + 14D state)

    The transform performs:
    1. Hybrid-relative conversion for poses (DELTA: translation relative to current EEF,
       rotation relative to initial orientation)
    2. Sample-and-hold for gripper during clutch-out (ABSOLUTE with hold logic)
    3. Zeroing of energy buttons during clutch-out for safety (ABSOLUTE with zeroing logic)
    4. Pass-through for thumbstick, clutch button, and state conditioning (ABSOLUTE: no transformation)
    5. Optional motion scaling to convert hand-controller-space to instrument-space

    Note: Only pose keys are converted to deltas. All other values remain as ABSOLUTE:
    - Gripper/energy: have clutch-aware logic (sample-and-hold / zeroing)
    - Thumbstick, clutch button, state conditioning: pure pass-through

    Args:
        apply_to: List of action keys to transform
        pose_keys: Dict mapping pose key names to their state reference keys
        gripper_keys: List of gripper key names (absolute, sample-and-hold during clutch)
        energy_keys: List of energy button key names (absolute, zeroed during clutch)
        thumbstick_keys: List of thumbstick key names (absolute, pass-through)
        engaged_left_key: Key name for left arm engagement state
        engaged_right_key: Key name for right arm engagement state
        translation_scaling_key: Optional key for translation scaling factor
        rotation_scaling_key: Optional key for rotation scaling factor
        input_rotation_format: Format of rotation in action data ("rot6d" or "quat")
        reference_rotation_format: Format of rotation in state reference ("rot6d" or "quat")
    """

    apply_to: list[str] = Field(..., description="The action keys to transform")
    pose_keys: dict[str, str] = Field(default_factory=dict, description="Dict mapping pose key -> state reference key")
    gripper_keys: list[str] = Field(default_factory=list, description="Gripper keys (sample-and-hold during clutch)")
    energy_keys: list[str] = Field(default_factory=list, description="Energy button keys (zeroed during clutch)")
    thumbstick_keys: list[str] = Field(
        default_factory=list,
        description="Thumbstick, clutch button, and state conditioning keys (absolute values, pass-through without any transformation)",
    )
    engaged_left_key: str = Field(default="state.hapticengaged_left", description="Key for left arm engagement")
    engaged_right_key: str = Field(default="state.hapticengaged_right", description="Key for right arm engagement")
    translation_scaling_key: str | None = Field(default=None, description="Key for translation scaling factor")
    rotation_scaling_key: str | None = Field(default=None, description="Key for rotation scaling factor")
    input_rotation_format: str = Field(default="rot6d", description="Rotation format in action data")
    reference_rotation_format: str = Field(default="rot6d", description="Rotation format in state reference")
    action_passthrough_keys: list[str] = Field(
        default_factory=list,
        description="Action keys to remove after processing (e.g. engagement status used for clutch logic)",
    )
    state_passthrough_keys: list[str] = Field(
        default_factory=list, description="State keys to remove after processing (e.g. scaling factors)"
    )

    def _get_arm_from_key(self, key: str) -> str | None:
        """Determine which arm a key belongs to."""
        if "left" in key.lower():
            return "left"
        elif "right" in key.lower():
            return "right"
        return None

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply hybrid-relative conversion with engagement-aware processing."""
        # Get engagement data
        engaged_left = None
        engaged_right = None
        ref_engaged_left = True
        ref_engaged_right = True

        if self.engaged_left_key in data:
            engaged_val = data[self.engaged_left_key]
            if isinstance(engaged_val, torch.Tensor):
                engaged_val = engaged_val.numpy()
            engaged_left = engaged_val.flatten().astype(bool)
            # Reference engagement from first timestep of state (delta_indices=[0])
            ref_engaged_left = bool(engaged_val.flatten()[0])

        if self.engaged_right_key in data:
            engaged_val = data[self.engaged_right_key]
            if isinstance(engaged_val, torch.Tensor):
                engaged_val = engaged_val.numpy()
            engaged_right = engaged_val.flatten().astype(bool)
            ref_engaged_right = bool(engaged_val.flatten()[0])

        # Get motion scaling factors if configured
        trans_scale = 1.0
        rot_scale = 1.0
        if self.translation_scaling_key and self.translation_scaling_key in data:
            scale_val = data[self.translation_scaling_key]
            if isinstance(scale_val, torch.Tensor):
                scale_val = scale_val.numpy()
            trans_scale = float(scale_val.flatten()[0])
        if self.rotation_scaling_key and self.rotation_scaling_key in data:
            scale_val = data[self.rotation_scaling_key]
            if isinstance(scale_val, torch.Tensor):
                scale_val = scale_val.numpy()
            rot_scale = float(scale_val.flatten()[0])

        # Process each key
        for key in self.apply_to:
            if key not in data:
                continue

            action = data[key]
            is_tensor = isinstance(action, torch.Tensor)
            action_np = action.numpy() if is_tensor else action
            original_dtype = action.dtype if is_tensor else action.dtype

            arm = self._get_arm_from_key(key)
            engaged = engaged_left if arm == "left" else engaged_right if arm == "right" else None
            ref_engaged = ref_engaged_left if arm == "left" else ref_engaged_right if arm == "right" else True

            # Handle pose keys - convert to hybrid-relative
            if key in self.pose_keys:
                state_key = self.pose_keys[key]
                if state_key not in data:
                    raise KeyError(f"State reference key '{state_key}' not found for pose key '{key}'")

                eef_pose = data[state_key]
                if isinstance(eef_pose, torch.Tensor):
                    eef_pose = eef_pose.numpy()
                eef_pose = eef_pose.flatten()  # Flatten to 1D

                if engaged is not None:
                    action_np = convert_to_hybrid_relative_with_engagement(
                        action_data=action_np,
                        eef_pose=eef_pose,
                        engaged=engaged,
                        input_rotation_format=self.input_rotation_format,
                        reference_rotation_format=self.reference_rotation_format,
                        ref_engaged=ref_engaged,
                    )

                # Apply motion scaling
                if trans_scale != 1.0 or rot_scale != 1.0:
                    action_np = apply_motion_scaling_to_hybrid_relative(action_np, trans_scale, rot_scale)

            # Handle gripper keys - sample-and-hold during clutch
            elif key in self.gripper_keys:
                if engaged is not None:
                    T = action_np.shape[0]
                    for t in range(T):
                        if not engaged[t] and t > 0:
                            action_np[t] = action_np[t - 1]

            # Handle energy keys - zero during clutch for safety
            elif key in self.energy_keys:
                if engaged is not None:
                    T = action_np.shape[0]
                    for t in range(T):
                        if not engaged[t]:
                            action_np[t] = 0.0

            # Handle thumbstick, clutch button, and state conditioning keys - pass through without modification
            # These are ABSOLUTE values (not deltas like pose):
            # - Thumbsticks: control endoscope and instrument straighten function
            # - Clutch buttons: engage/disengage arm control
            # - State conditioning: system state for MLP conditioning (haptic_engaged, armlinkedtohaptic, instrtype, color)
            # All may be active independently of arm engagement status
            elif key in self.thumbstick_keys:
                pass  # No modification needed, absolute values pass through

            # Convert back to tensor if needed
            if is_tensor:
                data[key] = torch.from_numpy(action_np).to(original_dtype)
            else:
                data[key] = action_np

        # Remove passthrough keys that were only needed for processing
        for key in self.action_passthrough_keys:
            data.pop(key, None)
        for key in self.state_passthrough_keys:
            data.pop(key, None)

        return data


# =============================================================================
# Generic per-key action config dataclass
# =============================================================================
# Mirrors gr00t-H's ActionConfig (gr00t/data/types.py) but as a simple
# Pydantic model suitable for the Cosmos transform pipeline.
# =============================================================================


class ActionKeyConfig(BaseModel):
    """Configuration for a single action key's delta/normalization behaviour.

    Mirrors the relevant fields of gr00t-H's ``ActionConfig``.

    Attributes:
        rep: Action representation.
            - ``"rel_xyz_rot6d"``: EEF pose → hybrid-relative xyz + rot6d (9D output).
            - ``"relative"``: Joint-space subtraction from reference state.
            - ``"delta"``: Data is already a delta — pass through unchanged.
            - ``"absolute"``: No delta conversion (grippers, energy, etc.).
        state_key: State key used as reference for relative conversions.
            For ``rel_xyz_rot6d`` / ``relative`` this is required.  For ``absolute``
            and ``delta`` it is ignored.
        input_rotation_format: Rotation format in the raw action data
            (``"quat"`` | ``"euler"`` | ``"rot6d"``).  Only used when rep is
            ``rel_xyz_rot6d``.
        reference_rotation_format: Rotation format in the reference state
            (``"quat"`` | ``"euler"`` | ``"rot6d"``).  Only used when rep is
            ``rel_xyz_rot6d``.
        input_quat_order: Quaternion ordering in the action data
            (``"xyzw"`` | ``"wxyz"``).  Default ``"xyzw"``.
        reference_quat_order: Quaternion ordering in the reference state
            (``"xyzw"`` | ``"wxyz"``).  Default ``"xyzw"``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rep: str = Field(
        ...,
        description="rel_xyz_rot6d | relative | delta | absolute",
    )
    state_key: str | None = Field(
        default=None,
        description="State key for reference (required for rel_xyz_rot6d / relative)",
    )
    input_rotation_format: str = Field(
        default="quat",
        description="Rotation format in raw action data (quat | euler | rot6d)",
    )
    reference_rotation_format: str = Field(
        default="quat",
        description="Rotation format in reference state (quat | euler | rot6d)",
    )
    input_quat_order: str = Field(
        default="xyzw",
        description="Quaternion ordering in action data (xyzw | wxyz)",
    )
    reference_quat_order: str = Field(
        default="xyzw",
        description="Quaternion ordering in reference state (xyzw | wxyz)",
    )


class GenericRelativeActionTransform(ModalityTransform):
    """Generic per-key delta action transform for all Open-H embodiments.

    This is the Cosmos equivalent of gr00t-H's
    ``StateActionProcessor.apply_action``.  It converts raw absolute action
    data into the delta representation that the model is trained on, following
    the **exact same** logic per ``ActionKeyConfig``:

    * **rel_xyz_rot6d** – calls ``convert_to_rel_xyz_rot6d`` (already present
      in this module) to produce 9D relative xyz + rot6d.  The reference EEF
      pose is read from the corresponding ``state_key`` at ``delta_indices=[0]``.
      Used for dVRK (JHU, UCB, Obuda, …), Hamlyn, Stanford, etc.
    * **relative** – simple subtraction: ``action - reference_state`` (joint
      space).  Used for USTC Torin.
    * **delta** – pass-through; the data is already a delta (Moon).
    * **absolute** – pass-through; grippers, energy buttons, etc.

    The transform runs **before** ``StateActionTransform`` (normalization) and
    **before** ``ConcatTransform``.

    Example usage in a transform pipeline::

        GenericRelativeActionTransform(
            apply_to=["action.psm1_pose", "action.psm1_gripper",
                       "action.psm2_pose", "action.psm2_gripper"],
            action_key_configs={
                "action.psm1_pose": ActionKeyConfig(
                    rep="rel_xyz_rot6d",
                    state_key="state.psm1_pose",
                    input_rotation_format="quat",
                    reference_rotation_format="quat",
                    input_quat_order="xyzw",
                    reference_quat_order="xyzw",
                ),
                "action.psm1_gripper": ActionKeyConfig(rep="absolute"),
                "action.psm2_pose": ActionKeyConfig(
                    rep="rel_xyz_rot6d",
                    state_key="state.psm2_pose",
                    ...
                ),
                "action.psm2_gripper": ActionKeyConfig(rep="absolute"),
            },
        )
    """

    action_key_configs: dict[str, ActionKeyConfig] = Field(
        ...,
        description="Per-key action config (keyed by the full 'action.xxx' key name)",
    )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue

            cfg = self.action_key_configs.get(key)
            if cfg is None:
                continue  # no config → pass through

            action_np = data[key]
            is_tensor = isinstance(action_np, torch.Tensor)
            if is_tensor:
                action_np = action_np.numpy()

            # ----------------------------------------------------------
            # REL_XYZ_ROT6D: EEF pose → relative xyz + rot6d (9D)
            # ----------------------------------------------------------
            if cfg.rep == "rel_xyz_rot6d":
                if cfg.state_key is None:
                    raise ValueError(f"state_key is required for rel_xyz_rot6d action key '{key}'")
                if cfg.state_key not in data:
                    raise KeyError(f"Reference state key '{cfg.state_key}' not found for action key '{key}'")

                eef_pose_raw = data[cfg.state_key]
                if isinstance(eef_pose_raw, torch.Tensor):
                    eef_pose_raw = eef_pose_raw.numpy()
                eef_pose = eef_pose_raw.flatten()  # (D,) – single reference

                # Note: zero-norm quaternions (e.g., episode boundary padding) are
                # handled gracefully inside convert_to_hybrid_relative() — they
                # produce identity rotation (= "no movement") instead of crashing.

                action_np = convert_to_hybrid_relative(
                    action_data=action_np,
                    eef_pose=eef_pose,
                    input_rotation_format=cfg.input_rotation_format,
                    reference_rotation_format=cfg.reference_rotation_format,
                    input_quat_order=cfg.input_quat_order,
                    reference_quat_order=cfg.reference_quat_order,
                )

            # ----------------------------------------------------------
            # RELATIVE (joint-space): action = action - reference_state
            # ----------------------------------------------------------
            elif cfg.rep == "relative":
                if cfg.state_key is None:
                    raise ValueError(f"state_key is required for relative action key '{key}'")
                if cfg.state_key not in data:
                    raise KeyError(f"Reference state key '{cfg.state_key}' not found for action key '{key}'")

                ref_state = data[cfg.state_key]
                if isinstance(ref_state, torch.Tensor):
                    ref_state = ref_state.numpy()
                # Reference is the last (or only) timestep of the state
                ref_state = ref_state[-1] if ref_state.ndim == 2 else ref_state.flatten()
                action_np = action_np - ref_state

            # ----------------------------------------------------------
            # DELTA / ABSOLUTE: pass-through (no conversion)
            # ----------------------------------------------------------
            elif cfg.rep in ("delta", "absolute"):
                pass  # nothing to do

            else:
                raise ValueError(f"Unknown action rep '{cfg.rep}' for key '{key}'")

            # Write back
            if is_tensor:
                data[key] = torch.from_numpy(action_np).to(data[key].dtype)
            else:
                data[key] = action_np

        return data


# =============================================================================
# SutureBot / dVRK Relative Action Transform
# =============================================================================
# Handles the pre-concatenated 20D dual-arm dVRK action format used by the
# SutureBot dataset:
#   Per arm: [xyz(3), rot6d(6), gripper(1)] = 10D
#   Dual-arm: arm1(10D) + arm2(10D) = 20D
#
# Computes global translation deltas and local rotation deltas (rot6d) relative
# to the base pose at t=0. Based on Stanford UMI's relative action formulation.
# =============================================================================


def _rotation_6d_to_matrix(rot6d):
    """Convert 6D rotation (first two rows of rotation matrix) to full 3x3 matrix.

    Uses Gram-Schmidt orthonormalization on the input rows.

    Args:
        rot6d: Array of shape (..., 6) containing [row1(3), row2(3)].

    Returns:
        Rotation matrices of shape (..., 3, 3).
    """
    shape = rot6d.shape[:-1]
    rot6d = rot6d.reshape(*shape, 2, 3)

    row1 = rot6d[..., 0, :]
    row1 = row1 / (np.linalg.norm(row1, axis=-1, keepdims=True) + 1e-8)

    row2 = rot6d[..., 1, :]
    row2 = row2 - np.sum(row1 * row2, axis=-1, keepdims=True) * row1
    row2 = row2 / (np.linalg.norm(row2, axis=-1, keepdims=True) + 1e-8)

    row3 = np.cross(row1, row2)

    return np.stack([row1, row2, row3], axis=-2)


def _compute_rel_actions_dvrk(actions):
    """Compute relative actions for a dual-arm dVRK robot.

    Global translation delta, local (tooltip frame) rotation delta in 6D format.
    actions[0] is the base pose, actions[1:] are targets.

    Input/output per-arm: [xyz(3), rot6d(6), gripper(1)] = 10D
    Dual-arm: [T, 20] → [T-1, 20]

    The relative rotation R_rel = R_base^T @ R_target is represented in 6D
    (first two rows of the rotation matrix, flattened).
    """
    if isinstance(actions, torch.Tensor):
        actions = actions.numpy()

    base = actions[0]
    targets = actions[1:]
    n_targets = targets.shape[0]
    rel_actions = np.zeros((n_targets, 20))

    for arm in range(2):
        i = arm * 10
        R_base = _rotation_6d_to_matrix(base[i + 3 : i + 9])
        R_tgt = _rotation_6d_to_matrix(targets[:, i + 3 : i + 9])

        rel_actions[:, i : i + 3] = targets[:, i : i + 3] - base[i : i + 3]
        R_rel = R_base.T @ R_tgt
        rel_actions[:, i + 3 : i + 9] = R_rel[:, :2, :].reshape(n_targets, 6)
        rel_actions[:, i + 9] = targets[:, i + 9]

    return rel_actions


class RelativeActionTransform(ModalityTransform):
    """Convert absolute actions to relative actions for dVRK/SutureBot datasets.

    Works on the pre-concatenated 20D dual-arm action vector (``action.action``).
    Input: [T, 20] absolute (xyz + rot6d + gripper per arm).
    Output: [T-1, 20] relative (delta_xyz + delta_rot6d + gripper per arm).
    """

    apply_to: list[str] = Field(..., description="Action keys to transform.")

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            actions = data[key]
            is_tensor = isinstance(actions, torch.Tensor)
            actions_np = actions.numpy() if is_tensor else actions
            rel_actions = _compute_rel_actions_dvrk(actions_np)
            data[key] = torch.from_numpy(rel_actions).to(actions.dtype) if is_tensor else rel_actions
        return data
