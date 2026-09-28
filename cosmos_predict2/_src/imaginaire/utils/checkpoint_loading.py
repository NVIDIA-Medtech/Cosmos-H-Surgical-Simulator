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

"""Restricted loading for tensor checkpoints; never retry with executable pickle."""

import copy

import numpy as np
import torch
from packaging.version import Version


def load_weights(file, **kwargs):
    if Version(torch.__version__).release[:2] < (2, 6):
        raise RuntimeError("Restricted checkpoints require PyTorch >= 2.6; install a supported CUDA extra")
    if kwargs.pop("weights_only", True) is not True or kwargs.get("pickle_module") is not None:
        raise ValueError("Unrestricted checkpoint loading is disabled; convert trusted legacy artifacts offline")
    return torch.load(file, weights_only=True, **kwargs)


def normalize_checkpoint_state(value):
    """Normalize producer-owned NumPy scalars without enabling new load globals.

    Preserve containers (including state_dict metadata), tensors and array types.
    This operates on in-memory producer state, never on untrusted serialized data.
    """
    if isinstance(value, np.generic):
        scalar = value.item()
        if isinstance(scalar, np.generic):
            raise TypeError("Checkpoint scalar has no lossless built-in representation")
        return scalar
    if isinstance(value, dict):
        result = copy.copy(value)
        for key, item in value.items():
            result[key] = normalize_checkpoint_state(item)
        if hasattr(value, "_metadata"):
            result._metadata = normalize_checkpoint_state(value._metadata)
        return result
    if isinstance(value, list):
        return [normalize_checkpoint_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_checkpoint_state(item) for item in value)
    return value


def save_weights(value, file, **kwargs):
    """Write producer state in the restricted reader's supported scalar format."""
    torch.save(normalize_checkpoint_state(value), file, **kwargs)
