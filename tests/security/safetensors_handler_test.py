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

import io

import pytest
import torch

from cosmos_predict2._src.imaginaire.utils.easy_io.handlers.registry_utils import file_handlers
from cosmos_predict2._src.imaginaire.utils.easy_io.handlers.safetensors_handler import SafetensorsHandler

# Module-level so the reduction is picklable by reference. It runs only when a
# reader unpickles the payload, which is exactly what must not happen.
_EXECUTED = []


def _mark_executed():
    _EXECUTED.append(True)
    return 0


class _Exploit:
    def __reduce__(self):
        return (_mark_executed, ())


def _state_dict():
    """Mirrors the dtypes and shapes a released checkpoint actually contains."""
    return {
        "net.blocks.0.weight": torch.randn(4, 8, dtype=torch.bfloat16),
        "net.accum_iteration": torch.tensor(12000, dtype=torch.int64),  # 0-dim scalar
        "net.blocks.0.self_attn.q_norm._extra_state": torch.zeros(5, dtype=torch.uint8),
    }


def _assert_identical(left, right):
    assert set(left) == set(right)
    for key, value in left.items():
        other = right[key]
        assert other.dtype == value.dtype, key
        assert other.shape == value.shape, key
        # Compare raw bytes so bfloat16 and 0-dim tensors are covered exactly.
        assert torch.equal(other.reshape(-1).view(torch.uint8), value.reshape(-1).view(torch.uint8)), key


def test_handler_is_registered():
    assert isinstance(file_handlers["safetensors"], SafetensorsHandler)


def test_path_roundtrip(tmp_path):
    handler = SafetensorsHandler()
    value = _state_dict()
    path = tmp_path / "weights.safetensors"
    handler.dump_to_path(value, str(path))
    _assert_identical(value, handler.load_from_path(str(path)))


def test_fileobj_roundtrip_matches_path(tmp_path):
    handler = SafetensorsHandler()
    value = _state_dict()
    path = tmp_path / "weights.safetensors"
    handler.dump_to_path(value, str(path))

    with open(path, "rb") as handle:
        from_fileobj = handler.load_from_fileobj(handle)
    _assert_identical(handler.load_from_path(str(path)), from_fileobj)


@pytest.mark.parametrize("kwargs", [{"weights_only": True}, {"map_location": "cpu"}, {"mmap": True}])
def test_torch_only_kwargs_are_accepted(tmp_path, kwargs):
    """Callers that do not know the format pass torch.load kwargs; they must not raise."""
    handler = SafetensorsHandler()
    value = _state_dict()
    path = tmp_path / "weights.safetensors"
    handler.dump_to_path(value, str(path))
    _assert_identical(value, handler.load_from_path(str(path), **kwargs))


def test_dump_to_str_is_rejected():
    with pytest.raises(NotImplementedError):
        SafetensorsHandler().dump_to_str(_state_dict())


def test_pickle_payload_is_not_executable():
    """A safetensors reader must never execute an embedded pickle opcode."""
    payload = io.BytesIO()
    torch.save({"x": _Exploit()}, payload)
    payload.seek(0)

    _EXECUTED.clear()
    # The bytes are a valid torch pickle but not a valid safetensors container,
    # so the handler rejects them rather than running the reduction.
    with pytest.raises(Exception):
        SafetensorsHandler().load_from_fileobj(payload)
    assert _EXECUTED == [], "safetensors handler executed a pickle reduction"
