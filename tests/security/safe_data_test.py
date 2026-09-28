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

import gzip
import io
import json
import pickle
import zipfile

import numpy as np
import pytest
import torch

from cosmos_predict2._src.imaginaire.utils import safe_data
from cosmos_predict2._src.imaginaire.utils.easy_io.handlers.data_handler import DataHandler
from cosmos_predict2._src.imaginaire.utils.easy_io.handlers.gzip_handler import GzipHandler
from cosmos_predict2._src.imaginaire.utils.easy_io.handlers.pickle_handler import PickleHandler


class UnexpectedObject:
    def __reduce__(self):
        return str, ("unexpected",)


def test_arrays_tensors_and_metadata_roundtrip():
    value = {
        "prompt": "sample",
        "windows": [{"embedding": np.ones((3, 4), dtype=np.float16)}],
        "tensor": torch.arange(4),
        "bf16": torch.ones(2, dtype=torch.bfloat16),
        "metadata": (True, None, 3),
    }
    restored = safe_data.loads(safe_data.dumps(value))
    assert restored["prompt"] == "sample"
    np.testing.assert_array_equal(restored["windows"][0]["embedding"], value["windows"][0]["embedding"])
    assert torch.equal(restored["tensor"], value["tensor"])
    assert restored["bf16"].dtype == torch.bfloat16
    assert restored["metadata"] == value["metadata"]


def test_handlers_roundtrip():
    value = {"array": np.arange(5), "prompt": "example"}
    for handler in [DataHandler(), GzipHandler()]:
        restored = handler.load_from_fileobj(io.BytesIO(handler.dump_to_str(value)))
        np.testing.assert_array_equal(restored["array"], value["array"])


def test_legacy_pickle_and_gzip_rejected():
    payload = pickle.dumps(UnexpectedObject())
    with pytest.raises(ValueError, match="Pickle data is disabled"):
        PickleHandler().load_from_fileobj(io.BytesIO(payload))
    with pytest.raises(zipfile.BadZipFile):
        GzipHandler().load_from_fileobj(io.BytesIO(gzip.compress(payload)))


@pytest.mark.parametrize("value", [UnexpectedObject(), np.array([object()], dtype=object), {3: "nonstring key"}])
def test_unknown_objects_cannot_be_written(value):
    with pytest.raises(ValueError):
        safe_data.dumps(value)


def archive_with_array(array):
    stream = io.BytesIO()
    data = io.BytesIO()
    np.save(data, array, allow_pickle=True)
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(
            "metadata.json", json.dumps({"version": 1, "data": ["array", {"name": "array_0", "tensor_dtype": None}]})
        )
        archive.writestr("array_0.npy", data.getvalue())
    return stream.getvalue()


def test_object_array_cannot_be_loaded():
    with pytest.raises(ValueError, match="numeric"):
        safe_data.loads(archive_with_array(np.array([UnexpectedObject()], dtype=object)))


def test_size_and_nesting_limits(monkeypatch):
    payload = safe_data.dumps(np.zeros(10000))
    monkeypatch.setattr(safe_data, "MAX_BYTES", 1024)
    with pytest.raises(ValueError):
        safe_data.loads(payload)
    nested = None
    for _ in range(70):
        nested = [nested]
    with pytest.raises(ValueError):
        safe_data.dumps(nested)


def test_malicious_shape_rejected_before_allocation():
    array = io.BytesIO()
    np.lib.format.write_array_header_1_0(array, {"descr": "<f8", "fortran_order": False, "shape": (10**12,)})
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(
            "metadata.json", json.dumps({"version": 1, "data": ["array", {"name": "array_0", "tensor_dtype": None}]})
        )
        z.writestr("array_0.npy", array.getvalue())
    with pytest.raises(ValueError, match="shape"):
        safe_data.loads(archive.getvalue())


def test_file_handlers(tmp_path):
    for handler, suffix in [(DataHandler(), "cdata"), (GzipHandler(), "gz")]:
        path = tmp_path / ("sample." + suffix)
        handler.dump_to_path({"value": [1, 2]}, path)
        assert handler.load_from_path(path) == {"value": [1, 2]}


def test_writer_accounts_for_expanded_headers_and_metadata(monkeypatch):
    monkeypatch.setattr(safe_data, "MAX_BYTES", 2048)
    with pytest.raises(ValueError, match="Expanded"):
        safe_data.dumps(np.zeros(2000, dtype=np.uint8))


def test_numpy_writer_rejects_object_arrays_and_pickle_override():
    from cosmos_predict2._src.imaginaire.utils.easy_io.handlers.np_handler import NumpyHandler

    with pytest.raises(ValueError):
        NumpyHandler().dump_to_str(np.array([UnexpectedObject()], dtype=object))
    with pytest.raises(ValueError):
        NumpyHandler().dump_to_str(np.zeros(2), allow_pickle=True)
