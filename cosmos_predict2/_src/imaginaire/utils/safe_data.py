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

"""Versioned JSON + numeric NPY archive, without Python object deserialization."""

import io
import json
import math
import zipfile

import numpy as np

MAX_BYTES = 256 * 1024 * 1024
MAX_METADATA_BYTES = 1024 * 1024
MAX_NODES = 100_000
MAX_DEPTH = 64
MAX_ARRAYS = 10_000


def dumps(value):
    arrays = {}
    nodes = 0
    total_bytes = 0

    def encode(obj, depth=0):
        nonlocal nodes, total_bytes
        nodes += 1
        if depth > MAX_DEPTH or nodes > MAX_NODES:
            raise ValueError("Data exceeds nesting or node limit")
        if obj is None or type(obj) in (bool, int, str, float):
            return ["scalar", obj]
        if type(obj) in (list, tuple):
            return ["tuple" if type(obj) is tuple else "list", [encode(x, depth + 1) for x in obj]]
        if type(obj) is dict:
            if not all(type(key) is str for key in obj):
                raise ValueError("Data dictionary keys must be strings")
            return ["dict", {key: encode(val, depth + 1) for key, val in obj.items()}]
        tensor_dtype = None
        if type(obj).__module__ == "torch" and type(obj).__name__ == "Tensor":
            import torch

            tensor_dtype = str(obj.dtype)
            obj = obj.detach().cpu()
            obj = obj.view(torch.uint16).numpy() if obj.dtype == torch.bfloat16 else obj.numpy()
        if type(obj) is np.ndarray and obj.dtype.kind in "biufc" and not obj.dtype.hasobject:
            total_bytes += obj.nbytes
            if total_bytes > MAX_BYTES or len(arrays) >= MAX_ARRAYS:
                raise ValueError("Data exceeds array size/count limit")
            name = f"array_{len(arrays)}"
            arrays[name] = obj
            return ["array", {"name": name, "tensor_dtype": tensor_dtype}]
        raise ValueError(f"Unsupported data type: {type(obj).__name__}")

    metadata = json.dumps({"version": 1, "data": encode(value)}, allow_nan=False).encode()
    if len(metadata) > MAX_METADATA_BYTES:
        raise ValueError("Data metadata exceeds size limit")
    with io.BytesIO() as stream:
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("metadata.json", metadata)
            for name, array in arrays.items():
                with io.BytesIO() as data:
                    np.save(data, array, allow_pickle=False)
                    archive.writestr(name + ".npy", data.getvalue())
            if sum(entry.file_size for entry in archive.infolist()) > MAX_BYTES:
                raise ValueError("Expanded data exceeds size limit")
        payload = stream.getvalue()
    if len(payload) > MAX_BYTES:
        raise ValueError("Data archive exceeds size limit")
    return payload


def loads(payload):
    if len(payload) > MAX_BYTES:
        raise ValueError("Data archive exceeds size limit")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(entries) > MAX_ARRAYS + 1 or len(names) != len(set(names)):
            raise ValueError("Invalid archive entries")
        if sum(entry.file_size for entry in entries) > MAX_BYTES:
            raise ValueError("Expanded data exceeds size limit")
        if "metadata.json" not in names or archive.getinfo("metadata.json").file_size > MAX_METADATA_BYTES:
            raise ValueError("Missing or oversized data metadata")
        try:
            document = json.loads(archive.read("metadata.json"))
        except (RecursionError, UnicodeError) as exc:
            raise ValueError("Invalid data metadata") from exc
        if (
            not isinstance(document, dict)
            or type(document.get("version")) is not int
            or document.get("version") != 1
            or set(document) != {"version", "data"}
        ):
            raise ValueError("Unsupported data schema")
        used = {"metadata.json"}
        nodes = 0

        def decode(node, depth=0):
            nonlocal nodes
            nodes += 1
            if depth > MAX_DEPTH or nodes > MAX_NODES:
                raise ValueError("Data exceeds nesting or node limit")
            if not isinstance(node, list) or len(node) != 2:
                raise ValueError("Invalid data node")
            kind, value = node
            if kind == "scalar" and (value is None or type(value) in (bool, int, str, float)):
                if type(value) is float and not math.isfinite(value):
                    raise ValueError("Nonfinite scalar")
                return value
            if kind in ("list", "tuple") and isinstance(value, list):
                values = [decode(x, depth + 1) for x in value]
                return tuple(values) if kind == "tuple" else values
            if kind == "dict" and isinstance(value, dict):
                return {key: decode(val, depth + 1) for key, val in value.items()}
            if kind == "array" and isinstance(value, dict) and set(value) == {"name", "tensor_dtype"}:
                name = value["name"]
                if not isinstance(name, str) or not name.startswith("array_") or not name[6:].isdigit():
                    raise ValueError("Invalid array name")
                filename = name + ".npy"
                if filename in used:
                    raise ValueError("Duplicate array reference")
                used.add(filename)
                with io.BytesIO(archive.read(filename)) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise ValueError("Unsupported array format")
                    if dtype.hasobject or dtype.kind not in "biufc" or len(shape) > MAX_DEPTH:
                        raise ValueError("Only numeric arrays are supported")
                    if any(type(dim) is not int or dim < 0 for dim in shape):
                        raise ValueError("Invalid array shape")
                    if math.prod(shape) * dtype.itemsize != len(stream.getbuffer()) - stream.tell():
                        raise ValueError("Array shape does not match payload size")
                    stream.seek(0)
                    result = np.load(stream, allow_pickle=False)
                tensor_dtype = value["tensor_dtype"]
                if tensor_dtype is not None:
                    import torch

                    result = torch.from_numpy(result)
                    if tensor_dtype == "torch.bfloat16" and result.dtype == torch.uint16:
                        result = result.view(torch.bfloat16)
                    if str(result.dtype) != tensor_dtype:
                        raise ValueError("Tensor dtype does not match stored array")
                return result
            raise ValueError("Unsupported data node")

        result = decode(document["data"])
        if used != set(names):
            raise ValueError("Unexpected archive members")
        return result


def load(stream):
    return loads(stream.read(MAX_BYTES + 1))


def dump(value, stream):
    stream.write(dumps(value))
