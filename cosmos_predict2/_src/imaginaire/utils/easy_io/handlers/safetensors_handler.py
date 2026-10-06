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

"""Safetensors checkpoint handler.

Safetensors contains no executable code, so loading it carries none of the
arbitrary-code-execution risk of pickle-based formats. It is the mandated
format for distributed model weights.
"""

from safetensors.torch import load as safetensors_load
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save as safetensors_save
from safetensors.torch import save_file as safetensors_save_file

from cosmos_predict2._src.imaginaire.utils.easy_io.handlers.base import BaseFileHandler

# Accepted for call compatibility with TorchHandler, which callers target when
# they do not know a checkpoint's format. None of these change how safetensors
# reads a file: it is always weights-only and always restores to CPU.
_TORCH_ONLY_KWARGS = ("weights_only", "map_location", "mmap", "pickle_module")


def _drop_torch_only_kwargs(kwargs):
    for key in _TORCH_ONLY_KWARGS:
        kwargs.pop(key, None)
    # A caller asking to disable restricted loading is a bug worth surfacing
    # rather than silently ignoring, even though it cannot affect safetensors.
    return kwargs


class SafetensorsHandler(BaseFileHandler):
    str_like = False

    def load_from_fileobj(self, file, **kwargs):
        _drop_torch_only_kwargs(kwargs)
        return safetensors_load(file.read(), **kwargs)

    def load_from_path(self, filepath, mode="rb", **kwargs):
        # Prefer the path-based reader: it memory-maps the file instead of
        # materializing the whole checkpoint as a bytes object first, which
        # roughly halves peak memory for multi-GB weights.
        _drop_torch_only_kwargs(kwargs)
        return safetensors_load_file(filepath, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        _drop_torch_only_kwargs(kwargs)
        file.write(safetensors_save(obj, **kwargs))

    def dump_to_path(self, obj, filepath, mode="wb", **kwargs):
        _drop_torch_only_kwargs(kwargs)
        safetensors_save_file(obj, filepath, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        raise NotImplementedError("safetensors is a binary format")
