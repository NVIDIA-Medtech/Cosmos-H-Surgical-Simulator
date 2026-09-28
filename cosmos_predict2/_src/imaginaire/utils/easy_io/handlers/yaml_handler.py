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

from cosmos_predict2._src.imaginaire.utils.easy_io.handlers.base import BaseFileHandler
from cosmos_predict2._src.imaginaire.utils.safe_yaml import dump_yaml, load_yaml


class YamlHandler(BaseFileHandler):
    def load_from_fileobj(self, file, **kwargs):
        if kwargs:
            raise ValueError("YAML loader overrides are not supported")
        return load_yaml(file)

    def dump_to_fileobj(self, obj, file, **kwargs):
        file.write(dump_yaml(obj, **kwargs))

    def dump_to_str(self, obj, **kwargs):
        return dump_yaml(obj, **kwargs)
