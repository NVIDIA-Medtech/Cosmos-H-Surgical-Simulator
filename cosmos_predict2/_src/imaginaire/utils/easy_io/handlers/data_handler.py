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

from cosmos_predict2._src.imaginaire.utils import safe_data
from cosmos_predict2._src.imaginaire.utils.easy_io.handlers.base import BaseFileHandler


class DataHandler(BaseFileHandler):
    str_like = False

    def load_from_path(self, filepath, **kwargs):
        with open(filepath, "rb") as file:
            return self.load_from_fileobj(file, **kwargs)

    def dump_to_path(self, obj, filepath, **kwargs):
        with open(filepath, "wb") as file:
            self.dump_to_fileobj(obj, file, **kwargs)

    def load_from_fileobj(self, file, **kwargs):
        if kwargs:
            raise ValueError("Data loader overrides are not supported")
        return safe_data.load(file)

    def dump_to_fileobj(self, obj, file, **kwargs):
        if kwargs:
            raise ValueError("Data writer overrides are not supported")
        safe_data.dump(obj, file)

    def dump_to_str(self, obj, **kwargs):
        if kwargs:
            raise ValueError("Data writer overrides are not supported")
        return safe_data.dumps(obj)
