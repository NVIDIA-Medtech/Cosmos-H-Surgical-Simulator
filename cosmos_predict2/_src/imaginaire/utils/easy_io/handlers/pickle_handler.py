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

"""Compatibility entry point: legacy pickle data must be migrated offline."""

from cosmos_predict2._src.imaginaire.utils.easy_io.handlers.base import BaseFileHandler


class PickleHandler(BaseFileHandler):
    str_like = False

    def load_from_fileobj(self, file, **kwargs):
        raise ValueError("Pickle data is disabled. Regenerate or migrate trusted artifacts to .cdata offline.")

    def load_from_path(self, filepath, **kwargs):
        return self.load_from_fileobj(None, **kwargs)

    def dump_to_path(self, obj, filepath, **kwargs):
        return self.dump_to_fileobj(obj, None, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        raise ValueError("Pickle output is disabled. Use .cdata for numeric arrays and data containers.")

    def dump_to_str(self, obj, **kwargs):
        raise ValueError("Pickle output is disabled. Use .cdata.")
