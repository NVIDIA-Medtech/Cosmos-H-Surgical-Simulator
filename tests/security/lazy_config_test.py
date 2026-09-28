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

import pytest
import yaml

from cosmos_predict2._src.imaginaire.lazy_config.lazy import LazyConfig


def test_yaml_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  frames: 13\n  name: demo\n")
    assert LazyConfig.load(str(path)).model.frames == 13
    assert LazyConfig.load(str(path), keys="model").name == "demo"


@pytest.mark.parametrize("text", ["!!python/object/apply:builtins.str [unexpected]", "[]", "1: value", "a: &x [*x]"])
def test_invalid_config_is_rejected(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    with pytest.raises((ValueError, yaml.YAMLError)):
        LazyConfig.load(str(path))
