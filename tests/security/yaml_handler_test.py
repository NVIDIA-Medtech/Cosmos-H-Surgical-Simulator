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
import yaml

from cosmos_predict2._src.imaginaire.utils.easy_io.handlers.yaml_handler import YamlHandler
from cosmos_predict2._src.imaginaire.utils.safe_yaml import load_yaml


def test_data_roundtrip():
    handler = YamlHandler()
    value = {"model": {"frames": 13, "enabled": True}, "values": [1, 2, None]}
    assert handler.load_from_fileobj(io.StringIO(handler.dump_to_str(value))) == value


@pytest.mark.parametrize(
    "document",
    [
        "!!python/object/apply:builtins.str [unexpected]",
        "!!python/name:os.system",
        "a: &loop [*loop]",
        "[" * 66 + "0" + "]" * 66,
        "value: [",
    ],
)
def test_rejects_unsafe_or_malformed_yaml(document):
    with pytest.raises((ValueError, yaml.YAMLError)):
        load_yaml(document)


def test_size_limit_and_no_loader_override():
    with pytest.raises(ValueError):
        load_yaml("x" * (4 * 1024 * 1024 + 1))
    with pytest.raises(ValueError):
        YamlHandler().load_from_fileobj(io.StringIO("a: 1"), Loader=yaml.UnsafeLoader)


def test_shared_containers_roundtrip():
    shared = [{"frames": [1, 2, 3]}]
    value = {"train": shared, "validation": shared}
    handler = YamlHandler()
    text = handler.dump_to_str(value)
    assert "&id" not in text and "*id" not in text
    assert handler.load_from_fileobj(io.StringIO(text)) == value
    output = io.StringIO()
    handler.dump_to_fileobj(value, output)
    assert load_yaml(output.getvalue()) == value


def test_writer_rejects_cycles_and_bounded_expansion(monkeypatch):
    cyclic = []
    cyclic.append(cyclic)
    output = io.StringIO()
    with pytest.raises(ValueError, match="Cyclic"):
        YamlHandler().dump_to_fileobj(cyclic, output)
    assert output.getvalue() == ""
    monkeypatch.setattr("cosmos_predict2._src.imaginaire.utils.safe_yaml.MAX_YAML_NODES", 20)
    value = [0]
    for _ in range(10):
        value = [value, value]
    with pytest.raises(ValueError, match="limit"):
        YamlHandler().dump_to_str(value)


def test_writer_enforces_bytes_depth_and_safe_dumper(monkeypatch):
    with pytest.raises(ValueError, match="overrides"):
        YamlHandler().dump_to_str({}, Dumper=yaml.Dumper)
    nested = [0]
    for _ in range(66):
        nested = [nested]
    with pytest.raises(ValueError, match="limit"):
        YamlHandler().dump_to_str(nested)
    monkeypatch.setattr("cosmos_predict2._src.imaginaire.utils.safe_yaml.MAX_YAML_BYTES", 12)
    with pytest.raises(ValueError, match="size limit"):
        YamlHandler().dump_to_str("x" * 13)
