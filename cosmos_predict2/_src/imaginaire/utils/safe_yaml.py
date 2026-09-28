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

"""Bounded, data-only YAML parsing shared by configuration and storage readers."""

import io

import yaml

MAX_YAML_BYTES = 4 * 1024 * 1024
MAX_YAML_DEPTH = 64
MAX_YAML_NODES = 100_000


def load_yaml(stream):
    """Reject object constructors, aliases, and unbounded documents before loading."""
    text = stream.read(MAX_YAML_BYTES + 1) if hasattr(stream, "read") else stream
    if not isinstance(text, (str, bytes)):
        raise TypeError("YAML input must be text, bytes, or a readable stream")
    size = len(text.encode("utf-8")) if isinstance(text, str) else len(text)
    if size > MAX_YAML_BYTES:
        raise ValueError("YAML document exceeds the size limit")
    depth = nodes = 0
    for event in yaml.parse(text, Loader=yaml.SafeLoader):
        nodes += 1
        if nodes > MAX_YAML_NODES:
            raise ValueError("YAML document exceeds the node limit")
        if isinstance(event, yaml.AliasEvent):
            raise ValueError("YAML aliases are not supported; expand them before loading")
        if isinstance(event, (yaml.MappingStartEvent, yaml.SequenceStartEvent)):
            depth += 1
            if depth > MAX_YAML_DEPTH:
                raise ValueError("YAML document exceeds the nesting limit")
        elif isinstance(event, (yaml.MappingEndEvent, yaml.SequenceEndEvent)):
            depth -= 1
    return yaml.safe_load(text)


class _DataDumper(yaml.SafeDumper):
    """Expand shared acyclic containers, with limits before graph expansion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._active = set()
        self._nodes = 0

    def ignore_aliases(self, data):
        return True

    def represent_data(self, data):
        self._nodes += 1
        if self._nodes > MAX_YAML_NODES or len(self._active) > MAX_YAML_DEPTH:
            raise ValueError("YAML output exceeds the node or nesting limit")
        identity = id(data)
        if identity in self._active:
            raise ValueError("Cyclic YAML data is not supported")
        self._active.add(identity)
        try:
            return super().represent_data(data)
        finally:
            self._active.remove(identity)


class _BoundedOutput(io.StringIO):
    size = 0

    def write(self, text):
        self.size += len(text.encode("utf-8"))
        if self.size > MAX_YAML_BYTES:
            raise ValueError("YAML output exceeds the size limit")
        return super().write(text)


def dump_yaml(value, **kwargs):
    """Return data-only text accepted by load_yaml, or fail before publishing it."""
    if "Dumper" in kwargs or kwargs.get("encoding") is not None:
        raise ValueError("YAML dumper/encoding overrides are not supported")
    output = _BoundedOutput()
    yaml.dump(value, stream=output, Dumper=_DataDumper, **kwargs)
    text = output.getvalue()
    load_yaml(text)  # Enforce the reader's exact event/depth limits too.
    return text
