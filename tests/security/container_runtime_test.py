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

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("uid,gid", [("0", "10001"), ("10001", "0"), ("-1", "1"), ("", "1")])
def test_reject_privileged_or_invalid_ids(uid, gid):
    result = subprocess.run(["bash", str(ROOT / "bin/setup-container-user.sh"), uid, gid], capture_output=True)
    assert result.returncode != 0


@pytest.mark.parametrize("entrypoint,installer", [("bin/entrypoint.sh", "uv"), ("docker/nightly-entrypoint.sh", "pip")])
def test_failed_install_stops_entrypoint(tmp_path, entrypoint, installer):
    stub = tmp_path / installer
    stub.write_text("#!/bin/sh\nexit 42\n")
    stub.chmod(0o700)
    marker = tmp_path / "executed"
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}", CUDA_NAME="cu128")
    result = subprocess.run([str(ROOT / entrypoint), "touch", str(marker)], env=env)
    assert result.returncode == 42
    assert not marker.exists()
