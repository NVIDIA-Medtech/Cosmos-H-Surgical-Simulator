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

"""Run via torchrun in the nightly image, from a directory outside the source tree."""

import os
import sys
from pathlib import Path

import torch

import cosmos_predict2

assert os.getuid() != 0
assert sys.prefix == "/opt/cosmos-venv", sys.executable
assert cosmos_predict2.__file__ is not None
assert Path(cosmos_predict2.__file__).is_relative_to("/workspace")
assert torch.ones(1).item() == 1
print(f"PASS: non-root worker imported editable project with {sys.executable}")
if os.environ.get("COSMOS_PROBE_CUDA") == "1":
    assert torch.ones(1, device="cuda").sum().item() == 1
    print(f"PASS: non-root CUDA worker on {torch.cuda.get_device_name()}")
