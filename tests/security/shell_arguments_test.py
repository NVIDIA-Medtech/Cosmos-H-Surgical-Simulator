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

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("kind", ["nemo", "groot"])
@pytest.mark.parametrize("value", ["ordinary", "two words", "--option", "x; echo BAD", "$(echo BAD)", "'quoted'"])
def test_training_arguments_are_passed_as_data(kind, value, monkeypatch):
    filename = "post_training_nemo_assets.py" if kind == "nemo" else "post_training_groot.py"
    spec = importlib.util.spec_from_file_location("training_example", ROOT / "examples/posttraining" / kind / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = next(v for v in vars(module).values() if isinstance(v, type) and hasattr(v, "run_training"))
    obj = cls.__new__(cls)
    obj.setup_environment_variables = lambda: None
    obj.experiment_name = obj.job_name = value
    obj.max_iters = obj.checkpoint_save_iter = 1
    obj.pipeline_state = {}
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs)))
    obj.run_training()
    args, kwargs = calls[0]
    assert isinstance(args, list)
    assert kwargs["shell"] is False
    assert args[0] == "torchrun"
    assert f"experiment={value}" in args
    if kind == "groot":
        assert f"job.name={value}" in args
