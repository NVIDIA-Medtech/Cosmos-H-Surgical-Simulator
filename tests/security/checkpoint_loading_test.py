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
import pickle

import pytest
import torch

from cosmos_predict2._src.imaginaire.utils.checkpoint_loading import load_weights
from cosmos_predict2._src.imaginaire.utils.easy_io.handlers.torch_handler import TorchHandler


class UnexpectedObject:
    def __reduce__(self):
        return str, ("unexpected object",)


def test_model_and_optimizer_roundtrip():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters())
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    value = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": 1}
    stream = io.BytesIO()
    torch.save(value, stream, pickle_protocol=2)
    stream.seek(0)
    restored = TorchHandler().load_from_fileobj(stream, map_location="cpu")
    model.load_state_dict(restored["model"])
    optimizer.load_state_dict(restored["optimizer"])
    assert restored["iteration"] == 1
    assert torch.equal(restored["model"]["weight"], value["model"]["weight"])


def test_custom_object_rejected_without_execution():
    stream = io.BytesIO()
    torch.save(UnexpectedObject(), stream)
    stream.seek(0)
    with pytest.raises(pickle.UnpicklingError):
        load_weights(stream)


@pytest.mark.parametrize("kwargs", [{"weights_only": False}, {"pickle_module": pickle}])
def test_no_unsafe_override(kwargs):
    with pytest.raises(ValueError):
        TorchHandler().load_from_fileobj(io.BytesIO(), **kwargs)


def test_old_torch_rejected(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.5.1")
    with pytest.raises(RuntimeError):
        load_weights(io.BytesIO())


def _broadcast_worker(rank, rendezvous):
    from datetime import timedelta

    import numpy as np
    import torch.distributed as dist

    from cosmos_predict2._src.imaginaire.checkpointer.safe_broadcast import broadcast_object

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    try:
        expected = {"model": {"weight": torch.arange(4)}, "iteration": 17, "scheduler": {"_last_lr": [np.float64(0.1)]}}
        result = broadcast_object(expected if rank == 0 else None, src_rank=0)
        assert result["iteration"] == 17
        assert torch.equal(result["model"]["weight"], expected["model"]["weight"])
        assert type(result["scheduler"]["_last_lr"][0]) is float
    finally:
        dist.destroy_process_group()


def test_checkpoint_broadcast_between_two_cpu_ranks(tmp_path):
    torch.multiprocessing.spawn(_broadcast_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


def test_actual_scheduler_checkpoint_resume():
    import hydra

    from cosmos_predict2._src.imaginaire.configs.lr_scheduler import LambdaLinearSchedulerConfig

    def setup():
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        schedule = hydra.utils.instantiate(LambdaLinearSchedulerConfig)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[schedule.schedule])
        return model, optimizer, scheduler

    model, optimizer, scheduler = setup()
    for _ in range(3):
        optimizer.zero_grad()
        model(torch.ones(1, 3)).sum().backward()
        optimizer.step()
        scheduler.step()
    stream = io.BytesIO()
    # Direct torch.save verifies the scheduler itself produces compatible state.
    torch.save(
        {"model": model.state_dict(), "optim": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, stream
    )
    stream.seek(0)
    restored = load_weights(stream)
    restored_model, restored_optimizer, restored_scheduler = setup()
    restored_model.load_state_dict(restored["model"])
    restored_optimizer.load_state_dict(restored["optim"])
    restored_scheduler.load_state_dict(restored["scheduler"])
    for current_model, current_optimizer, current_scheduler in (
        (model, optimizer, scheduler),
        (restored_model, restored_optimizer, restored_scheduler),
    ):
        current_optimizer.zero_grad()
        current_model(torch.ones(1, 3)).sum().backward()
        current_optimizer.step()
        current_scheduler.step()
    assert torch.equal(model.weight, restored_model.weight)
    assert scheduler.get_last_lr() == restored_scheduler.get_last_lr()
    assert type(restored_optimizer.param_groups[0]["lr"]) is float


def test_writer_normalizes_numpy_scalars_and_preserves_metadata():
    from collections import OrderedDict

    import numpy as np

    value = OrderedDict(weight=torch.ones(2), state={"lr": np.float64(0.1), "step": np.int64(3)})
    value._metadata = {"": {"version": 1}}
    stream = io.BytesIO()
    TorchHandler().dump_to_fileobj(value, stream)
    stream.seek(0)
    restored = load_weights(stream)
    assert type(restored["state"]["lr"]) is float
    assert type(restored["state"]["step"]) is int
    assert restored._metadata == value._metadata
    assert isinstance(value["state"]["lr"], np.float64)  # no producer mutation
