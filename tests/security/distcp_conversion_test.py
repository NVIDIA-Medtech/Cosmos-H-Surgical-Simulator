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
import torch

from cosmos_predict2._src.imaginaire.utils.checkpoint_loading import load_weights
from scripts import convert_distcp_to_pt as converter


def test_distributed_conversion_requires_explicit_trust_before_io(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / ".metadata").write_bytes(b"untrusted input must never be read")
    destination = tmp_path / "destination"
    destination.mkdir()
    existing = destination / "model.pt"
    existing.write_bytes(b"existing output must survive refusal")
    args = converter.Args(input_dir=str(source), output_dir=destination, ema=False)
    monkeypatch.setattr(converter.tyro, "cli", lambda *a, **kw: args)
    monkeypatch.setattr(converter, "dcp_to_torch_save", lambda *a: pytest.fail("untrusted metadata reached converter"))
    with pytest.raises(ValueError, match="--trust-checkpoint"):
        converter.main()
    assert existing.read_bytes() == b"existing output must survive refusal"


def test_trusted_distributed_tensor_conversion(tmp_path, monkeypatch):
    import torch.distributed.checkpoint as dcp

    source = tmp_path / "source"
    value = {"net.weight": torch.arange(4)}
    dcp.save(value, checkpoint_id=source)
    args = converter.Args(input_dir=str(source), output_dir=tmp_path / "output", ema=False, trust_checkpoint=True)
    monkeypatch.setattr(converter.tyro, "cli", lambda *a, **kw: args)
    converter.main()
    restored = load_weights(args.output_dir / "model.pt")
    assert torch.equal(restored["net.weight"], value["net.weight"])
