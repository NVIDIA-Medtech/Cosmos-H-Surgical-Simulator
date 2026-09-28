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

import pickle

import numpy as np
import pytest

from cosmos_predict2._src.imaginaire.datasets.decoders.pkl_loader import pkl_decoder as legacy_decoder
from cosmos_predict2._src.imaginaire.datasets.webdataset.decoders.data import data_decoder
from cosmos_predict2._src.imaginaire.datasets.webdataset.decoders.pickle import pkl_decoder
from cosmos_predict2._src.imaginaire.utils import safe_data


@pytest.mark.parametrize("decoder", [data_decoder, pkl_decoder, legacy_decoder])
def test_numeric_embeddings_roundtrip(decoder):
    sample = [{"caption": np.ones((3, 4), dtype=np.float16)}]
    result = decoder("sample.cdata", safe_data.dumps(sample))
    np.testing.assert_array_equal(result[0]["caption"], sample[0]["caption"])
    assert decoder("sample.mp4", b"unrelated") is None


@pytest.mark.parametrize("decoder", [data_decoder, pkl_decoder, legacy_decoder])
@pytest.mark.parametrize("suffix", ["pkl", "pickle", "pyd", "PKL"])
def test_pickle_rejected_before_deserialization(decoder, suffix, monkeypatch):
    monkeypatch.setattr(pickle, "loads", lambda *args: pytest.fail("pickle must never be called"))
    with pytest.raises(ValueError, match="Pickle dataset entries"):
        decoder("sample." + suffix, b"untrusted pickle bytes")


@pytest.mark.parametrize("suffix", ["pkl", "pickle", "pyd", "pkl.gz", "pth"])
def test_webdataset_pipeline_has_no_executable_fallback(suffix):
    import gzip
    import io

    import torch
    import webdataset as wds

    from cosmos_predict2._src.imaginaire.datasets.webdataset.decoders.data import decoding_error_handler

    # Built-in .pyd/.pth handlers must not provide an alternate unsafe path.
    class UnexpectedObject:
        def __reduce__(self):
            return str, ("executed",)

    if suffix == "pth":
        stream = io.BytesIO()
        torch.save(UnexpectedObject(), stream)
        payload = stream.getvalue()
    else:
        payload = pickle.dumps(UnexpectedObject())
        if suffix.endswith(".gz"):
            payload = gzip.compress(payload)
    decoder = wds.autodecode.Decoder([data_decoder])
    with pytest.raises(wds.autodecode.DecodingError) as caught:
        decoder({"__key__": "test", "embedding." + suffix: payload})
    if suffix != "pth":
        with pytest.raises(wds.autodecode.DecodingError):
            decoding_error_handler(caught.value)


def test_webdataset_pipeline_numeric_data():
    import io

    import torch
    import webdataset as wds

    sample = [{"caption": np.ones((3, 4), dtype=np.float16)}]
    stream = io.BytesIO()
    torch.save({"tensor": torch.ones(2)}, stream)
    decoder = wds.autodecode.Decoder([data_decoder])
    result = decoder(
        {"__key__": "sample", "embedding.cdata": safe_data.dumps(sample), "weights.pth": stream.getvalue()}
    )
    np.testing.assert_array_equal(result["embedding.cdata"][0]["caption"], sample[0]["caption"])
    assert torch.equal(result["weights.pth"]["tensor"], torch.ones(2))
