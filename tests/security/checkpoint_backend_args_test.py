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

"""Remote checkpoints must name their backend rather than inherit one.

The s3 backend cannot be built with no arguments: exactly one of ``profile``
or ``s3_credential_path`` is required. Reading an ``s3://`` checkpoint without
backend arguments therefore only works when some earlier, unrelated call has
registered a default backend in the process.
"""

import pytest

from cosmos_predict2._src.predict2.utils.model_loader import S3_TRAINING_CREDENTIAL_PATH, _backend_args_for


@pytest.mark.parametrize(
    "uri",
    [
        "/lustre/checkpoints/iter_000012000/model_ema_bf16.safetensors",
        "/tmp/model_ema_bf16.pt",
        "relative/path/model_ema_bf16.safetensors",
    ],
)
def test_local_paths_need_no_backend(uri):
    assert _backend_args_for(uri) is None


@pytest.mark.parametrize(
    "uri",
    [
        "s3://bucket/checkpoints/iter_000012000/model_ema_bf16.safetensors",
        "s3://bucket/model_ema_bf16.pt",
    ],
)
def test_s3_uris_name_their_credential(uri):
    assert _backend_args_for(uri) == {
        "backend": "s3",
        "s3_credential_path": S3_TRAINING_CREDENTIAL_PATH,
    }


def test_s3_backend_cannot_be_built_without_arguments():
    """Guards the premise: without arguments the backend raises, so the
    helper's explicit credential is load-bearing rather than decorative."""
    from cosmos_predict2._src.imaginaire.utils.easy_io.backends.registry_utils import prefix_to_backends

    with pytest.raises(ValueError, match="profile or s3_credential_path"):
        prefix_to_backends["s3"]()
