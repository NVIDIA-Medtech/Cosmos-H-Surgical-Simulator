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

from cosmos_predict2._src.imaginaire.utils import safe_data
from cosmos_predict2._src.imaginaire.utils.checkpoint_loading import load_weights


class UnsafeDatasetFormatError(ValueError):
    """A legacy executable format must terminate the data pipeline."""


def data_decoder(key: str, data: bytes):
    extension = key.rsplit(".", 1)[-1].lower()
    if extension in ("pkl", "pickle", "pyd"):
        raise UnsafeDatasetFormatError(
            "Pickle dataset entries are disabled. Regenerate or migrate trusted shards to .cdata."
        )
    if extension == "pth":
        return load_weights(io.BytesIO(data), map_location="cpu")
    if extension == "cdata":
        return safe_data.loads(data)
    return None


def decoding_error_handler(error):
    """Do not skip legacy-only shards forever; preserve other corrupt-sample handling."""
    from webdataset.handlers import warn_and_continue

    cause = error
    while cause is not None:
        if isinstance(cause, UnsafeDatasetFormatError):
            raise error
        cause = cause.__cause__
    return warn_and_continue(error)
