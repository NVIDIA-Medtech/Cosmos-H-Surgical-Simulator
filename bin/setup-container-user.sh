#!/usr/bin/env bash
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


# Build-time only: provision the unprivileged application account and writable paths.
set -eu

app_uid=${1-10001}
app_gid=${2-10001}
case "$app_uid:$app_gid" in
    *[!0-9:]*|:*|*:) echo "Application UID/GID must be numeric" >&2; exit 1 ;;
esac
if [ "$app_uid" -eq 0 ] || [ "$app_gid" -eq 0 ]; then
    echo "Application UID/GID must be nonzero" >&2
    exit 1
fi
# A host-matching ID may already belong to a base-image account.
groupadd --non-unique --gid "$app_gid" cosmos
useradd --non-unique --uid "$app_uid" --gid cosmos --create-home --shell /bin/bash cosmos
install -d -o cosmos -g cosmos /workspace /home/cosmos/.cache /home/cosmos/.local/bin
chown -R cosmos:cosmos /workspace /home/cosmos
