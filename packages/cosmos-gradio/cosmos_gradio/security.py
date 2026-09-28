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

"""Fail-closed authentication and a single trusted-operator authorization policy."""

import hmac
import os
from dataclasses import dataclass, field
from pathlib import Path

import gradio as gr

from cosmos_gradio.file_access import file_serving_options

ACTIONS = frozenset({"generate", "upload", "list", "view", "logs"})


@dataclass(frozen=True)
class AccessPolicy:
    username: str
    password: str = field(repr=False)

    def __post_init__(self):
        if not self.username.strip() or len(self.password) < 16:
            raise ValueError("Set COSMOS_GRADIO_USERNAME and a password of at least 16 characters")

    @classmethod
    def from_environment(cls, environ=None):
        environ = os.environ if environ is None else environ
        password = environ.get("COSMOS_GRADIO_PASSWORD", "")
        password_file = environ.get("COSMOS_GRADIO_PASSWORD_FILE")
        if password_file:
            if password:
                raise ValueError("Configure only one Gradio password source")
            password = Path(password_file).read_text().rstrip("\r\n")
        return cls(environ.get("COSMOS_GRADIO_USERNAME", ""), password)

    def authenticate(self, username, password):
        if not isinstance(username, str) or not isinstance(password, str):
            return False
        user_ok = hmac.compare_digest(username.encode(), self.username.encode())
        password_ok = hmac.compare_digest(password.encode(), self.password.encode())
        return user_ok and password_ok

    def authorize(self, request, action):
        username = getattr(request, "username", None)
        if action not in ACTIONS or username != self.username:
            raise PermissionError("Operator authorization required")


def protected(callback, policy, action, *, takes_input=True):
    """Check the server-authenticated principal for every application action."""
    if action not in ACTIONS:
        raise ValueError("Unknown operator action")
    if takes_input:

        def guarded(value, request: gr.Request):
            policy.authorize(request, action)
            return callback(value)
    else:

        def guarded(request: gr.Request):
            policy.authorize(request, action)
            return callback()

    return guarded


def launch_options(policy):
    """Require TLS for non-loopback listeners; local proxies may use loopback."""
    host = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    cert = os.environ.get("GRADIO_SSL_CERTFILE")
    key = os.environ.get("GRADIO_SSL_KEYFILE")
    if bool(cert) != bool(key):
        raise ValueError("Configure both GRADIO_SSL_CERTFILE and GRADIO_SSL_KEYFILE")
    if host not in ("127.0.0.1", "::1", "localhost") and not (cert and key):
        raise ValueError("Non-loopback Gradio listeners require TLS; otherwise use a local TLS proxy or SSH tunnel")
    if cert and not all(Path(p).is_file() for p in (cert, key)):
        raise ValueError("Gradio TLS certificate/key file is missing")
    return dict(
        auth=policy.authenticate,
        server_name=host,
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "8080")),
        ssl_certfile=cert,
        ssl_keyfile=key,
        share=False,
        debug=False,
        show_error=False,
        max_file_size="500MB",
        **file_serving_options(),
    )
