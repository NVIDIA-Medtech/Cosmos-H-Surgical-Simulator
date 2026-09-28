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

import secrets
from types import SimpleNamespace

import gradio as gr
import pytest
from cosmos_gradio.security import AccessPolicy, launch_options, protected
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def policy():
    return AccessPolicy("operator", secrets.token_urlsafe(24))


def test_credentials_required(monkeypatch):
    for key in ("COSMOS_GRADIO_USERNAME", "COSMOS_GRADIO_PASSWORD", "COSMOS_GRADIO_PASSWORD_FILE"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError):
        AccessPolicy.from_environment()


def test_private_listener_and_remote_tls(monkeypatch, policy):
    for key in ("GRADIO_SERVER_NAME", "GRADIO_SSL_CERTFILE", "GRADIO_SSL_KEYFILE"):
        monkeypatch.delenv(key, raising=False)
    opts = launch_options(policy)
    assert opts["server_name"] == "127.0.0.1"
    assert opts["share"] is False and opts["debug"] is False
    assert opts["auth"]("operator", policy.password)
    assert not opts["auth"]("operator", "incorrect")
    monkeypatch.setenv("GRADIO_SERVER_NAME", "0.0.0.0")
    with pytest.raises(ValueError, match="TLS"):
        launch_options(policy)


@pytest.mark.parametrize("action", ["generate", "upload", "list", "view", "logs"])
def test_each_action_checks_principal(policy, action):
    guarded = protected(lambda value: value, policy, action)
    for request in (None, SimpleNamespace(username=None), SimpleNamespace(username="someone_else")):
        with pytest.raises(PermissionError):
            guarded("data", request)
    assert guarded("data", SimpleNamespace(username="operator")) == "data"


def test_gradio_routes_enforce_login(policy, tmp_path):
    asset = tmp_path / "sample.txt"
    asset.write_text("private fixture")
    with gr.Blocks() as blocks:
        output = gr.Textbox()
        gr.Button().click(
            protected(lambda: "authorized", policy, "generate", takes_input=False),
            outputs=output,
            api_name="probe",
            queue=False,
        )
    app = gr.mount_gradio_app(FastAPI(), blocks, path="/", auth=policy.authenticate, allowed_paths=[str(tmp_path)])
    with TestClient(app) as client:
        assert client.get("/config").status_code == 401
        assert client.get(f"/gradio_api/file={asset}").status_code == 401
        assert client.post("/gradio_api/upload", files={"files": ("test.txt", b"hello")}).status_code == 401
        assert (
            client.post("/gradio_api/queue/join", json={"data": [], "fn_index": 0, "session_hash": "test"}).status_code
            == 401
        )
        assert client.post("/gradio_api/api/probe", json={"data": []}).status_code == 401
        assert client.post("/login", data={"username": "operator", "password": "incorrect"}).status_code == 400
        assert client.post("/login", data={"username": "operator", "password": policy.password}).status_code == 200
        assert client.get("/config").status_code == 200
        assert client.get(f"/gradio_api/file={asset}").text == "private fixture"
        response = client.post("/gradio_api/api/probe", json={"data": []})
        assert response.status_code == 200, response.text
        assert response.json()["data"] == ["authorized"]


def test_password_file_and_conflicting_sources(tmp_path):
    secret = tmp_path / "operator-secret"
    password = secrets.token_urlsafe(24)
    secret.write_text(password + "\n")
    env = {"COSMOS_GRADIO_USERNAME": "operator", "COSMOS_GRADIO_PASSWORD_FILE": str(secret)}
    policy = AccessPolicy.from_environment(env)
    assert policy.authenticate("operator", password)
    assert password not in repr(policy)
    env["COSMOS_GRADIO_PASSWORD"] = password
    with pytest.raises(ValueError, match="one Gradio password source"):
        AccessPolicy.from_environment(env)
