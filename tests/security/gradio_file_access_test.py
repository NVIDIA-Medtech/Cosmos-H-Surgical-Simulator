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

import os
from pathlib import Path

import pytest
from cosmos_gradio.file_access import media_snapshot, open_confined, read_text
from cosmos_gradio.gradio_app.gradio_file_server import (
    _format_files_list,
    _handle_view_file_dropdown_select_event,
)


@pytest.mark.parametrize(
    "selection",
    [
        "../secret.txt",
        "/etc/passwd",
        "nested/../../secret.txt",
        "./file.txt",
        "nested//file.txt",
        "..\\secret.txt",
        "%2e%2e/secret.txt",
    ],
)
def test_traversal_denied(tmp_path, selection):
    with pytest.raises((ValueError, OSError)):
        with open_confined(tmp_path, selection):
            pytest.fail("Traversal accepted")


@pytest.mark.parametrize("directory_link", [False, True])
def test_symlink_escape_not_listed_or_read(tmp_path, directory_link):
    root = tmp_path / "uploads"
    root.mkdir()
    outside = tmp_path / "uploads-sibling"
    outside.mkdir()
    (outside / "secret.txt").write_text("outside secret")
    if directory_link:
        (root / "link").symlink_to(outside, target_is_directory=True)
        selection = "link/secret.txt"
    else:
        (root / "secret.txt").symlink_to(outside / "secret.txt")
        selection = "secret.txt"
    assert _format_files_list(output_dir=str(root)) == []
    result = _handle_view_file_dropdown_select_event(selection, str(root))
    assert result[3].value == "Unable to view the selected file"


def test_symlink_swap_during_open_denied(tmp_path, monkeypatch):
    root = tmp_path / "uploads"
    root.mkdir()
    (root / "file.txt").write_text("safe")
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    real_open = os.open

    def swapping_open(path, flags, **kwargs):
        if path == "file.txt":
            (root / "file.txt").unlink()
            (root / "file.txt").symlink_to(outside)
        return real_open(path, flags, **kwargs)

    monkeypatch.setattr(os, "open", swapping_open)
    with pytest.raises(OSError):
        with open_confined(root, "file.txt"):
            pytest.fail("Symlink race accepted")


def test_normal_files_and_private_snapshot(tmp_path):
    (tmp_path / "nested").mkdir()
    text = tmp_path / "nested/file.txt"
    text.write_text("ordinary text")
    (tmp_path / "data.json").write_text('{"ok": true}')
    assert {value for _, value in _format_files_list(output_dir=str(tmp_path))} == {"nested/file.txt", "data.json"}
    assert _handle_view_file_dropdown_select_event("nested/file.txt", str(tmp_path))[3].value == "ordinary text"
    assert _handle_view_file_dropdown_select_event("data.json", str(tmp_path))[2].value == {"ok": True}
    with open_confined(tmp_path, "nested/file.txt") as stream:
        snapshot = Path(media_snapshot(stream, ".mp4"))
    text.write_text("changed after read")
    assert snapshot.read_text() == "ordinary text"
    assert not snapshot.is_relative_to(tmp_path)
    assert snapshot.stat().st_mode & 0o077 == 0


def test_fifo_and_large_text_denied(tmp_path, monkeypatch):
    os.mkfifo(tmp_path / "pipe.txt")
    with pytest.raises(ValueError, match="regular"):
        with open_confined(tmp_path, "pipe.txt"):
            pass
    (tmp_path / "large.txt").write_text("123456")
    monkeypatch.setattr("cosmos_gradio.file_access.MAX_TEXT_BYTES", 5)
    with open_confined(tmp_path, "large.txt") as stream, pytest.raises(ValueError, match="limit"):
        read_text(stream)


def test_authenticated_viewer_blocks_replaced_dropdown_file(tmp_path):
    import secrets

    import gradio as gr
    from cosmos_gradio.gradio_app.gradio_file_server import create_gradio_blocks
    from cosmos_gradio.security import AccessPolicy
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    root = tmp_path / "uploads"
    root.mkdir()
    target = root / "visible.txt"
    target.write_text("inside")
    secret = tmp_path / "secret.txt"
    secret.write_text("outside secret")
    policy = AccessPolicy("operator", secrets.token_urlsafe(24))
    blocks = create_gradio_blocks(str(root), access_policy=policy)
    fn_index = next(index for index, fn in blocks.fns.items() if len(fn.outputs) == 4)
    app = gr.mount_gradio_app(FastAPI(), blocks, path="/", auth=policy.authenticate)
    with TestClient(app) as client:
        client.post("/login", data={"username": policy.username, "password": policy.password})
        body = {"data": ["visible.txt"], "fn_index": fn_index}
        result = client.post("/gradio_api/api/predict", json=body)
        assert result.status_code == 200, result.text
        assert result.json()["data"][3]["value"] == "inside"
        target.unlink()
        target.symlink_to(secret)
        result = client.post("/gradio_api/api/predict", json=body)
        assert result.status_code == 200, result.text
        assert result.json()["data"][3]["value"] == "Unable to view the selected file"
        assert "outside secret" not in result.text


def test_direct_route_denies_mutable_roots_and_serves_upload_snapshot(tmp_path, monkeypatch):
    import json
    import secrets

    import gradio as gr
    from cosmos_gradio.file_access import file_serving_options
    from cosmos_gradio.gradio_app.gradio_file_server import create_gradio_blocks
    from cosmos_gradio.security import AccessPolicy
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    root = tmp_path / "uploads"
    root.mkdir()
    target = root / "visible.txt"
    target.write_text("inside")
    outside = tmp_path / "outside.txt"
    outside.write_text("OUTSIDE_SYNTHETIC_FIXTURE")
    policy = AccessPolicy("operator", secrets.token_urlsafe(24))
    blocks = create_gradio_blocks(str(root), access_policy=policy)
    app = gr.mount_gradio_app(FastAPI(), blocks, path="/", auth=policy.authenticate, **file_serving_options())
    real_check = gr.utils.is_allowed_file

    def check_then_swap(path, *args, **kwargs):
        result = real_check(path, *args, **kwargs)
        if Path(path) == target:
            target.unlink()
            target.symlink_to(outside)
        return result

    with TestClient(app) as client:
        client.post("/login", data={"username": policy.username, "password": policy.password})
        with monkeypatch.context() as patch:
            patch.setattr(gr.utils, "is_allowed_file", check_then_swap)
            response = client.get(f"/gradio_api/file={target}")
        assert response.status_code == 403
        assert "OUTSIDE_SYNTHETIC_FIXTURE" not in response.text
        uploaded = client.post("/gradio_api/upload", files={"files": ("sample.txt", b"uploaded fixture")})
        assert uploaded.status_code == 200, uploaded.text
        temporary = uploaded.json()[0]
        response = client.post(
            "/gradio_api/api/upload_file", json={"data": [{"path": temporary, "meta": {"_type": "gradio.FileData"}}]}
        )
        assert response.status_code == 200, response.text
        result = json.loads(response.json()["data"][0])
        assert "error" not in result, result
        assert Path(result["path"]).read_bytes() == b"uploaded fixture"
        assert client.get(f"/gradio_api/file={result['path']}").status_code == 403
        source = Path(result["path"])
        source.unlink()
        source.symlink_to(outside)
        response = client.get(f"/gradio_api/file={result['download_path']}")
        assert response.status_code == 200 and response.content == b"uploaded fixture"
        # A mutable alias initially pointing INTO an allowed snapshot must not
        # pass Gradio's resolving allow check, then be swapped to an outside file.
        for route in ("file=", "file/"):
            target.unlink()
            target.symlink_to(result["download_path"])
            with monkeypatch.context() as patch:
                patch.setattr(gr.utils, "is_allowed_file", check_then_swap)
                response = client.get(f"/gradio_api/{route}{target}")
            assert response.status_code == 403
            assert "OUTSIDE_SYNTHETIC_FIXTURE" not in response.text
        client.cookies.clear()
        assert client.get(f"/gradio_api/file={result['download_path']}").status_code == 401
    assert Path(blocks.GRADIO_CACHE).stat().st_mode & 0o077 == 0


def test_generated_video_and_log_download_use_snapshots(tmp_path):
    import secrets
    import shutil
    import subprocess

    import gradio as gr
    from cosmos_gradio.file_access import file_serving_options
    from cosmos_gradio.gradio_app.gradio_ui import create_gradio_UI
    from cosmos_gradio.security import AccessPolicy
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    if not shutil.which("ffmpeg"):
        pytest.skip("video integration requires the application's ffmpeg dependency")
    uploads = tmp_path / "uploads"
    outputs = tmp_path / "outputs"
    uploads.mkdir()
    outputs.mkdir()
    media = outputs / "result.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=32x32:d=0.2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(media),
        ],
        check=True,
    )
    expected = media.read_bytes()
    log = tmp_path / "app.log"
    log.write_text("private fixture log")
    policy = AccessPolicy("operator", secrets.token_urlsafe(24))
    blocks = create_gradio_UI(
        lambda _: (str(media), "done"), "test", "{}", "", str(uploads), str(outputs), str(log), policy
    )
    app = gr.mount_gradio_app(FastAPI(), blocks, path="/", auth=policy.authenticate, **file_serving_options())
    with TestClient(app) as client:
        client.post("/login", data={"username": policy.username, "password": policy.password})
        response = client.post("/gradio_api/api/generate_video", json={"data": ["{}"]})
        assert response.status_code == 200, response.text
        video = response.json()["data"][0]["video"]["path"]
        media.unlink()
        media.symlink_to(log)
        assert client.get(f"/gradio_api/file={media}").status_code == 403
        assert client.get(f"/gradio_api/file={video}").content == expected
        ranged = client.get(f"/gradio_api/file={video}", headers={"Range": "bytes=0-15"})
        assert ranged.status_code == 206 and ranged.content == expected[:16]
        log_index = next(
            index for index, fn in blocks.fns.items() if any(isinstance(out, gr.File) for out in fn.outputs)
        )
        response = client.post("/gradio_api/api/predict", json={"data": [], "fn_index": log_index})
        assert response.status_code == 200, response.text
        download = response.json()["data"][0]["path"]
        log.write_text("changed after snapshot")
        assert client.get(f"/gradio_api/file={log}").status_code == 403
        assert client.get(f"/gradio_api/file={download}").text == "private fixture log"
