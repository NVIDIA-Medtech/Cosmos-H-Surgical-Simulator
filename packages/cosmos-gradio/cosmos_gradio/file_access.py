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

"""Confined reads on Linux: never follow symlinks, including during path traversal."""

import hashlib
import os
import re
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path

from starlette.middleware import Middleware
from starlette.responses import PlainTextResponse

MAX_TEXT_BYTES = 8 * 1024 * 1024
MAX_MEDIA_BYTES = 500 * 1024 * 1024
# Gradio receives private snapshots, never paths that an upload writer can swap.
_SNAPSHOTS = tempfile.TemporaryDirectory(prefix="cosmos-viewer-")
_GRADIO_CACHE = tempfile.TemporaryDirectory(prefix="cosmos-gradio-")


def configure_file_serving():
    """Keep Gradio's implicitly served upload/example caches private too.

    Call before constructing Blocks/components and before launching/mounting.
    Upload/output source directories must never be added to allowed_paths.
    """
    os.environ["GRADIO_TEMP_DIR"] = _GRADIO_CACHE.name
    os.environ["GRADIO_EXAMPLES_CACHE"] = str(Path(_GRADIO_CACHE.name) / "examples")
    return [_SNAPSHOTS.name]


class _PrivateFileURLs:
    """Reject aliases outside private roots before Gradio resolves symlinks.

    Gradio's allow check resolves paths, but FileResponse later reopens the
    original path. Only URLs lexically inside our private roots may reach it.
    Other OS users cannot replace files or directories within these roots.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            match = re.search(r"/(?:gradio_api/)?file[=/](.*)", scope["path"])
            if match:
                path = match.group(1)
                candidate = Path(path)
                allowed = (
                    candidate.is_absolute()
                    and ".." not in path.split("/")
                    and "\x00" not in path
                    and any(candidate.is_relative_to(root) for root in (_SNAPSHOTS.name, _GRADIO_CACHE.name))
                )
                if not allowed:
                    return await PlainTextResponse("File not available", status_code=403)(scope, receive, send)
        await self.app(scope, receive, send)


def file_serving_options():
    """Options shared by Blocks.launch and mount_gradio_app."""
    return {
        "allowed_paths": configure_file_serving(),
        "app_kwargs": {"middleware": [Middleware(_PrivateFileURLs)]},
    }


def snapshot_path(root, path):
    """Copy a producer's path via the confined reader before publishing it."""
    root = Path(os.path.abspath(root))
    source = Path(os.path.abspath(path))
    selection = source.relative_to(root).as_posix()
    with open_confined(root, selection) as stream:
        return media_snapshot(stream, source.suffix)


@contextmanager
def open_confined(root, selection):
    if not isinstance(selection, str) or not selection or "\\" in selection or "\x00" in selection:
        raise ValueError("Invalid file selection")
    parts = selection.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("Select a relative file inside the configured directory")
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("Only regular files can be viewed")
            yield stream
    finally:
        os.close(directory)


def read_text(stream):
    data = stream.read(MAX_TEXT_BYTES + 1)
    if len(data) > MAX_TEXT_BYTES:
        raise ValueError("Text file exceeds viewer size limit")
    return data.decode("utf-8")


def media_snapshot(stream, suffix):
    digest = hashlib.sha256()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=_SNAPSHOTS.name, delete=False) as output:
            temporary = Path(output.name)
            size = 0
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_MEDIA_BYTES:
                    raise ValueError("Media exceeds viewer size limit")
                digest.update(chunk)
                output.write(chunk)
        destination = Path(_SNAPSHOTS.name) / (digest.hexdigest() + suffix)
        os.replace(temporary, destination)
        return str(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
