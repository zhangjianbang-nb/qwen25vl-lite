"""The ``camera`` subcommand: it grabs frames, posts each one as a data URL, and prints the
description the server returns.

The camera itself cannot be unit-tested, so the frame source is monkeypatched: ``_open_camera``
and ``_grab_rgb`` are the only two places that touch OpenCV, and a stub HTTP server stands in
for ``serve``.  That keeps these tests independent of both the hardware and the cv2 install.
"""

from __future__ import annotations

import base64
import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest
from PIL import Image

from qwen25vl_lite import cli
from qwen25vl_lite.cli import build_parser

_REPLY = {
    "id": "cmpl-test",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "A desk and a monitor."},
                 "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 421, "completion_tokens": 8, "total_tokens": 429},
}


class _ChatHandler(BaseHTTPRequestHandler):
    reply: dict = {}
    requests: list = []

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler's spelling
        length = int(self.headers.get("Content-Length", 0))
        type(self).requests.append(json.loads(self.rfile.read(length)))
        body = json.dumps(self.reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence the default stderr access log
        pass


def _serve(reply: dict):
    handler = type("ChatHandler", (_ChatHandler,), {"reply": reply, "requests": []})
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1], handler.requests


class _FakeCap:
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


def _fake_camera(monkeypatch):
    """Point the camera at a synthetic 48x64 frame, and return the list of opened handles."""
    opened = []

    def open_camera(device):
        opened.append(_FakeCap())
        return opened[-1]

    monkeypatch.setattr(cli, "_open_camera", open_camera)
    monkeypatch.setattr(cli, "_grab_rgb",
                        lambda cap: np.zeros((48, 64, 3), dtype=np.uint8))
    return opened


def _camera_args(url: str, *extra: str):
    return build_parser().parse_args(["camera", "--url", url, "--interval", "0", *extra])


def test_rgb_to_data_url_roundtrips_through_jpeg():
    rgb = np.zeros((20, 30, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255  # a red image, so the JPEG is not degenerate

    url = cli._rgb_to_data_url(rgb, quality=90)
    assert url.startswith("data:image/jpeg;base64,")

    img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert img.format == "JPEG"
    assert img.size == (30, 20), "PIL reports (width, height); the array is 20 rows x 30 cols"


def test_camera_describes_each_frame_and_posts_the_image(monkeypatch, capsys):
    server, port, seen = _serve(_REPLY)
    opened = _fake_camera(monkeypatch)
    try:
        assert cli.cmd_camera(_camera_args(f"http://127.0.0.1:{port}", "--max-frames", "3")) == 0
    finally:
        server.shutdown()

    out = capsys.readouterr().out
    assert out.count("A desk and a monitor.") == 3
    assert len(seen) == 3, "one request per frame"
    assert opened and opened[0].released, "the camera must be released on the way out"

    content = seen[0]["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1]["text"] == "Describe what you see in one short sentence."
    assert seen[0]["temperature"] == 0.0, "frame description must be deterministic"


def test_camera_quiet_mode_prints_only_the_description(monkeypatch, capsys):
    server, port, _ = _serve(_REPLY)
    _fake_camera(monkeypatch)
    try:
        assert cli.cmd_camera(
            _camera_args(f"http://127.0.0.1:{port}", "--max-frames", "2", "--quiet")) == 0
    finally:
        server.shutdown()

    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert lines == ["A desk and a monitor."] * 2, lines


def test_camera_json_mode_emits_one_parseable_object_per_frame(monkeypatch, capsys):
    server, port, _ = _serve(_REPLY)
    _fake_camera(monkeypatch)
    try:
        assert cli.cmd_camera(
            _camera_args(f"http://127.0.0.1:{port}", "--max-frames", "2", "--json")) == 0
    finally:
        server.shutdown()

    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert len(lines) == 2
    for i, line in enumerate(lines):
        obj = json.loads(line)
        assert obj["frame"] == i
        assert obj["text"] == "A desk and a monitor."
        assert obj["error"] is None
        assert obj["latency_s"] > 0
        assert obj["fps"] > 0


def test_camera_save_dir_writes_every_captured_frame(monkeypatch, capsys, tmp_path):
    server, port, _ = _serve(_REPLY)
    _fake_camera(monkeypatch)
    try:
        assert cli.cmd_camera(_camera_args(f"http://127.0.0.1:{port}", "--max-frames", "3",
                                          "--save-dir", str(tmp_path))) == 0
    finally:
        server.shutdown()

    saved = sorted(p.name for p in tmp_path.glob("*.jpg"))
    assert saved == ["frame-00000.jpg", "frame-00001.jpg", "frame-00002.jpg"], saved


def test_camera_fails_loudly_when_the_server_never_answers(monkeypatch, capsys):
    # Port 1 is reserved and nothing listens on it, so every frame is refused.
    _fake_camera(monkeypatch)
    assert cli.cmd_camera(_camera_args("http://127.0.0.1:1", "--max-frames", "2",
                                       "--timeout", "0.5")) == 1
    out = capsys.readouterr().out
    assert "ERROR" in out and out.count("ERROR") == 2, out


def test_camera_rejects_a_negative_interval():
    args = build_parser().parse_args(["camera", "--interval", "-1"])
    with pytest.raises(SystemExit, match="negative"):
        cli.cmd_camera(args)


def test_open_camera_without_opencv_says_how_to_install_it(monkeypatch):
    # A None entry in sys.modules makes `import cv2` raise ImportError.
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(SystemExit, match="qwen25vl-lite\\[camera\\]"):
        cli._open_camera(0)


def test_downscale_preserves_the_aspect_ratio():
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    assert cli._downscale(rgb, 320).shape == (240, 320, 3)
    assert cli._downscale(rgb, 0).shape == (480, 640, 3), "0 means native"
    assert cli._downscale(rgb, 1280).shape == (480, 640, 3), "it must never upscale"


def test_camera_omits_the_vision_budget_by_default(monkeypatch):
    """With no flags the request carries no vision fields, so the server's own default (full)
    applies.  Sending "full" explicitly would be indistinguishable, but it would also pin the
    budget against a server whose default changes."""
    server, port, seen = _serve(_REPLY)
    _fake_camera(monkeypatch)
    try:
        assert cli.cmd_camera(_camera_args(f"http://127.0.0.1:{port}", "--max-frames", "1")) == 0
    finally:
        server.shutdown()

    body = seen[0]
    assert "vision_tier" not in body
    assert "max_vision_tokens" not in body


def test_camera_sends_the_request_level_vision_budget(monkeypatch):
    """The image budget is a *request* field: the server reads it from the body, not from its
    own --vision-tier flag, so this is the only way to shrink the vision token count over HTTP."""
    server, port, seen = _serve(_REPLY)
    _fake_camera(monkeypatch)
    try:
        assert cli.cmd_camera(_camera_args(f"http://127.0.0.1:{port}", "--max-frames", "1",
                                          "--vision-tier", "thumbnail",
                                          "--max-vision-tokens", "128")) == 0
    finally:
        server.shutdown()

    body = seen[0]
    assert body["vision_tier"] == "thumbnail"
    assert body["max_vision_tokens"] == 128


def test_camera_rejects_an_unknown_vision_tier():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["camera", "--vision-tier", "nope"])

