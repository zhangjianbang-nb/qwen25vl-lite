"""The ``monitor`` subcommand: it renders the live decode rate from a server's /stats, it shows
an idle engine as idle rather than holding a stale rate, and it degrades gracefully when the
server is unreachable instead of raising.

A stub HTTP server stands in for a real ``serve`` instance, so the tests exercise the actual
HTTP path and the actual formatting without needing a model or a GPU.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from qwen25vl_lite.cli import _sparkline, build_parser, cmd_monitor


class _StatsHandler(BaseHTTPRequestHandler):
    payload: dict = {}

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's spelling
        body = json.dumps(self.payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence the default stderr access log
        pass


def _serve(payload: dict):
    handler = type("StatsHandler", (_StatsHandler,), {"payload": payload})
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def _monitor_args(url: str, *extra: str):
    return build_parser().parse_args(["monitor", "--url", url, "--count", "1", "--rows", *extra])


def test_monitor_renders_the_live_decode_rate(capsys):
    payload = {
        "decode_tokens_per_s": 73.2, "decode_step_ms": 13.6,
        "decode_active": 1.0, "decode_idle_s": 0.02,
        "running": 3, "waiting": 1, "swapped": 0,
        "kv_blocks_used": 120, "kv_blocks_total": 4000,
        "prefix_hit_rate": 0.42,
    }
    server, port = _serve(payload)
    try:
        assert cmd_monitor(_monitor_args(f"http://127.0.0.1:{port}")) == 0
    finally:
        server.shutdown()

    out = capsys.readouterr().out
    assert "decode tok/s" in out, "the header must label the columns"
    assert "73.2" in out, out
    assert "13.60" in out, out
    assert "120/4000" in out, out
    assert "42.0%" in out, out
    assert "—" in out, "an active engine must not be labelled idle"


def test_monitor_shows_how_long_the_engine_has_been_idle(capsys):
    """Once decoding stops the server reports a zero rate; the monitor must show that, plus how
    long it has been quiet, instead of holding the last rate it saw."""
    payload = {"decode_tokens_per_s": 0.0, "decode_step_ms": 0.0,
               "decode_active": 0.0, "decode_idle_s": 3.4,
               "running": 0, "waiting": 0, "kv_blocks_used": 0, "kv_blocks_total": 4000}
    server, port = _serve(payload)
    try:
        assert cmd_monitor(_monitor_args(f"http://127.0.0.1:{port}")) == 0
    finally:
        server.shutdown()

    out = capsys.readouterr().out
    assert "idle" in out, "the header must label the idle column"
    assert "3.4s" in out, out
    assert "0.0" in out, out


def test_monitor_json_mode_emits_one_parseable_object_per_sample(capsys):
    payload = {"decode_tokens_per_s": 73.2, "decode_active": 1.0, "decode_idle_s": 0.01,
               "running": 1, "kv_blocks_used": 5, "kv_blocks_total": 4000}
    server, port = _serve(payload)
    try:
        args = build_parser().parse_args(
            ["monitor", "--url", f"http://127.0.0.1:{port}", "--count", "2", "--json"])
        assert cmd_monitor(args) == 0
    finally:
        server.shutdown()

    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert len(lines) == 2, lines
    for line in lines:
        obj = json.loads(line)
        assert obj["decode_tokens_per_s"] == 73.2
        assert obj["error"] is None
        assert "time" in obj


def test_monitor_reports_an_unreachable_server_without_crashing(capsys):
    # Port 1 is reserved and nothing listens on it, so the connection is refused immediately.
    assert cmd_monitor(_monitor_args("http://127.0.0.1:1", "--timeout", "0.5")) == 0
    out = capsys.readouterr().out
    assert "unreachable" in out, out


def test_monitor_tolerates_a_stats_payload_missing_the_new_fields(capsys):
    """An older server (started before the rolling window existed) returns no rate keys; the
    monitor must still render, defaulting those columns to zero."""
    server, port = _serve({"running": 2, "waiting": 0, "kv_blocks_used": 8,
                           "kv_blocks_total": 64})
    try:
        assert cmd_monitor(_monitor_args(f"http://127.0.0.1:{port}")) == 0
    finally:
        server.shutdown()

    out = capsys.readouterr().out
    assert "0.0" in out and "8/64" in out, out


def test_sparkline_is_fixed_width_and_distinguishes_steady_from_idle():
    assert _sparkline([], 8) == " " * 8
    assert len(_sparkline([1.0, 2.0, 3.0], 8)) == 3, "it renders what it has, the row pads the rest"
    assert _sparkline([1.0], 0) == "", "width 0 disables the history column"

    steady = _sparkline([70.0] * 5, 5)
    idle = _sparkline([0.0] * 5, 5)
    assert len(set(steady)) == 1 and len(set(idle)) == 1
    assert steady != idle, "a steady rate must not look like an idle engine"
