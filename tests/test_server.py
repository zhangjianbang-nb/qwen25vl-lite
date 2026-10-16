"""Protocol-level tests for the OpenAI-compatible surface.

These drive the streaming path directly, without a socket: ``make_handler`` returns a plain
class, so ``_run`` can be bound to a stand-in object that records the SSE frames it is handed.
The point is the *terminal* frame -- it must agree with the object type of the stream it ends
and it must repeat the real finish reason, not a hard-coded ``"stop"``.

``output_token_ids`` on each update is the **cumulative** sequence produced so far, exactly as
the engine's ``RequestOutput`` defines it; the fake service mirrors that contract.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, List, Optional

from qwen25vl_lite.server import PreparedPrompt, SamplingParams, make_handler


@dataclass
class _Update:
    output_token_ids: List[int]
    finish_reason: Optional[str] = None
    seq_id: int = 0
    error: Optional[str] = None

    @property
    def num_prompt_tokens(self) -> int:
        return 1

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)


class _FakeEngine:
    """Just enough engine for the handler: it must be cancellable when a stop string hits."""

    def __init__(self) -> None:
        self.cancelled: List[int] = []

    def cancel(self, seq_id: int, reason: str = "stop") -> bool:
        self.cancelled.append(seq_id)
        return True


class _FakeService:
    model_name = "fake"

    def __init__(self, updates: List[_Update]) -> None:
        self._updates = updates
        self.engine = _FakeEngine()

    def decode(self, ids: List[int]) -> str:
        return "".join(chr(0x40 + i) for i in ids)

    def generate(self, prepared, params):
        return 0, iter(self._updates)


class _Recorder:
    """Minimal stand-in for the request handler: only what ``_run`` touches."""

    def __init__(self, service: _FakeService) -> None:
        self.service_ref = service
        self.frames: List[Dict] = []
        self.json: List[Dict] = []
        self.wfile = io.BytesIO()

    def _sse_start(self) -> None:
        pass

    def _sse(self, obj: Dict) -> None:
        self.frames.append(obj)

    def _send_json(self, obj: Dict, status: int = 200) -> None:
        self.json.append(obj)


def _run(updates: List[_Update], chat: bool, body: Optional[Dict] = None) -> _Recorder:
    handler_cls = make_handler(_FakeService(updates))
    rec = _Recorder(_FakeService(updates))
    handler_cls._run(rec, PreparedPrompt(token_ids=[1], pixel_values=None,
                                         image_grid_thw=None, vision_token_count=0),
                     SamplingParams(temperature=0.0, max_tokens=8), dict(body or {"stream": True}),
                     chat)
    return rec


def test_stream_repeats_the_real_finish_reason_in_its_terminal_frame():
    rec = _run([_Update([1]), _Update([1, 2]), _Update([1, 2, 3], finish_reason="length")],
               chat=True)
    assert rec.frames[-1]["choices"][0]["finish_reason"] == "length"


def test_stream_terminal_frame_matches_the_object_type_it_ends():
    rec = _run([_Update([1], finish_reason="stop")], chat=True)
    assert rec.frames[-1]["object"] == "chat.completion.chunk"

    rec = _run([_Update([1], finish_reason="stop")], chat=False)
    assert rec.frames[-1]["object"] == "text_completion"


def test_stop_sequence_ends_the_stream_at_the_sequence():
    # `decode` maps token i to chr(0x40 + i): [1, 2, 3, 4] renders as "ABCD".
    rec = _run([_Update([1]), _Update([1, 2]), _Update([1, 2, 3]), _Update([1, 2, 3, 4])],
               chat=True, body={"stream": True, "stop": ["C"]})
    text = "".join(f["choices"][0].get("delta", {}).get("content", "") for f in rec.frames)
    assert text == "AB"
    assert rec.frames[-1]["choices"][0]["finish_reason"] == "stop"
    # Ending the stream must also stop the engine: otherwise it runs on to max_tokens.
    assert rec.service_ref.engine.cancelled == [0]


def test_non_streaming_response_is_a_single_json_object():
    rec = _run([_Update([1, 2, 3], finish_reason="stop")], chat=True, body={"stream": False})
    assert len(rec.json) == 1
    payload = rec.json[0]
    assert payload["choices"][0]["message"]["content"] == "ABC"
    assert payload["choices"][0]["finish_reason"] == "stop"


def test_completions_returns_token_ids_in_both_modes():
    """With no tokenizer `text` is an empty string, so the endpoint would be useless without the
    token ids.  They are an extension field and are present in the JSON response *and* in every
    streaming chunk -- the streaming path used to omit them, so the README's claim that the ids
    are always returned was untrue."""
    rec = _run([_Update([1, 2, 3], finish_reason="stop")], chat=False, body={"stream": False})
    choice = rec.json[0]["choices"][0]
    assert choice["token_ids"] == [1, 2, 3]
    assert choice["text"] == "ABC"  # the fake service has a decoder; the field is still there

    rec = _run([_Update([1]), _Update([1, 2], finish_reason="stop")], chat=False,
               body={"stream": True})
    carried = [f["choices"][0] for f in rec.frames if f["choices"][0].get("text")]
    assert carried and all(c.get("token_ids") for c in carried), (
        "a streaming text_completion chunk omitted token_ids"
    )
    assert rec.frames[-1]["choices"][0]["token_ids"] == [1, 2]


def test_non_streaming_stop_reports_stop_not_length():
    """A stop string truncates the text, so the reason is "stop" -- the streaming path already
    said so, and the two paths must not disagree about the same semantics."""
    rec = _run([_Update([1, 2, 3], finish_reason="length")], chat=True,
               body={"stream": False, "stop": ["C"]})
    assert rec.json[0]["choices"][0]["finish_reason"] == "stop"


def test_streaming_reports_a_scheduling_failure_instead_of_hanging():
    """Two failure modes used to be invisible: an exception from `engine.step()` escaped past the
    HTTP handler (the client saw a reset connection), and a request the scheduler never admits
    looped forever yielding empty updates.  Both are now surfaced as a request error."""

    class _BrokenEngine:
        def cancel(self, seq_id, reason="stop"):
            return True

        def step(self):
            raise RuntimeError("scheduler made no progress")

    from qwen25vl_lite.server import EngineService

    class _Stub:
        def __init__(self):
            self.engine = _BrokenEngine()

    stub = _Stub()
    first = next(EngineService._stream_until(stub, 0))
    assert first.finish_reason == "error" and "no progress" in (first.error or ""), (
        "a step() failure must become a request error, not an exception out of the handler"
    )


def test_streaming_error_frame_is_emitted():
    """The engine can also report a failure on an output; the server must turn it into an error
    frame instead of dropping it (which left the client with a bare "stop")."""
    rec = _run([_Update([], finish_reason="error", error="request could not be scheduled")],
               chat=True)
    assert rec.frames[-1]["choices"][0]["finish_reason"] == "error"
    assert rec.frames[-1]["error"]["message"] == "request could not be scheduled"
