"""OpenAI-compatible HTTP server.

Uses only the standard library (``http.server``), so serving a model needs no web framework.
A tokenizer is required to turn chat messages into token ids; if ``transformers`` is
installed its ``AutoTokenizer`` is used, otherwise only ``/v1/completions`` with explicit
``prompt_token_ids`` works.

Endpoints
---------
``GET  /health``                liveness + backend report
``GET  /stats``                 scheduler / cache counters
``GET  /v1/models``             model list
``POST /v1/chat/completions``   OpenAI chat format, supports ``image_url`` content parts
``POST /v1/completions``        legacy completion format

Streaming (``"stream": true``) is supported for both.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

from .engine import LLMEngine, RequestOutput
from .imageproc import (QWEN_MEAN, QWEN_STD, ImageBudget, image_mean_std, process_image,
                        vision_tokens)
from .sampling import SamplingParams
from .vision_budget import TIERS, tier_budget

__all__ = ["EngineService", "run_server", "build_placeholder_block", "parse_image_url"]

VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"


def build_placeholder_block(n_tokens: int) -> str:
    """The exact markup a Qwen2.5-VL prompt must contain for one image.

    The count must equal the number of vision features the tower will produce, otherwise
    ``scatter_image_features`` raises -- which is a *good* failure, because the alternative is
    a silently misaligned multimodal prompt.
    """
    if n_tokens <= 0:
        raise ValueError("an image must occupy at least one vision token")
    return VISION_START + IMAGE_PAD * n_tokens + VISION_END


def parse_image_url(url: str):
    """Decode an ``image_url`` value: ``data:`` URLs are decoded inline, ``http(s)`` fetched."""
    from PIL import Image

    if url.startswith("data:"):
        header, _, payload = url.partition(",")
        if "base64" in header:
            raw = base64.b64decode(payload)
        else:
            raw = payload.encode()
        return Image.open(io.BytesIO(raw))
    if url.startswith(("http://", "https://")):
        import urllib.request

        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - user-supplied URL
            return Image.open(io.BytesIO(resp.read()))
    raise ValueError(f"unsupported image_url scheme: {url[:32]!r}")


@dataclass
class PreparedPrompt:
    token_ids: List[int]
    pixel_values: Optional[torch.Tensor]
    image_grid_thw: Optional[torch.Tensor]
    vision_token_count: int


class EngineService:
    """Turns HTTP requests into engine calls, and engine outputs back into JSON."""

    def __init__(self, engine: LLMEngine, tokenizer=None, model_name: str = "qwen2.5-vl",
                 model_dir: Optional[str] = None):
        self.engine = engine
        # Normalisation statistics are a property of the checkpoint, not a constant: using
        # mean = std = 0.5 instead of the checkpoint's CLIP statistics changes every pixel the
        # vision tower sees and degrades every answer, silently.
        self.image_mean, self.image_std = (
            image_mean_std(model_dir) if model_dir else (QWEN_MEAN, QWEN_STD))
        self.tokenizer = tokenizer
        self.model_name = model_name

        # The HTTP server is threaded but `LLMEngine` is not thread-safe: its block manager,
        # paged KV cache and prefix cache are mutated by both `add_request` and `step`, and
        # the model itself is one set of CUDA buffers.  Letting each request thread drive the
        # engine -- which is what calling `step()` from the handler does -- races on all of
        # it: two concurrent image requests against the same picture produced
        # "图图图图中有中有两个两个两个…" and dropped connections outright.
        #
        # So exactly one thread owns the engine.  Request threads only enqueue a request and
        # wait on a condition variable, which is also what lets the engine batch them: the
        # driver keeps calling `step()` and the scheduler admits whatever is waiting.
        #
        # Two locks, deliberately: `_engine_lock` is held for the duration of `step()`
        # (a quarter of a second per token), and `_cond` guards only the published
        # snapshots.  With a single lock the driver held it across every step, so waiting
        # request threads were starved until generation finished -- the whole answer then
        # arrived in one SSE burst instead of token by token.
        self._engine_lock = threading.Lock()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._live: Dict[int, RequestOutput] = {}
        self._driver_error: Optional[BaseException] = None
        threading.Thread(target=self._drive, name="qwen25vl-engine", daemon=True).start()

    def _drive(self) -> None:
        """The only thread that ever touches the engine."""
        while True:
            with self._engine_lock:
                if not self.engine.has_work():
                    idle = True
                else:
                    idle = False
                    try:
                        outputs = self.engine.step()
                        snapshots = [
                            (s.seq_id, list(s.prompt_token_ids), list(s.output_token_ids))
                            for s in self.engine.running
                        ]
                        error = None
                    except BaseException as exc:  # noqa: BLE001
                        error = exc
            if idle:
                time.sleep(0.02)
                continue
            with self._cond:
                if error is not None:
                    # Publish the failure instead of dying silently: every waiting request
                    # must be released, not left to hang until its client times out.
                    self._driver_error = error
                    self._cond.notify_all()
                    return
                for seq_id, prompt_ids, out_ids in snapshots:
                    self._live[seq_id] = RequestOutput(
                        seq_id=seq_id,
                        prompt_token_ids=prompt_ids,
                        output_token_ids=out_ids,
                        finish_reason="",
                    )
                for out in outputs:      # finished outputs win over the running snapshot
                    self._live[out.seq_id] = out
                self._cond.notify_all()

    # -- prompt construction ----------------------------------------------------------
    def prepare(
        self,
        messages: Sequence[Dict[str, Any]],
        tier: str = "full",
        max_vision_tokens: Optional[int] = None,
    ) -> PreparedPrompt:
        if self.tokenizer is None:
            raise RuntimeError(
                "chat requests need a tokenizer; install transformers and pass one, or use "
                "/v1/completions with explicit prompt_token_ids"
            )
        vision_tier = TIERS.get(tier)
        if vision_tier is None:
            raise ValueError(f"unknown vision tier {tier!r}; known: {sorted(TIERS)}")
        budget = tier_budget(vision_tier)
        if max_vision_tokens is not None:
            budget = ImageBudget.from_tokens(max_vision_tokens)

        pixel_values: List[torch.Tensor] = []
        grids: List[torch.Tensor] = []
        rendered: List[Dict[str, Any]] = []

        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                rendered.append({"role": message["role"], "content": content})
                continue
            pieces: List[str] = []
            for part in content:
                kind = part.get("type")
                if kind == "text":
                    pieces.append(part.get("text", ""))
                elif kind in ("image_url", "image"):
                    url = part.get("image_url", {}).get("url") if kind == "image_url" else part.get("image")
                    image = parse_image_url(url)
                    proc = process_image(image, budget, mean=self.image_mean, std=self.image_std)
                    pixel_values.append(proc["pixel_values"])
                    grids.append(proc["grid_thw"])
                    n = vision_tokens(proc["grid_thw"][0], self.engine.runner.cfg.vision_config.spatial_merge_size)
                    pieces.append(build_placeholder_block(n))
                else:
                    raise ValueError(f"unsupported content part type {kind!r}")
            rendered.append({"role": message["role"], "content": "".join(pieces)})

        text = self.tokenizer.apply_chat_template(rendered, tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(text, add_special_tokens=False)
        ids = list(encoded["input_ids"])
        return PreparedPrompt(
            token_ids=ids,
            pixel_values=torch.cat(pixel_values, dim=0) if pixel_values else None,
            image_grid_thw=torch.cat(grids, dim=0) if grids else None,
            vision_token_count=sum(
                vision_tokens(g[0], self.engine.runner.cfg.vision_config.spatial_merge_size) for g in grids
            ),
        )

    # -- generation -------------------------------------------------------------------
    def generate(
        self,
        prepared: PreparedPrompt,
        params: SamplingParams,
    ) -> Tuple[int, Iterator[RequestOutput]]:
        with self._cond:
            if self._driver_error is not None:
                raise RuntimeError(f"engine driver stopped: {self._driver_error!r}")
        # `add_request` encodes the image through the model, so it needs the engine lock and
        # must not overlap a `step()` -- but it is brief, unlike a full decode step.
        with self._engine_lock:
            seq_id = self.engine.add_request(
                prepared.token_ids,
                params,
                pixel_values=prepared.pixel_values,
                image_grid_thw=prepared.image_grid_thw,
            )
        return seq_id, self._stream_until(seq_id)

    def _stream_until(self, seq_id: int) -> Iterator[RequestOutput]:
        """Yield each time the sequence gains a token, until it finishes.

        This never touches the engine directly -- it only reads the snapshots the driver
        publishes, which is what makes concurrent requests safe.
        """
        emitted = -1
        while True:
            with self._cond:
                while True:
                    if self._driver_error is not None:
                        raise RuntimeError(f"engine driver stopped: {self._driver_error!r}")
                    update = self._live.get(seq_id)
                    if update is not None and len(update.output_token_ids) != emitted:
                        break
                    self._cond.wait(1.0)
                self._live.pop(seq_id, None)   # consumed; the driver republishes on progress
            emitted = len(update.output_token_ids)
            if os.environ.get("QWEN_DBG"):
                print(f"[dbg {time.time():.2f}] yield seq={seq_id} tokens={emitted} "
                      f"finish={update.finish_reason!r}", file=sys.stderr, flush=True)
            yield update
            if update.finish_reason:
                return

    def decode(self, token_ids: Sequence[int]) -> str:
        if self.tokenizer is None:
            return ""
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)


# --------------------------------------------------------------------------------------
# HTTP plumbing
# --------------------------------------------------------------------------------------


def _parse_params(body: Dict[str, Any]) -> SamplingParams:
    """Decoding parameters, engine-safe.

    String ``stop`` sequences are deliberately *not* forwarded: the engine has no tokenizer,
    so it refuses them rather than silently ignoring them.  They are resolved here against
    decoded text by :func:`_apply_stops`, which is where a tokenizer exists.
    """
    return SamplingParams(
        temperature=float(body.get("temperature", 0.0) or 0.0),
        top_p=float(body.get("top_p", 1.0) or 1.0),
        top_k=int(body.get("top_k", -1) or -1),
        max_tokens=int(body.get("max_tokens", body.get("max_completion_tokens", 256)) or 256),
        repetition_penalty=float(body.get("repetition_penalty", 1.0) or 1.0),
        presence_penalty=float(body.get("presence_penalty", 0.0) or 0.0),
        frequency_penalty=float(body.get("frequency_penalty", 0.0) or 0.0),
        seed=body.get("seed"),
        ignore_eos=bool(body.get("ignore_eos", False)),
    )


def _stops(body: Dict[str, Any]) -> Tuple[str, ...]:
    stop = body.get("stop") or ()
    if isinstance(stop, str):
        stop = (stop,)
    return tuple(str(s) for s in stop)


def _apply_stops(text: str, stops: Tuple[str, ...]) -> str:
    """Truncate ``text`` at the earliest occurrence of any stop sequence."""
    cut = len(text)
    for s in stops:
        if not s:
            continue
        idx = text.find(s)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]


def make_handler(service: EngineService):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        service_ref = service

        def log_message(self, fmt, *args):  # keep the server quiet by default
            pass

        # -- helpers ------------------------------------------------------------------
        def _send_json(self, obj: Dict[str, Any], status: int = 200) -> None:
            payload = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _read_body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode())

        def _sse_start(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

        def _sse(self, obj: Dict[str, Any]) -> None:
            self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()

        # -- routes -------------------------------------------------------------------
        def do_GET(self):  # noqa: N802
            if self.path.startswith("/health"):
                report = self.service_ref.engine.runner.backend_report
                self._send_json({
                    "status": "ok",
                    "model": self.service_ref.model_name,
                    "attn_backends": self.service_ref.engine.runner.backends,
                    "backend_report": report.summary() if report else "pinned by configuration",
                    "vision_tiers": sorted(TIERS),
                })
            elif self.path.startswith("/stats"):
                self._send_json(self.service_ref.engine.stats())
            elif self.path.startswith("/v1/models"):
                self._send_json({
                    "object": "list",
                    "data": [{"id": self.service_ref.model_name, "object": "model", "owned_by": "local"}],
                })
            else:
                self._send_json({"error": "not found"}, status=404)

        def do_POST(self):  # noqa: N802
            try:
                body = self._read_body()
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": {"message": f"invalid JSON body: {exc}"}}, status=400)
                return

            if self.path.startswith("/v1/chat/completions"):
                self._handle_chat(body)
            elif self.path.startswith("/v1/completions"):
                self._handle_completion(body)
            else:
                self._send_json({"error": "not found"}, status=404)

        def _handle_chat(self, body: Dict[str, Any]) -> None:
            tier = body.get("vision_tier", "full")
            try:
                prepared = self.service_ref.prepare(body.get("messages", []), tier=tier,
                                                max_vision_tokens=body.get("max_vision_tokens"))
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": {"message": str(exc), "type": "invalid_request_error"}}, status=400)
                return
            params = _parse_params(body)
            self._run(prepared, params, body, chat=True)

        def _handle_completion(self, body: Dict[str, Any]) -> None:
            ids = body.get("prompt_token_ids")
            if ids is None:
                self._send_json({"error": {"message": "prompt_token_ids is required"}}, status=400)
                return
            prepared = PreparedPrompt(token_ids=list(ids), pixel_values=None,
                                      image_grid_thw=None, vision_token_count=0)
            params = _parse_params(body)
            self._run(prepared, params, body, chat=False)

        def _run(self, prepared: PreparedPrompt, params: SamplingParams, body: Dict[str, Any], chat: bool) -> None:
            created = int(time.time())
            completion_id = f"cmpl-{uuid.uuid4().hex[:16]}"
            try:
                _seq_id, stream = self.service_ref.generate(prepared, params)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": {"message": str(exc), "type": "invalid_request_error"}}, status=400)
                return

            streaming = bool(body.get("stream", False))
            stops = _stops(body)
            if streaming:
                self._sse_start()
                emitted = 0
                first = True
                for update in stream:
                    text = _apply_stops(self.service_ref.decode(update.output_token_ids), stops)
                    chunk = text[emitted:]
                    emitted = len(text)
                    hit_stop = len(text) < len(self.service_ref.decode(update.output_token_ids))
                    if hit_stop:
                        # A stop sequence was produced: emit what is left of the visible text
                        # and end the stream here rather than letting the model run on.
                        if chunk:
                            delta = {"role": "assistant", "content": chunk} if first else {"content": chunk}
                            first = False
                            if chat:
                                self._sse({
                                    "id": completion_id, "object": "chat.completion.chunk",
                                    "created": created, "model": self.service_ref.model_name,
                                    "choices": [{"index": 0, "delta": delta, "finish_reason": "stop"}],
                                })
                            else:
                                self._sse({
                                    "id": completion_id, "object": "text_completion",
                                    "created": created, "model": self.service_ref.model_name,
                                    "choices": [{"index": 0, "text": chunk, "finish_reason": "stop"}],
                                })
                        break
                    if not chunk:
                        continue
                    delta = {"role": "assistant", "content": chunk} if first else {"content": chunk}
                    first = False
                    if chat:
                        self._sse({
                            "id": completion_id, "object": "chat.completion.chunk", "created": created,
                            "model": self.service_ref.model_name,
                            "choices": [{"index": 0, "delta": delta,
                                         "finish_reason": update.finish_reason or None}],
                        })
                    else:
                        self._sse({
                            "id": completion_id, "object": "text_completion", "created": created,
                            "model": self.service_ref.model_name,
                            "choices": [{"index": 0, "text": chunk,
                                         "finish_reason": update.finish_reason or None}],
                        })
                # The terminal event must carry the same object type as the stream it ends:
                # a `/v1/completions` stream reporting `chat.completion.chunk` here was an
                # inconsistency clients could reasonably trip over.
                if chat:
                    self._sse({"id": completion_id, "object": "chat.completion.chunk", "created": created,
                               "model": self.service_ref.model_name,
                               "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                else:
                    self._sse({"id": completion_id, "object": "text_completion", "created": created,
                               "model": self.service_ref.model_name,
                               "choices": [{"index": 0, "text": "", "finish_reason": "stop"}]})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return

            final = None
            for update in stream:
                final = update
            assert final is not None
            text = _apply_stops(self.service_ref.decode(final.output_token_ids), stops)
            if chat:
                self._send_json({
                    "id": completion_id, "object": "chat.completion", "created": created,
                    "model": self.service_ref.model_name,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                 "finish_reason": final.finish_reason}],
                    "usage": {
                        "prompt_tokens": final.num_prompt_tokens,
                        "completion_tokens": final.num_output_tokens,
                        "total_tokens": final.num_prompt_tokens + final.num_output_tokens,
                    },
                })
            else:
                self._send_json({
                    "id": completion_id, "object": "text_completion", "created": created,
                    "model": self.service_ref.model_name,
                    "choices": [{"index": 0, "text": text, "finish_reason": final.finish_reason}],
                    "usage": {
                        "prompt_tokens": final.num_prompt_tokens,
                        "completion_tokens": final.num_output_tokens,
                        "total_tokens": final.num_prompt_tokens + final.num_output_tokens,
                    },
                })

    return Handler


def run_server(service: EngineService, host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(service))
    server.daemon_threads = True
    print(f"serving {service.model_name} on http://{host}:{port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
