#!/usr/bin/env bash
# Example launcher for `qwen25vl-lite serve`, tuned for an 8 GB card (RTX 5060 / 4060 class).
#
# This is a *reference* configuration, not a required one.  Every value below is overridable
# from the environment:
#
#   MODEL=./Qwen2.5-VL-7B-Instruct PORT=8000 ./scripts/serve.sh
#
# The three opt-in accelerators are worth enabling together, and only together -- see
# README.md, "Acceleration".  Individually each one loses in some regime:
#   --int4-kernel        fused INT4 dequantise + GEMM
#   --paged-attn-kernel  fused paged decode attention (flash-decoding)
#   --cuda-graph         capture the decode step into one CUDA graph
# Combined they measured +40% / +42% / +62% at 512 / 2048 / 4096-token contexts.
#
# Each one self-checks at startup and refuses itself if the check fails, so enabling them is
# not a correctness risk -- a refusal is printed and the reference path is kept.
set -euo pipefail

# `qwen25vl-lite` lives in whichever environment you pip-installed it into.  If your shell is
# activated elsewhere you will get "command not found"; either activate that env, or set
# BIN=/path/to/env/bin/qwen25vl-lite here.
BIN=${BIN:-qwen25vl-lite}
MODEL=${MODEL:-./Qwen2.5-VL-3B-Instruct}
PORT=${PORT:-8081}

# bfloat16, not float16.  With --dtype float16 the vision tower overflows to NaN on a large
# patch grid (e.g. a 1312x1199 image at the `full` tier), so every logit is NaN and the model
# emits one repeated token.  HuggingFace transformers fails identically, so this is the
# checkpoint's precision limit rather than an engine bug.  config.json declares bfloat16.
#
# PYTHONUNBUFFERED: when run in the background, stdout is not a tty and Python block-buffers,
# so the self-check output only appears when the process exits and the log looks empty.
export PYTHONUNBUFFERED=1

exec "$BIN" serve \
  --model "$MODEL" \
  --device cuda \
  --dtype bfloat16 \
  --weight-dtype int4 \
  --group-size 128 \
  --kv-dtype float16 \
  --num-blocks 4000 \
  --block-size 16 \
  --max-model-len 32768 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --vision-tier full \
  --int4-kernel \
  --paged-attn-kernel \
  --cuda-graph \
  --host 127.0.0.1 \
  --port "$PORT"
