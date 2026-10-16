"""qwen25vl-lite -- a dependency-light Qwen2.5-VL inference engine for consumer GPUs.

The package is deliberately layered so each stage can be verified on its own:

``config``         configuration parsing and parameter accounting
``rope``           MRoPE (text) and 2-D RoPE (vision)
``quant``          weight-only INT8 / INT4 / activation-aware INT4
``attention``      runtime SDPA backend probing and the paged-attention reference path
``imageproc``      smart-resize and patchification matching the HF processor
``vision``         the vision tower
``text``           the text decoder
``model``          assembly, streaming weight loading, forward pass
``cache``          block manager, paged KV cache, radix prefix cache
``vision_budget``  resolution budgets and token reduction strategies
``planner``        device-memory planning that keeps three context numbers separate
``offload``        static CPU/GPU layer placement with prefetch
``sampling``       penalties and top-k/top-p sampling
``engine``         continuous batching, chunked prefill, preemption
``server``         OpenAI-compatible HTTP API
``cli``            command line entry points

The engine core does not import ``transformers``: the config is parsed from
``config.json`` directly.  ``transformers`` appears in two places only -- the test suite,
where it is the reference implementation this engine is checked against numerically, and the
serving layer, which uses its tokenizer.  Everything the engine itself needs is pure torch.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import Qwen25VLConfig, TextConfig, VisionConfig  # noqa: F401
from .model import Qwen25VLLite, build_model, load_weights  # noqa: F401
from .engine import EngineConfig, LLMEngine, ModelRunner, RequestOutput  # noqa: F401
from .sampling import SamplingParams  # noqa: F401

__all__ = [
    "__version__",
    "Qwen25VLConfig",
    "TextConfig",
    "VisionConfig",
    "Qwen25VLLite",
    "build_model",
    "load_weights",
    "EngineConfig",
    "LLMEngine",
    "ModelRunner",
    "RequestOutput",
    "SamplingParams",
]
