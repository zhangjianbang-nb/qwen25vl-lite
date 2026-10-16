"""Qwen2.5-VL configuration objects.

Parses a HuggingFace ``config.json`` -- both the legacy ``rope_scaling`` key used by
Qwen2.5-VL releases and the newer ``rope_parameters`` key used by transformers >= 5 --
and exposes the parameter accounting that the memory planner consumes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple


@dataclass
class VisionConfig:
    """Vision tower hyper-parameters (``vision_config`` in config.json)."""

    depth: int = 32
    hidden_size: int = 1280
    intermediate_size: int = 3420
    num_heads: int = 16
    in_chans: int = 3
    out_hidden_size: int = 2048
    patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    window_size: int = 112
    fullatt_block_indexes: Tuple[int, ...] = (7, 15, 23, 31)
    tokens_per_second: float = 2.0
    hidden_act: str = "silu"

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def rotary_dim(self) -> int:
        """Width of the vision RoPE fed into each of the two spatial axes."""
        return self.head_dim // 2

    @property
    def spatial_merge_unit(self) -> int:
        return self.spatial_merge_size * self.spatial_merge_size

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "VisionConfig":
        raw = dict(raw)
        if "in_chans" not in raw and "in_channels" in raw:
            raw["in_chans"] = raw.pop("in_channels")
        known = set(cls.__dataclass_fields__)
        kwargs = {k: v for k, v in raw.items() if k in known}
        if "fullatt_block_indexes" in kwargs:
            kwargs["fullatt_block_indexes"] = tuple(int(i) for i in kwargs["fullatt_block_indexes"])
        return cls(**kwargs)

    def num_params(self) -> int:
        """Number of parameters in the vision tower, including biases."""
        dim = self.hidden_size
        inter = self.intermediate_size
        total = self.in_chans * dim * self.temporal_patch_size * self.patch_size * self.patch_size

        per_block = 3 * dim * dim + 3 * dim  # qkv weight + bias
        per_block += dim * dim + dim  # proj weight + bias
        per_block += 3 * dim * inter + 3 * inter  # gate/up/down weight + bias
        per_block += 2 * dim  # two RMSNorm weights
        total += per_block * self.depth

        merged = dim * self.spatial_merge_unit
        total += dim  # merger.ln_q
        total += merged * merged + merged  # mlp.0
        total += merged * self.out_hidden_size + self.out_hidden_size  # mlp.2
        return total


@dataclass
class TextConfig:
    """Text decoder hyper-parameters (``text_config`` in config.json)."""

    hidden_size: int = 2048
    intermediate_size: int = 11008
    num_hidden_layers: int = 36
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    vocab_size: int = 151936
    max_position_embeddings: int = 128000
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    mrope_section: Tuple[int, ...] = (16, 24, 24)
    hidden_act: str = "silu"
    attention_dropout: float = 0.0
    sliding_window: Optional[int] = 32768
    use_sliding_window: bool = False
    tie_word_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def layer_types(self) -> Tuple[str, ...]:
        """Per-layer attention flavour. Qwen2.5-VL ships ``use_sliding_window=false``."""
        if self.use_sliding_window:
            return tuple("sliding_attention" if (i + 1) % 2 else "full_attention" for i in range(self.num_hidden_layers))
        return tuple("full_attention" for _ in range(self.num_hidden_layers))

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TextConfig":
        raw = dict(raw)
        known = set(cls.__dataclass_fields__)
        kwargs = {k: v for k, v in raw.items() if k in known}
        if "mrope_section" in kwargs:
            kwargs["mrope_section"] = tuple(int(i) for i in kwargs["mrope_section"])
        return cls(**kwargs)

    def num_params(self) -> int:
        h = self.hidden_size
        hd = self.head_dim
        inter = self.intermediate_size
        total = self.vocab_size * h  # embed_tokens
        per_layer = h * (self.num_attention_heads * hd) + self.num_attention_heads * hd
        per_layer += 2 * (h * (self.num_key_value_heads * hd) + self.num_key_value_heads * hd)
        per_layer += self.num_attention_heads * hd * h  # o_proj (no bias)
        per_layer += 3 * h * inter  # SwiGLU (no bias)
        per_layer += 2 * h  # two RMSNorm weights
        total += per_layer * self.num_hidden_layers
        total += h  # final norm
        if not self.tie_word_embeddings:
            total += self.vocab_size * h
        return total


@dataclass
class Qwen25VLConfig:
    """Top-level Qwen2.5-VL config."""

    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    eos_token_id: int = 151645
    bos_token_id: int = 151643
    pad_token_id: int = 151643
    model_type: str = "qwen2_5_vl"
    tie_word_embeddings: bool = True
    torch_dtype: str = "bfloat16"
    _name_or_path: str = ""

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Qwen25VLConfig":
        text_raw = dict(raw.get("text_config", {}))
        # Qwen2.5-VL releases keep the MRoPE section in `rope_scaling`; transformers 5
        # moved it to `rope_parameters`. Accept either so the engine loads both.
        for key in ("rope_scaling", "rope_parameters"):
            block = raw.get(key) or text_raw.get(key)
            if isinstance(block, dict):
                if "mrope_section" in block:
                    text_raw.setdefault("mrope_section", block["mrope_section"])
                if "rope_theta" in block:
                    text_raw.setdefault("rope_theta", block["rope_theta"])
        for key in ("rope_theta", "mrope_section"):
            if key in raw:
                text_raw.setdefault(key, raw[key])
        text_raw.setdefault("tie_word_embeddings", raw.get("tie_word_embeddings", True))
        vision = VisionConfig.from_dict(raw.get("vision_config", {}))
        text = TextConfig.from_dict(text_raw)
        return cls(
            text_config=text,
            vision_config=vision,
            image_token_id=int(raw.get("image_token_id", 151655)),
            video_token_id=int(raw.get("video_token_id", 151656)),
            vision_start_token_id=int(raw.get("vision_start_token_id", 151652)),
            vision_end_token_id=int(raw.get("vision_end_token_id", 151653)),
            eos_token_id=int(raw.get("eos_token_id", 151645)),
            bos_token_id=int(raw.get("bos_token_id", 151643)),
            pad_token_id=int(raw.get("pad_token_id", 151643)),
            model_type=raw.get("model_type", "qwen2_5_vl"),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", True)),
            torch_dtype=raw.get("torch_dtype", "bfloat16"),
            _name_or_path=raw.get("_name_or_path", ""),
        )

    @classmethod
    def from_json(cls, path: str) -> "Qwen25VLConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "Qwen25VLConfig":
        path = model_dir if model_dir.endswith(".json") else os.path.join(model_dir, "config.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"no config.json under {model_dir!r}; point --model at a HuggingFace snapshot directory"
            )
        cfg = cls.from_json(path)
        cfg._name_or_path = os.path.abspath(os.path.dirname(path) if path.endswith("config.json") else path)
        return cfg

    def num_params(self) -> Dict[str, int]:
        vis = self.vision_config.num_params()
        txt = self.text_config.num_params()
        return {"vision": vis, "text": txt, "total": vis + txt}

    def describe(self) -> str:
        p = self.num_params()
        t = self.text_config
        v = self.vision_config
        return (
            f"{self.model_type}: text {t.num_hidden_layers}L x {t.hidden_size}d "
            f"(GQA {t.num_attention_heads}/{t.num_key_value_heads}), vocab {t.vocab_size}; "
            f"vision {v.depth}L x {v.hidden_size}d, patch {v.patch_size}, merge {v.spatial_merge_size}, "
            f"window {v.window_size}; params vision {p['vision']/1e6:.1f}M text {p['text']/1e6:.1f}M "
            f"total {p['total']/1e6:.1f}M"
        )


# Byte widths used by the planner. "int4" is stored at 0.5 bytes per weight plus
# one fp16 scale and one fp16 zero-point per `group_size` weights.
DTYPE_BYTES = {
    "float32": 4.0,
    "float16": 2.0,
    "bfloat16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}


def bytes_per_weight(dtype: str, group_size: Optional[int] = None) -> float:
    """Bytes per stored weight for a weight-only quantisation scheme."""
    if dtype in ("int8", "float16", "bfloat16", "float32"):
        return DTYPE_BYTES[dtype]
    if dtype in ("int4", "awq"):
        g = group_size or 128
        return 0.5 + 4.0 / g  # packed nibbles + fp16 scale + fp16 zero
    raise ValueError(f"unknown weight dtype {dtype!r}")
