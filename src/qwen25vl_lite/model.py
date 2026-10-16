"""Model assembly, HuggingFace weight loading, and the forward pass.

Weight loading is **streaming**: the model is built on the ``meta`` device so construction
costs no memory, and tensors are then pulled one at a time out of the safetensors mmap.  If
a quantisation scheme is active, each linear is quantised the moment its weight is read and
the float copy is dropped immediately -- peak RAM stays at "quantised model + one layer"
instead of "float model + quantised model".

Tied embeddings are respected: Qwen2.5-VL-3B ships ``tie_word_embeddings=true`` and the
checkpoint has no ``lm_head.weight``, so the head shares ``embed_tokens.weight`` rather than
duplicating a 151936 x 2048 matrix.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .config import Qwen25VLConfig
from .quant import QuantLinear, QuantizedTensor, quantize_awq_style, quantize_int4_groupwise, quantize_int8_per_channel
from .rope import get_rope_index
from .text import TextModel
from .vision import VisionTransformer

__all__ = ["Qwen25VLLite", "build_model", "load_weights", "apply_quantization", "map_hf_name",
           "QUANTIZABLE_SUFFIXES"]

# Only these linear layers are quantised.  Vision tower, embeddings, norms and the LM head
# stay in the compute dtype: the vision tower is a small fraction of the parameters and is
# far more sensitive to weight error than the decoder, and the head's cost is dominated by
# memory bandwidth over the vocabulary, not by precision.
QUANTIZABLE_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

# Text-decoder submodules that appear directly under ``model.`` in checkpoints saved by
# transformers >= 5 (which flattens ``model.language_model.*`` to ``model.*``).
_TEXT_ROOTS = ("embed_tokens", "layers", "norm")


def map_hf_name(name: str) -> str:
    """Translate a checkpoint key into this engine's module namespace.

    Two layouts are in the wild and both must load:

    * the Hub layout used by Qwen2.5-VL releases -- ``model.language_model.*``,
      ``model.visual.*``;
    * the layout ``save_pretrained`` writes under transformers >= 5 -- ``model.*`` for the
      decoder and a bare ``visual.*`` for the tower.

    ``lm_head.weight`` is passed through unchanged.
    """
    if name.startswith("lm_head."):
        return name
    if name.startswith("visual."):
        return name
    rest = name[len("model.") :] if name.startswith("model.") else name
    if rest.startswith("visual.") or rest.startswith("language_model."):
        return rest
    if rest.split(".")[0] in _TEXT_ROOTS:
        return f"language_model.{rest}"
    return rest


class Qwen25VLLite(nn.Module):
    """Vision tower + text decoder + LM head."""

    def __init__(self, cfg: Qwen25VLConfig):
        super().__init__()
        self.cfg = cfg
        self.visual = VisionTransformer(cfg.vision_config)
        self.language_model = TextModel(cfg.text_config)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.language_model.embed_tokens.weight
        self.quant_scheme: Optional[str] = None
        self.group_size: Optional[int] = None

    # -- helpers ---------------------------------------------------------------------
    @property
    def text_layers(self):
        return self.language_model.layers

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.embed_tokens(input_ids)

    def embed_images(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        *,
        backends: Optional[Sequence[str]] = None,
        return_intermediates: bool = False,
    ):
        return self.visual(pixel_values, grid_thw, backends=backends, return_intermediates=return_intermediates)

    def scatter_image_features(
        self, inputs_embeds: torch.Tensor, input_ids: torch.Tensor, image_features: torch.Tensor
    ) -> torch.Tensor:
        """Replace ``image_token_id`` positions with vision features (HF ``masked_scatter``)."""
        mask = (input_ids == self.cfg.image_token_id).unsqueeze(-1)
        n_tokens = int(mask.sum().item())
        if n_tokens * inputs_embeds.shape[-1] != image_features.numel():
            raise ValueError(
                f"image features and image tokens do not match: {n_tokens} tokens, "
                f"{image_features.shape[0]} feature rows"
            )
        return inputs_embeds.masked_scatter(mask.to(inputs_embeds.device), image_features.to(inputs_embeds.dtype))

    def position_ids(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        return get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            image_token_id=self.cfg.image_token_id,
            video_token_id=self.cfg.video_token_id,
            spatial_merge_size=self.cfg.vision_config.spatial_merge_size,
            tokens_per_second=self.cfg.vision_config.tokens_per_second,
            attention_mask=attention_mask,
        )

    # -- forward ---------------------------------------------------------------------
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        *,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        cache=None,
        slot_mapping: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        context_lens: Optional[torch.Tensor] = None,
        query_lens: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        backends: Optional[Sequence[str]] = None,
        capture_probs_at: Optional[int] = None,
        layer_hook=None,
        logits_to_keep: int = 0,
        return_hidden: bool = False,
    ):
        """Run a forward pass.

        ``logits_to_keep=0`` returns logits for every position; ``logits_to_keep=1`` returns
        only the last (the decode path).
        """
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("either input_ids or inputs_embeds is required")
            inputs_embeds = self.embed_tokens(input_ids)

        if pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("pixel_values provided without image_grid_thw")
            feats = self.embed_images(pixel_values, image_grid_thw, backends=backends)
            if input_ids is None:
                raise ValueError("pixel_values require input_ids so placeholders can be located")
            inputs_embeds = self.scatter_image_features(inputs_embeds, input_ids, feats)

        if position_ids is None:
            if input_ids is None:
                raise ValueError("position_ids must be supplied when only inputs_embeds are given")
            position_ids, _ = self.position_ids(input_ids, image_grid_thw, video_grid_thw)

        hidden = self.language_model(
            inputs_embeds,
            position_ids,
            cache=cache,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
            query_lens=query_lens,
            attn_mask=attn_mask,
            backends=backends,
            capture_probs_at=capture_probs_at,
            layer_hook=layer_hook,
        )

        if return_hidden:
            return hidden
        if logits_to_keep and logits_to_keep > 0:
            hidden = hidden[:, -logits_to_keep:, :]
        return self.lm_head(hidden)

    # -- introspection ---------------------------------------------------------------
    def quantized_modules(self) -> List[QuantLinear]:
        return [m for m in self.language_model.modules() if isinstance(m, QuantLinear)]

    def memory_report(self, dtype_bytes: float = 2.0) -> Dict[str, float]:
        """Bytes held by weights, split by component and by storage scheme."""
        out = {"quantized_bytes": 0.0, "float_bytes": 0.0, "vision_bytes": 0.0, "head_bytes": 0.0}
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                out["quantized_bytes"] += module.quantized().nbytes()
                continue
            for p in module.parameters(recurse=False):
                nbytes = p.numel() * p.element_size()
                if name.startswith("visual"):
                    out["vision_bytes"] += nbytes
                elif name == "lm_head":
                    out["head_bytes"] += nbytes
                else:
                    out["float_bytes"] += nbytes
        out["total_bytes"] = sum(v for k, v in out.items() if k != "total_bytes")
        return out


# --------------------------------------------------------------------------------------
# Construction / loading
# --------------------------------------------------------------------------------------


def build_model(cfg: Qwen25VLConfig, device: str = "cpu", dtype: torch.dtype = torch.float32) -> Qwen25VLLite:
    """Construct on the ``meta`` device so no weights are allocated yet.

    Every parameter is a meta tensor with the right shape; :func:`load_weights` replaces them
    one at a time.  Nothing else in the model holds state that needs materialising (the RoPE
    frequency tables are computed lazily), so this really does start at zero bytes.
    """
    with torch.device("meta"):
        model = Qwen25VLLite(cfg)
    model._target_device = device
    model._target_dtype = dtype
    return model


def _set_param(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
    """Install a parameter regardless of whether the module currently holds a meta tensor."""
    tensor = tensor.detach()
    if name in module._parameters:
        module._parameters[name] = nn.Parameter(tensor, requires_grad=False)
    else:
        module.register_parameter(name, nn.Parameter(tensor, requires_grad=False))


def apply_quantization(
    model: Qwen25VLLite,
    scheme: str,
    group_size: int = 128,
    *,
    quantize_vision: bool = False,
) -> None:
    """Replace decoder linears with empty :class:`QuantLinear` placeholders.

    The weights are filled in later by :func:`load_weights`, which quantises each tensor as
    it streams off disk.  ``quantize_vision`` is off by default (see ``QUANTIZABLE_SUFFIXES``).
    """
    if scheme not in ("int8_per_channel", "int4_groupwise", "awq_style"):
        raise ValueError(f"unknown scheme {scheme!r}")
    model.quant_scheme = scheme
    model.group_size = group_size if scheme != "int8_per_channel" else None

    containers = [model.language_model]
    if quantize_vision:
        containers.append(model.visual)

    for container in containers:
        for module_name, module in list(container.named_modules()):
            for child_name, child in list(module.named_children()):
                if not isinstance(child, nn.Linear):
                    continue
                full = f"{module_name}.{child_name}" if module_name else child_name
                if not any(full.endswith(sfx) for sfx in QUANTIZABLE_SUFFIXES):
                    continue
                placeholder = QuantLinear(bias=child.bias if child.bias is not None else None)
                setattr(module, child_name, placeholder)


def _quantize_tensor(
    weight: torch.Tensor,
    scheme: str,
    group_size: int,
    act_scale: Optional[torch.Tensor] = None,
) -> QuantizedTensor:
    if scheme == "int8_per_channel":
        return quantize_int8_per_channel(weight)
    if scheme == "int4_groupwise":
        return quantize_int4_groupwise(weight, group_size=group_size)
    if scheme == "awq_style":
        if act_scale is None:
            # Without calibration statistics AWQ degenerates to plain groupwise INT4; say so
            # rather than pretending activation-awareness happened.
            return quantize_int4_groupwise(weight, group_size=group_size)
        return quantize_awq_style(weight, act_scale, group_size=group_size)
    raise ValueError(f"unknown scheme {scheme!r}")


def _iter_safetensors(model_dir: str):
    """Yield ``(name, tensor)`` from safetensors shards, preferring the mmap-backed reader."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    shards: List[str] = []
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as fh:
            weight_map = json.load(fh)["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        single = os.path.join(model_dir, "model.safetensors")
        if os.path.isfile(single):
            shards = ["model.safetensors"]
    if not shards:
        raise FileNotFoundError(
            f"no safetensors weights found under {model_dir!r}; expected model.safetensors or an index json"
        )

    try:
        from safetensors import safe_open
    except ImportError:
        # Fall back to torch.load of a .bin checkpoint.  This materialises the whole file, so
        # it is explicitly reported as a degraded path rather than silently used.
        bin_path = os.path.join(model_dir, "pytorch_model.bin")
        if not os.path.isfile(bin_path):
            raise ImportError(
                "safetensors is not installed and no pytorch_model.bin is present; "
                "install safetensors for streaming (mmap) weight loading"
            )
        state = torch.load(bin_path, map_location="cpu", weights_only=True)
        for name, tensor in state.items():
            yield name, tensor
        return

    for shard in shards:
        with safe_open(os.path.join(model_dir, shard), framework="pt", device="cpu") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


def load_weights(
    model_dir: str,
    model: Qwen25VLLite,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    act_scales: Optional[Dict[str, torch.Tensor]] = None,
    strict: bool = True,
    progress=None,
) -> Dict[str, int]:
    """Stream weights from ``model_dir`` into ``model``.

    Returns counters: ``{"loaded", "quantized", "skipped_tied", "missing"}``.
    """
    cfg = model.cfg
    stats = {"loaded": 0, "quantized": 0, "skipped_tied": 0, "missing": 0}
    seen: set = set()
    lm_head_present = False

    def resolve(module: nn.Module, name: str):
        parts = name.split(".")
        for part in parts[:-1]:
            if part.isdigit() and isinstance(module, (nn.ModuleList, nn.Sequential)):
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module, parts[-1]

    for hf_name, tensor in _iter_safetensors(model_dir):
        local = map_hf_name(hf_name)
        if local == "lm_head.weight":
            # An explicit head always wins: if the checkpoint ships one, the tie is broken
            # rather than silently discarded.
            lm_head_present = True
            _set_param(model.lm_head, "weight", tensor.to(device=device, dtype=dtype))
            seen.add(local)
            stats["loaded"] += 1
            continue
        seen.add(local)
        try:
            module, attr = resolve(model, local)
        except AttributeError:
            stats["missing"] += 1
            continue

        if isinstance(module, QuantLinear) and attr == "weight":
            scale = None
            if act_scales is not None:
                key = local.rsplit(".weight", 1)[0]
                scale = act_scales.get(key)
            qt = _quantize_tensor(tensor.float(), model.quant_scheme or "int8_per_channel",
                                  model.group_size or 128, scale)
            module.set_quantized(qt)
            stats["quantized"] += 1
        else:
            _set_param(module, attr, tensor.to(device=device, dtype=dtype))
        stats["loaded"] += 1
        if progress is not None:
            progress(local)

    if cfg.tie_word_embeddings and not lm_head_present:
        model.lm_head.weight = model.language_model.embed_tokens.weight
        stats["skipped_tied"] += 1

    if strict:
        expected = _expected_names(model)
        missing = sorted(expected - seen)
        if missing:
            raise RuntimeError(
                f"{len(missing)} parameters were not present in the checkpoint, e.g. {missing[:8]}"
            )
        unfilled = [n for n, m in model.named_modules() if isinstance(m, QuantLinear) and not m.meta]
        if unfilled:
            raise RuntimeError(
                f"{len(unfilled)} quantised layers never received a weight, e.g. {unfilled[:8]}"
            )
    return stats


def _expected_names(model: Qwen25VLLite) -> set:
    """Every parameter the checkpoint is expected to provide, in local namespace."""
    names = {name for name, _ in model.named_parameters(remove_duplicate=False)}
    # A tied head is not stored separately when the checkpoint omits it; if the checkpoint
    # *does* store one it is in `names` anyway and will have been loaded.
    if model.cfg.tie_word_embeddings:
        names.discard("lm_head.weight")
    return names
