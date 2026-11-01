"""Device-placement fixes on top of torchada for DiffSynth's VRAM offload on MUSA.

DiffSynth keeps parameters on CPU (`offload_device="cpu"`) and moves a module onto the
accelerator inside its VRAM layer wrapper. Qwen3-VL's vision tower is partly offloaded, so
`pos_embed` weight and the text hidden states can end up on different devices.

Two fixes:
1. `Qwen3VLVisionModel.forward` syncs the whole vision tower onto the input device once.
2. `F.embedding` / `F.linear` keep operands on one device, preferring the accelerator.
"""
import logging

import torch
import torch.nn.functional as F

log = logging.getLogger("musa_compat")

_orig_embedding = F.embedding
_stats = {"embedding": 0, "vision": 0}

_orig_linear = F.linear
_stats_linear = {"moved": 0}


def _accelerator_side(a, b):
    """Return the accelerator device if exactly one side is CPU, else `a`."""
    if a.type == "cpu" and b.type != "cpu":
        return b
    return a


def _embedding(weight, input, *args, **kwargs):
    if isinstance(input, torch.Tensor) and input.device != weight.device:
        target = _accelerator_side(weight.device, input.device)
        _stats["embedding"] += 1
        if _stats["embedding"] <= 3:
            log.warning("F.embedding: weight %s / index %s -> %s", weight.device, input.device, target)
        if weight.device != target:
            weight = weight.to(target)
        if input.device != target:
            input = input.to(target)
    return _orig_embedding(weight, input, *args, **kwargs)


def _linear(input, weight, *args, **kwargs):
    if isinstance(input, torch.Tensor) and input.device != weight.device:
        _stats_linear["moved"] += 1
        if _stats_linear["moved"] <= 3:
            log.warning("F.linear: input %s -> weight %s", input.device, weight.device)
        input = input.to(weight.device)
    return _orig_linear(input, weight, *args, **kwargs)


def _patch_qwen3vl_vision():
    try:
        from transformers.models.qwen3_vl import modeling_qwen3_vl as modeling
    except Exception as exc:
        log.warning("qwen3_vl patch skipped: %s", exc)
        return
    cls = modeling.Qwen3VLVisionModel
    orig_forward = cls.forward

    def forward(self, hidden_states, grid_thw=None, **kwargs):
        first = next(self.parameters(), None)
        if torch.is_tensor(hidden_states) and first is not None and first.device != hidden_states.device:
            if _stats["vision"] < 3:
                log.warning("Qwen3VLVisionModel: moving tower %s -> %s", first.device, hidden_states.device)
            _stats["vision"] += 1
            self.to(hidden_states.device)
        return orig_forward(self, hidden_states, grid_thw=grid_thw, **kwargs)

    cls.forward = forward


def apply():
    F.embedding = _embedding
    F.linear = _linear
    _patch_qwen3vl_vision()
    return _stats
