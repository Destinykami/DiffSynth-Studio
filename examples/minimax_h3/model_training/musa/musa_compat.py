"""Device-placement and operator fixes on top of torchada for DiffSynth on MUSA.

Fixes:
1. `Qwen3VLVisionModel.forward` syncs the whole vision tower onto the input device once under VRAM offload.
2. `F.embedding` / `F.linear` keep operands on one device, preferring the accelerator.
3. `mate.mha_interface.FlashAttnVarlenFunc.backward`: fixes `flash_attention_3` non-varlen 4D
   (`cu_seqlens_q=None`) backward crash (`ValueError: too many values to unpack (expected 2)`).
   In `flash_attn_3.interface.flash_attn_func`, 4D `[B, S, H, D]` inputs are forwarded to
   `mate_flash_attn_varlen_func(..., cu_seqlens_q=None)`, where `mubin` forward already produces
   3D padded `softmax_lse` of shape `[B, H, S]`, but `backward` unconditionally called
   `lse_varlen_to_padded` expecting 2D `[H, total_Q]` and `cu_seqlens_q != None`.
"""
import logging

import torch
import torch.nn.functional as F

log = logging.getLogger("musa_compat")

_orig_embedding = F.embedding
_stats = {"embedding": 0, "vision": 0, "mate_fa3_bwd": 0}

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


def _patch_mate_fa3_4d_backward():
    try:
        import mate.mha_interface as mha
    except Exception:
        return
    if getattr(mha.FlashAttnVarlenFunc, "_musa_4d_bwd_patched", False):
        return
    orig_bwd = mha.FlashAttnVarlenFunc.backward

    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        if cu_seqlens_q is None and q.dim() == 4 and softmax_lse.dim() == 3:
            _stats["mate_fa3_bwd"] += 1
            b, sq, hq, d = q.shape
            _, sk, hk, dv = v.shape
            cu_q = torch.arange(0, (b + 1) * sq, step=sq, dtype=torch.int32, device=q.device)
            cu_k = torch.arange(0, (b + 1) * sk, step=sk, dtype=torch.int32, device=k.device)
            dq = torch.empty((b * sq, hq, d), dtype=q.dtype, device=q.device)
            dk = torch.empty((b * sk, hk, d), dtype=k.dtype, device=k.device)
            dv_t = torch.empty((b * sk, hk, dv), dtype=v.dtype, device=v.device)
            dout_flat = dout.reshape(b * sq, hq, dv)
            if d % 8 != 0:
                dout_flat = F.pad(dout_flat, [0, 8 - d % 8])
            mha._flash_attn_varlen_backward(
                dout_flat,
                q.reshape(b * sq, hq, d),
                k.reshape(b * sk, hk, d),
                v.reshape(b * sk, hk, dv),
                out.reshape(b * sq, hq, dv),
                softmax_lse.contiguous(),
                dq,
                dk,
                dv_t,
                cu_q,
                cu_k,
                sq,
                sk,
                dropout_p=0.0,
                softmax_scale=ctx.softmax_scale,
                causal=ctx.causal,
                window_size_left=ctx.window_size[0],
                window_size_right=ctx.window_size[1],
                softcap=ctx.softcap,
                alibi_slopes=None,
                deterministic=ctx.deterministic,
                rng_state=None,
            )
            return (dq.view_as(q), dk.view_as(k), dv_t.view_as(v)) + (None,) * 28
        return orig_bwd(ctx, dout, *args)

    mha.FlashAttnVarlenFunc.backward = staticmethod(backward)
    mha.FlashAttnVarlenFunc._musa_4d_bwd_patched = True


def apply():
    F.embedding = _embedding
    F.linear = _linear
    _patch_qwen3vl_vision()
    _patch_mate_fa3_4d_backward()
    return _stats
