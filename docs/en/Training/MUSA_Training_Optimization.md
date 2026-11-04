# MiniMax-H3 Training Performance Optimization and Precision Verification on Moore Threads MUSA GPUs

This document details the 8 in-framework lossless training optimizations, operator micro-benchmarks, numerical precision audits, and 8-GPU end-to-end ablation studies for **MiniMax-H3 (33B Audio-Video Generation DiT)** on **8 × Moore Threads MTT S5000 (80 GB)** GPUs (with native `flash_attention_3` via `mate>=0.2.8.post1` and `mccl` distributed communication).

All optimizations are integrated directly into the core `DiffSynth-Studio` framework without requiring external wrapper scripts or runtime monkey-patches—running the standard `examples/minimax_h3/model_training/train.py` entrypoint enables them automatically.

---

## 1. Core Optimizations and Implementation Details

### 1.1 Eliminate Per-Layer `cu_seqlens.tolist()` GPU-to-CPU Synchronization (Commit `d8870d71`)
- **Bottleneck**:
  - In `diffsynth/models/minimax_h3_dit.py`, `_sdpa_varlen_attention` previously called `cu_seqlens.tolist()` inside every DiT block, triggering **52 host-device synchronizations per step** (**102 syncs/step** with activation checkpointing recomputation), stalling asynchronous kernel launches and communication-computation overlap.
- **Optimization**:
  - Pass host-side `cu_seqlens_bounds: tuple[int, ...]` directly from `MiniMaxH3Unit_PackedSequenceBuilder` and `model_fn_minimax_h3` (`diffsynth/pipelines/minimax_h3_audio_video.py`), achieving **zero D2H synchronization** throughout training.

### 1.2 Skip Trailing Alignment Padding in FlashAttention-3 (Commit `d8870d71`)
- **Bottleneck**:
  - MiniMax-H3 sequence packing uses a `[0, used, aligned_seq_len]` layout (e.g., `[0, 6630, 6656]` for 56 frames and `[0, 14950, 14976]` for 124 frames). The trailing 26 tokens `[used:aligned_seq_len]` are inert alignment padding (`token_tags == -1`) discarded by the final `index_select` in `MiniMaxH3DiT.forward`, yet the original loop launched a second FlashAttention-3 forward and backward pass on them in every block.
- **Optimization**:
  - Detect the `[0, active_seqlen, total]` layout in `_sdpa_varlen_attention`, execute `attention_forward` only on the active prefix `[:active_seqlen]`, and zero out the trailing padding directly, eliminating 100 redundant FA3 kernel launches per step.

### 1.3 Native Fused `_fused_swiglu` Kernel in `MiniMaxH3MLP` (Commit `c4927061`)
- **Bottleneck**:
  - `MiniMaxH3MLP.forward` previously executed `gate, up = hidden.chunk(2, dim=-1); F.silu(gate) * up`, materializing large intermediate tensors (`[14976, 28672]` at 124 frames) and taking `471.3 ms/step` across 50 blocks.
- **Optimization**:
  - Dispatch `MiniMaxH3MLP` through `torch.ops.aten._fused_swiglu_forward` and `torch.ops.aten._fused_swiglu_backward` via `_FusedSwiGLUFunction`, reducing 50-layer MLP activation time to **`237.1 ms` (1.99× speedup at 124f; 2.27× at 56f)**.

### 1.4 Native Fused `torch.rope` 3D Rotary Position Embedding (Commit `c4927061`)
- **Bottleneck**:
  - `_apply_rope` previously recomputed `torch.cos(freqs)` and `torch.sin(freqs)` in `float32` for both `q` and `k` across all 50 DiT blocks (100 redundant trigonometric evaluations plus multiple `chunk`/`cat` copies per step), taking `413.7 ms/step` at 124 frames.
- **Optimization**:
  - Dispatch the 96-dim rotary slice directly to native `torch.rope` (or precompute `(cos, sin)` once in `MiniMaxH3DiT.forward` on fallback backends), cutting 50-layer RoPE time to **`218.4 ms` (1.89× speedup at 124f; 2.17× at 56f)**.

### 1.5 Contiguous QKV `unbind` Before `q_norm` / `k_norm` (Commit `c4926898`)
- **Bottleneck**:
  - `MiniMaxH3Attention.forward` previously sliced `q = qkv[:, :, 0, :]` and `k = qkv[:, :, 1, :]`, passing non-contiguous views (`stride=(21504, 384, 1)`) into `self.q_norm` and `self.k_norm`. On MUSA, non-contiguous inputs trigger a slow `nn.RMSNorm` backward path (`814.2 ms/step` across 50 layers at 124 frames).
- **Optimization**:
  - Replace the non-contiguous slices with `q, k, v = qkv.permute(2, 0, 1, 3).contiguous().unbind(0)` so `q_norm` and `k_norm` receive contiguous tensors, reducing 50-layer time from `814.2 ms` to **`248.4 ms` (3.28× speedup, saving 565.7 ms/step)** while remaining **100% bit-exact (`max_diff = 0.0`)** in both forward and backward passes.

### 1.6 Bit-Exact Triton Fused AdaLN Modulation Kernels (Commit `c4926898`)
- **Bottleneck**:
  - `_modulate_scale_shift` and `_modulate_gate` previously expanded `[9, 5376]` modulation tables into `[S, 5376]` tensors via `index_select(0, indices)`. Even when `adaln_proj` is frozen during LoRA training (`requires_grad=False`), standard PyTorch autograd saved multiple `[S, 5376]` tensors per uncheckpointed block, wasting **`614.2 MB` per block** (`767.8 MB` vs `153.6 MB`).
- **Optimization**:
  - Implement stride-aware Triton kernels (`_FusedAdaLNScaleShiftFunction` and `_FusedAdaLNGateFunction`) using native `bfloat16` arithmetic matching PyTorch's exact operation order, and save only `(scale/gate, indices)` when `adaln_proj` is frozen.
  - **Benefit**: **2.00×–2.06×** faster AdaLN forward+backward, **614.2 MB (−80%)** less saved activation memory per block, and **100% bit-exact (`max_diff = 0.0`)** forward and backward outputs.

### 1.7 Fused `_FusedLoRALinearFunction` for PEFT LoRA Layers (Commit `f1156136`)
- **Bottleneck**:
  - When LoRA weights are kept in `float32` while activations `x` are in `bfloat16`, standard PEFT `LoraLinear.forward` casts the large input activation `x = x.to(lora_A.weight.dtype)` to `float32` and runs slow `fp32` GEMMs without in-place `addmm_`.
- **Optimization**:
  - Implement `_FusedLoRALinearFunction` in `diffsynth/diffusion/training_module.py` (automatically applied in `add_lora_to_model`): keep master LoRA weights in `float32` for the optimizer, cast the small `lora_A / lora_B` matrices to `bfloat16` during GEMM, and accumulate in-place via `out.addmm_(xa, l_b.t(), beta=1.0, alpha=scaling)`, speeding up 50-layer `qkv_proj` from `1214.9 ms` to **`765.9 ms` (1.59×)**.

### 1.8 FSDP Base/LoRA Separation, Fast Trainable-Only Checkpoint Saving, and Bidirectional Selective Activation Checkpointing (Commits `3f9a5015`, `8c2be8c9`)
- **Bottleneck**:
  1. **65 GB/step Redundant FSDP `reduce_scatter` & `fp32` Bloat**: Under default FSDP1, each `MiniMaxH3DiTBlock` (`1.30 GB` frozen base + `6 MB` trainable LoRA) is flattened into a single `FlatParameter` with `requires_grad=True`. Consequently, Accelerate upcasts all 33B frozen weights to `float32` under `mixed_precision="bf16"`, and FSDP allocates unsharded grad buffers and executes `reduce_scatter_tensor` on **50 × 1.30 GB = 65 GB/step** of frozen weights.
  2. **110s FSDP Checkpoint Saving**: `accelerator.get_state_dict(model)` unshards 130 GB of `fp32` base weights to Rank 0 CPU before filtering out 300 MB of LoRA tensors, taking **107–114 seconds**.
  3. **Full Recompute Despite Idle VRAM**: After separating base and LoRA weights, peak allocated VRAM is only `18–26 GiB` out of `80 GB`, yet all 50 blocks are recomputed during backward.
- **Optimization**:
  1. **FSDP Base/LoRA Separation (`diffsynth/diffusion/runner.py`)**: Automatically add `lora_A / lora_B` to `fsdp_plugin.ignored_modules` during pure LoRA fine-tuning so the 50 frozen base `FlatParameter`s have `requires_grad=False` and stay in `bfloat16` (skipping 65 GB/step of frozen `reduce_scatter`), while synchronizing replicated `fp32` LoRA gradients with a single bucketed `dist.all_reduce(flat_grad, op=AVG)`.
  2. **Fast Trainable-Only Save (`diffsynth/diffusion/logger.py`)**: Gather only `requires_grad=True` parameters in `ModelLogger.save_model`, cutting FSDP save time from **`112s` to `1.78s` (63× faster)**.
  3. **Bidirectional Selective Activation Checkpointing (`--gradient_checkpointing_interval`)**:
     - `1` (default): checkpoint every block;
     - `k > 1`: skip checkpointing every `k`-th block (keeping `1/k` of blocks in VRAM; e.g., `interval=5` for 124f FSDP, `interval=9` for 124f ZeRO-3);
     - `-k` (`k >= 2`): only checkpoint every `k`-th block (keeping `(k-1)/k` of blocks in VRAM; e.g., `interval=-3` for 56f FSDP keeps 66% of blocks uncheckpointed, reaching **`3.79 s/it`**).

---

## 2. Operator Micro-Benchmarks and Numerical Precision Audit

Tested on MTT S5000 with fixed random seed (`seed=42`) across 50 `MiniMaxH3DiTBlock` layers:

| Optimization | 56f (`S=6,656`) 50-Layer Time | 124f (`S=14,976`) 50-Layer Time | Forward Parity (`max_diff` / `cos`) | Backward Grad Parity (`max_diff` / `cos`) | Verdict |
|---|---|---|---|---|---|
| **Opt 1+2**: Zero D2H `cu_seqlens` sync + Skip padding FA3 | `945.7 ms` → **`521.1 ms` (−424.7 ms)** | `2541.9 ms` → **`2304.3 ms` (−237.6 ms)** | `max_diff = 0.0` | `dq/dk/dv cos = 1.0000000` | **Bit-Exact / Identical** |
| **Opt 3**: Native `_fused_swiglu` | `220.6 ms` → **`97.3 ms` (2.27×)** | `471.3 ms` → **`237.1 ms` (1.99×)** | `cos = 0.9999950` | `dx cos = 0.9999955` | **Lossless bf16** |
| **Opt 4**: Native `torch.rope` 3D RoPE | `207.5 ms` → **`95.8 ms` (2.17×)** | `413.7 ms` → **`218.4 ms` (1.89×)** | `max_diff = 0.03125` (1 ULP) | `dq/dk cos = 0.9999964` | **Lossless bf16** |
| **Opt 5**: Contiguous QKV `permute+unbind` | `380.0 ms` → **`131.1 ms` (2.90×)** | `814.2 ms` → **`248.4 ms` (3.28×)** | **`max_diff = 0.0`** | **`max_diff = 0.0`** | **100% Bit-Exact** |
| **Opt 6**: Triton Fused AdaLN (`ScaleShift` + `Gate`) | `149.7 ms` → **`72.5 ms` (2.06×)**<br>*(Saved mem: `341 MB` → `68 MB`/blk)* | `324.1 ms` → **`162.0 ms` (2.00×)**<br>*(Saved mem: `768 MB` → `154 MB`/blk)* | **`max_diff = 0.0`** | **`max_diff = 0.0`** | **100% Bit-Exact** |
| **Opt 7**: `_FusedLoRALinearFunction` (`qkv_proj` × 50) | `554.1 ms` → **`360.0 ms` (1.54×)** | `1214.9 ms` → **`765.9 ms` (1.59×)** | `cos = 0.9999934` | `dx cos = 0.9999920`<br>`dA/dB cos = 0.9999959` | **Lossless bf16** |
| **Combined Full-Block Audit** (Opt 1–7 all enabled vs Ref) | — | — | Single-step Loss rel. diff **`3.90e-5`** | Input `dx cos = 0.9999977`<br>`lora_B` grad `max_diff = 4.49e-9` | **End-to-End Lossless** |

---

## 3. 8-GPU End-to-End Ablation Benchmarks & Loss Trajectories (8 × MTT S5000 80 GB)

Evaluated with identical random seeds (model/LoRA init `seed=42`, per-step noise `seed=1234 + step`) and default `flash_attention_3` (`mate 0.2.8.post1`) across 4 training steps:

### 3.1 8-GPU PyTorch FSDP Progressive Ablation

#### (1) 56-Frame T2VA (`480×832×56`, `S=6,656`)

| Progressive Optimization Stage | Step 1 Time | **Steady-State (Steps 2–4 avg)** | Best Step | **Ckpt Save Time** | Peak Allocated VRAM | Steps 1–4 Loss Trajectory (Fixed Seed) |
|---|---|---|---|---|---|---|
| **0. Baseline** (`fix/fa3-top-level-import`) | `26.84 s` | `7.20 s/it` | `7.15 s` | `112.19 s` | `37.05 GiB` | `[0.589031, 0.141761, 0.621581, 0.615383]` |
| **1. `+` Zero D2H sync & Skip padding FA3** (`d8870d71`) | `26.43 s` | `7.10 s/it` | `7.09 s` | `112.24 s` | `37.05 GiB` | `[0.589031, 0.142773, 0.619632, 0.616861]` |
| **2. `+` Fused `_fused_swiglu` & `torch.rope`** (`c4927061`) | `27.27 s` | `6.76 s/it` | `6.75 s` | `108.50 s` | `36.74 GiB` | `[0.589320, 0.142830, 0.619349, 0.613960]` |
| **3. `+` Contiguous QKV `unbind` & Triton AdaLN** (`c4926898`) | `22.86 s` | `6.50 s/it` | `6.27 s` | `113.89 s` | `36.70 GiB` | `[0.589320, 0.143452, 0.621373, 0.617049]` |
| **4. `+` Fused `_FusedLoRALinearFunction`** (`f1156136`) | `22.11 s` | `6.16 s/it` | `6.03 s` | `110.18 s` | `36.70 GiB` | `[0.589320, 0.143978, 0.619889, 0.615309]` |
| **5. `+` FSDP Base/LoRA Separation & Fast Save** (`3f9a5015`, `int=1`) | `18.44 s` | `4.68 s/it` (−35.0%) | `4.54 s` | **`1.79 s` (62.7×)** | **`18.72 GiB` (−49.5%)** | `[0.589320, 0.145121, 0.620282, 0.614669]` |
| **6a. `+` Selective AC `interval=2`** (`8c2be8c9`) | `17.17 s` | `4.04 s/it` (−43.9%) | `3.97 s` | **`1.78 s` (63.0×)** | `53.76 GiB` | `[0.589320, 0.143843, 0.617449, 0.616538]` |
| **★ 6b. `+` Selective AC `interval=-3`** (`8c2be8c9`) | **`17.10 s`** | **`3.79 s/it` (−47.4%, 1.90×)** | **`3.79 s`** | **`2.11 s`** | `63.57 GiB` | **`[0.589320, 0.143652, 0.620536, 0.614410]`** |

#### (2) 124-Frame FL2VA First/Last Frame + Audio (`480×832×124`, `S=14,976`)

| Progressive Optimization Stage | Step 1 Time | **Steady-State (Steps 2–4 avg)** | Best Step | **Ckpt Save Time** | Peak Allocated VRAM | Steps 1–4 Loss Trajectory (Fixed Seed) |
|---|---|---|---|---|---|---|
| **0. Baseline** (`fix/fa3-top-level-import`) | `37.38 s` | `18.14 s/it` | `18.14 s` | `106.86 s` | `41.43 GiB` | `[1.621342, 0.114358, 0.821202, 1.245091]` |
| **1. `+` Zero D2H sync & Skip padding FA3** (`d8870d71`) | `37.18 s` | `18.05 s/it` | `18.04 s` | `106.72 s` | `41.41 GiB` | `[1.621342, 0.114014, 0.822722, 1.242465]` |
| **2. `+` Fused `_fused_swiglu` & `torch.rope`** (`c4927061`) | `35.92 s` | `17.28 s/it` | `17.27 s` | `106.99 s` | `40.76 GiB` | `[1.622339, 0.114790, 0.823149, 1.245477]` |
| **3. `+` Contiguous QKV `unbind` & Triton AdaLN** (`c4926898`) | `31.35 s` | `15.92 s/it` | `15.82 s` | `110.81 s` | `40.27 GiB` | `[1.622339, 0.114605, 0.821387, 1.243726]` |
| **4. `+` Fused `_FusedLoRALinearFunction`** (`f1156136`) | `30.80 s` | `15.35 s/it` | `15.22 s` | `107.54 s` | `40.27 GiB` | `[1.622339, 0.114198, 0.823836, 1.240772]` |
| **5. `+` FSDP Base/LoRA Separation & Fast Save** (`3f9a5015`, `int=1`) | `25.76 s` | `12.53 s/it` (−30.9%) | `12.43 s` | **`1.84 s` (58.1×)** | **`26.49 GiB` (−36.1%)** | `[1.622339, 0.113733, 0.824336, 1.242391]` |
| **★ 6. `+` Selective AC `interval=5`** (`8c2be8c9`) | **`25.17 s`** | **`11.96 s/it` (−34.1%, 1.52×)** | **`11.80 s`** | **`1.79 s` (59.7×)** | `60.28 GiB` | **`[1.622339, 0.114154, 0.826281, 1.243440]`** |

---

### 3.2 8-GPU DeepSpeed ZeRO-3 Progressive Ablation

| Sequence Spec | Progressive Optimization Stage | Step 1 Time | **Steady-State (Steps 2–4 avg)** | Best Step | Ckpt Save Time | Peak Allocated VRAM | Steps 1–4 Loss Trajectory (Fixed Seed) |
|---|---|---|---|---|---|---|---|
| **56f T2VA** | **0. Baseline** (`fix/fa3-top-level-import`) | `14.18 s` | `6.67 s/it` | `6.45 s` | `20.05 s` | `17.25 GiB` | `[0.587947, 0.139359, 0.620329, 0.617694]` |
| **56f T2VA** | **★ 1–4. `+` All Fused Operators** (`f1156136`, `int=1`) | **`10.04 s`** | **`5.89 s/it` (−11.7%)** | **`5.74 s`** | `24.16 s` | **`16.67 GiB`** | **`[0.588515, 0.140965, 0.617915, 0.615669]`** |
| **124f FL2VA** | **0. Baseline** (`fix/fa3-top-level-import`) | `24.84 s` | `16.22 s/it` | `16.02 s` | `20.15 s` | `25.91 GiB` | `[0.560166, 0.094707, 0.499458, 0.534068]` |
| **124f FL2VA** | **1–4. `+` All Fused Operators** (`f1156136`, `int=1`) | **`20.21 s`** | `14.87 s/it` (−8.3%) | **`13.97 s`** | `24.05 s` | **`24.45 GiB`** | `[0.560116, 0.094670, 0.500470, 0.534425]` |
| **124f FL2VA** | **★ 6. `+` Selective AC `interval=9`** (`8c2be8c9`) | `20.79 s` | **`14.73 s/it` (−9.2%)** | `14.38 s` | `22.10 s` | `44.74 GiB` | **`[0.560116, 0.094854, 0.500816, 0.533867]`** |

---

## 4. Recommended Best-Practice Training Settings

When fine-tuning MiniMax-H3 LoRA on 8 × MTT S5000 (80 GB), **8-GPU PyTorch FSDP** is strongly recommended over DeepSpeed ZeRO-3 for both training throughput and checkpoint saving speed:

1. **Enable `fsdp_forward_prefetch: true` in `fsdp8.yaml`**;
2. **Unset `PYTORCH_MUSA_ALLOC_CONF` (`unset PYTORCH_MUSA_ALLOC_CONF`)** before launching to avoid virtual memory page remapping jitter;
3. **Pass the optimal `--gradient_checkpointing_interval` according to frame count**:
   - **56-frame training (`--num_frames 56`)**: `--use_gradient_checkpointing --gradient_checkpointing_interval -3` (**`3.79 s/it`** steady-state, **`1.8 s`** save);
   - **124-frame training (`--num_frames 124`)**: `--use_gradient_checkpointing --gradient_checkpointing_interval 5` (**`11.96 s/it`** steady-state, **`1.8 s`** save; or keep default `--gradient_checkpointing_interval 1` for `12.53 s/it` at only `26.49 GiB` peak VRAM).
