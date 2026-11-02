# MiniMax-H3 在摩尔线程 MUSA（8 × MTT S5000）上的训练启动脚本与深度无损优化说明

本目录提供 **MiniMax-H3（33B 音视频联合生成 DiT）** 在摩尔线程 **MTT S5000（80 GB × 8，14-Link MTLink 全互联）** 上的两阶段微调启动脚本、分布式配置以及两轮无损训练优化详解与消融实验数据。

---

## 0. 核心成果速览（8 × MTT S5000 80 GB，6 步同口径实测）

| 训练场景 | 初始基线（ZeRO-3） | 第一轮优化后 | **第二轮深度优化后（当前）** | **累计提速 / 关键突破** |
|---|---|---|---|---|
| **56 帧 T2VA（Stage 2）** | 稳态 `7.55 s/it`（均 `7.99s`），保存 `20.9s` | ZeRO-3 `5.65 s/it` / FSDP `6.00 s/it` | **FSDP 稳态 `3.78 s/it`（均 `5.90s`，保存 `1.3s`）**<br>ZeRO-3 稳态 `5.44 s/it`（最快 `5.22s`） | **稳态提速 2.00×（`7.55s → 3.78s`）**，保存提速 **16×** |
| **124 帧 FL2VA 全长首尾帧+音频（Stage 2 LoRA）** | 稳态 `18.74 s/it`（均 `19.74s`），保存 `20.4s` | ZeRO-3 `15.32 s/it` / FSDP `16.04 s/it` | **FSDP 稳态 `11.74 s/it`（均 `13.85s`，保存 `1.3s`）**<br>ZeRO-3 稳态 `14.75 s/it`（最快 `14.55s`） | **稳态提速 1.60×（`18.74s → 11.74s`）**，保存提速 **15.7×** |
| **33B 全参数微调（Full-Param SFT，535 张量）** | 124 帧 ZeRO-3 直接 OOM（需开 CPU offload `51.43 s/it`） | — | **8 卡 FSDP 纯显存内全参训练：**<br>• **56 帧 T2VA 稳态 `9.21 s/it`**（保存 62 GiB `58.4s`）<br>• **124 帧 FL2VA 稳态 `17.85 s/it`**（保存 62 GiB `58.6s`） | **8 卡 80 GB 免 CPU Offload 跑通 33B 全参 SFT**（124 帧比 ZeRO-3 CPU Offload **快 2.88×**） |
| **124 帧 FL2VA 数据预处理（Stage 1）** | 单卡 `20.8 s/样本`（不含加载） | 8 卡 DP `6.64 s/样本`（稳态 `~2.25 s/样本`） | **8 卡 DP 稳态 `~2.25 s/样本`** | 稳态吞吐提升 **~9.2×** |
| **50 层 `MiniMaxH3DiTBlock + LoRA` 单卡纯算力** | 56f `4.77 s` / 124f `12.30 s`（全检查点） | 56f `4.34 s` / 124f `11.49 s` | **56f `3.69 s`（无检查点 `2.64 s`）**<br>**124f `10.05 s`（无检查点 `7.29 s`）** | 单层算子耗时再降 **15%–20%**，单层保存激活降 **27%** |

---

## 1. 目录结构与脚本清单

```text
examples/minimax_h3/model_training/musa/
├── README.md                          # 本文档：快速启动 + 两轮优化原理 + 消融实验数据
├── train_musa.py                      # MUSA 训练入口包装器（torchada / FSDP 冻结基座与 LoRA 分离 / bf16 分片 / 步级计时）
├── musa_compat.py                     # Qwen3-VL 视觉塔对齐 + mate flash_attention_3 4D backward 修复补丁
├── accelerate_config_fsdp_8gpu.yaml   # 8 卡 PyTorch FSDP 配置（推荐首选：最快训练 + 1.3s 保存）
├── accelerate_config_zero3_8gpu.yaml  # 8 卡 DeepSpeed ZeRO-3 配置
├── stage1_data_process.sh             # 阶段一：数据预处理（支持单卡或 8 卡数据并行）
├── stage2_fsdp_8gpu.sh                # ★ 阶段二 LoRA 首选：8 卡 FSDP（56 帧稳态 3.78 s/it，124 帧稳态 11.74 s/it，保存 1.3 s）
├── stage2_zero3_56f_t2va.sh           # 阶段二 LoRA：8 卡 ZeRO-3 56 帧 T2VA 训练（稳态 5.44 s/it）
├── stage2_zero3_124f_fl2va.sh         # 阶段二 LoRA：8 卡 ZeRO-3 124 帧 FL2VA 全长首尾帧训练（稳态 14.75 s/it）
├── stage2_full_fsdp_8gpu.sh           # ★ 阶段二 33B 全参 SFT 首选：8 卡 FSDP（56 帧 9.21 s/it，124 帧 17.85 s/it，免 CPU offload）
├── stage2_full_zero3_8gpu.sh          # 阶段二 33B 全参 SFT：8 卡 ZeRO-3（56 帧 11.04 s/it；124 帧需 CPU offload 51.43 s/it）
├── build_av_stability_dataset.py      # 构建 16 条（1 条真实 + 15 条多模式合成）音视频长稳基准数据集
└── run_stability_test.sh              # 一键长稳与收敛验证（自动构建数据 + 8 卡 Stage 1 + 8 卡 FSDP 训练 + 固定锚点评估导出 CSV）
```

---

## 2. 快速开始

如果权重存放在本地自定义目录（例如 `/data/yanami/models`），启动前可导出：
```bash
export DIFFSYNTH_MODEL_BASE_PATH=/data/yanami/models
export DIFFSYNTH_SKIP_DOWNLOAD=true
```

### 2.1 阶段一：数据预处理（编码视频/音频 Latent 与文本/首尾帧 Embedding）

```bash
# 56 帧 T2VA（单卡，16.4 s/样本）
FRAMES=56 EXTRA_INPUTS="input_audio" NUM_PROCESSES=1 \
  bash examples/minimax_h3/model_training/musa/stage1_data_process.sh

# 124 帧 FL2VA 首尾帧 + 音频（8 卡数据并行，稳态 ~2.25 s/样本）
FRAMES=124 EXTRA_INPUTS="input_audio,input_image,end_image" NUM_PROCESSES=8 DATASET_REPEAT=16 \
  bash examples/minimax_h3/model_training/musa/stage1_data_process.sh
```

### 2.2 阶段二：8 卡 LoRA 微调训练（推荐 FSDP）

```bash
# ★ 推荐首选 1：8 卡 FSDP — 56 帧 T2VA（稳态 3.78 s/it，保存 1.3 s）
FRAMES=56  bash examples/minimax_h3/model_training/musa/stage2_fsdp_8gpu.sh

# ★ 推荐首选 2：8 卡 FSDP — 124 帧 FL2VA 全长首尾帧+音频（稳态 11.74 s/it，保存 1.3 s）
FRAMES=124 bash examples/minimax_h3/model_training/musa/stage2_fsdp_8gpu.sh

# 备选：8 卡 DeepSpeed ZeRO-3（56 帧稳态 5.44 s/it；124 帧稳态 14.75 s/it）
bash examples/minimax_h3/model_training/musa/stage2_zero3_56f_t2va.sh
bash examples/minimax_h3/model_training/musa/stage2_zero3_124f_fl2va.sh
```

### 2.3 阶段二：8 卡 33B 全参数微调（Full-Parameter SFT，`--trainable_models "dit"`）

```bash
# ★ 推荐首选：8 卡 FSDP 纯显存内全参 SFT（无需 CPU optimizer offload，产出 62 GiB / 535 张量完整 ckpt）
FRAMES=56  bash examples/minimax_h3/model_training/musa/stage2_full_fsdp_8gpu.sh   # 稳态 9.21 s/it，保存 58.4 s
FRAMES=124 bash examples/minimax_h3/model_training/musa/stage2_full_fsdp_8gpu.sh   # 稳态 17.85 s/it，保存 58.6 s

# 备选：8 卡 DeepSpeed ZeRO-3 全参 SFT（56 帧显存内 11.04 s/it；124 帧自动启用 CPU optimizer offload 51.43 s/it）
FRAMES=56  bash examples/minimax_h3/model_training/musa/stage2_full_zero3_8gpu.sh
FRAMES=124 bash examples/minimax_h3/model_training/musa/stage2_full_zero3_8gpu.sh
```

### 2.4 可选：开启 SwanLab 可视化看板（支持在线同步与无密钥/离线自动降级）

所有阶段二训练脚本与 `run_stability_test.sh` 均内置了**可选的 SwanLab 可视化支持**（默认不开启，不影响任何原有脚本行为；即使开启但未配置 API Key 或外网不通，也会自动降级为 `offline` 本地落盘模式，**绝不中断训练**）：

```bash
# 方式 1：已通过 `swanlab login` 保存过凭证时，只需设 ENABLE_SWANLAB=1
ENABLE_SWANLAB=1 SWANLAB_RUN_NAME="fsdp-56f-run1" \
  FRAMES=56 bash examples/minimax_h3/model_training/musa/stage2_fsdp_8gpu.sh

# 方式 2：直接通过环境变量传入 SWANLAB_API_KEY（无需预先 login）
SWANLAB_API_KEY="<YOUR_SWANLAB_API_KEY>" \
  SWANLAB_PROJECT_NAME="MiniMax-H3-MUSA" \
  SWANLAB_RUN_NAME="stability-56f" \
  FRAMES=56 DATASET_REPEAT=50 TRAIN_EVAL_EVERY=10 \
  bash examples/minimax_h3/model_training/musa/run_stability_test.sh

# 方式 3：无外网 / 无 API Key 环境下仅记录本地 SwanLab 日志（事后可 `swanlab sync`）
ENABLE_SWANLAB=1 SWANLAB_MODE=offline \
  FRAMES=56 bash examples/minimax_h3/model_training/musa/stage2_fsdp_8gpu.sh
```

开启后会自动向 SwanLab 上报以下指标曲线：
- `loss`（每步原始训练 Loss）、`train/ema_loss`（平滑 Loss）、`train/grad_norm`（LoRA 梯度二范数）；
- `train/step_time_s`（单步耗时）、`train/alloc_gib` / `train/resv_gib`（显存监控）；
- `eval/fixed_anchor_mean_loss` 及 `eval/fixed_anchor_t200` / `t500` / `t800`（当启用 `TRAIN_EVAL_EVERY` 时）。

---

## 3. 核心优化点详解（全部验证无损 bf16 精度）

### 3.1 消除每步 50+ 次 GPU→CPU 同步阻塞 + 跳过 Padding 段无效 SDPA（Commit `b7315f5a`）
- **问题定位**：
  1. `diffsynth/models/minimax_h3_dit.py` 的 `_sdpa_varlen_attention` 在每个 Block 内调用 `cu_seqlens.tolist()`，每步触发 **52 次**（含重算则 **102 次**）GPU→CPU 同步，打断异步 kernel 发射与通信重叠；
  2. `[0, used, aligned_seq_len]` 布局中末尾对齐填充段 `[used:aligned_seq_len]` 的输出最终被 `index_select` 丢弃，但原实现仍对该段启动第 2 次 SDPA 前反向。
- **优化方案**：在 `MiniMaxH3Unit_SequencePacker` 与 `model_fn_minimax_h3` 中透传 CPU 侧 `cu_seqlens_bounds` tuple（全程 **零 D2H 同步**），并对末尾填充段直接置零跳过 SDPA。

### 3.2 接入 MUSA 原生融合算子 `_fused_swiglu` 与 `torch.rope`（Commit `7c0eb3ac`）
- **SwiGLU**：运行时探测 `torch.ops.aten._fused_swiglu_forward / _backward`，将 124 帧下 50 层 MLP 激活耗时从 `555.8 ms` 降至 **`132.7 ms`（4.19×）**；
- **3D RoPE**：运行时探测原生 `torch.rope` 作用于前 96 维旋转切片（消除每步 100 次冗余 fp32 `cos/sin` 计算），50 层耗时从 `525.0 ms` 降至 **`271.0 ms`（1.94×）**。

### 3.3 消除非连续 `qkv[:,:,0,:]` 切片导致的 `nn.RMSNorm` 反向 4.1× 退化
- **问题定位**：
  - `MiniMaxH3Attention.forward` 原代码通过 `q = qkv[:, :, 0, :]; k = qkv[:, :, 1, :]` 取出步长非连续（`stride=(21504, 384, 1)`）的视图直接送入 `self.q_norm` / `self.k_norm`。
  - 实测在 MUSA 上，非连续张量触发 `nn.RMSNorm` 反向慢路径：124 帧下 50 层 `q_norm + k_norm` 反向耗时高达 **`674.5 ms`**（连续张量仅需 **`163.8 ms`，相差 4.1 倍**）！
- **优化方案**：
  - 改为 `q, k, v = qkv.permute(2, 0, 1, 3).contiguous().unbind(0)`，一次轻量重排同时产出连续的 `q, k, v`，50 层 `q_norm + k_norm + RoPE` 总耗时从 `1268.6 ms` 降至 **`694.8 ms`（省 574 ms/step）**，且前反向数值与原版 **逐比特 100% 一致（`max_diff = 0.0`）**。

### 3.4 Triton 融合 AdaLN 调制算子（`_FusedAdaLNScaleShiftFunction` / `_FusedAdaLNGateFunction`）
- **问题定位**：
  - 原 `_modulate_scale_shift` 与 `_modulate_gate` 通过 `scale.index_select(0, indices)` 将 `[9, 5376]` 的调制表展开为 `[S, 5376]` 的巨型张量（124 帧下每次分配 `161 MB`），且 PyTorch Autograd 为计算 `dscale` 与 `dgate` 在每个 Block 保存了 4 份 `[S, 5376]` 张量（每层白占 **`644 MB`** 激活显存，50 层达 `32 GB`），而在 LoRA 微调中 `adaln_proj` 是冻结的（`requires_grad=False`）！
- **优化方案**：
  - 编写 stride-aware Triton 融合算子：在寄存器内按行读取 `indices[row]` 并完成 `x * (1 + scale) + shift` 与 `x + gate * other`，不物化任何 `[S, 5376]` 展开张量；当 `adaln_proj` 冻结时不保存 `x` / `other` 副本（保存激活从 `644 MB/层` 降至 **`0 MB/层`**），使单层未检查点总激活从 `4.53 GiB` 降至 `3.06 GiB`（124 帧） / `2.02 GiB` 降至 `1.36 GiB`（56 帧），50 层耗时从 `303.6 ms` 降至 **`152.1 ms`（2.0×）**。

### 3.5 FSDP 冻结基座与可训练 LoRA 彻底分离 + 自定义 `_FusedLoRALinearFunction`
- **问题定位（通信与反向双瓶颈）**：
  1. 在 `use_orig_params=True` 下，由于每个 `MiniMaxH3DiTBlock`（`1.30 GB` 冻结基座 + `6 MB` LoRA）混装在同一个 `FlatParameter` 中，`_flat_param.requires_grad` 为 `True`。结果 FSDP 在每层反向时不仅要分配 **1.30 GB 的未分片梯度缓冲**，还要在 `_post_backward_hook` 中对整块 **1.30 GB（99.5% 全是无需梯度的冻结权重！）执行 `reduce_scatter_tensor`** —— 每步多传 **65 GB** 无效梯度通信！
  2. 此外，PEFT 默认 `LoraLinear.forward` 经过多层 Python 包装与张量加和，未融合 `addmm_`。
- **优化方案**：
  1. **分离冻结基座与 LoRA（`train_musa.py` 默认开启 `FSDP_SEPARATE_LORA=1`）**：在 `Accelerator.prepare_model` 中将 416 个 `lora_A` / `lora_B` 模块加入 `fsdp_plugin.ignored_modules`，使 50 个 `MiniMaxH3DiTBlock` 的 `FlatParameter` 变为纯冻结（`requires_grad=False`，保持 `bfloat16`）。FSDP 由此 **完全跳过 50 × 1.30 GB = 65 GB/step 的无效梯度分配与 `reduce_scatter`**！而全部 416 个可训练 LoRA 参数（共 155M 参数，`fp32` 仅 `591.5 MiB`）在各卡常驻 `fp32` 主权重，反向结束后通过单次扁平化 `dist.all_reduce(flat_grad, op=AVG)` 在 MTLink（`220 GB/s` 带宽）上仅耗时 **~3 ms** 完成同步。
  2. **融合 `_FusedLoRALinearFunction`（`diffsynth/diffusion/training_module.py`）**：将 `base_layer` 与 `lora_A / lora_B` 融合为基于 `torch.mm` + 原地 `out.addmm_(xa, lora_B.T, alpha=scaling)` 的自定义 `autograd.Function`，省去对冻结基座求 `dW_base` 的分支判断与中间结果分配。

### 3.6 双向选择性梯度检查点（支持正/负 `--gradient_checkpointing_interval`）
- **增强设计（Commit `aa4513ce` + 扩展）**：
  - `--gradient_checkpointing_interval k` 支持正数与负数两种语义：
    - `k == 1`（默认）：全部 50 层开启梯度检查点；
    - `k > 1`：**每隔 `k` 层跳过检查点**（保留 `1/k` 的 Block 激活在显存中，适合 124 帧长序列，如 `k=4` 保留 13/50 层，峰值显存 `70.4 GiB`）；
    - `k < -1`：**仅每隔 `|k|` 层开启检查点**（保留 `(|k|-1)/|k|` 的 Block 激活在显存中，适合 56 帧中等序列，如 `k=-5` 保留 **40/50 层（80%）免重算**，仅检查点 10 层，峰值显存 `73.3 GiB`，单步稳态直接压至 **`3.78 s/it`**！）。

### 3.7 关闭 ZeRO-3 下的 `PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True`
- **避坑结论**：在 DeepSpeed ZeRO-3 下开启 `expandable_segments:True` 会与 ZeRO-3 每层动态申请/释放参数缓冲的模式冲突，触发 MUSA 虚拟页重映射与 `allocator cache flushes`（使 124 帧稳态从 `14.75 s/it` 劣化至 `38.36 s/it`）。所有 ZeRO-3 脚本均已显式 `unset PYTORCH_MUSA_ALLOC_CONF`。

### 3.8 修复 `mate` 库 `flash_attention_3` 4D 非 varlen 反向崩溃（`musa_compat._patch_mate_fa3_4d_backward`）
- **根因澄清**：
  - 早期记录曾认为 `mate` 的 varlen 反向整体损坏。经源码核查发现：**真正的 3D `flash_attn_varlen_func(cu_seqlens_q != None)` 在 `mate` 中前向与反向均完全正常**（`mubin` 后端产出 2D `softmax_lse [H, total_Q]`，`lse_varlen_to_padded` 正常转为 3D `[B, H, max_Q]`）。
  - 报错只发生在 **4D 非 varlen 接口 `flash_attn_interface.flash_attn_func(q, k, v)`**（DiffSynth 的 `attention_forward` 所调入口）：`flash_attn_3/interface.py:72` 将 4D `[B, S, H, D]` 透传给 `mate_flash_attn_varlen_func(..., cu_seqlens_q=None)`；`FlashAttnVarlenFunc.forward` 的 `else` 分支（`is_varlen=False`）已直接产出 **3D** `softmax_lse [B, H, S]`，但 `FlashAttnVarlenFunc.backward` 漏写了 `cu_seqlens_q is None` 分支，无条件调用只接受 2D 输入的 `lse_varlen_to_padded(softmax_lse, None, None)`，导致 `H, total_Q = lse_flat.shape` 抛出 `ValueError: too many values to unpack (expected 2)`。
- **补丁修复**：
  - 在 `musa_compat.py` 中对 `FlashAttnVarlenFunc.backward` 增加 `cu_seqlens_q is None and q.dim() == 4` 分支处理（直接复用 3D `softmax_lse` 并构造等长 `cu_seqlens` 调用 `_flash_attn_varlen_backward`）。修复后 `DIFFSYNTH_ATTENTION_IMPLEMENTATION=flash_attention_3` 在训练反向中 100% 通过，与 `flash_attention_2` / `torch`（SDPA）数值一致且速度相同（8 卡 FSDP 56 帧稳态同为 `3.78 s/it`）。

---

## 4. 精度无损验证（`tools/train_opt_parity_check.py`）

在 MTT S5000 上固定随机种子与输入，开启全部算子优化（无同步 `cu_seqlens_bounds` + Padding SDPA 跳过 + MUSA `_fused_swiglu` + 原生 `torch.rope` + QKV 连续 `unbind` + Triton AdaLN + `_FusedLoRALinearFunction`）与未优化原始参考实现逐项对拍：

| 检查项 | 实测结果 | 判定 |
|---|---|---|
| `permute(2,0,1,3).contiguous().unbind(0)` vs 原版 `qkv[:,:,0,:]` | 前向 `max_diff = 0.0`，反向梯度 `max_diff = 0.0` | **逐比特完全一致** |
| Triton AdaLN（`fp32` 精度下）vs 原版 `index_select` | 前向 `max_diff = 0.0`（`cos = 0.99999994`），反向 `max_diff = 0.0` | **数学等价** |
| 全模型（含全部 7 项算子优化，`bf16`）单步 Loss 对拍 | `loss_ref = 2.720632` vs `loss_opt = 2.720966`（相对误差 **`1.2e-4`**） | **无损（bf16 舍入量级）** |
| 全模型 48 个 LoRA 梯度张量对拍（`interval=1 / 2 / 0`） | 最大绝对误差 **`4.77e-7`**，最小余弦相似度 **`0.999901`** | **梯度高度一致** |
| 产物 Checkpoint 完整性与 8 卡 TP 推理闭环 | `416` 个 `bfloat16` 张量，`NaN/Inf = 0`；挂载 8 卡 TP 生成 124 帧音视频 `infer_s = 46.9s` 通过校验 | **训推闭环通过** |

---

## 5. 完整消融实验总表（8 × MTT S5000 80 GB）

### 5.1 单卡 50 层 `MiniMaxH3DiTBlock + LoRA` 纯算力演进（无通信下界）

| 序列规格 | 版本 | 单层 fwd | 单层 bwd | 50 层无检查点 | 50 层 `interval=2` | 50 层全检查点 | 单层保存激活 |
|---|---|---|---|---|---|---|---|
| **56 帧 (`S=6,656`)** | 原始基线 | 26.32 ms | 42.73 ms | 3.45 s | 4.11 s | 4.77 s | 2.015 GiB |
| **56 帧 (`S=6,656`)** | 第一轮（SwiGLU + RoPE + Padding 跳过） | 23.95 ms | 38.91 ms | 3.14 s | 3.74 s | 4.34 s | 1.626 GiB |
| **56 帧 (`S=6,656`)** | **第二轮（+ QKV unbind + Triton AdaLN + Fused LoRA）** | **21.11 ms** | **31.62 ms** | **2.64 s** | **3.16 s** | **3.69 s (−22.6%)** | **1.360 GiB (−32.5%)** |
| **124 帧 (`S=14,976`)** | 原始基线 | 64.99 ms | 116.00 ms | 9.05 s | 10.67 s | 12.30 s | 4.534 GiB |
| **124 帧 (`S=14,976`)** | 第一轮（SwiGLU + RoPE + Padding 跳过） | 60.78 ms | 108.26 ms | 8.45 s | 9.97 s | 11.49 s | 3.061 GiB |
| **124 帧 (`S=14,976`)** | **第二轮（+ QKV unbind + Triton AdaLN + Fused LoRA）** | **55.07 ms** | **90.80 ms** | **7.29 s** | **8.67 s** | **10.05 s (−18.3%)** | **3.061 GiB (−32.5%)** |

### 5.2 8 卡端到端训练 6 步同口径消融对比

| 基准负载 | 训练方案 | 首步耗时 | 稳态步时（第 2~6 步均） | 最快单步 | 6 步总均 | 保存 ckpt | 峰值显存 |
|---|---|---|---|---|---|---|---|
| **56 帧 T2VA** | 初始 ZeRO-3 基线（offload + `expandable_segments` + `int=1`） | 10.16 s | 7.55 s/it | 7.37 s | 7.99 s/it | 20.9 s | 13.7 GiB |
| **56 帧 T2VA** | 第一轮 ZeRO-3（去 offload + 关 `expandable_segments` + `int=2`） | 9.38 s | 5.65 s/it | 5.48 s | 6.27 s/it | 20.9 s | 58.7 GiB |
| **56 帧 T2VA** | **第二轮 ZeRO-3（+ QKV unbind + Triton AdaLN，`stage2_zero3_56f_t2va.sh`）** | 9.85 s | **5.44 s/it (−27.9%)** | **5.22 s** | 6.17 s/it | 21.1 s | 51.6 GiB |
| **56 帧 T2VA** | 第一轮 FSDP（bf16 混装 FlatParam + prefetch + `int=2`） | 20.30 s | 6.00 s/it | 6.00 s | 8.39 s/it | 1.8 s | 67.0 GiB |
| **56 帧 T2VA** | 第二轮 FSDP（+ LoRA 分离消 65GB RS + 新算子，`int=2`） | 16.66 s | 4.11 s/it | 4.11 s | 6.20 s/it | 1.3 s | 53.7 GiB |
| **56 帧 T2VA** | **★ 第二轮 FSDP（`int=-5` 80% 层免重算，`FRAMES=56 stage2_fsdp_8gpu.sh`）** | **16.50 s** | **3.78 s/it (−49.9%，2.00×）** | **3.78 s** | **5.90 s/it** | **1.3 s (16×)** | 73.3 GiB |
| **124 帧 FL2VA** | 初始 ZeRO-3 基线（offload + `expandable_segments` + `int=1`） | 24.77 s | 18.74 s/it | 18.44 s | 19.74 s/it | 20.4 s | 17.3 GiB |
| **124 帧 FL2VA** | 第一轮 ZeRO-3（去 offload + 关 `expandable_segments` + `int=9`） | 20.86 s | 15.32 s/it | 15.04 s | 16.25 s/it | 21.1 s | 49.6 GiB |
| **124 帧 FL2VA** | **第二轮 ZeRO-3（+ 新算子 + `int=7`，`stage2_zero3_124f_fl2va.sh`）** | **19.77 s** | **14.75 s/it (−21.3%)** | **14.55 s** | **15.59 s/it** | 23.9 s | 48.3 GiB |
| **124 帧 FL2VA** | 第一轮 FSDP（bf16 混装 FlatParam + prefetch + `int=5`） | 30.11 s | 16.04 s/it | 15.96 s | 18.38 s/it | 1.8 s | 75.7 GiB |
| **124 帧 FL2VA** | **★ 第二轮 FSDP（+ LoRA 分离消 65GB RS + 新算子 + `int=4`，`FRAMES=124 stage2_fsdp_8gpu.sh`）** | **24.41 s** | **11.74 s/it (−37.4%，1.60×）** | **11.72 s** | **13.85 s/it** | **1.3 s (15.7×)** | 70.4 GiB |

### 5.3 8 卡 33B 全参数微调（Full-Parameter SFT，535 张量 / 62 GiB）实测对比

| 基准负载 | 训练方案 | 首步耗时 | 稳态步时（第 2~4 步均） | 最快单步 | 保存 62 GiB ckpt | 峰值显存 | 说明 |
|---|---|---|---|---|---|---|---|
| **56 帧 T2VA** | 8 卡 ZeRO-3（无 CPU Offload，`int=1`） | 18.61 s | 11.04 s/it | 11.00 s | 61.8 s | 70.8 GiB | fp32 主权重 + m/v 占 ~58.7 GiB/卡 |
| **56 帧 T2VA** | 8 卡 FSDP（`bf16` 分片 + `FusedAdamW`，`int=1`） | 30.50 s | 9.59 s/it | 9.58 s | 61.2 s | 39.3 GiB | 静态仅 ~32.6 GiB/卡，空余 >40 GiB 显存 |
| **56 帧 T2VA** | **★ 8 卡 FSDP（`bf16` 分片 + `int=2`，`FRAMES=56 stage2_full_fsdp_8gpu.sh`）** | **29.71 s** | **9.21 s/it** | **9.16 s** | **58.4 s** | 75.7 GiB | 25/50 层免重算，**比 ZeRO-3 快 16.6%** |
| **124 帧 FL2VA** | 8 卡 ZeRO-3（无 CPU Offload，`int=1`） | — | ❌ OOM | — | — | >79.9 GiB | 58.7 GiB 静态 + ~25 GiB 激活超 80 GB |
| **124 帧 FL2VA** | 8 卡 ZeRO-3（开启 `offload_optimizer_device: cpu`，`int=1`） | 68.71 s | 51.43 s/it | 50.04 s | 61.8 s | 25.3 GiB | 受限于 PCIe D2H/H2D 优化器更新 |
| **124 帧 FL2VA** | 8 卡 FSDP（`bf16` 分片 + `FusedAdamW`，`int=1`，**免 CPU Offload**） | 32.60 s | 18.31 s/it | 18.28 s | 55.4 s | 43.6 GiB | **纯显存内跑通 124 帧全参 SFT** |
| **124 帧 FL2VA** | **★ 8 卡 FSDP（`bf16` 分片 + `int=7`，`FRAMES=124 stage2_full_fsdp_8gpu.sh`）** | **33.71 s** | **17.85 s/it** | **17.81 s** | **58.6 s** | 71.8 GiB | 8/50 层免重算，**比 ZeRO-3 CPU Offload 快 2.88×** |

> **全参 SFT 训推闭环校验**：8 卡 FSDP 124 帧全参 SFT 产出的 `epoch-0.safetensors`（`62 GiB`）包含 `535` 个 `bfloat16` 张量，与原始 13 分片 `FL2VA/transformer/model*.safetensors` 的键名集合 **100% 匹配（535/535）**；直接通过 8 卡 TP 加载该 62 GiB 全参 checkpoint 推理生成 124 帧 / 50 步音视频（`tp8_full_sft_124f.mp4`），`load_s = 53.4 s`、`infer_s = 49.3 s`，经 `probe_av.py` 校验视频与立体声音轨完全正常。

---

## 6. 长时间训练稳定性与 Loss 收敛验证（16 样本混合音视频基准集）

### 6.1 数据集与双轨评估方法
- **16 样本混合音视频基准集（`build_av_stability_dataset.py`）**：
  - `sample_00_real.mp4`：来自 `MiniMax-H3-FL2VA` 的真实人物英语说话音视频（124 帧 `832×480` 24fps + 32kHz 立体声）；
  - `sample_01..15_*.mp4`：15 条涵盖不同背景/色块、8 种运动轨迹（水平、垂直、对角、圆周轨道、正余弦振荡、李萨如曲线等）、15 组立体声双频和弦（220 Hz ~ 880 Hz）以及不同长度 Prompt 的音视频样本，用于同时压测多样本 DataLoader 随机洗牌与动态 `cu_seqlens` 序列长度下的显存分配稳定性。
- **确定性固定锚点评估（Fixed-`(t, noise)` Anchor Evaluation，`TRAIN_EVAL_EVERY=N`）**：
  - 由于 Flow-Matching 训练每步随机采样 `timestep ~ U(0, 1000)` 和高斯噪声 `noise`，不同时间步的天然损失尺度相差 5~10 倍；
  - `train_musa.py` 内置无侵入评估钩子：每隔 `N` 步在完全固定的随机种子（`seed_v=12345, seed_a=67890`）与三个代表性时间步 **`t_idx ∈ {200, 500, 800}`** 上执行 8 卡 `no_grad` 全量同步评估，并将每步 `step_time_s, raw_loss, ema_loss, grad_norm, alloc_gib, resv_gib, fixed_eval_mean, fixed_eval_t200/t500/t800` 导出至 `<OUTPUT_PATH>/stability_metrics.csv`。

一键复现命令：
```bash
# 100 步 56 帧 T2VA 长稳与收敛测试（每 10 步做一次固定锚点评估）
FRAMES=56 DATASET_REPEAT=50 TRAIN_EVAL_EVERY=10 \
  bash examples/minimax_h3/model_training/musa/run_stability_test.sh

# 40 步 124 帧 FL2VA 全长首尾帧+音频长稳与收敛测试（每 8 步做一次固定锚点评估）
FRAMES=124 DATASET_REPEAT=20 TRAIN_EVAL_EVERY=8 \
  bash examples/minimax_h3/model_training/musa/run_stability_test.sh
```

### 6.2 实测结果 1：56 帧 T2VA 100 步长稳实验（8 卡 FSDP，`CKPT_INTERVAL=-5`）
- **步时与显存稳定性（Steps 2..100）**：
  - 稳态步时均值 **`3.80 s/it`**（最小 **`3.78 s`**，最大 `4.21 s`，99 步内无任何性能衰减）；
  - 每步结束显存 `alloc_gib` 从第 10 步到第 100 步恒定为 **`9.451 GiB`**，`resv_gib` 恒定为 **`76.047 GiB`（连续 90 步波动 `0.000 GiB`，零显存泄漏）**；
  - 全程 100 步 `NaN/Inf = 0`，梯度范数 `grad_norm` 稳定在 `0.020 ~ 0.134`。
- **Loss 收敛曲线（固定锚点 `mean_loss` 下降 `-66.3%`，其中 `t=500` 严格单调下降 `-76.0%`）**：

| Step | 单步耗时 `step_s` | 训练滑动均值 `ema_loss` | 梯度范数 `grad_norm` | 常驻显存 `alloc` / `resv` | **固定锚点总均 `FIXED_EVAL`** | `t=200` (低噪) | `t=500` (中噪) | `t=800` (高噪) |
|---|---|---|---|---|---|---|---|---|
| **1** | 16.44 s | 0.1494 | 0.0273 | 9.452 / 74.480 GiB | **0.1903 (基线)** | 0.2817 | 0.1294 | 0.1598 |
| **10** | 3.80 s | 0.1529 | 0.0509 | 9.451 / 76.047 GiB | **0.1386 (−27.2%)** | 0.1935 | 0.0864 | 0.1360 |
| **20** | 3.79 s | 0.1297 | 0.0449 | 9.451 / 76.047 GiB | **0.0932 (−51.0%)** | 0.1224 | 0.0613 | 0.0959 |
| **30** | 3.79 s | 0.1081 | 0.0284 | 9.451 / 76.047 GiB | **0.0856 (−55.0%)** | 0.1139 | 0.0546 | 0.0883 |
| **40** | 4.10 s | 0.0906 | 0.0517 | 9.451 / 76.047 GiB | **0.0736 (−61.3%)** | 0.0885 | 0.0466 | 0.0858 |
| **50** | 3.80 s | 0.0845 | 0.1338 | 9.451 / 76.047 GiB | **0.0644 (−66.2%)** | 0.0700 | 0.0429 | 0.0804 |
| **70** | 3.79 s | 0.0857 | 0.0290 | 9.451 / 76.047 GiB | **0.0632 (−66.8%)** | 0.0763 | 0.0369 | 0.0765 |
| **90** | 4.21 s | 0.0820 | 0.0335 | 9.451 / 76.047 GiB | **0.0618 (−67.5%)** | 0.0784 | 0.0325 | 0.0746 |
| **100** | 3.79 s | **0.0834 (−44.2%)** | 0.0205 | **9.451 / 76.047 GiB** | **0.0642 (−66.3%)** | **0.0905 (−67.9%)** | **0.0310 (−76.0%)** | **0.0712 (−55.4%)** |

### 6.3 实测结果 2：124 帧 FL2VA 全长首尾帧+音频 40 步长稳实验（8 卡 FSDP，`CKPT_INTERVAL=4`）
- **步时与显存稳定性（Steps 2..40）**：
  - 稳态步时均值 **`11.76 s/it`**（最小 **`11.65 s`**，最大 `12.12 s`）；
  - 每步结束显存 `alloc_gib` 全程 40 步恒定为 **`9.489 GiB`**，`resv_gib` 稳定在 **`72.643 ~ 72.645 GiB`**（峰值分配 `70.40 GiB`，零泄漏）；
- **Loss 收敛曲线（固定锚点 `mean_loss` 每次评估严格单调下降，40 步累计下降 `-38.2%`）**：

| Step | 单步耗时 `step_s` | 训练滑动均值 `ema_loss` | 梯度范数 `grad_norm` | 常驻显存 `alloc` / `resv` | **固定锚点总均 `FIXED_EVAL`** | `t=200` (低噪) | `t=500` (中噪) | `t=800` (高噪) |
|---|---|---|---|---|---|---|---|---|
| **1** | 24.05 s | 0.4397 | 0.0311 | 9.489 / 71.725 GiB | **0.4572 (基线)** | 0.2068 | 0.3960 | 0.7687 |
| **8** | 11.76 s | 0.4214 | 0.0777 | 9.489 / 72.643 GiB | **0.4185 (−8.5%)** | 0.1830 | 0.3746 | 0.6979 |
| **16** | 12.12 s | 0.3764 | 0.0637 | 9.489 / 72.643 GiB | **0.3531 (−22.8%)** | 0.1417 | 0.3397 | 0.5779 |
| **24** | 11.73 s | 0.3218 | 0.1058 | 9.489 / 72.643 GiB | **0.3215 (−29.7%)** | 0.1143 | 0.3245 | 0.5257 |
| **32** | 11.96 s | 0.2902 | 0.0807 | 9.489 / 72.645 GiB | **0.3004 (−34.3%)** | 0.1082 | 0.3018 | 0.4913 |
| **40** | 11.73 s | **0.2579 (−41.3%)** | 0.0717 | **9.489 / 72.645 GiB** | **0.2827 (−38.2%)** | **0.1042 (−49.6%)** | **0.2924 (−26.2%)** | **0.4517 (−41.2%)** |


