# MiniMax-H3 在摩尔线程 MUSA（8 × MTT S5000）上的训练启动脚本与无损优化说明

本目录提供 **MiniMax-H3（33B 音视频联合生成 DiT）** 在摩尔线程 **MTT S5000（80 GB × 8，PCIe 拓扑）** 上的两阶段微调启动脚本、分布式配置以及无损训练优化说明。

---

## 1. 目录结构与脚本清单

```text
examples/minimax_h3/model_training/musa/
├── README.md                          # 本文档：快速启动 + 优化原理 + 消融实验数据
├── train_musa.py                      # MUSA 训练入口包装器（torchada / 视觉塔对齐 / FSDP bf16 分片保持 / 步级计时）
├── musa_compat.py                     # Qwen3-VL 视觉塔与 F.embedding 设备对齐补丁
├── accelerate_config_zero3_8gpu.yaml  # 8 卡 DeepSpeed ZeRO-3 配置
├── accelerate_config_fsdp_8gpu.yaml   # 8 卡 PyTorch FSDP 配置（开启 forward_prefetch + BACKWARD_PRE）
├── stage1_data_process.sh             # 阶段一：数据预处理（支持单卡或 8 卡数据并行）
├── stage2_zero3_56f_t2va.sh           # 阶段二：8 卡 ZeRO-3 56 帧 T2VA 训练（稳态 5.65 s/it）
├── stage2_zero3_124f_fl2va.sh         # 阶段二：8 卡 ZeRO-3 124 帧 FL2VA 全长首尾帧训练（稳态 15.32 s/it）
└── stage2_fsdp_8gpu.sh                # 阶段二：8 卡 FSDP 56 帧 / 124 帧训练（稳态 6.00s / 16.04s，保存仅 1.8 s）
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

### 2.2 阶段二：8 卡 LoRA 微调训练（ZeRO-3 或 FSDP）

```bash
# 方案 A：8 卡 DeepSpeed ZeRO-3（56 帧 T2VA，稳态 5.65 s/it）
bash examples/minimax_h3/model_training/musa/stage2_zero3_56f_t2va.sh

# 方案 B：8 卡 DeepSpeed ZeRO-3（124 帧 FL2VA 全长 ~15k token，稳态 15.32 s/it）
bash examples/minimax_h3/model_training/musa/stage2_zero3_124f_fl2va.sh

# 方案 C：8 卡 PyTorch FSDP（保存仅 1.8 s，比 ZeRO-3 快 11 倍）
FRAMES=56  bash examples/minimax_h3/model_training/musa/stage2_fsdp_8gpu.sh
FRAMES=124 bash examples/minimax_h3/model_training/musa/stage2_fsdp_8gpu.sh
```

所有脚本产出的 LoRA 权重均为标准 `epoch-0.safetensors`（`296 MiB` / `416` 个 `bfloat16` 张量），可直接通过 `examples/minimax_h3/model_training/validate_lora/MiniMax-H3-FL2VA.py` 加载推理。

---

## 3. 核心优化点详解（无损 bf16 精度）

本次优化在不改变模型结构与 `bfloat16` 数学精度的前提下，从 **算子层、重计算策略、内存分配器、分布式框架** 四个维度消除了训练瓶颈，核心改动拆分为 4 个独立 commit：

### 3.1 消除每步 50+ 次 GPU→CPU 同步阻塞 + 跳过 Padding 段无效 SDPA（Commit `b7315f5a`）
- **问题定位**：
  1. `diffsynth/models/minimax_h3_dit.py` 中的 `_sdpa_varlen_attention` 在每个 DiT Block 内调用 `cu_seqlens.tolist()`，每步训练触发 **52 次**（开启梯度检查点重算则达 **102 次**）GPU→CPU 同步，打断异步 kernel 发射并阻碍通信与计算重叠。
  2. MiniMax-H3 序列打包采用 `[0, used, aligned_seq_len]` 布局，末尾 `[used:aligned_seq_len]` 为对齐填充（`token_tags == -1`），其输出在 `MiniMaxH3DiT.forward` 末尾被 `index_select` 直接丢弃，但原实现仍对该段启动第 2 次 SDPA 前向与反向计算。
- **优化方案**：
  - 在 `MiniMaxH3Unit_SequencePacker` 及 `model_fn_minimax_h3` 中直接透传 CPU 侧 `cu_seqlens_bounds: tuple[int, ...]`，实现训练全程 **零 D2H 同步**；
  - 对 `[0, used, aligned_seq_len]` 布局，仅对有效前缀 `[:used]` 执行 SDPA，末尾填充段直接置零，每层省去一次 SDPA 前反向开销。

### 3.2 接入 MUSA 原生融合算子 `_fused_swiglu` 与 `torch.rope`（Commit `7c0eb3ac`）
- **问题定位**：
  1. `MiniMaxH3MLP` 原实现为 `gate, up = hidden.chunk(2, dim=-1); F.silu(gate) * up`，在 124 帧（`S=14,976, 2×FFN=28,672`）下 50 层前反向耗时 `555.8 ms/step`，且产生多份大张量中间激活；
  2. `_apply_rope` 在 50 个 Block 中对 Q、K 重复计算 `torch.cos(freqs)` / `torch.sin(freqs)`（每步重复 100 次 fp32 三角函数 + 多次 `chunk/cat`），50 层前反向耗时 `525.0 ms/step`。
- **优化方案**：
  - 运行时自动探测 `torch.ops.aten._fused_swiglu_forward / _fused_swiglu_backward`，将 `MiniMaxH3MLP` 激活融合为单算子，50 层耗时从 `555.8 ms` 降至 **`132.7 ms`（提速 4.19×）**；
  - 运行时探测原生 `torch.rope`，对前 96 维旋转切片直接调用融合旋转位置编码（回退路径则在 `MiniMaxH3DiT.forward` 入口一次性预计算 `(cos, sin)` 复用到 50 层），50 层耗时从 `525.0 ms` 降至 **`271.0 ms`（提速 1.94×）**。
- **无损精度验证**：
  - 固定随机种子与输入对拍：单步 Loss 相对原版差异仅 **`4.84e-5`**（`2.720516` vs `2.720564`），全部 LoRA 梯度张量最大绝对误差 `2.53e-7`、最小余弦相似度 **`0.999941`**。

### 3.3 选择性梯度检查点（Selective Activation Checkpointing）与缓存标志覆盖修复（Commit `aa4513ce`）
- **问题定位**：
  1. 原 `examples/minimax_h3/model_training/train.py` 在 Stage 2 从磁盘缓存读取 `inputs_shared` 时，直接沿用了 Stage 1 缓存字典里保存的 `use_gradient_checkpointing_offload` 旧值，导致 Stage 2 命令行参数无法覆盖；
  2. 8 卡分片后每卡静态权重 + 优化器仅占 `~14–17 GiB`，80 GB 显存空置超过 `60 GiB`，却对全部 50 层无差别做重计算（多算一整遍 50 层 Forward）甚至通过 PCIe 卸载激活到 CPU。
- **优化方案**：
  - 修复 `MiniMaxH3TrainingModule.forward`：从缓存加载时始终用当前训练进程的梯度检查点配置覆盖 `inputs_shared`；
  - 新增 `--gradient_checkpointing_interval k`：当 `k > 1` 时每隔 `k` 层跳过梯度检查点（直接保留该层激活，省去反向时的重算 Forward）：
    - **56 帧 T2VA（单层未检查点激活 `2.02 GB`）**：设置 `interval=2`（25/50 层免重算），峰值显存 `58.7 GiB`，稳态再省 `~0.51 s/it`；
    - **124 帧 FL2VA（单层未检查点激活 `4.53 GB`）**：设置 `interval=9`（6/50 层免重算），峰值显存 `49.6 GiB`（安全低于 DeepSpeed 分配器 flush 阈值），稳态再省 `~0.41 s/it`。

### 3.4 消除 MUSA `expandable_segments:True` 在 ZeRO-3 下的虚拟内存重映射抖动
- **问题定位**：
  - 在 6 步消融实验中发现：若在 DeepSpeed ZeRO-3 下设置 `PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True`，当峰值显存超过 50 GiB 时，ZeRO-3 每层动态申请/释放参数 `all_gather` 缓冲会频繁触发 MUSA 驱动的物理页重映射与 `pytorch allocator cache flushes`，导致 56 帧稳态从 `5.65 s/it` 劣化至 `10.32 s/it`，124 帧从 `15.32 s/it` 劣化至 `38.36 s/it`。
- **优化方案**：
  - 在 ZeRO-3 启动脚本中显式 `unset PYTORCH_MUSA_ALLOC_CONF` 并去掉冗余 `--use_gradient_checkpointing_offload` 与 `--find_unused_parameters`，稳态步时抖动完全消失（方差 `< 0.15 s`）。

### 3.5 FSDP `bfloat16` 分片保持 + `forward_prefetch` + 可训练分片轻量保存（Commit `4b273d95`）
- **问题定位**：
  1. `accelerate.prepare_model` 在 `mixed_precision: bf16` 下检测到含 LoRA 的 `MiniMaxH3DiTBlock` 的 `_flat_param.requires_grad=True`，会将整块 `FlatParameter`（其中 99.5% 为冻结基座，共 33B 参数）全部 upcast 为 `float32`，不仅静态显存翻倍，还在每次 `all_gather` 前产生额外转换开销；
  2. 原 `ModelLogger.save_model` 在 FSDP 下调用 `accelerator.get_state_dict(model)`，触发 130 GB `fp32` 全模型 unshard 到 Rank 0 CPU（耗时 ~95 s），导致非 0 Rank 提前退出撕裂 MCCL 通信域，且 `unwrap_model` 残留内层 `_fsdp_wrapped_module` 前缀导致丢掉 400 个主 Block LoRA 张量。
- **优化方案**：
  - 在 `train_musa.py` 中保持 51 个 FSDP `FlatParameter` 为 `bfloat16`，并在 `accelerate_config_fsdp_8gpu.yaml` 中开启 `fsdp_forward_prefetch: true` 与 `limit_all_gathers: true`；
  - 在 `diffsynth/diffusion/logger.py` 中原生实现 `_gather_fsdp_trainable_state_dict`：保存时仅 `all_gather` `requires_grad=True` 的 LoRA 分片（总通信量从 `130 GB` 降至 `~310 MB`），自动清理 `_fsdp_wrapped_module` 前缀并补齐分布式屏障，使 8 卡 FSDP 保存耗时从崩溃/95s 降至 **`1.8 s`（比 ZeRO-3 快 11 倍）**。

---

## 4. 消融实验与性能总表（8 × MTT S5000 80 GB）

### 4.1 单卡 50 层 `MiniMaxH3DiTBlock + LoRA` 纯算力理论下界（无多卡通信）

| 序列规格 | 算子版本 | 单层 fwd | 单层 bwd | 50 层无检查点 | 50 层 `interval=2` | 50 层全检查点 | 单层未检查点激活 |
|---|---|---|---|---|---|---|---|
| **56 帧 T2VA (`S=6,656`)** | 优化前 | 26.32 ms | 42.73 ms | 3.45 s | 4.11 s | 4.77 s | 2.02 GB |
| **56 帧 T2VA (`S=6,656`)** | **优化后** | **23.95 ms** | **38.91 ms** | **3.14 s** | **3.74 s** | **4.34 s (−9.0%)** | 2.02 GB |
| **124 帧 FL2VA (`S=14,976`)** | 优化前 | 64.99 ms | 116.00 ms | 9.05 s | 10.67 s | 12.30 s | 4.53 GB |
| **124 帧 FL2VA (`S=14,976`)** | **优化后** | **60.78 ms** | **108.26 ms** | **8.45 s** | **9.97 s** | **11.49 s (−6.6%)** | 4.53 GB |

### 4.2 8 卡端到端训练 6 步同口径消融对比

| 基准负载 | 训练方案 | 首步耗时 | 稳态步时（第 2~6 步均） | 最快单步 | 6 步总均 | 保存 ckpt | 峰值显存 |
|---|---|---|---|---|---|---|---|
| **56 帧 T2VA** | 原 ZeRO-3 基线（offload + `expandable_segments` + `int=1`） | 10.16 s | 7.55 s/it | 7.37 s | 7.99 s/it | 20.9 s | 13.7 GiB |
| **56 帧 T2VA** | `+` 算子融合 + 去 offload + 关 `expandable_segments` (`int=1`) | 9.69 s | 6.16 s/it | 5.97 s | 6.75 s/it | 20.2 s | 17.1 GiB |
| **56 帧 T2VA** | **`+` Selective AC `interval=2` (`stage2_zero3_56f_t2va.sh`)** | **9.38 s** | **5.65 s/it (−25.2%)** | **5.48 s** | **6.27 s/it** | 20.9 s | 58.7 GiB |
| **56 帧 T2VA** | **8 卡 FSDP 优化版 (`FRAMES=56 stage2_fsdp_8gpu.sh`, `int=2`)** | 20.30 s | **6.00 s/it (−20.0%)** | **6.00 s** | 8.39 s/it | **1.8 s (11×)** | 67.0 GiB |
| **124 帧 FL2VA** | 原 ZeRO-3 基线（offload + `expandable_segments` + `int=1`） | 24.77 s | 18.74 s/it | 18.44 s | 19.74 s/it | 20.4 s | 17.3 GiB |
| **124 帧 FL2VA** | `+` 算子融合 + 去 offload + 关 `expandable_segments` (`int=1`) | 19.63 s | 15.73 s/it | 15.25 s | 16.38 s/it | 20.4 s | 25.4 GiB |
| **124 帧 FL2VA** | **`+` Selective AC `interval=9` + `FusedAdamW` (`stage2_zero3_124f_fl2va.sh`)** | **19.96 s** | **15.32 s/it (−18.2%)** | **15.04 s** | **16.25 s/it** | 21.1 s | 49.6 GiB |
| **124 帧 FL2VA** | **8 卡 FSDP 优化版 (`FRAMES=124 stage2_fsdp_8gpu.sh`, `int=5`)** | 30.11 s | **16.04 s/it** | **15.96 s** | 18.38 s/it | **1.8 s (11×)** | 75.7 GiB |
