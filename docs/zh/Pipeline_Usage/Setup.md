# 安装依赖

从源码安装（推荐）：

```
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

从 pypi 安装（存在版本更新延迟，如需使用最新功能，请从源码安装）

```
pip install diffsynth
```

为保证框架的轻量化，基础安装选项只会安装必要的依赖包，我们提供了一些额外的安装选项：

* `[audio]`: 用于音频模型的支持，例如 ACE-Step、MiniMax-Music3 等
* `[quant]`: 用于参数量化，启用 NF4、INT8、NVFP4 等精度
* `[training]`: 用于分布式大规模预训练
* `[logger]`: 用于启用 TensorBoard、SwanLab 等训练日志记录器
* `[npu]`: 用于 x86 架构的 Ascend NPU 设备
* `[npu_aarch64]`: 用于 aarch64/ARM 架构的 Ascend NPU 设备
* 特定模型的依赖
   * `[controlnet]`: 用于 `examples/flux` 和 `examples/qwen_image` 下 ControlNet 示例所需的 ControlNet 注解器（Canny、深度图、OpenPose 等）
   * `[infiniteyou]`: https://arxiv.org/abs/2503.16418
   * `[ses]`: https://arxiv.org/abs/2602.03208
   * `[nexusgen]`: https://arxiv.org/pdf/2504.21356
* `[all]`: 包含除“特定模型的依赖”以外的所有依赖

你可以使用命令 `pip install -e ".[audio,quant]"` 或 `pip install diffsynth[audio,quant]` 来安装多组依赖包。

## GPU/NPU 支持

### NVIDIA GPU

按照以上方式安装即可。

### AMD GPU

需安装支持 ROCm 的 `torch` 包，以 ROCm 6.4（本文更新于 2025 年 12 月 15 日）、Linux 系统为例，请运行以下命令

```shell
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
```

### Apple Silicon

在 Apple Silicon 设备上，无需修改安装步骤。但由于显存和内存是统一的，请将代码中的 `"cuda"` 全部修改为 `"mps"` 或 `"cpu"`。

### Ascend NPU

1. 通过官方文档安装 [CANN](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)

2. 从源码安装
   ```shell
   git clone https://github.com/modelscope/DiffSynth-Studio.git
   cd DiffSynth-Studio
   # aarch64/ARM
   pip install -e .[npu_aarch64] 
   # x86
   pip install -e .[npu] --extra-index-url "https://download.pytorch.org/whl/cpu"
   ```

使用 Ascend NPU 时，请将 Python 代码中的 `"cuda"` 改为 `"npu"`，详见[NPU 支持](../Pipeline_Usage/GPU_support.md#ascend-npu)。

### Moore Threads MUSA GPU

在摩尔线程（Moore Threads）MUSA GPU 环境下（基于预装 `torch` 与 `torch_musa` 2.7.x 的官方 `training-suite:v2.1.7` 镜像），请按以下步骤完成依赖配置：

1. 安装系统 FFmpeg 共享库（启用 `torchaudio` 的原生 FFmpeg 音视频解码后端，用于从 `.mp4` 训练视频中直接读取音轨；先使用 `dpkg-divert` 避开 `mthreads` 容器运行时只读挂载的宿主机驱动文件冲突）：
   ```shell
   mount | awk '$3 ~ "^/usr/lib/x86_64-linux-gnu/" {print $3}' | xargs -r -I{} dpkg-divert --local --no-rename --divert {}.pkg {}
   apt-get update && apt-get install -y --no-install-recommends \
       libavutil56 libavcodec58 libavformat58 libavfilter7 libavdevice58
   ```

2. 安装 `torchada` 适配层与 `DiffSynth-Studio`（会自动约束适配 `torch 2.7` 的 `peft>=0.18.1,<0.19.0` 与原生支持 `mccl` 的 `accelerate>=1.12.0`）：
   ```shell
   git clone https://github.com/modelscope/DiffSynth-Studio.git
   cd DiffSynth-Studio
   pip install torchada "torchcodec==0.5" --index-url https://pypi.org/simple
   pip install -e . --index-url https://pypi.org/simple
   ```

3. 安装或升级 MUSA `FlashAttention-3` 与 `MATE` 算子库（需 `>=0.2.8.post1`）：
   ```shell
   # 安装 MATE（含 TileLang 反向后处理依赖）与可选 MATE-MUBIN
   pip install -U "mate[all]" --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple --extra-index-url https://pypi.org/simple
   pip install -U mate-mubin --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple

   # 安装 FlashAttention-3 接口包
   pip install -U flash_attn_3 --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple
   ```

使用 MUSA GPU 运行推理与分布式训练（FSDP / DeepSpeed ZeRO-3）的详细启动方式，请参见 [MUSA GPU 支持](../Pipeline_Usage/GPU_support.md#moore-threads-musa-gpu)。

## 其他安装问题

如果在安装过程中遇到问题，可能是由上游依赖包导致的，请参考这些包的文档：

* [torch](https://pytorch.org/get-started/locally/)
* [Ascend/pytorch](https://github.com/Ascend/pytorch)
* [sentencepiece](https://github.com/google/sentencepiece)
* [cmake](https://cmake.org)
