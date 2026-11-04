# Installing Dependencies

Install from source (recommended):

```
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

Install from PyPI (there may be delays in version updates; for latest features, install from source):

```
pip install diffsynth
```

To keep the framework lightweight, the base installation only installs the necessary dependencies. We provide some additional installation options:

* `[audio]`: Support for audio models, e.g., ACE-Step, MiniMax-Music3, etc.
* `[quant]`: For parameter quantization, enabling precisions such as NF4, INT8, NVFP4.
* `[training]`: For distributed large-scale pretraining.
* `[logger]`: To enable training loggers such as TensorBoard, SwanLab, etc.
* `[npu]`: For Ascend NPU devices with x86 architecture.
* `[npu_aarch64]`: For Ascend NPU devices with aarch64/ARM architecture.
* Dependencies of specific models
   * `[controlnet]`: For the ControlNet annotator detectors (Canny, depth, OpenPose, etc.) used by the ControlNet examples under `examples/flux` and `examples/qwen_image`.
   * `[infiniteyou]`: https://arxiv.org/abs/2503.16418
   * `[ses]`: https://arxiv.org/abs/2602.03208
   * `[nexusgen]`: https://arxiv.org/pdf/2504.21356
* `[all]`: Includes all dependencies except the "dependencies of specific models" above.

You can install multiple sets of dependencies with `pip install -e ".[audio,quant]"` or `pip install diffsynth[audio,quant]`.

## GPU/NPU Support

### NVIDIA GPU

Install as described above.

### AMD GPU

You need to install the `torch` package with ROCm support. Taking ROCm 6.4 (as of the article update date: December 15, 2025) on Linux as an example, run the following command:

```shell
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
```

### Apple Silicon

On Apple Silicon devices, no changes to the installation steps are needed. However, since VRAM and memory are unified, replace all `"cuda"` in the code with `"mps"` or `"cpu"`.

### Ascend NPU

1. Install [CANN](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit) through official documentation.

2. Install from source
   ```shell
   git clone https://github.com/modelscope/DiffSynth-Studio.git
   cd DiffSynth-Studio
   # aarch64/ARM
   pip install -e .[npu_aarch64] 
   # x86
   pip install -e .[npu] --extra-index-url "https://download.pytorch.org/whl/cpu"
   ```
When using Ascend NPU, please replace `"cuda"` with `"npu"` in your Python code. For details, see [NPU Support](../Pipeline_Usage/GPU_support.md#ascend-npu).

### Moore Threads MUSA GPU

On Moore Threads MUSA GPUs (starting from the official `training-suite:v2.1.7` container image pre-installed with `torch` and `torch_musa` 2.7.x), install the dependencies as follows:

1. Install system FFmpeg shared libraries (required by `torchaudio`'s native FFmpeg backend to decode audio tracks directly from `.mp4` videos; divert read-only host-mounted MUSA driver libraries first so `dpkg` does not attempt to overwrite them):
   ```shell
   mount | awk '$3 ~ "^/usr/lib/x86_64-linux-gnu/" {print $3}' | xargs -r -I{} dpkg-divert --local --no-rename --divert {}.pkg {}
   apt-get update && apt-get install -y --no-install-recommends \
       libavutil56 libavcodec58 libavformat58 libavfilter7 libavdevice58
   ```

2. Install the `torchada` adapter and `DiffSynth-Studio` (which automatically pins `peft>=0.18.1,<0.19.0` for `torch 2.7` compatibility and `accelerate>=1.12.0` for native `mccl` backend support):
   ```shell
   git clone https://github.com/modelscope/DiffSynth-Studio.git
   cd DiffSynth-Studio
   pip install torchada "torchcodec==0.5" --index-url https://pypi.org/simple
   pip install -e . --index-url https://pypi.org/simple
   ```

3. Install or upgrade MUSA `FlashAttention-3` and `MATE` operator packages (`>=0.2.8.post1`):
   ```shell
   # Install MATE (with TileLang backward post-processing dependencies) and optional MATE-MUBIN
   pip install -U "mate[all]" --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple --extra-index-url https://pypi.org/simple
   pip install -U mate-mubin --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple

   # Install FlashAttention-3 interface
   pip install -U flash_attn_3 --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple
   ```

For instructions on running inference and multi-GPU distributed training (FSDP / DeepSpeed ZeRO-3) on MUSA GPUs, see [MUSA GPU Support](../Pipeline_Usage/GPU_support.md#moore-threads-musa-gpu).

## Other Installation Issues

If you encounter issues during installation, they may be caused by upstream dependencies. Please refer to the documentation for these packages:

* [torch](https://pytorch.org/get-started/locally/)
* [Ascend/pytorch](https://github.com/Ascend/pytorch)
* [sentencepiece](https://github.com/google/sentencepiece)
* [cmake](https://cmake.org)
