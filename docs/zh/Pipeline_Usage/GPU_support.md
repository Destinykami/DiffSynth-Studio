# GPU/NPU 支持

`DiffSynth-Studio` 支持多种 GPU/NPU，本文介绍如何在这些设备上运行模型推理和训练。

在开始前，请参考[安装依赖](../Pipeline_Usage/Setup.md)安装好 GPU/NPU 相关的依赖包。

## NVIDIA GPU

本项目提供的所有样例代码默认支持 NVIDIA GPU，无需额外修改。

## AMD GPU

AMD 提供了基于 ROCm 的 torch 包，所以大多数模型无需修改代码即可运行，少数模型由于依赖特定的 cuda 指令无法运行。

### Apple Silicon

在 Apple Silicon 设备上，由于显存和内存是统一的，请将代码中的 `"cuda"` 全部修改为 `"mps"` 或 `"cpu"`。

## Ascend NPU
### 推理
使用 Ascend NPU 时，需把代码中的 `"cuda"` 改为 `"npu"`。

例如，Wan2.1-T2V-1.3B 的推理代码：

```diff
import torch
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.core.device.npu_compatible_device import get_device_name

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
-   "preparing_device": "cuda",
+   "preparing_device": "npu",
    "computation_dtype": torch.bfloat16,
-   "computation_device": "cuda",
+   "computation_device": "npu",
}
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
-   device="cuda",
+   device="npu",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
-   vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
+   vram_limit=torch.npu.mem_get_info(get_device_name())[1] / (1024 ** 3) - 2,
)

video = pipe(
    prompt="纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
)
save_video(video, "video.mp4", fps=15, quality=5)
```

#### USP(Unified Sequence Parallel)
如果想要在NPU上使用该特性，请通过如下方式安装额外的第三方库：
```shell
pip install git+https://github.com/feifeibear/long-context-attention.git
pip install git+https://github.com/xdit-project/xDiT.git
```

### 训练
当前已为每类模型添加NPU的启动脚本样例，脚本存放在`examples/xxx/special/npu_training`目录下，例如 `examples/wanvideo/model_training/special/npu_training/Wan2.2-T2V-A14B-NPU.sh`。

在NPU训练脚本中，添加了可以优化性能的NPU特有环境变量，并针对特定模型开启了相关参数。

#### 环境变量
```shell
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```
`expandable_segments:<value>`: 使能内存池扩展段功能，即虚拟内存特征。

```shell
export CPU_AFFINITY_CONF=1
```
设置0或未设置: 表示不启用绑核功能

1: 表示开启粗粒度绑核

2: 表示开启细粒度绑核

#### 特定模型需要开启的参数
| 模型        | 参数 | 备注                |
|-----------|------|-------------------|
| Wan 14B系列 | --initialize_model_on_cpu | 14B模型需要在cpu上进行初始化 |
| Qwen-Image系列 | --initialize_model_on_cpu | 模型需要在cpu上进行初始化 |
| Z-Image 系列 | --enable_npu_patch | 使用NPU融合算子来替换Z-image模型中的对应算子以提升模型在NPU上的性能 |

## Moore Threads MUSA GPU

借助 `torchada` 适配层，`DiffSynth-Studio` 可以在摩尔线程（Moore Threads）MUSA GPU 上透明复用标准 `"cuda"` 接口，无需将业务代码中的 `"cuda"` 逐一替换为 `"musa"`，且默认启用高性能 `flash_attention_3`（基于 `mate` / `mate-mubin`）与 `mccl` 分布式通信后端。

在运行前，请先按照 [安装依赖 - Moore Threads MUSA GPU](../Pipeline_Usage/Setup.md#moore-threads-musa-gpu) 完成系统库与 Python 依赖包的安装。

### 推理

在推理脚本顶部导入 `torchada`（如果脚本中包含 `torch.cuda.is_available()` 判断，可在导入前设置 `TORCHADA_FORCE_CUDA_AVAILABLE=1`），其余代码与 NVIDIA GPU 完全一致：

```python
import os
os.environ.setdefault("TORCHADA_FORCE_CUDA_AVAILABLE", "1")
import torchada  # 自动将标准 torch.cuda 调用重定向至 torch_musa

# 后续直接运行或导入任何官方 examples/* 推理代码，无需修改 device="cuda"
```

也可以通过一个通用的 3 行启动包装器直接运行任意现有的推理或训练脚本，实现示例脚本零改动：

```python
# run_musa.py
import runpy, sys
import torchada

sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
```

运行推理示例：
```shell
MUSA_VISIBLE_DEVICES=0 python run_musa.py examples/minimax_h3/model_inference/MiniMax-H3-FL2VA.py
```

### 分布式训练（以 MiniMax-H3 8 卡训练为例）

`accelerate>=1.12.0` 原生支持识别 MUSA 设备并自动选用 `mccl` 通信后端，支持两阶段拆分训练（Stage 1 特征预处理 + Stage 2 PyTorch FSDP 或 DeepSpeed ZeRO-3 训练）。

#### 1. 准备 Accelerate 分布式配置文件

- **8 卡 PyTorch FSDP 配置（`fsdp8.yaml`）**：
  ```yaml
  compute_environment: LOCAL_MACHINE
  debug: false
  distributed_type: FSDP
  downcast_bf16: 'no'
  enable_cpu_affinity: false
  fsdp_config:
    fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
    fsdp_backward_prefetch: BACKWARD_PRE
    fsdp_cpu_ram_efficient_loading: false
    fsdp_forward_prefetch: false
    fsdp_offload_params: false
    fsdp_sharding_strategy: FULL_SHARD
    fsdp_state_dict_type: SHARDED_STATE_DICT
    fsdp_sync_module_states: false
    fsdp_transformer_layer_cls_to_wrap: MiniMaxH3DiTBlock
    fsdp_use_orig_params: true
  machine_rank: 0
  main_training_function: main
  mixed_precision: bf16
  num_machines: 1
  num_processes: 8
  rdzv_backend: static
  same_network: true
  use_cpu: false
  ```

- **8 卡 DeepSpeed ZeRO-3 配置（`zero3_8gpu.yaml`）**：
  ```yaml
  compute_environment: LOCAL_MACHINE
  debug: false
  deepspeed_config:
    gradient_accumulation_steps: 1
    offload_optimizer_device: none
    offload_param_device: none
    zero3_init_flag: false
    zero3_save_16bit_model: true
    zero_stage: 3
  distributed_type: DEEPSPEED
  downcast_bf16: 'no'
  enable_cpu_affinity: false
  machine_rank: 0
  main_training_function: main
  mixed_precision: bf16
  num_machines: 1
  num_processes: 8
  rdzv_backend: static
  same_network: true
  use_cpu: false
  ```

#### 2. 启动两阶段 8 卡训练

使用上面保存的 `run_musa.py`（内部仅 `import torchada`）即可直接启动官方 `train.py`：

```shell
export MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Stage 1：8 卡并行数据预处理（解码 MP4 视频与音轨，提取首尾帧视觉特征、文本特征与音视频 VAE 潜变量）
accelerate launch --num_processes 8 run_musa.py \
  examples/minimax_h3/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-FL2VA \
  --dataset_metadata_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-FL2VA/metadata.csv \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio,input_image,end_image" \
  --height 480 \
  --width 832 \
  --num_frames 124 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3:FL2VA/video_vae/source/model.safetensors,MiniMax/MiniMax-H3:FL2VA/audio_vae/model.safetensors" \
  --output_path "./models/train/MiniMax-H3-FL2VA-split-cache" \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --task "sft:data_process"

# Stage 2（方式 A：8 卡 PyTorch FSDP 训练）
accelerate launch --config_file fsdp8.yaml run_musa.py \
  examples/minimax_h3/model_training/train.py \
  --dataset_base_path ./models/train/MiniMax-H3-FL2VA-split-cache \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio,input_image,end_image" \
  --height 480 \
  --width 832 \
  --num_frames 124 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/transformer/model*.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-FL2VA-fsdp" \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --task "sft:train"

# Stage 2（方式 B：8 卡 DeepSpeed ZeRO-3 训练）
accelerate launch --config_file zero3_8gpu.yaml run_musa.py \
  examples/minimax_h3/model_training/train.py \
  --dataset_base_path ./models/train/MiniMax-H3-FL2VA-split-cache \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio,input_image,end_image" \
  --height 480 \
  --width 832 \
  --num_frames 124 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/transformer/model*.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-FL2VA-zero3" \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --task "sft:train"
```

#### 已验证的关键依赖版本参考

| 组件 | 推荐/已验证版本 | 说明 |
|------|-----------------|------|
| `torch` / `torch_musa` | `2.7.1.post1` | MUSA PyTorch 基础框架 |
| `torchada` | `>=0.1.102` | CUDA → MUSA 透明适配层 |
| `mate` / `mate-mubin` | `>=0.2.8.post1` | MUSA FlashAttention-3 高性能前反向算子（支持 `head_dim` 64~256） |
| `flash-attn-3` | `>=0.2.8.post1+musa` | FlashAttention-3 Python 接口（`flash_attn_interface`） |
| `tilelang-musa` | `0.1.12+musa.5` | 通过 `pip install "mate[all]"` 自动安装，提供 FA3 反向后处理内核 |
| `accelerate` | `>=1.12.0` | 原生支持 `MULTI_MUSA` 与 `mccl` 后端 |
| `peft` | `>=0.18.1,<0.19.0` | 兼容 `torch 2.7` / `torchao 0.9` 环境 |
| `torchaudio` + FFmpeg 库 | `2.7.1` + `libav*58` | 原生支持从 `.mp4` 容器直接解码音轨 |
| `torchcodec` | `0.5` | 适配 `torch 2.7` 的 `diffsynth.utils.data.audio` 默认后端 |