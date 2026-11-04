# GPU/NPU Support

`DiffSynth-Studio` supports various GPUs and NPUs. This document explains how to run model inference and training on these devices.

Before you begin, please follow the [Installation Guide](../Pipeline_Usage/Setup.md) to install the required GPU/NPU dependencies.

## NVIDIA GPU

All sample code provided by this project supports NVIDIA GPUs by default, requiring no additional modifications.

## AMD GPU

AMD provides PyTorch packages based on ROCm, so most models can run without code changes. A small number of models may not be compatible due to their reliance on CUDA-specific instructions.

### Apple Silicon

On Apple Silicon devices, since VRAM and memory are unified, replace all `"cuda"` in the code with `"mps"` or `"cpu"`.

## Ascend NPU
### Inference
When using Ascend NPU, you need to replace `"cuda"` with `"npu"` in your code.

For example, here is the inference code for **Wan2.1-T2V-1.3B**, modified for Ascend NPU:

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
    prompt="Documentary-style photography: a lively puppy running swiftly across lush green grass. The puppy has brownish-yellow fur, upright ears, and an alert, joyful expression. Sunlight bathes its body, making the fur appear exceptionally soft and shiny. The background is an open field with occasional wildflowers, and faint blue sky with scattered white clouds in the distance. Strong perspective captures the motion of the running puppy and the vitality of the surrounding grass. Mid-shot, side-moving viewpoint.",
    negative_prompt="Overly vibrant colors, overexposed, static, blurry details, subtitles, artistic style, painting, still image, overall grayish tone, worst quality, low quality, JPEG artifacts, ugly, distorted, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, fused fingers, motionless scene, cluttered background, three legs, many people in background, walking backward",
    seed=0, tiled=True,
)
save_video(video, "video.mp4", fps=15, quality=5)
```

#### USP(Unified Sequence Parallel)
If you want to use this feature on NPU, please install additional third-party libraries as follows:
```shell
pip install git+https://github.com/feifeibear/long-context-attention.git
pip install git+https://github.com/xdit-project/xDiT.git
```


### Training
NPU startup script samples have been added for each type of model,the scripts are stored in the `examples/xxx/special/npu_training`, for example `examples/wanvideo/model_training/special/npu_training/Wan2.2-T2V-A14B-NPU.sh`.

In the NPU training scripts, NPU specific environment variables that can optimize performance have been added, and relevant parameters have been enabled for specific models.

#### Environment variables
```shell
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```
`expandable_segments:<value>`: Enable the memory pool expansion segment function, which is the virtual memory feature.

```shell
export CPU_AFFINITY_CONF=1
```
Set 0 or not set: indicates not enabling the binding function

1: Indicates enabling coarse-grained kernel binding

2: Indicates enabling fine-grained kernel binding

#### Parameters for specific models
| Model          | Parameter                 | Note              |
|----------------|---------------------------|-------------------|
| Wan 14B series | --initialize_model_on_cpu | The 14B model needs to be initialized on the CPU |
| Qwen-Image series | --initialize_model_on_cpu | The model needs to be initialized on the CPU |
| Z-Image series | --enable_npu_patch | Using NPU fusion operator to replace the corresponding operator in Z-image model to improve the performance of the model on NPU |

## Moore Threads MUSA GPU

With the `torchada` adapter, `DiffSynth-Studio` runs transparently on Moore Threads MUSA GPUs using standard `"cuda"` APIs without rewriting `"cuda"` to `"musa"` across scripts, while automatically enabling `flash_attention_3` (powered by `mate` / `mate-mubin`) and the native `mccl` distributed backend.

Before running, complete the system library and Python package setup in [Installing Dependencies - Moore Threads MUSA GPU](../Pipeline_Usage/Setup.md#moore-threads-musa-gpu).

### Inference

Import `torchada` at the top of your script (set `TORCHADA_FORCE_CUDA_AVAILABLE=1` before importing if the script explicitly checks `torch.cuda.is_available()`). No other code modifications are required:

```python
import os
os.environ.setdefault("TORCHADA_FORCE_CUDA_AVAILABLE", "1")
import torchada  # Transparently redirects standard torch.cuda calls to torch_musa

# Run or import any official examples/* script directly with device="cuda"
```

You can also use a 3-line launcher script to run any existing inference or training script unmodified:

```python
# run_musa.py
import runpy, sys
import torchada

sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
```

Run an inference example:
```shell
MUSA_VISIBLE_DEVICES=0 python run_musa.py examples/minimax_h3/model_inference/MiniMax-H3-FL2VA.py
```

### Distributed Training (8-GPU MiniMax-H3 Example)

`accelerate>=1.12.0` natively detects MUSA GPUs and selects the `mccl` communication backend, supporting both two-stage split preprocessing (Stage 1) and distributed training with PyTorch FSDP or DeepSpeed ZeRO-3 (Stage 2).

#### 1. Prepare Accelerate Configuration Files

- **8-GPU PyTorch FSDP (`fsdp8.yaml`)**:
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

- **8-GPU DeepSpeed ZeRO-3 (`zero3_8gpu.yaml`)**:
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

#### 2. Launch Two-Stage 8-GPU Training

Using `run_musa.py` above, launch the official `train.py` directly:

```shell
export MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Stage 1: 8-GPU data preprocessing (decode MP4 video & audio, extract first/last frame visual features, text features, and VAE latents)
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

# Stage 2 (Option A: 8-GPU PyTorch FSDP training)
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

# Stage 2 (Option B: 8-GPU DeepSpeed ZeRO-3 training)
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

#### Verified Package Versions

| Package | Verified Version | Notes |
|---------|------------------|-------|
| `torch` / `torch_musa` | `2.7.1.post1` | Base PyTorch framework for MUSA |
| `torchada` | `>=0.1.102` | Transparent CUDA → MUSA adapter |
| `mate` / `mate-mubin` | `>=0.2.8.post1` | MUSA FlashAttention-3 forward/backward kernels (`head_dim` 64–256) |
| `flash-attn-3` | `>=0.2.8.post1+musa` | FlashAttention-3 Python interface (`flash_attn_interface`) |
| `tilelang-musa` | `0.1.12+musa.5` | Installed via `pip install "mate[all]"` for FA3 backward post-processing |
| `accelerate` | `>=1.12.0` | Native `MULTI_MUSA` and `mccl` backend support |
| `peft` | `>=0.18.1,<0.19.0` | Compatible with `torch 2.7` / `torchao 0.9` |
| `torchaudio` + FFmpeg libs | `2.7.1` + `libav*58` | Requires `libavutil56 libavcodec58 libavformat58 libavfilter7 libavdevice58` for native FFmpeg decoding |
| `torchcodec` | `0.5` | Compatible with `torch 2.7` for `diffsynth.utils.data.audio` |