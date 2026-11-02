#!/bin/bash
# Stage 2: Optimized 8-GPU DeepSpeed ZeRO-3 LoRA training for MiniMax-H3 on MUSA (124 frames FL2VA full-length).
# Measured on 8 x MTT S5000 (80 GB):
#   First step: 19.77 s | Steady-state: 14.75 s/it (min 14.55 s) | Peak VRAM: 48.3 GiB
#
# Key optimizations applied:
# 1. Unsets PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True (eliminates virtual-memory remap thrashing under ZeRO-3).
# 2. Disables redundant --use_gradient_checkpointing_offload and --find_unused_parameters.
# 3. Uses selective activation checkpointing (--gradient_checkpointing_interval 7, keeping 8/50 blocks in VRAM with zero allocator flushes).
# 4. Uses torch_musa.optim.FusedAdamW + MUSA native fused SwiGLU, torch.rope, Triton AdaLN, contiguous QKV unbind, and padding SDPA skip.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

export TORCHADA_FORCE_CUDA_AVAILABLE=1
export PYTHONUNBUFFERED=1
export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export DIFFSYNTH_ATTENTION_IMPLEMENTATION="${DIFFSYNTH_ATTENTION_IMPLEMENTATION:-torch}"
unset PYTORCH_MUSA_ALLOC_CONF

CACHE_PATH="${CACHE_PATH:-./models/train/MiniMax-H3-FL2VA-124f-cache}"
OUTPUT_PATH="${OUTPUT_PATH:-./models/train/MiniMax-H3-FL2VA-124f-zero3-lora}"
DATASET_REPEAT="${DATASET_REPEAT:-48}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LORA_RANK="${LORA_RANK:-32}"
CKPT_INTERVAL="${CKPT_INTERVAL:-7}"

SWANLAB_ARGS=()
if [ "${ENABLE_SWANLAB:-0}" = "1" ] || [ -n "${SWANLAB_API_KEY:-}" ]; then
  SWANLAB_ARGS=(--enable_swanlab_log --swanlab_project "${SWANLAB_PROJECT_NAME:-${SWANLAB_PROJECT:-MiniMax-H3-MUSA}}")
  unset SWANLAB_PROJECT
fi

accelerate launch --config_file "${SCRIPT_DIR}/accelerate_config_zero3_8gpu.yaml" \
  "${SCRIPT_DIR}/train_musa.py" \
  --dataset_base_path "${CACHE_PATH}" \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio,input_image,end_image" \
  --height 480 \
  --width 832 \
  --num_frames 124 \
  --dataset_repeat "${DATASET_REPEAT}" \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/transformer/model*.safetensors" \
  --learning_rate "${LEARNING_RATE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --customized_optimizer "torch_musa.optim.FusedAdamW" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank "${LORA_RANK}" \
  --use_gradient_checkpointing \
  --gradient_checkpointing_interval "${CKPT_INTERVAL}" \
  "${SWANLAB_ARGS[@]}" \
  --task "sft:train"
