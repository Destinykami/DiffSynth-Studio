#!/bin/bash
# Stage 2: Optimized 8-GPU PyTorch FSDP LoRA training for MiniMax-H3 on MUSA (56f T2VA or 124f FL2VA).
# Measured on 8 x MTT S5000 (80 GB, 14-link MTLink full-mesh):
#   56f T2VA:   First step: 16.50 s | Steady-state: 3.78 s/it (1.99x faster than 7.55s baseline!) | Save: 1.3 s | Peak VRAM: 73.3 GiB
#   124f FL2VA: First step: 24.41 s | Steady-state: 11.74 s/it (1.60x faster than 18.74s baseline!) | Save: 1.3 s | Peak VRAM: 70.4 GiB
#
# Usage:
#   FRAMES=56  bash examples/minimax_h3/model_training/musa/stage2_fsdp_8gpu.sh
#   FRAMES=124 bash examples/minimax_h3/model_training/musa/stage2_fsdp_8gpu.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

export TORCHADA_FORCE_CUDA_AVAILABLE=1
export PYTHONUNBUFFERED=1
export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_MUSA_ALLOC_CONF="${PYTORCH_MUSA_ALLOC_CONF:-expandable_segments:True}"
export DIFFSYNTH_ATTENTION_IMPLEMENTATION="${DIFFSYNTH_ATTENTION_IMPLEMENTATION:-torch}"

FRAMES="${FRAMES:-56}"
DATASET_REPEAT="${DATASET_REPEAT:-48}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LORA_RANK="${LORA_RANK:-32}"

if [ "${FRAMES}" = "124" ]; then
  CACHE_PATH="${CACHE_PATH:-./models/train/MiniMax-H3-FL2VA-124f-cache}"
  EXTRA_INPUTS="${EXTRA_INPUTS:-input_audio,input_image,end_image}"
  CKPT_INTERVAL="${CKPT_INTERVAL:-4}"
  OUTPUT_PATH="${OUTPUT_PATH:-./models/train/MiniMax-H3-FL2VA-124f-fsdp-lora}"
else
  CACHE_PATH="${CACHE_PATH:-./models/train/MiniMax-H3-T2VA-56f-cache}"
  EXTRA_INPUTS="${EXTRA_INPUTS:-input_audio}"
  CKPT_INTERVAL="${CKPT_INTERVAL:--5}"
  OUTPUT_PATH="${OUTPUT_PATH:-./models/train/MiniMax-H3-T2VA-56f-fsdp-lora}"
fi

SWANLAB_ARGS=()
if [ "${ENABLE_SWANLAB:-0}" = "1" ] || [ -n "${SWANLAB_API_KEY:-}" ]; then
  SWANLAB_ARGS=(--enable_swanlab_log --swanlab_project "${SWANLAB_PROJECT_NAME:-${SWANLAB_PROJECT:-MiniMax-H3-MUSA}}")
  unset SWANLAB_PROJECT
fi

accelerate launch --config_file "${SCRIPT_DIR}/accelerate_config_fsdp_8gpu.yaml" \
  "${SCRIPT_DIR}/train_musa.py" \
  --dataset_base_path "${CACHE_PATH}" \
  --data_file_keys "video,input_audio" \
  --extra_inputs "${EXTRA_INPUTS}" \
  --height 480 \
  --width 832 \
  --num_frames "${FRAMES}" \
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
