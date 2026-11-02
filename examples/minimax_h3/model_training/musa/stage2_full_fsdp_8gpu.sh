#!/bin/bash
# Stage 2: Optimized 8-GPU PyTorch FSDP Full-Parameter SFT for MiniMax-H3 (33B DiT, 535 tensors) on MUSA.
# Trains all 33B DiT parameters in-GPU without CPU optimizer offload!
#
# Measured on 8 x MTT S5000 (80 GB, 14-link MTLink full-mesh, 4 steps):
#   56f T2VA   (CKPT_INTERVAL=2): First step: 29.71 s | Steady-state:  9.21 s/it (min  9.16 s) | Save 62 GiB: 58.4 s | Peak VRAM: 75.7 GiB
#   56f T2VA   (CKPT_INTERVAL=1): First step: 30.50 s | Steady-state:  9.59 s/it (min  9.58 s) | Save 62 GiB: 61.2 s | Peak VRAM: 39.3 GiB
#   124f FL2VA (CKPT_INTERVAL=7): First step: 33.71 s | Steady-state: 17.85 s/it (min 17.81 s) | Save 62 GiB: 58.6 s | Peak VRAM: 71.8 GiB
#   124f FL2VA (CKPT_INTERVAL=1): First step: 32.60 s | Steady-state: 18.31 s/it (min 18.28 s) | Save 62 GiB: 55.4 s | Peak VRAM: 43.6 GiB
#
# Usage:
#   FRAMES=56  bash examples/minimax_h3/model_training/musa/stage2_full_fsdp_8gpu.sh
#   FRAMES=124 bash examples/minimax_h3/model_training/musa/stage2_full_fsdp_8gpu.sh
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
DATASET_REPEAT="${DATASET_REPEAT:-32}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"

if [ "${FRAMES}" = "124" ]; then
  CACHE_PATH="${CACHE_PATH:-./models/train/MiniMax-H3-FL2VA-124f-cache}"
  EXTRA_INPUTS="${EXTRA_INPUTS:-input_audio,input_image,end_image}"
  CKPT_INTERVAL="${CKPT_INTERVAL:-7}"
  OUTPUT_PATH="${OUTPUT_PATH:-./models/train/MiniMax-H3-FL2VA-124f-full-fsdp}"
else
  CACHE_PATH="${CACHE_PATH:-./models/train/MiniMax-H3-T2VA-56f-cache}"
  EXTRA_INPUTS="${EXTRA_INPUTS:-input_audio}"
  CKPT_INTERVAL="${CKPT_INTERVAL:-2}"
  OUTPUT_PATH="${OUTPUT_PATH:-./models/train/MiniMax-H3-T2VA-56f-full-fsdp}"
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
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --gradient_checkpointing_interval "${CKPT_INTERVAL}" \
  "${SWANLAB_ARGS[@]}" \
  --task "sft:train"
