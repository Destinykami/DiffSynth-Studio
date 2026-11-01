#!/bin/bash
# Stage 2: 8-GPU DeepSpeed ZeRO-3 Full-Parameter SFT for MiniMax-H3 (33B DiT, 535 tensors) on MUSA.
# Note: For 124-frame FL2VA full-parameter SFT without CPU offload, prefer stage2_full_fsdp_8gpu.sh
# (17.85 s/it in-GPU vs 51.43 s/it with ZeRO-3 CPU optimizer offload).
#
# Measured on 8 x MTT S5000 (80 GB, 4 steps):
#   56f T2VA (no CPU offload):              First step: 18.61 s | Steady-state: 11.04 s/it | Save 62 GiB: 61.8 s | Peak VRAM: 70.8 GiB
#   124f FL2VA (offload_optimizer=cpu):     First step: 68.71 s | Steady-state: 51.43 s/it | Save 62 GiB: 61.8 s | Peak VRAM: 25.3 GiB
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

export TORCHADA_FORCE_CUDA_AVAILABLE=1
export PYTHONUNBUFFERED=1
export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export DIFFSYNTH_ATTENTION_IMPLEMENTATION="${DIFFSYNTH_ATTENTION_IMPLEMENTATION:-torch}"
unset PYTORCH_MUSA_ALLOC_CONF

FRAMES="${FRAMES:-56}"
DATASET_REPEAT="${DATASET_REPEAT:-32}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
CKPT_INTERVAL="${CKPT_INTERVAL:-1}"

if [ "${FRAMES}" = "124" ]; then
  CACHE_PATH="${CACHE_PATH:-./models/train/MiniMax-H3-FL2VA-124f-cache}"
  EXTRA_INPUTS="${EXTRA_INPUTS:-input_audio,input_image,end_image}"
  OUTPUT_PATH="${OUTPUT_PATH:-./models/train/MiniMax-H3-FL2VA-124f-full-zero3}"
  # 124f full-parameter SFT under ZeRO-3 requires CPU optimizer offload (58.7 GB static fp32 master states + 25 GB activations > 80 GB)
  TMP_CFG="$(mktemp /tmp/accelerate_zero3_full_XXXXXX.yaml)"
  trap 'rm -f "${TMP_CFG}"' EXIT
  sed 's/offload_optimizer_device: none/offload_optimizer_device: cpu/' \
    "${SCRIPT_DIR}/accelerate_config_zero3_8gpu.yaml" > "${TMP_CFG}"
  CFG_FILE="${TMP_CFG}"
else
  CACHE_PATH="${CACHE_PATH:-./models/train/MiniMax-H3-T2VA-56f-cache}"
  EXTRA_INPUTS="${EXTRA_INPUTS:-input_audio}"
  OUTPUT_PATH="${OUTPUT_PATH:-./models/train/MiniMax-H3-T2VA-56f-full-zero3}"
  CFG_FILE="${SCRIPT_DIR}/accelerate_config_zero3_8gpu.yaml"
fi

accelerate launch --config_file "${CFG_FILE}" \
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
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --gradient_checkpointing_interval "${CKPT_INTERVAL}" \
  --task "sft:train"
