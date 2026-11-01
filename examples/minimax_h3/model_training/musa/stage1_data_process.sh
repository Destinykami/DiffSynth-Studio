#!/bin/bash
# Stage 1: Data preprocessing for MiniMax-H3 on MUSA (1-GPU or 8-GPU Data Parallel).
# Encodes video latents, audio latents, and Qwen3-VL text/keyframe embeddings into disk cache.
#
# Usage:
#   # 56-frame T2VA (single GPU)
#   FRAMES=56 EXTRA_INPUTS="input_audio" NUM_PROCESSES=1 bash examples/minimax_h3/model_training/musa/stage1_data_process.sh
#   # 124-frame FL2VA first/last-frame + audio (8-GPU data parallel, ~2.25 s/sample steady-state)
#   FRAMES=124 EXTRA_INPUTS="input_audio,input_image,end_image" NUM_PROCESSES=8 bash examples/minimax_h3/model_training/musa/stage1_data_process.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

export TORCHADA_FORCE_CUDA_AVAILABLE=1
export PYTHONUNBUFFERED=1
unset PYTORCH_MUSA_ALLOC_CONF

FRAMES="${FRAMES:-124}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
DATASET_BASE_PATH="${DATASET_BASE_PATH:-data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-FL2VA}"
DATASET_METADATA_PATH="${DATASET_METADATA_PATH:-${DATASET_BASE_PATH}/metadata.csv}"
EXTRA_INPUTS="${EXTRA_INPUTS:-input_audio,input_image,end_image}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

if [ "${FRAMES}" = "56" ] && [ "${EXTRA_INPUTS}" = "input_audio" ]; then
  DEFAULT_CACHE="./models/train/MiniMax-H3-T2VA-56f-cache"
else
  DEFAULT_CACHE="./models/train/MiniMax-H3-FL2VA-${FRAMES}f-cache"
fi
OUTPUT_PATH="${OUTPUT_PATH:-${DEFAULT_CACHE}}"

if [ "${NUM_PROCESSES}" -gt 1 ]; then
  LAUNCHER=(accelerate launch --multi_gpu --num_processes "${NUM_PROCESSES}" --mixed_precision bf16)
else
  export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0}"
  LAUNCHER=(python)
fi

"${LAUNCHER[@]}" "${SCRIPT_DIR}/train_musa.py" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --data_file_keys "video,input_audio" \
  --extra_inputs "${EXTRA_INPUTS}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --num_frames "${FRAMES}" \
  --dataset_repeat "${DATASET_REPEAT}" \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3:FL2VA/video_vae/source/model.safetensors,MiniMax/MiniMax-H3:FL2VA/audio_vae/model.safetensors" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --task "sft:data_process"
