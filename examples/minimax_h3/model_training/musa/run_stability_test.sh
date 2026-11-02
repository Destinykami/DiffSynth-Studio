#!/bin/bash
# End-to-end long-run stability and loss-convergence verification for MiniMax-H3 on MUSA.
# 1. Builds the 16-sample mixed (1 real + 15 synthetic multi-pattern) audiovisual dataset if missing.
# 2. Runs 8-GPU Stage-1 data preprocessing if the latent/embedding cache is missing.
# 3. Launches 8-GPU FSDP training with deterministic fixed-anchor evaluation (TRAIN_EVAL_EVERY)
#    and exports per-step telemetry (step_time_s, raw_loss, ema_loss, grad_norm, alloc_gib,
#    resv_gib, fixed_eval_mean, fixed_eval_t200/t500/t800) to `<OUTPUT_PATH>/stability_metrics.csv`.
#
# Usage:
#   # 100-step 56-frame T2VA stability run (~6.5 min, eval every 10 steps)
#   FRAMES=56 DATASET_REPEAT=50 TRAIN_EVAL_EVERY=10 bash examples/minimax_h3/model_training/musa/run_stability_test.sh
#
#   # 40-step 124-frame FL2VA full-length stability run (~8.5 min, eval every 8 steps)
#   FRAMES=124 DATASET_REPEAT=20 TRAIN_EVAL_EVERY=8 bash examples/minimax_h3/model_training/musa/run_stability_test.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

FRAMES="${FRAMES:-56}"
AV_BENCH_DIR="${AV_BENCH_DIR:-./data/av_stability_dataset}"

if [ ! -f "${AV_BENCH_DIR}/metadata.csv" ]; then
  echo "[stability] building 16-sample mixed audiovisual benchmark dataset in ${AV_BENCH_DIR} ..."
  AV_BENCH_DIR="${AV_BENCH_DIR}" python "${SCRIPT_DIR}/build_av_stability_dataset.py"
fi

if [ "${FRAMES}" = "124" ]; then
  EXTRA_INPUTS="${EXTRA_INPUTS:-input_audio,input_image,end_image}"
  CACHE_PATH="${CACHE_PATH:-./models/train/MiniMax-H3-FL2VA-124f-stability-cache}"
  OUTPUT_PATH="${OUTPUT_PATH:-./models/train/MiniMax-H3-FL2VA-124f-stability}"
  DATASET_REPEAT="${DATASET_REPEAT:-20}"
  TRAIN_EVAL_EVERY="${TRAIN_EVAL_EVERY:-8}"
  CKPT_INTERVAL="${CKPT_INTERVAL:-4}"
else
  EXTRA_INPUTS="${EXTRA_INPUTS:-input_audio}"
  CACHE_PATH="${CACHE_PATH:-./models/train/MiniMax-H3-T2VA-56f-stability-cache}"
  OUTPUT_PATH="${OUTPUT_PATH:-./models/train/MiniMax-H3-T2VA-56f-stability}"
  DATASET_REPEAT="${DATASET_REPEAT:-50}"
  TRAIN_EVAL_EVERY="${TRAIN_EVAL_EVERY:-10}"
  CKPT_INTERVAL="${CKPT_INTERVAL:--5}"
fi

if [ ! -d "${CACHE_PATH}/0" ]; then
  echo "[stability] running 8-GPU Stage-1 preprocessing into ${CACHE_PATH} ..."
  FRAMES="${FRAMES}" EXTRA_INPUTS="${EXTRA_INPUTS}" NUM_PROCESSES=8 DATASET_REPEAT=1 \
    DATASET_BASE_PATH="${AV_BENCH_DIR}" OUTPUT_PATH="${CACHE_PATH}" \
    bash "${SCRIPT_DIR}/stage1_data_process.sh"
fi

echo "[stability] launching 8-GPU FSDP stability & convergence run (FRAMES=${FRAMES}, DATASET_REPEAT=${DATASET_REPEAT}, TRAIN_EVAL_EVERY=${TRAIN_EVAL_EVERY}) ..."
TRAIN_EVAL_EVERY="${TRAIN_EVAL_EVERY}" \
  FRAMES="${FRAMES}" \
  CACHE_PATH="${CACHE_PATH}" \
  EXTRA_INPUTS="${EXTRA_INPUTS}" \
  CKPT_INTERVAL="${CKPT_INTERVAL}" \
  DATASET_REPEAT="${DATASET_REPEAT}" \
  OUTPUT_PATH="${OUTPUT_PATH}" \
  bash "${SCRIPT_DIR}/stage2_fsdp_8gpu.sh"
