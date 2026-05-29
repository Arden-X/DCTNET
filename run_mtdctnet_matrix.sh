#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/path/to/20260405_NN_10}"
TRAIN_CSV="${TRAIN_CSV:-DCTNET/outputs/splits/train.csv}"
VAL_CSV="${VAL_CSV:-DCTNET/outputs/splits/val.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-DCTNET/outputs/runs}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-0.001}"
PATIENCE="${PATIENCE:-15}"
HIT_TOL="${HIT_TOL:-10}"
MAX_ROWS="${MAX_ROWS:-0}"

extra_max_rows=()
if [[ "${MAX_ROWS}" != "0" ]]; then
  extra_max_rows=(--max-rows "${MAX_ROWS}")
fi

common=(
  --data-root "${DATA_ROOT}"
  --train-csv "${TRAIN_CSV}"
  --val-csv "${VAL_CSV}"
  --output-dir "${OUTPUT_DIR}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --device "${DEVICE}"
)

for model in MT-DCTNet-IQ MT-DCTNet-Corr MT-DCTNet-Dual; do
  echo "===== Classification: ${model} ====="
  "${PYTHON}" DCTNET/train_classifier.py \
    --task multiclass6 \
    --model-name "${model}" \
    "${common[@]}" \
    "${extra_max_rows[@]}"
done

echo "===== Sync only: MT-DCTNet ====="
"${PYTHON}" DCTNET/train_multitask_sync.py \
  --model-name MT-DCTNet \
  --data-root "${DATA_ROOT}" \
  --train-csv "${TRAIN_CSV}" \
  --val-csv "${VAL_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --device "${DEVICE}" \
  --lambda-cls 0 \
  --lambda-loc 1 \
  --select-by loc_mae \
  --hit-tol "${HIT_TOL}" \
  --early-stop-patience "${PATIENCE}" \
  "${extra_max_rows[@]}"

echo "===== Multi-task: MT-DCTNet ====="
"${PYTHON}" DCTNET/train_multitask_sync.py \
  --model-name MT-DCTNet \
  --data-root "${DATA_ROOT}" \
  --train-csv "${TRAIN_CSV}" \
  --val-csv "${VAL_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --device "${DEVICE}" \
  --lambda-cls 1 \
  --lambda-loc 3 \
  --select-by joint \
  --hit-tol "${HIT_TOL}" \
  --early-stop-patience "${PATIENCE}" \
  "${extra_max_rows[@]}"
