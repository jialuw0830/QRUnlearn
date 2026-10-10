#!/bin/bash
# Find forget-specific outliers: score = F_forget / (F_retain + eps), top-k% per module.
# Usage: bash scripts/find_outliers.sh [books|news]

CORPUS=${1:-books}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
MODEL_ROOT=${MODEL_ROOT:-/playpen-shared/huaizhi/jialu/models}
case "$CORPUS" in
  books) MODEL_DIR="$MODEL_ROOT/MUSE-Books_target" ;;
  news)  MODEL_DIR="$MODEL_ROOT/MUSE-News_target" ;;
  *) echo "Unknown corpus: $CORPUS"; exit 1 ;;
esac
SENSITIVITY_PERCENTS=(0.1 0.5)  # top-k% per module

cd "${PROJECT_ROOT}/baselines"
for SENS in "${SENSITIVITY_PERCENTS[@]}"; do
    python -m baselines.outlier \
        --model_path "$MODEL_DIR" \
        --tokenizer_path "$MODEL_ROOT/Llama-2-7b-hf" \
        --forget_txt "${PROJECT_ROOT}/data/$CORPUS/raw/forget.txt" \
        --retain_txt "${PROJECT_ROOT}/data/$CORPUS/raw/retain1.txt" \
        --output_folder "${PROJECT_ROOT}/outputs/outliers/$CORPUS" \
        --sensitivity_percent "$SENS" \
        --max_length 2048 \
        --batch_size 1
done
