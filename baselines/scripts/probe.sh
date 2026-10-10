#!/bin/bash
# Probe per-layer forget/retain gradient log-ratio.
# Usage: bash scripts/probe.sh [books|news]

CORPUS=${1:-books}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
MODEL_ROOT=${MODEL_ROOT:-/playpen-shared/huaizhi/jialu/models}
case "$CORPUS" in
  books) MODEL_DIR="$MODEL_ROOT/MUSE-Books_target" ;;
  news)  MODEL_DIR="$MODEL_ROOT/MUSE-News_target" ;;
  *) echo "Unknown corpus: $CORPUS"; exit 1 ;;
esac

cd "${PROJECT_ROOT}/baselines"
python -m baselines.probe \
    --model_dir "$MODEL_DIR" \
    --tokenizer_dir "$MODEL_ROOT/Llama-2-7b-hf" \
    --forget_data_file "${PROJECT_ROOT}/data/$CORPUS/raw/forget.txt" \
    --retain_data_file "${PROJECT_ROOT}/data/$CORPUS/raw/retain1.txt" \
    --max_len 2048 \
    --batch_size 2 \
    --output_file "${PROJECT_ROOT}/outputs/probe/${CORPUS}.json"
