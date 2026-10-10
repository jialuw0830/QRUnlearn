#!/bin/bash
# Evaluate every model under outputs/ckpt/<corpus>; writes <model>_<quant>_metrics.csv next to each model.
# Usage: bash eval.sh [books|news]
# quantize_4bit=0, quantize_8bit=0 -> full precision; 4bit=1 -> 4-bit; 8bit=1 -> 8-bit
export PYTHONNOUSERSITE=1
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_ROOT=${MODEL_ROOT:-/playpen-shared/huaizhi/jialu/models}
CORPUS=${1:-books}
quantize_4bit=1
quantize_8bit=0

CKPT_DIR="${PROJECT_ROOT}/outputs/ckpt/${CORPUS}"
quantize_method=$([ ${quantize_4bit} -eq 1 ] && echo "4bit" || ([ ${quantize_8bit} -eq 1 ] && echo "8bit" || echo "fp"))

# Find all model directories (full model or LoRA adapter)
mapfile -t MODEL_DIRS < <(find "${CKPT_DIR}" \( -name "config.json" -o -name "adapter_config.json" \) -type f -exec dirname {} \;)
echo "Found ${#MODEL_DIRS[@]} models to evaluate"

cd "${PROJECT_ROOT}"
for MODEL_DIR in "${MODEL_DIRS[@]}"; do
  MODEL_NAME=$(basename "${MODEL_DIR}")
  OUTPUT_FILE="$(dirname "${MODEL_DIR}")/${MODEL_NAME}_${quantize_method}_metrics.csv"

  if [ -f "${OUTPUT_FILE}" ]; then
    echo "Skipping (file exists): ${OUTPUT_FILE}"
    continue
  fi

  echo "Evaluating: ${MODEL_DIR} -> ${OUTPUT_FILE}"
  python eval.py \
    --model_dirs "${MODEL_DIR}" \
    --tokenizer_dir "${MODEL_ROOT}/Llama-2-7b-hf" \
    --names "${MODEL_NAME}" \
    --corpus "${CORPUS}" \
    --quantize_4bit ${quantize_4bit} \
    --quantize_8bit ${quantize_8bit} \
    --metrics knowmem_f verbmem_f privleak knowmem_r \
    --out_file "${OUTPUT_FILE}"
done
