# alpha is for utility constraint, threshold is for filtering out salient modules
export PYTHONNOUSERSITE=1
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
MODEL_ROOT=${MODEL_ROOT:-/playpen-shared/huaizhi/jialu/models}
CORPUS=('books')
FORGET="${PROJECT_ROOT}/data/$CORPUS/raw/forget.txt"
RETAIN="${PROJECT_ROOT}/data/$CORPUS/raw/retain1.txt"
TARGET_DIR="${MODEL_ROOT}/MUSE-Books_target"
LLAMA_DIR="${MODEL_ROOT}/Llama-2-7b-hf"
MAX_LEN=2048
PER_DEVICE_BATCH_SIZE=1
RUN_EVAL=true

# LoRA finetune options
USE_LORA=false
LORA_R=32
LORA_ALPHA=64
LORA_DROPOUT=0.05

# Learning rate search
LR_CANDIDATES=('1e-5')

# Experiment configs: "algo:loss_scale:epochs"
EXPERIMENT_CONFIGS=(
    # "npo_avg_klr:3:5"
    # "ga_avg_klr:5:1"
    "ga_gdr:5:1"
    # "npo_gdr:3:5"
)

# Outlier sensitivity: 0.1%, 0.5%
SENSITIVITY_PERCENTS=(0.1)

# Outlier mode: "layer" (per-module)
OUTLIER_MODE="layer"

OUTLIER_BASE_DIR="${PROJECT_ROOT}/outputs/outliers/books"  # from scripts/find_outliers.sh

ADD_NOISE=true
NOISE_MODE='row'
NOISE_SCALE=1.0
LORA_DROPOUT_FORMAT=$(echo "${LORA_DROPOUT}" | sed 's/\./_/')
LORA_TAG=$([ "$USE_LORA" = true ] && echo "_lora_r${LORA_R}_a${LORA_ALPHA}_d${LORA_DROPOUT_FORMAT}" || echo "")

GRADNORM_JSON="${PROJECT_ROOT}/outputs/probe/books.json"  # from scripts/probe.sh
GRADNORM_TOPK=10
GRADNORM_MODE="high"

LR_FORMAT="$LR"
NOISE_SCALE_FORMAT=$(echo "${NOISE_SCALE}" | sed 's/\./_/')
TOPK_TAG="_tk${GRADNORM_TOPK}"

# Loop over learning rates
for LR in "${LR_CANDIDATES[@]}"; do
    LR_FORMAT="$LR"
    echo ""
    echo "=========================================="
    echo "Starting experiments with LR=$LR"
    echo "=========================================="
    
    # Loop over experiment configurations
    for EXP_CONFIG in "${EXPERIMENT_CONFIGS[@]}"; do
    # Parse experiment config: algo:loss_scale:epochs
    IFS=':' read -r ALGO LOSS_SCALE_CURRENT EPOCHS <<< "$EXP_CONFIG"
    
    LOSS_SCALE_FORMAT=$(echo "${LOSS_SCALE_CURRENT}" | sed 's/\./_/')
    LOSS_SCALE_TAG="_lossscale${LOSS_SCALE_FORMAT}"
    EPOCH_TAG="_epoch${EPOCHS}"
    
    echo "=========================================="
    echo "Processing: ALGO=$ALGO, LOSS_SCALE=$LOSS_SCALE_CURRENT, EPOCHS=$EPOCHS"
    echo "=========================================="
    
    # Loop over sensitivity percentages
    for SENS_PERCENT in "${SENSITIVITY_PERCENTS[@]}"; do
        SENS_FORMAT=$(echo "${SENS_PERCENT}" | sed 's/\./_/')
        OUTLIERS_NPZ="${OUTLIER_BASE_DIR}/forget_specific_outliers_sens${SENS_FORMAT}.npz"

        if [ ! -f "$OUTLIERS_NPZ" ]; then
            echo "ERROR: Outlier file not found: $OUTLIERS_NPZ"
            echo "Skipping algo=$ALGO sensitivity=$SENS_PERCENT..."
            continue
        fi

        OUTLIER_FILE=$(basename "${OUTLIERS_NPZ}")
        OUTLIER_TYPE=$(echo "${OUTLIER_FILE}" | sed 's/_outliers\.npz$//' | sed 's/forget_specific_outliers_/fs_/')
        SENSITIVITY=$(echo "${OUTLIER_TYPE}" | sed -n 's/.*sens\([0-9_]*\).*/\1/p')
        NOISE_TAG="_${NOISE_MODE}_fs_sens${SENSITIVITY}_n${NOISE_SCALE_FORMAT}"
        OUT_DIR="${PROJECT_ROOT}/outputs/ckpt/books/${CORPUS}_${ALGO}${TOPK_TAG}_${LR_FORMAT}${NOISE_TAG}_${OUTLIER_MODE}${LOSS_SCALE_TAG}${EPOCH_TAG}${LORA_TAG}"
        METRICS_CSV="${OUT_DIR}_4bit_metrics.csv"

        if [ -d "$OUT_DIR" ]; then
            echo "Skip (model dir exists): $OUT_DIR"
            continue
        fi
        if [ -f "$METRICS_CSV" ]; then
            echo "Skip (metrics exists): $METRICS_CSV"
            continue
        fi

        echo "Run: algo=$ALGO  sensitivity=$SENS_PERCENT%  epochs=$EPOCHS  loss_scale=$LOSS_SCALE_CURRENT  out_dir=$OUT_DIR"

        CMD_ARGS=""
        if [ "$ADD_NOISE" = true ] && [ -n "$OUTLIERS_NPZ" ]; then
            CMD_ARGS="$CMD_ARGS --add_noise --noise_mode $NOISE_MODE --outlier_indices_path $OUTLIERS_NPZ --noise_scale $NOISE_SCALE"
        fi
        if [ -n "$GRADNORM_JSON" ]; then
            CMD_ARGS="$CMD_ARGS --gradnorm_json $GRADNORM_JSON --gradnorm_mode $GRADNORM_MODE"
            [ -n "$GRADNORM_TOPK" ] && CMD_ARGS="$CMD_ARGS --gradnorm_topk $GRADNORM_TOPK"
        fi
        if [ "$USE_LORA" = true ]; then
            CMD_ARGS="$CMD_ARGS --use_lora --lora_r $LORA_R --lora_alpha $LORA_ALPHA --lora_dropout $LORA_DROPOUT"
        fi

        python "${PROJECT_ROOT}/baselines/unlearn.py" \
            --algo $ALGO \
            --model_dir $TARGET_DIR --tokenizer_dir $LLAMA_DIR \
            --data_file $FORGET --retain_data_file $RETAIN \
            --out_dir "$OUT_DIR" \
            --max_len $MAX_LEN --epochs $EPOCHS --lr $LR \
            --alpha 1.0 --threshold 90 \
            --loss_scale $LOSS_SCALE_CURRENT \
            --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
            $CMD_ARGS
    done
    done
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo ""
if [ "${RUN_EVAL}" = true ]; then
    echo "=========================================="
    echo "Running evaluation script..."
    echo "=========================================="
    cd "${PROJECT_ROOT}"
    bash eval.sh books
    echo "Evaluation completed!"
fi

