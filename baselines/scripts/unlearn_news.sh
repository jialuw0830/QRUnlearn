# 实验配置：ga_avg_klr, npo_avg_klr, ga_gdr, npo_gdr
# 每个配置在三个 sensitivity (0.05, 0.1, 0.5) 下运行
# alpha is for utility constraint, threshold is for filtering out salient modules
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
MODEL_ROOT=${MODEL_ROOT:-/playpen-shared/huaizhi/jialu/models}
CORPUS=('news')
FORGET="${PROJECT_ROOT}/data/$CORPUS/raw/forget.txt"
RETAIN="${PROJECT_ROOT}/data/$CORPUS/raw/retain1.txt"
TARGET_DIR="${MODEL_ROOT}/MUSE-News_target"
LLAMA_DIR="${MODEL_ROOT}/Llama-2-7b-hf"
MAX_LEN=2048
PER_DEVICE_BATCH_SIZE=2

# Learning rate search
LR_CANDIDATES=('1e-5')

# 实验配置：格式 "algo:loss_scale:epochs:sensitivity"
EXPERIMENT_CONFIGS=(
    "ga_gdr:1:10:0.1"
    "npo_gdr:1:10:0.5"
)

# Outlier sensitivity: 0.05%, 0.1%, 0.5%
SENSITIVITY_PERCENTS=(0.1)

OUTLIER_MODE="layer"
OUTLIER_BASE_DIR="${PROJECT_ROOT}/outputs/outliers/news"  # from scripts/find_outliers.sh

ADD_NOISE=true
NOISE_MODE='row'
NOISE_SCALE=1.0

GRADNORM_JSON="${PROJECT_ROOT}/outputs/probe/news.json"  # from scripts/probe.sh
GRADNORM_TOPK=10
GRADNORM_PERCENT=""
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
    # Parse experiment config: algo:loss_scale:epochs:sensitivity
    IFS=':' read -r ALGO LOSS_SCALE_CURRENT EPOCHS SENS_PERCENT_CONFIG <<< "$EXP_CONFIG"
    
    # Use config sensitivity if provided, otherwise use first from SENSITIVITY_PERCENTS
    if [ -n "$SENS_PERCENT_CONFIG" ]; then
        SENSITIVITY_PERCENTS_USE=("$SENS_PERCENT_CONFIG")
    else
        SENSITIVITY_PERCENTS_USE=("${SENSITIVITY_PERCENTS[@]}")
    fi
    
    LOSS_SCALE_FORMAT=$(echo "${LOSS_SCALE_CURRENT}" | sed 's/\./_/')
    LOSS_SCALE_TAG="_lossscale${LOSS_SCALE_FORMAT}"
    EPOCH_TAG="_epoch${EPOCHS}"
    
    echo "=========================================="
    echo "Processing: ALGO=$ALGO, LOSS_SCALE=$LOSS_SCALE_CURRENT, EPOCHS=$EPOCHS"
    echo "=========================================="
    
    # Loop over sensitivity percentages
    for SENS_PERCENT in "${SENSITIVITY_PERCENTS_USE[@]}"; do
        SENS_FORMAT=$(echo "${SENS_PERCENT}" | sed 's/\./_/')

        OUTLIERS_NPZ="${OUTLIER_BASE_DIR}/forget_specific_outliers_sens${SENS_FORMAT}.npz"

        if [ ! -f "$OUTLIERS_NPZ" ]; then
            echo "ERROR: Outlier file not found: $OUTLIERS_NPZ"
            echo "Skipping algo=$ALGO sensitivity=$SENS_PERCENT..."
            continue
        fi

        OUTLIER_FILE=$(basename "${OUTLIERS_NPZ}")
        OUTLIER_TYPE=$(echo "${OUTLIER_FILE}" | sed 's/_outliers\.npz$//')
        NOISE_TAG=$([ "$ADD_NOISE" = true ] && echo "_row_${OUTLIER_TYPE}_noise${NOISE_SCALE_FORMAT}" || echo "")
        MODE_TAG="_${OUTLIER_MODE}"

        OUT_DIR="${PROJECT_ROOT}/outputs/ckpt/news/${CORPUS}_${ALGO}${TOPK_TAG}_${LR_FORMAT}${NOISE_TAG}${MODE_TAG}${LOSS_SCALE_TAG}${EPOCH_TAG}"
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
            [ -n "$GRADNORM_PERCENT" ] && CMD_ARGS="$CMD_ARGS --gradnorm_percent $GRADNORM_PERCENT"
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
