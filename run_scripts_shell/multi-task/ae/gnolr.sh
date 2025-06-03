source ~/anaconda3/etc/profile.d/conda.sh
conda activate base

if [ $# -lt 2 ] || [ $# -gt 6 ]; then
    echo "Usage:"
    echo "  Train:   $0 <device> <inference> [REGION1 REGION2 ...]"
    echo "  Example: $0 cuda:0 false"
    echo "  Inference: $0 <device> <inference> [REGION1 REGION2 ...]"
    echo "  Example: $0 cuda:0 true ES FR"
    echo "Note: You can provide up to 4 regions (default: ES FR NL US)."
    exit 1
fi

DEVICE=$1
INFERENCE=$2
shift 2
LIST=("$@")

if [ ${#LIST[@]} -eq 0 ]; then
    LIST=("ES" "FR" "NL" "US")
fi

# default arguments
THREADS=64
VERSION="v1"
MODEL="gnolr"
LOSS="multi_gnolr"
TASK_INDICES="[0,1]"
EPOCHS=10
VALID_INTERVAL=1  # ori 4
BATCH=1024
LR="5"
LR_DECAY_RATE="0"
LR_DECAY_STEP="0"

OUTPUT=2
DIM=16
MLP_LAYER="(128,64,32)"
DROPOUT="[0, 0, 0]"
POS_WEIGHT="[1,1]"

for REGION in "${LIST[@]}"
do
    DATA_DIR="/GNOLR/data_process/ae/$REGION"
    LOG_DIR="/GNOLR/data_process/ae/$REGION/log/multi-task"
    CACHE_DIR="/GNOLR/cache"

    if [ "$REGION" == "ES" ]; then
        S="[1.0,1.0]"
        M="[0.027315173022208962,0.0006033428212804473]"
    elif [ "$REGION" == "FR" ]; then
        S="[3.0,3.0]"
        M="[0.02048677439284746,0.0005340257089524359]"
    elif [ "$REGION" == "US" ]; then
        S="[2.0,2.0]"
        M="[0.01668737395847271,0.0003955184364728915]"
    elif [ "$REGION" == "NL" ]; then
        S="[3.0,3.0]"
        M="[0.024008110193059744,0.0008388634611777494]"
    else
        echo "Invalid region"
        exit 1
    fi

    python -u /GNOLR/main.py --data_dir "$DATA_DIR" --log_dir "$LOG_DIR" --cache_dir "$CACHE_DIR" --data_loader_worker "4" --device "$DEVICE" --model "$MODEL" --loss "$LOSS" --l2_normalization "true" --valid_interval "$VALID_INTERVAL" --batch_size "$BATCH" --epochs "$EPOCHS" --dimension "$DIM" --mlp_layer "$MLP_LAYER" --dropout "$DROPOUT" --lr "$LR" --lr_decay_rate "$LR_DECAY_RATE" --lr_decay_step "$LR_DECAY_STEP" --output "$OUTPUT" --task_indices "$TASK_INDICES" --version "$VERSION" --num_threads "$THREADS" --pos_weight "$POS_WEIGHT"  --inference "$INFERENCE" --is_list "false" --S1 "$S" --m "$M"
done

if [ $? -ne 0 ]; then
    echo "Python script failed"
    exit 1
fi

echo "Python script succeeded!"
