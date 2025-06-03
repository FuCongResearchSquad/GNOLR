source ~/anaconda3/etc/profile.d/conda.sh
conda activate base

if [ $# -ne 2 ]; then
    echo "Usage:"
    echo "  Train:   $0 <device> <inference>"
    echo "  Example: $0 cuda:0 false"
    echo "  Inference: $0 <device> <inference>"
    echo "  Example: $0 cuda:0 true"
    exit 1
fi

DEVICE=$1
INFERENCE=$2

# default arguments
DATA_DIR="/GNOLR/data_process/aliccp"
LOG_DIR="/GNOLR/data_process/aliccp/log/multi-task"
CACHE_DIR="/GNOLR/cache"
THREADS=64
VERSION="v1"
MODEL="ipw"
LOSS="ipw_bceloss"
TASK_INDICES="[0,1]"
EPOCHS=20
VALID_INTERVAL=1  # ori 4
BATCH=1024
LR="0.01"
LR_DECAY_RATE="0"
LR_DECAY_STEP="0"

OUTPUT=2
DIM=16
MLP_LAYER="(128,64,32)"
DROPOUT="[0, 0, 0]"
POS_WEIGHT="[10,50]"

python -u /GNOLR/main.py --data_dir "$DATA_DIR" --log_dir "$LOG_DIR" --cache_dir "$CACHE_DIR" --data_loader_worker "4" --device "$DEVICE" --model "$MODEL" --loss "$LOSS" --l2_normalization "true" --valid_interval "$VALID_INTERVAL" --batch_size "$BATCH" --epochs "$EPOCHS" --dimension "$DIM" --mlp_layer "$MLP_LAYER" --dropout "$DROPOUT" --lr "$LR" --lr_decay_rate "$LR_DECAY_RATE" --lr_decay_step "$LR_DECAY_STEP" --output "$OUTPUT" --task_indices "$TASK_INDICES" --version "$VERSION" --num_threads "$THREADS" --pos_weight "$POS_WEIGHT"  --inference "$INFERENCE" --is_list "false"

if [ $? -ne 0 ]; then
    echo "Python script failed"
    exit 1
fi

echo "Python script succeeded!"
