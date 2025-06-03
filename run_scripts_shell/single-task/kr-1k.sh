source ~/anaconda3/etc/profile.d/conda.sh
conda activate base

if [ $# -ne 2 ]; then
    echo "train: ./run.sh 1 false"
    echo "inference: ./run.sh 1 true"
    exit 1
fi

DEVICE=$1
INFERENCE=$2

# default arguments
DATA_DIR="/GNOLR/data_process/kuai_rand/KuaiRand-1K"
LOG_DIR="/GNOLR/data_process/kuai_rand/KuaiRand-1K/log/single-task"
CACHE_DIR="/GNOLR/cache"
THREADS=64
VERSION="v1"
MODEL="dssm"
OUTPUT=1
TASK_INDICES="[1]"
EPOCHS=100
VALID_INTERVAL=4
BATCH=1024
BATCH_L=32
LR=0.05
LR_DECAY_RATE=0
LR_DECAY_STEP=0
S="2.8"
M="0.019580012753444488"
POS_WEIGHT="[10,1]"

seed_list=("42")
baseline=("bceloss" "jrc" "single_gnolr")
baseline_l=("bceloss" "ranknet" "lambdarank" "listnet" "jrc" "set2setrank" "setrank" "single_gnolr" "single_gnolr_l")

for seed in "${seed_list[@]}"
do
    for loss in "${baseline[@]}"
    do
        python -u /GNOLR/main.py --data_dir "$DATA_DIR" --log_dir "$LOG_DIR" --cache_dir "$CACHE_DIR" --data_loader_worker "4" --device "$DEVICE" --model "$MODEL" --loss "$loss" --l2_normalization "true" --seed "$seed" --valid_interval "$VALID_INTERVAL" --batch_size "$BATCH" --epochs "$EPOCHS" --lr "$LR" --lr_decay_rate "$LR_DECAY_RATE" --lr_decay_step "$LR_DECAY_STEP" --output "$OUTPUT" --task_indices "$TASK_INDICES" --version "$VERSION" --num_threads "$THREADS" --inference "$INFERENCE" --is_list "false" --S1 "$S" --S2 "$S" --m "$M" --pos_weight "$POS_WEIGHT"
        echo "$loss done"
    done
    for loss in "${baseline_l[@]}"
    do
        python -u /GNOLR/main.py --data_dir "$DATA_DIR" --log_dir "$LOG_DIR" --cache_dir "$CACHE_DIR" --data_loader_worker "4" --device="$DEVICE" --model "$MODEL" --loss "$loss" --l2_normalization "true" --seed="$seed" --valid_interval "$VALID_INTERVAL" --batch_size "$BATCH_L" --epochs "$EPOCHS" --lr "$LR" --lr_decay_rate "$LR_DECAY_RATE" --lr_decay_step "$LR_DECAY_STEP" --output "$OUTPUT" --task_indices="$TASK_INDICES" --version "$VERSION" --num_threads="$THREADS" --inference "$INFERENCE" --is_list="true" --S1 "$S" --S2 "$S" --m "$M" --pos_weight "$POS_WEIGHT"
        echo "$loss done"
    done
done

echo "Finish."
