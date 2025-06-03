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
THREADS=64
VERSION="v1"
MODEL="dssm"
OUTPUT=1
TASK_INDICES="[0]"
EPOCHS=10
VALID_INTERVAL=1
BATCH=1024
BATCH_L=32
LR=5
LR_DECAY_RATE=0
LR_DECAY_STEP=0
POS_WEIGHT="[10,1]"

seed_list=("42")
baseline=("bceloss" "jrc" "single_gnolr")
baseline_l=("bceloss" "ranknet" "lambdarank" "listnet" "jrc" "set2setrank" "setrank" "single_gnolr" "single_gnolr_l")

LIST=("ES" "FR" "NL" "US")

for REGION in "${LIST[@]}"
do
    DATA_DIR="/GNOLR/data_process/ae/$REGION"
    LOG_DIR="/GNOLR/data_process/ae/$REGION/log/single-task"
    CACHE_DIR="/GNOLR/cache"

    if [ "$REGION" == "ES" ]; then
        S="7.00"
        M="0.027315173022208962"
    elif [ "$REGION" == "FR" ]; then
        S="7.00"
        M="0.02048677439284746"
    elif [ "$REGION" == "US" ]; then
        S="8.00"
        M="0.01668737395847271"
    elif [ "$REGION" == "NL" ]; then
        S="2.80"
        M="0.024008110193059744"
    else
        echo "Invalid region"
        exit 1
    fi

    for seed in "${seed_list[@]}"
    do
        for loss in "${baseline[@]}"
        do
            python -u /GNOLR/main.py --data_dir "$DATA_DIR" --log_dir "$LOG_DIR" --cache_dir "$CACHE_DIR" --data_loader_worker "4" --device "$DEVICE" --model "$MODEL" --loss "$loss" --l2_normalization "true" --seed "$seed" --valid_interval "$VALID_INTERVAL" --batch_size "$BATCH" --epochs "$EPOCHS" --lr "$LR" --lr_decay_rate "$LR_DECAY_RATE" --lr_decay_step "$LR_DECAY_STEP" --output "$OUTPUT" --task_indices "$TASK_INDICES" --version "$VERSION" --num_threads "$THREADS" --inference "$INFERENCE" --is_list "false" --S1 "$S" --S2 "$S" --m "$M" --pos_weight "$POS_WEIGHT"
            echo "$loss done"
        done
        for loss in "${baseline_l[@]}"
        do
            python -u /GNOLR/main.py --data_dir "$DATA_DIR" --log_dir "$LOG_DIR" --cache_dir "$CACHE_DIR" --data_loader_worker "4" --device="$DEVICE" --model "$MODEL" --loss "$loss" --l2_normalization "true" --seed="$seed" --valid_interval "$VALID_INTERVAL" --batch_size "$BATCH_L" --epochs "$EPOCHS" --lr "$LR" --lr_decay_rate "$LR_DECAY_RATE" --lr_decay_step "$LR_DECAY_STEP" --output "$OUTPUT" --task_indices "$TASK_INDICES" --version "$VERSION" --num_threads="$THREADS" --inference "$INFERENCE" --is_list="true" --S1 "$S" --S2 "$S" --m "$M" --pos_weight "$POS_WEIGHT"
            echo "$loss done"
        done
    done
done

echo "Finish."
