import os
import json
import torch
import random
import logging
import argparse
import warnings
import numpy as np
import multiprocessing as mp
from script.DSSMTrainable import DSSMTrainable
from data_process.data_loader import data_loader, recall_data_loader

warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)


# Please DO NOT change the default hyper-parameters here. Modify them in the running script instead.
def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--S1", type=str, default="1")  # gnolr
    parser.add_argument("--S2", type=str, default="1")  # gnolr + listnet
    parser.add_argument("--m", type=str, default="0.5")  # a = -ln(m)

    # Make sure the dataset directory includes the 'train' and 'test' folders, along with the 'feature.json' file.
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data_process/movie_lens/ml-1m",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./data_process/movie_lens/ml-1m/log/single-task",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
    )
    parser.add_argument("--data_loader_worker", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")

    # single-task: "dssm"
    # multi-task: "nsb" "esmm" "ipw" "dr" "dcmt" "nise" "tafe" "nolr" "gnolr"
    parser.add_argument("--model", type=str, default="nsb")

    # single-task: "bceloss" "ranknet" "lambdarank" "listnet" "setrank" "set2setrank" "jrc"
    # multi-task: "multi_bceloss" "esmm_bceloss" "ipw_bceloss" "dr_bceloss" "dcmt_bceloss" "nise_bceloss" "tafe_bceloss" "multi_naive_olr" "multi_gnolr"
    parser.add_argument("--loss", type=str, default="multi_bceloss")

    parser.add_argument("--similarity", type=str, default="dot")
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--use_senet", type=str, default="false")
    parser.add_argument("--activation", type=str, default="LeakyReLU")
    parser.add_argument("--l2_normalization", type=str, default="true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid_interval", type=int, default=4)

    # single-task: 32
    # multi-task: 1024
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--dimension", type=int, default=16)
    parser.add_argument("--mlp_layer", type=str, default="(128, 64, 32)")
    parser.add_argument("--dropout", type=str, default="[0, 0, 0]")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--lr_decay_rate", type=float, default=0)
    parser.add_argument("--lr_decay_step", type=int, default=0)

    # single-task: 1
    # multi-task: (dr, dcmt)=3, other=2
    parser.add_argument("--output", type=int, default=2)

    # single-task: (ml-1m, ml-20m, aliccp, ae, retailrocket)=[0], (kr-pure, kr-1k)=[1]
    # multi-task: (aliccp, ae, retailrocket)=[0,1], (kr-pure, kr-1k)=[1,2]
    parser.add_argument("--task_indices", type=str, default="[0,1]")

    # sample reweighting
    # single-task: [1]
    # multi-task: [1,1]
    parser.add_argument("--pos_weight", type=str, default="[1,1]")

    # dssm, MMoE, MMFI
    parser.add_argument("--base_tower", type=str, default="base")

    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--num_threads", type=int, default=64)
    parser.add_argument("--inference", type=str, default="false")
    parser.add_argument("--is_list", type=str, default="false")
    parser.add_argument("--is_trainable_a", type=str, default="false")

    # arguments for recall_exp
    parser.add_argument("--is_recall", type=str, default="false")
    parser.add_argument(
        "--recall_dir", type=str, default="/GNOLR/data_process/movie_lens/ml-1m"
    )
    parser.add_argument("--embedding_type", type=str, default="user/item")
    parser.add_argument("--dataset_type", type=str, default="kr")

    args = parser.parse_args()

    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.INFO


def main():
    logging.info(json.dumps(vars(opt)))
    print(json.dumps(vars(opt)))

    (
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        user_feature,
        item_feature,
        label_feature,
    ) = data_loader(
        dir=opt.data_dir,
        batch_size=opt.batch_size,
        data_loader_worker=opt.data_loader_worker,
        is_list=opt.is_list,
        CACHE_DIR=opt.cache_dir,
    )
    print("----------data load finish----------")

    save_model_path = os.path.join(
        opt.data_dir, "parameter", opt.model.lower(), opt.version
    )
    os.makedirs(save_model_path, exist_ok=True)
    save_model = os.path.join(save_model_path, opt.model.lower() + "_" + opt.loss)

    trainable = None
    model_list = [
        "dssm",
        "nsb",
        "esmm",
        "ipw",
        "dr",
        "dcmt",
        "nise",
        "tafe",
        "nolr",
        "gnolr",
    ]
    if opt.inference.lower() == "true":
        train_dataloader = None

    if opt.model.lower() in model_list:
        trainable = DSSMTrainable(
            user_feature=user_feature,
            item_feature=item_feature,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            user_dnn_size=eval(opt.mlp_layer),
            item_dnn_size=eval(opt.mlp_layer),
            similarity=opt.similarity,
            output=opt.output,
            loss_func=opt.loss,
            dropout=eval(opt.dropout),
            activation=opt.activation,
            use_senet=opt.use_senet,
            valid_interval=opt.valid_interval,
            dimensions=opt.dimension,
            model_path=save_model,
            device=opt.device,
            model=opt.model,
            l2_normalization=opt.l2_normalization,
            S1=eval(opt.S1),
            S2=eval(opt.S2),
            m=eval(opt.m),
            tower=opt.base_tower,
        )

    task_indices = eval(opt.task_indices)

    if opt.model.lower() in model_list:
        if opt.inference.lower() == "false":
            trainable.test_dataloader = valid_dataloader
            trainable.train(
                epochs=opt.epochs,
                optimizer="Adam",
                lr=opt.lr,
                lr_decay_rate=opt.lr_decay_rate,
                lr_decay_step=opt.lr_decay_step,
                task_indices=task_indices,
                pos_weight=eval(opt.pos_weight),
                trainable_a=opt.is_trainable_a,
            )

            trainable.test_dataloader = test_dataloader
            if opt.model.lower() == "dssm":
                metrics = trainable.test(
                    task_indices, is_list=opt.is_list, inference="true"
                )
            elif opt.model.lower() in ["nolr", "gnolr"]:
                metrics = trainable.test_olr(opt.output, task_indices, inference="true")
            else:
                metrics = trainable.test_multi_tasks(
                    opt.output, task_indices, inference="true"
                )

            print(str(metrics))
            logging.info(str(metrics))
        else:
            if opt.model.lower() == "dssm":
                metrics = trainable.test(
                    task_indices, is_list=opt.is_list, inference="true"
                )
            elif opt.model.lower() in ["nolr", "gnolr"]:
                metrics = trainable.test_olr(opt.output, task_indices, inference="true")
            else:
                metrics = trainable.test_multi_tasks(
                    opt.output, task_indices, inference="true"
                )

            print(str(metrics))
            logging.info(str(metrics))


def recall():
    logging.info(json.dumps(vars(opt)))
    print(json.dumps(vars(opt)))

    train_dataloader, test_dataloader, user_feature, item_feature, label_feature = (
        recall_data_loader(
            dir=opt.data_dir,
            batch_size=opt.batch_size,
            data_loader_worker=opt.data_loader_worker,
            embedding_type=opt.embedding_type,
            is_list=opt.is_list,
            DATASET_TYPE=opt.dataset_type,
            CACHE_DIR=opt.cache_dir,
        )
    )
    print("----------data load finish----------")

    save_model_path = os.path.join(
        opt.data_dir, "parameter", opt.model.lower(), opt.version
    )
    os.makedirs(save_model_path, exist_ok=True)
    saved_model_path = os.path.join(save_model_path, opt.model.lower() + "_" + opt.loss)

    print("[Recall Exp]Saved Checkpoint Path: ", saved_model_path)

    trainable = None

    if opt.model.lower() == "gnolr" or opt.model.lower() == "nsb":
        print(f"[RecallExp]Load DSSMTrainable [Model] {opt.model.lower()}")
        trainable = DSSMTrainable(
            user_feature=user_feature,
            item_feature=item_feature,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            user_dnn_size=eval(opt.mlp_layer),
            item_dnn_size=eval(opt.mlp_layer),
            similarity=opt.similarity,
            output=opt.output,
            loss_func=opt.loss,
            dropout=opt.dropout,
            activation=opt.activation,
            use_senet=opt.use_senet,
            valid_interval=opt.valid_interval,
            dimensions=opt.dimension,
            model_path=saved_model_path,
            device=opt.device,
            model=opt.model,
            l2_normalization=opt.l2_normalization,
            S1=eval(opt.S1),
            S2=eval(opt.S2),
            m=eval(opt.m),
            tower=opt.base_tower,
        )

    # Dump Item/User Tensor
    recall_index_path = os.path.join(opt.recall_dir, opt.model.lower() + "_" + opt.loss)
    print(
        f"[RecallExp]DSSMTrainable Dump Start...[Embedding Type] {opt.embedding_type}, [PATH] {recall_index_path}"
    )
    trainable.recall_exp_dump_tensor(
        embedding_type=opt.embedding_type,
        task_indices=eval(opt.task_indices),
        recall_index_path=recall_index_path,
    )


if __name__ == "__main__":
    opt = parse_opt()
    if opt.num_threads != 0:
        torch.set_num_threads(opt.num_threads)

    mp.set_start_method("spawn")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    os.makedirs(opt.log_dir, exist_ok=True)

    file_handler = logging.FileHandler(
        os.path.join(opt.log_dir, opt.model + "_" + opt.version + ".log"), mode="a"
    )

    file_handler.setLevel(logging.INFO)
    file_handler.addFilter(InfoFilter())
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    set_seed(opt.seed)
    if opt.is_recall.lower() == "false":
        main()
    elif opt.is_recall.lower() == "true":
        recall()
