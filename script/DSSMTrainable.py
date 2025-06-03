"""
Twin-Tower train
"""

import os
import math
import torch
import logging
import numpy as np
from tqdm import *
from util.loss import get_loss
from util.utils import get_optimizer
from util.metric import get_metric
from util.get_model import get_model
from sklearn.metrics import roc_auc_score


class DSSMTrainable:
    def __init__(
        self,
        user_feature=None,
        item_feature=None,
        train_dataloader=None,
        test_dataloader=None,
        output=1,
        similarity="dot",
        user_dnn_size=(64, 32),
        item_dnn_size=(64, 32),
        loss_func="bceloss",
        model="dssm",
        activation="LeakyReLU",
        device="cpu",
        dropout=[0, 0],
        l2_normalization=False,
        use_senet="false",
        model_path=None,
        dimensions=16,
        valid_interval=4,
        S1=1.5,
        S2=1.5,
        m=0.2,
        tower="base",
    ):
        self.S1 = S1
        self.S2 = S2
        self.m = m
        self.loss_func = loss_func
        self.use_senet = use_senet
        self.model_path = model_path
        self.device = device
        self.output = output
        self.test_dataloader = test_dataloader
        self.valid_interval = valid_interval
        self.a_param = torch.empty(self.output).normal_().to(self.device)

        if train_dataloader is None:
            self.model = torch.load(
                model_path + ".pt", map_location=torch.device(device)
            )
        else:
            self.train_dataloader = train_dataloader
            print(tower)
            self.model = get_model(
                model,
                user_feature,
                item_feature,
                user_dnn_size=user_dnn_size,
                item_dnn_size=item_dnn_size,
                output=self.output,
                similarity=similarity,
                dropout=dropout,
                activation=activation,
                use_senet=self.use_senet,
                dimensions=dimensions,
                l2_normalization=l2_normalization,
                loss=loss_func,
                tower=tower,
            )
        self.model = self.model.to(device=self.device)

    def train(
        self,
        epochs=100,
        optimizer="Adam",
        lr=1e-4,
        lr_decay_rate=0,
        lr_decay_step=0,
        task_indices=[0],
        pos_weight=[1, 1],
        trainable_a="false",
    ):
        if trainable_a.lower() == "false":
            optimizer = get_optimizer(optimizer, self.model.parameters(), lr)
        elif trainable_a.lower() == "true":
            print("Using trainable A")
            self.a_param.requires_grad_()
            optimizer = get_optimizer(
                optimizer, list(self.model.parameters()) + [self.a_param], lr
            )
        else:
            raise ValueError(f"Unknown trainable_a {trainable_a}")

        val_auc = 0
        best_val_auc = 0
        early_stop = 0

        for epoch in range(epochs):
            epoch = epoch + 1
            self.model.train()

            total_loss = 0

            for data in tqdm(self.train_dataloader):
                # For single-task, x contains a batch_size of users,
                #   and [batch] is equal to the sum of the number of interactions between each user
                # x: [batch, feature_num]
                # y: [batch, task]
                # sum(group_list) = batch
                x, y, group_list = data

                x, y = x.to(self.device), y.to(self.device)

                # y: [batch, task] -> [batch, sub_task]
                y = y[:, task_indices]

                # y_logit: [batch, task]
                # user_emb: [batch, task, user_dnn_size[-1]]
                # item_emb: [batch, task, item_dnn_size[-1]]
                y_logit, user_emb, item_emb = self.model(x, group=group_list)

                loss = get_loss(
                    loss=self.loss_func,
                    true=y,
                    pred=y_logit,
                    group=group_list,
                    S1=self.S1,
                    S2=self.S2,
                    m=self.m,
                    pos_weight=pos_weight,
                    a_param=self.a_param,
                )

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                total_loss += loss.item()

            print("epoch:{}, train loss:{:.5}   ".format(epoch, total_loss))

            if epoch % self.valid_interval == 0:
                if self.output == 1 and len(task_indices) == 1:
                    metrics = self.test(task_indices)

                    print("epoch:{},".format(epoch) + str(metrics))
                    logging.info("epoch:{},".format(epoch) + str(metrics))

                    val_auc = metrics["auc"]
                    early_stop += 1

                    if val_auc > best_val_auc:
                        self.save_model()
                        best_val_auc = val_auc
                        early_stop = 0
                    if early_stop > 2:
                        break
                else:
                    if self.loss_func.lower() in [
                        "multi_naive_olr",
                        "multi_naive_olr_with_multi_embd",
                        "multi_gnolr_without_nested_opt",
                        "multi_gnolr",
                        "multi_gnolr_with_multi_feedback",
                        "multi_gnolr_with_learnable_a",
                        "multi_gnolr_with_pos_neg",
                    ]:
                        metrics = self.test_olr(self.output, task_indices)
                    else:
                        metrics = self.test_multi_tasks(self.output, task_indices)

                    print("epoch:{},".format(epoch) + str(metrics))
                    logging.info("epoch:{},".format(epoch) + str(metrics))

                    val_auc = 1.0
                    for i in range(len(task_indices)):
                        val_auc *= metrics[f"task_{i}"]["auc"]
                    early_stop += 1

                    if val_auc > best_val_auc:
                        self.save_model()
                        best_val_auc = val_auc
                        early_stop = 0
                    if early_stop > 2:
                        break

        print("----------finish train----------")
        return best_val_auc

    def test_multi_tasks(self, output, task_indices, inference="false"):
        if inference.lower() == "true":
            self.model = torch.load(
                self.model_path + ".pt", map_location=torch.device(self.device)
            )
        self.model.eval()

        num_tasks = len(task_indices)
        all_metrics = [None for _ in range(num_tasks)]
        all_labels = [list() for _ in range(num_tasks)]
        all_pre_pro = [list() for _ in range(num_tasks)]

        with torch.no_grad():
            for data in tqdm(self.test_dataloader):
                x, y, group_list = data
                x, y = x.to(self.device), y.to(self.device)

                y = y[:, task_indices]

                y_pre, user_emb, item_emb = self.model.predict(x, group=group_list)

                y_multi = torch.unbind(y, dim=1)
                if output == 1:
                    y_pre_multi = torch.squeeze(y_pre, dim=1)

                    for i in range(num_tasks):
                        all_pre_pro[i].extend(
                            y_pre_multi.cpu().detach().numpy().squeeze()
                        )
                        all_labels[i].extend(
                            y_multi[i].cpu().detach().numpy().squeeze()
                        )
                else:
                    y_pre_multi = torch.unbind(y_pre, dim=1)

                    for i in range(num_tasks):
                        all_pre_pro[i].extend(
                            y_pre_multi[i].cpu().detach().numpy().squeeze()
                        )
                        all_labels[i].extend(
                            y_multi[i].cpu().detach().numpy().squeeze()
                        )

        for i in range(num_tasks):
            all_metrics[i] = dict()
            all_metrics[i]["auc"] = roc_auc_score(
                np.array(all_labels[i]), np.array(all_pre_pro[i])
            )

        return_metrics = {}
        for i in range(num_tasks):
            return_metrics[f"task_{i}"] = all_metrics[i]

        return return_metrics

    def test_olr(self, output, task_indices, inference="false"):
        if inference.lower() == "true":
            self.model = torch.load(
                self.model_path + ".pt", map_location=torch.device(self.device)
            )
        self.model.eval()

        num_tasks = len(task_indices)
        all_metrics = [None for _ in range(num_tasks)]
        all_labels = [list() for _ in range(num_tasks)]
        all_pre_pro = [list() for _ in range(num_tasks)]

        with torch.no_grad():
            for data in tqdm(self.test_dataloader):
                x, y, group_list = data
                x, y = x.to(self.device), y.to(self.device)

                y = y[:, task_indices]

                if self.loss_func.lower() == "multi_gnolr":
                    y_pre, user_emb, item_emb = self.model.get_olr_prob_v2_right(
                        x,
                        S=self.S1,
                        a=[-math.log(m_i) for m_i in self.m],
                        group=group_list,
                    )
                elif self.loss_func.lower() == "multi_naive_olr":
                    y_pre, user_emb, item_emb = self.model.get_olr_prob_naive(
                        x,
                        S=self.S1,
                        a=[-math.log(m_i) for m_i in self.m],
                        group=group_list,
                    )
                elif self.loss_func.lower() == "multi_naive_olr_with_multi_embd":
                    y_pre, user_emb, item_emb = self.model.get_olr_prob_with_multi_embd(
                        x,
                        S=self.S1,
                        a=[-math.log(m_i) for m_i in self.m],
                        group=group_list,
                    )
                elif self.loss_func.lower() == "multi_gnolr_with_learnable_a":
                    y_pre, user_emb, item_emb = (
                        self.model.get_olr_prob_v2_with_learnable_a(
                            x,
                            S=self.S1,
                            a_param=self.a_param,
                            group=group_list,
                        )
                    )
                elif self.loss_func.lower() == "multi_gnolr_with_multi_feedback":
                    y_pre, user_emb, item_emb = (
                        self.model.get_olr_prob_v2_with_multi_feedback(
                            x,
                            S=self.S1,
                            a=[-math.log(m_i) for m_i in self.m],
                            group=group_list,
                        )
                    )
                elif self.loss_func.lower() == "multi_gnolr_without_nested_opt":
                    y_pre, user_emb, item_emb = self.model.get_olr_prob_v2_right(
                        x,
                        S=self.S1,
                        a=[-math.log(m_i) for m_i in self.m],
                        group=group_list,
                    )
                elif self.loss_func.lower() == "multi_gnolr_with_pos_neg":
                    y_pre, user_emb, item_emb = self.model.get_olr_prob_v2_pos_neg(
                        x,
                        S=self.S1,
                        a=[-math.log(m_i) for m_i in self.m],
                        group=group_list,
                    )

                if output == 1:
                    y_multi = torch.unbind(y, dim=1)
                    y_pre_multi = torch.unbind(y_pre, dim=1)

                    for i in range(num_tasks):
                        all_pre_pro[i].extend(
                            y_pre_multi[i].cpu().detach().numpy().squeeze()
                        )
                        all_labels[i].extend(
                            y_multi[i].cpu().detach().numpy().squeeze()
                        )
                else:
                    y_multi = torch.unbind(y, dim=1)
                    y_pre_multi = torch.unbind(y_pre, dim=1)

                    assert len(y_pre_multi) == num_tasks and len(y_multi) == num_tasks

                    for i in range(num_tasks):
                        all_pre_pro[i].extend(
                            y_pre_multi[i].cpu().detach().numpy().squeeze()
                        )
                        all_labels[i].extend(
                            y_multi[i].cpu().detach().numpy().squeeze()
                        )

        for i in range(num_tasks):
            all_metrics[i] = dict()
            all_metrics[i]["auc"] = roc_auc_score(
                np.array(all_labels[i]), np.array(all_pre_pro[i])
            )

            if self.loss_func.lower() == "multi_gnolr_with_learnable_a":
                all_metrics[i]["a"] = self.a_param[i]

        return_metrics = {}
        for i in range(num_tasks):
            return_metrics[f"task_{i}"] = all_metrics[i]

        return return_metrics

    def test(self, task_indices=[0], is_list="false", inference="false"):
        if inference.lower() == "true":
            self.model = torch.load(
                self.model_path + ".pt", map_location=torch.device(self.device)
            )

        self.model.eval()
        labels, pre_pro = list(), list()
        metrics = None

        with torch.no_grad():
            for data in tqdm(self.test_dataloader):
                x, y, group_list = data
                x, y = x.to(self.device), y.to(self.device)

                y = y[:, task_indices]
                y_pre, user_emb, item_emb = self.model.predict(x, group=group_list)
                pre_pro.extend(y_pre[:, -1].cpu().detach().numpy().squeeze())
                labels.extend(y.cpu().detach().numpy().squeeze())

                if is_list.lower() == "true":
                    metrics = get_metric(
                        metrics=metrics, true=y, pred=y_pre, group=group_list, ats=[5]
                    )

        if is_list.lower() == "true":
            metrics = {
                key: torch.mean(values).item() for key, values in metrics.items()
            }
        else:
            metrics = dict()

        metrics["auc"] = roc_auc_score(np.array(labels), np.array(pre_pro))
        return metrics

    def recall_exp_dump_tensor(self, embedding_type, task_indices, recall_index_path):
        self.model.eval()

        if os.path.exists(recall_index_path + "_" + embedding_type + ".npy"):
            os.remove(recall_index_path + "_" + embedding_type + ".npy")

        # dump label in test dataset
        if embedding_type == "user":
            if os.path.exists(recall_index_path + "_label.npy"):
                os.remove(recall_index_path + "_label.npy")

        with torch.no_grad():
            for data in tqdm(self.test_dataloader):
                x, y, sample_id, group_list, user_embeddings, item_embeddings = data
                (
                    x,
                    y,
                    sample_id,
                ) = (
                    x.to(self.device),
                    y.to(self.device),
                    sample_id.to(self.device),
                )

                user_embeddings = (
                    user_embeddings.to(self.device)
                    if user_embeddings is not None
                    else user_embeddings
                )
                item_embeddings = (
                    item_embeddings.to(self.device)
                    if item_embeddings is not None
                    else item_embeddings
                )

                # y: [batch, task] -> [batch, sub_task]
                y = y[:, task_indices]

                y_pre, user_index_emb, item_index_emb = self.model.get_embedding(
                    x, user_embeddings, item_embeddings
                )

                if embedding_type == "user":
                    user_id = sample_id[:, 0].cpu().numpy().reshape(-1, 1)  # user_id
                    index_emb = user_index_emb.cpu().numpy()
                    index_emb = np.concatenate((user_id, index_emb), axis=1)

                    # dump label
                    sample_id = sample_id.cpu().numpy()
                    label = y.cpu().numpy()
                    label_dt = np.concatenate((sample_id, label), axis=1)
                else:
                    assert embedding_type == "item"
                    item_id = sample_id[:, 1].cpu().numpy().reshape(-1, 1)  # item_id
                    index_emb = item_index_emb.cpu().numpy()
                    index_emb = np.concatenate((item_id, index_emb), axis=1)

                with open(recall_index_path + "_" + embedding_type + ".npy", "ab") as f:
                    np.save(f, index_emb)

                # dump label
                if embedding_type == "user":
                    with open(recall_index_path + "_label.npy", "ab") as f:
                        np.save(f, label_dt)

        print(f"All item embeddings saved to {recall_index_path}")

    def save_model(self):
        torch.save(self.model, self.model_path + ".pt")
        torch.save(self.model.state_dict(), self.model_path + ".pkl")
