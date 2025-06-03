"""
DSSM CIKM-2013: https://dl.acm.org/doi/10.1145/2505515.2505665
"""

import sys
import torch
import torch.nn as nn
from model.Tower import Tower
from model.MMoETower import MMoETower
from model.MMFiTower import MMFiModel
from util.utils import embedding_list_to_slate, slate_to_list

sys.path.append("../")


class DSSM(nn.Module):
    def __init__(
        self,
        user_datatypes,
        item_datatypes,
        user_dnn_size=(256, 128),
        l2_normalization=False,
        similarity="dot",
        item_dnn_size=(256, 128),
        dropout=[0, 0],
        activation="ReLU",
        use_senet="false",
        dimensions=16,
        output=1,
        loss="becloss",
        tower="base",
    ):
        super().__init__()
        self.user_dnn_size = user_dnn_size
        self.item_dnn_size = item_dnn_size
        self.dropout = dropout
        self.user_datatypes = user_datatypes
        self.item_datatypes = item_datatypes
        self.l2_normalization = l2_normalization
        self.similarity = None
        self.loss = loss
        if self.loss == "jrc":
            self.user_dnn_size = self.user_dnn_size[:-1] + (self.user_dnn_size[-1] * 2,)
            self.item_dnn_size = self.item_dnn_size[:-1] + (self.item_dnn_size[-1] * 2,)
        # Tower can be reused
        if tower == "base":
            print("use base")
            self.user_tower = Tower(
                self.user_datatypes,
                self.user_dnn_size,
                self.dropout,
                activation=activation,
                use_senet=use_senet,
                dimensions=dimensions,
                output=output,
            )
            self.item_tower = Tower(
                self.item_datatypes,
                self.item_dnn_size,
                self.dropout,
                activation=activation,
                use_senet=use_senet,
                dimensions=dimensions,
                output=output,
            )
        elif tower == "MMoE":
            print("use MMoE")
            self.user_tower = MMoETower(
                self.user_datatypes,
                self.user_dnn_size,
                self.dropout,
                activation=activation,
                use_senet=use_senet,
                dimensions=dimensions,
                output=output,
            )
            self.item_tower = MMoETower(
                self.item_datatypes,
                self.item_dnn_size,
                self.dropout,
                activation=activation,
                use_senet=use_senet,
                dimensions=dimensions,
                output=output,
            )
        elif tower == "MMFi":
            self.user_tower = MMFiModel(
                self.user_datatypes,
                self.user_dnn_size,
                self.dropout,
                activation=activation,
                use_senet=use_senet,
                dimensions=dimensions,
                output=output,
            )
            self.item_tower = MMFiModel(
                self.item_datatypes,
                self.item_dnn_size,
                self.dropout,
                activation=activation,
                use_senet=use_senet,
                dimensions=dimensions,
                output=output,
            )

    def forward(self, data, group=None):
        user_embeddings = self.user_tower(data)
        item_embeddings = self.item_tower(data)
        if self.loss == "jrc":
            user_embeddings = user_embeddings.view(
                user_embeddings.shape[0],
                user_embeddings.shape[1] * 2,
                user_embeddings.shape[2] // 2,
            )
            item_embeddings = item_embeddings.view(
                item_embeddings.shape[0],
                item_embeddings.shape[1] * 2,
                item_embeddings.shape[2] // 2,
            )
        if self.l2_normalization.lower() == "true":
            # [batch, task , dim]
            # print("[Debug] L2_Norm Open.")
            user_embeddings = user_embeddings / torch.norm(
                user_embeddings, p=2, dim=-1, keepdim=True
            )
            item_embeddings = item_embeddings / torch.norm(
                item_embeddings, p=2, dim=-1, keepdim=True
            )
        if self.similarity != None:
            # Multi-task is not currently supported, deprecated
            user_embeddings = user_embeddings[:, 0, :]
            item_embeddings = item_embeddings[:, 0, :]
            user_embeddings, item_embeddings = embedding_list_to_slate(
                user_embeddings, item_embeddings, group
            )
            similarities = self.similarity(
                input_embeddings=user_embeddings, item_embeddings=item_embeddings
            )
            similarities = slate_to_list(similarities, group)
            similarities = similarities.unsqueeze(dim=-1)
        else:
            similarities = (user_embeddings * item_embeddings).sum(dim=-1)

        return similarities, user_embeddings, item_embeddings

    def predict(self, data, group=None, use_sigmoid=True):
        similarities, user_embeddings, item_embeddings = self.forward(data, group=group)
        if self.loss.lower() == "jrc":
            # The length of the final output of JRC is doubled
            similarities = similarities.view(
                similarities.shape[0], similarities.shape[1] // 2, 2
            )
            similarities = similarities[:, :, 1] - similarities[:, :, 0]

        if use_sigmoid:
            # print('using sigmoid')
            similarities = torch.sigmoid(similarities)

        return similarities, user_embeddings, item_embeddings

    def get_olr_prob_naive(self, data, S, a, group=None):
        with torch.no_grad():
            similarities, user_embeddings, item_embeddings = self(data, group=group)

            S = torch.tensor(S).to(similarities.device)
            logit = similarities * S

            out_sim = torch.sigmoid(logit - torch.tensor([a]).to(similarities.device))

        return out_sim, user_embeddings, item_embeddings

    def get_olr_prob_v2_right(self, data, S, a, group=None):
        with torch.no_grad():
            similarities, user_embeddings, item_embeddings = self(data, group=group)

            S = torch.tensor(S).to(similarities.device)
            logit = similarities * S

            logit[:, 1] += logit[:, 0]

            out_sim = torch.sigmoid(logit - torch.tensor([a]).to(similarities.device))

        return out_sim, user_embeddings, item_embeddings

    def get_olr_prob_v2_with_learnable_a(self, data, S, a_param, group=None):
        with torch.no_grad():
            similarities, user_embeddings, item_embeddings = self(data, group=group)

            S = torch.tensor(S).to(similarities.device)
            logit = similarities * S

            logit[:, 1] += logit[:, 0]

            # a_actual = torch.cumsum(torch.exp(a_param), dim=0)
            # out_sim = torch.sigmoid(logit - a_actual)

            out_sim = torch.sigmoid(logit - a_param)

        return out_sim, user_embeddings, item_embeddings

    def get_olr_prob_v2_with_multi_feedback(self, data, S, a, group=None):
        with torch.no_grad():
            similarities, user_embeddings, item_embeddings = self(data, group=group)

            S = torch.tensor(S).to(similarities.device)
            logit = similarities * S

            # logit[:, 1] += logit[:, 0]

            for i in range(1, logit.shape[1]):
                logit[:, i] += logit[:, i - 1]

            out_sim = torch.sigmoid(logit - torch.tensor([a]).to(similarities.device))

        return out_sim, user_embeddings, item_embeddings

    def get_olr_prob_with_multi_embd(self, data, S, a, group=None):
        with torch.no_grad():
            similarities, user_embeddings, item_embeddings = self(data, group=group)

            S = torch.tensor(S).to(similarities.device)
            logit = similarities * S

            # logit[:, 1] += logit[:, 0]

            out_sim = torch.sigmoid(logit - torch.tensor([a]).to(similarities.device))

        return out_sim, user_embeddings, item_embeddings

    def get_olr_prob_v2_pos_neg(self, data, S, a, group=None):
        with torch.no_grad():
            similarities, user_embeddings, item_embeddings = self(data, group=group)

            S = torch.tensor(S).to(similarities.device)
            logit = similarities * S

            logit[:, 1] += logit[:, 0]
            logit[:, 2] += logit[:, 0]

            out_sim = torch.sigmoid(logit - torch.tensor([a]).to(similarities.device))

        return out_sim, user_embeddings, item_embeddings

    def get_embedding(self, data, user_embeddings, item_embeddings):
        """
        For Naive OLR
        user_embedding / item_embedding: [batch_size, 1, hidden_size] -> [batch_size, 1 * hidden_size]

        For GNOLR
        user_embedding / item_embedding: [batch_size, 2, hidden_size] -> [batch_size, 2 * hidden_size]
        """
        with torch.no_grad():
            _, _, similarities, user_embeddings, item_embeddings = self(
                data, user_embeddings, item_embeddings
            )

        reshaped_user_embeddings = user_embeddings.reshape(user_embeddings.shape[0], -1)
        reshaped_item_embeddings = item_embeddings.reshape(item_embeddings.shape[0], -1)

        return similarities, reshaped_user_embeddings, reshaped_item_embeddings
