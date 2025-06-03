"""
Naive Share Bottom
"""

import sys
import torch
import torch.nn as nn
from model.Tower import Tower
from model.MMoETower import MMoETower
from model.MMFiTower import MMFiModel

sys.path.append("../")


class NSB(nn.Module):
    def __init__(
        self,
        user_datatypes,
        item_datatypes,
        user_dnn_size=(256, 128),
        l2_normalization=False,
        similarity="dot",  # tower: base, MMoE
        item_dnn_size=(256, 128),
        dropout=[0, 0],
        activation="ReLU",
        use_senet=False,
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
        self.loss = loss
        if tower == "base":
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
        # print('no dot')

    def forward(self, data, group=None):
        # user vec & # item vec
        user_embeddings = self.user_tower(data)
        item_embeddings = self.item_tower(data)

        # l2 norm
        if self.l2_normalization.lower() == "true":
            # print("[Debug][NSE][L2 Norm]")
            user_embeddings = user_embeddings / torch.norm(
                user_embeddings, p=2, dim=-1, keepdim=True
            )
            item_embeddings = item_embeddings / torch.norm(
                item_embeddings, p=2, dim=-1, keepdim=True
            )

        # inner product
        similarities = (user_embeddings * item_embeddings).sum(dim=-1)

        return similarities, user_embeddings, item_embeddings

    def predict(self, data, group=None, use_sigmoid=True, residual_add=False):
        # inner product
        similarities, user_embeddings, item_embeddings = self.forward(data, group=group)

        # pctr, pctcvr, pctcvrlike, pctcvrlikefollow
        out_sim = torch.sigmoid(similarities)

        # out_sim[:, 1] = out_sim[:, 0] * out_sim[:, 1]

        return out_sim, user_embeddings, item_embeddings
