"""
Embedding Module
"""

import torch
import torch.nn as nn
from model.SENet import SENet, LightSE


class EmbeddingModule(nn.Module):
    def __init__(self, datatypes, use_se_net, dimensions=16):
        super().__init__()
        self.dim = 0
        self.embedding_dim = 0
        self.embs = nn.ModuleList()
        self.datatypes = datatypes
        self.use_se_net = use_se_net
        self.num = 0
        for datatype in datatypes:
            if (
                datatype["type"] == "SparseEncoder"
                or datatype["type"] == "BucketSparseEncoder"
            ):
                self.embs.append(nn.Embedding(datatype["length"], dimensions))
                self.dim += dimensions
                self.num += 1
            if datatype["type"] == "MultiSparseEncoder":
                self.embs.append(
                    nn.EmbeddingBag(
                        datatype["length"], dimensions, mode="mean", padding_idx=0
                    )
                )
                self.dim += dimensions
                self.num += 1
            elif datatype["type"] == "DenseEncoder":
                self.embs.append(nn.Embedding(len(datatype["length"]), dimensions))
                self.dim += dimensions
                self.num += 1

        if self.use_se_net.lower() != "false":
            if self.use_se_net == "LightSE":
                self.se_net = LightSE(self.num)
            else:
                self.se_net = SENet(self.num)

    def run_emb(self, emb, input):
        return emb(input)

    def forward(self, x):
        emb_output = []
        se_net_input = []
        for index in range(len(self.datatypes)):
            datatype = self.datatypes[index]
            if datatype["type"] == "MultiSparseEncoder":
                vec = self.embs[index](
                    x[:, datatype["index"] : datatype["index"] + datatype["size"]].int()
                )
                if self.use_se_net.lower() != "false":
                    se_net_input.append(torch.mean(vec, dim=1).view(-1, 1))
                emb_output.append(vec)
            elif datatype["type"] == "SparseEncoder":
                vec = self.embs[index](x[:, datatype["index"]].int())
                emb_output.append(vec)
                if self.use_se_net.lower() != "false":
                    se_net_input.append(torch.mean(vec, dim=1).view(-1, 1))
            elif datatype["type"] == "DenseEncoder":
                thersholds = torch.tensor(datatype["length"]).to(x.device)
                input_data = torch.bucketize(x[:, datatype["index"]], thersholds)
                vec = self.embs[index](input_data)
                emb_output.append(vec)
                if self.use_se_net.lower() != "false":
                    se_net_input.append(torch.mean(vec, dim=1).view(-1, 1))

        if len(se_net_input) != 0 and self.use_se_net.lower() != "false":
            se_net_output = self.se_net(torch.cat(se_net_input, dim=1))
            for i in range(self.num):
                emb_output[i] = emb_output[i] * se_net_output[-1, i : i + 1]

        output = torch.cat(emb_output, dim=1)

        return output.float()
