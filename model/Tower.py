import torch
import torch.nn as nn
from util.utils import get_activation
from model.EmbeddingModule import EmbeddingModule


class Tower(nn.Module):
    def __init__(
        self,
        datatypes,
        dnn_size=[256, 128],
        dropout=[0, 0],
        activation="ReLU",
        use_senet="false",
        dimensions=16,
        output=1,
    ):
        super().__init__()
        self.dnns = nn.ModuleList()
        self.embeddings = EmbeddingModule(datatypes, use_senet, dimensions=dimensions)

        for _ in range(output):
            input_dims = self.embeddings.dim + self.embeddings.embedding_dim
            layers = []
            for i in range(len(dnn_size)):
                layers.append(nn.Linear(input_dims, dnn_size[i]))

                if i != len(dnn_size) - 1:
                    layers.append(nn.Dropout(dropout[i]))
                    layers.append(get_activation(activation))

                input_dims = dnn_size[i]
            self.dnns.append(nn.Sequential(*layers))

    def forward(self, x):
        dnn_input = self.embeddings(x)

        results = []

        for dnn in self.dnns:
            results.append(dnn(dnn_input))
        results = torch.stack(results, dim=1)
        return results

    def run_dnn(self, dnn, dnn_input):
        return dnn(dnn_input)
