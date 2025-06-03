import torch.nn as nn
from model.EmbeddingModule import EmbeddingModule
from util.utils import get_activation
import torch
from torch.nn import ParameterList, Parameter


class MMoETower(nn.Module):
    def __init__(
        self,
        datatypes,
        dnn_size=(256, 128),
        dropout=[0, 0],
        activation="ReLU",
        use_senet="false",
        dimensions=16,
        output=1,
    ):
        """
        MMOE model input parameters
        :param user_feature_dict: user feature dict include: {feature_name: (feature_unique_num, feature_index)}
        :param item_feature_dict: item feature dict include: {feature_name: (feature_unique_num, feature_index)}
        :param emb_dim: int embedding dimension
        :param n_expert: int number of experts in mmoe
        :param mmoe_hidden_dim: mmoe layer input dimension
        :param hidden_dim: list task tower hidden dimension
        :param dropouts: list of task dnn drop out probability
        :param output_size: int task output size
        :param expert_activation: activation function like 'relu' or 'sigmoid'
        :param num_task: int default 2 multitask numbers
        """
        super().__init__()
        self.embeddings = EmbeddingModule(datatypes, use_senet, dimensions=dimensions)
        self.expert_activation = get_activation(activation)
        self.num_task = output
        # mmoe_hidden_dim = 512
        n_expert = 8

        # output_dimensions = dnn_size[-1]
        # dnn_size = dnn_size[:-1]
        hidden_size = self.embeddings.dim + self.embeddings.embedding_dim

        mmoe_hidden_dim = 512
        # experts
        self.experts = torch.nn.Parameter(
            torch.rand(hidden_size, mmoe_hidden_dim, n_expert), requires_grad=True
        )
        self.experts.data.normal_(0, 1)
        self.experts_bias = torch.nn.Parameter(
            torch.rand(mmoe_hidden_dim, n_expert), requires_grad=True
        )
        # gates
        self.gates = ParameterList(
            [Parameter(torch.randn(hidden_size, n_expert)) for _ in range(output)]
        )
        # self.gates_bias = ParameterList([
        #     Parameter(torch.randn(n_expert)) for _ in range(output_dimensions)
        # ])

        for i in range(self.num_task):
            setattr(self, "task_{}_dnn".format(i + 1), nn.ModuleList())
            hid_dim = (mmoe_hidden_dim,) + dnn_size
            for j in range(len(hid_dim) - 2):
                getattr(self, "task_{}_dnn".format(i + 1)).add_module(
                    "ctr_hidden_{}".format(j), nn.Linear(hid_dim[j], hid_dim[j + 1])
                )
                getattr(self, "task_{}_dnn".format(i + 1)).add_module(
                    "ctr_batchnorm_{}".format(j), get_activation(activation)
                )
                getattr(self, "task_{}_dnn".format(i + 1)).add_module(
                    "ctr_dropout_{}".format(j), nn.Dropout(dropout[j])
                )
            getattr(self, "task_{}_dnn".format(i + 1)).add_module(
                "task_last_layer", nn.Linear(hid_dim[-2], hid_dim[-1])
            )
        print(
            f"activation:{activation}expert:{n_expert}, mmoe_hidden_dim:{mmoe_hidden_dim}, hid_dim:{hid_dim}, dropout:{dropout}"
        )

    def forward(self, x):
        hidden = self.embeddings(x)

        # mmoe
        experts_out = torch.einsum(
            "ij, jkl -> ikl", hidden, self.experts
        )  # batch * mmoe_hidden_size * num_experts
        experts_out += self.experts_bias
        if self.expert_activation is not None:
            experts_out = self.expert_activation(experts_out)

        gates_out = list()
        for idx, gate in enumerate(self.gates):
            gate_out = torch.einsum("ab, bc -> ac", hidden, gate)  # batch * num_experts
            # if self.gates_bias:
            #     gate_out += self.gates_bias[idx]
            gate_out = nn.Softmax(dim=-1)(gate_out)
            gates_out.append(gate_out)

        outs = list()
        for gate_output in gates_out:
            expanded_gate_output = torch.unsqueeze(
                gate_output, 1
            )  # batch * 1 * num_experts
            weighted_expert_output = experts_out * expanded_gate_output.expand_as(
                experts_out
            )  # batch * mmoe_hidden_size * num_experts
            outs.append(
                torch.sum(weighted_expert_output, 2)
            )  # batch * mmoe_hidden_size

        # task tower
        task_outputs = list()
        for i in range(self.num_task):
            x = outs[i]
            for mod in getattr(self, "task_{}_dnn".format(i + 1)):
                x = mod(x)
            task_outputs.append(x)
        task_outputs = torch.stack(task_outputs, dim=1)
        return task_outputs

    def run_dnn(self, dnn, dnn_input):
        return dnn(dnn_input)
