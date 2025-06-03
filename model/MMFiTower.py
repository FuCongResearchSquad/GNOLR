import numpy as np
import torch
from torch import nn
from itertools import combinations
from model.EmbeddingModule import EmbeddingModule

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)

class InnerProductLayer(torch.nn.Module):
    """ output: product_sum_pooling (bs x 1),
                Bi_interaction_pooling (bs * dim),
                inner_product (bs x f2/2),
                elementwise_product (bs x f2/2 x emb_dim)
    """
    def __init__(self, num_fields=None, output="product_sum_pooling"):
        super(InnerProductLayer, self).__init__()
        self.num_fields = num_fields
        self._output_type = output
        if output not in ["product_sum_pooling", "Bi_interaction_pooling", "inner_product", "elementwise_product", "all_product", "matrix_product"]:
            raise ValueError("InnerProductLayer output={} is not supported.".format(output))
        if num_fields is None:
            if output in ["inner_product", "elementwise_product", "all_product"]:
                raise ValueError("num_fields is required when InnerProductLayer output={}.".format(output))
        else:
            p, q = zip(*list(combinations(range(num_fields), 2)))
            self.field_p = torch.nn.Parameter(torch.LongTensor(p), requires_grad=False)
            self.field_q = torch.nn.Parameter(torch.LongTensor(q), requires_grad=False)
            self.interaction_units = int(num_fields * (num_fields - 1) / 2)
            self.upper_triange_mask = torch.nn.Parameter(torch.triu(torch.ones(num_fields, num_fields), 1).type(torch.ByteTensor),
                                                   requires_grad=False)
            all = torch.LongTensor(range(num_fields))
            l = all.unsqueeze(0).expand(num_fields, -1).clone()
            r = all.unsqueeze(1).expand(-1, num_fields).clone()
            self.field_l = torch.nn.Parameter(l, requires_grad=False)
            self.field_r = torch.nn.Parameter(r, requires_grad=False)
            self.interaction_units = int(num_fields * (num_fields - 1) / 2)

    def forward(self, feature_emb, matrix=None):
        if self._output_type in ["product_sum_pooling", "Bi_interaction_pooling"]:
            sum_of_square = torch.sum(feature_emb, dim=1) ** 2
            square_of_sum = torch.sum(feature_emb ** 2, dim=1)
            bi_interaction = (sum_of_square - square_of_sum) * 0.5
            if self._output_type == "Bi_interaction_pooling":
                return bi_interaction
            else:
                return bi_interaction.sum(dim=-1, keepdim=True)
        elif self._output_type == "elementwise_product":
            emb1 = torch.index_select(feature_emb, 1, self.field_p)
            emb2 = torch.index_select(feature_emb, 1, self.field_q)
            return emb1 * emb2
        elif self._output_type == "matrix_product" and matrix is not None:
            emb1 = torch.index_select(feature_emb, 1, self.field_p)
            emb1 = torch.matmul(emb1.unsqueeze(2), matrix).squeeze(2)
            emb2 = torch.index_select(feature_emb, 1, self.field_q)
            return emb1 * emb2
        elif self._output_type == "all_product":
            emb1 = [torch.index_select(feature_emb, 1, self.field_l[f]) for f in range(self.num_fields)]
            emb2 = [torch.index_select(feature_emb, 1, self.field_r[f]) for f in range(self.num_fields)]
            return [emb1[f] * emb2[f] for f in range(self.num_fields)]
        elif self._output_type == "inner_product":
            inner_product_matrix = torch.bmm(feature_emb, feature_emb.transpose(1, 2))
            flat_upper_triange = torch.masked_select(inner_product_matrix, self.upper_triange_mask)
            return flat_upper_triange.view(-1, self.interaction_units)


class CrossInteractionLayer(torch.nn.Module):
    def __init__(self, input_dim):
        super(CrossInteractionLayer, self).__init__()
        self.weight = torch.nn.Linear(input_dim, 1, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(input_dim))

    def forward(self, X_0, X_i):
        interaction_out = self.weight(X_i) * X_0 + self.bias
        return interaction_out
class LinearLayer(torch.nn.Module):
    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias

class MMFiModel(nn.Module):

    def __init__(self, datatypes, dnn_size=(256, 128), dropout=[0, 0], activation='ReLU', use_senet="false", dimensions=16, output=1):
        super(MMFiModel, self).__init__()
        self.embeddings = EmbeddingModule(
            datatypes, use_se_net = "false", dimensions=dimensions)
        categorical_field_dims = self.embeddings.num
        self.embeddings_linear = EmbeddingModule(
            datatypes, use_se_net = "false", dimensions=dimensions)
        self.interact_dim = int((categorical_field_dims) * (categorical_field_dims-1) / 2)
        self.num_fields = int(categorical_field_dims)
        self.embed_dim = dimensions
        self.embed_output_dim = dimensions*2

        self.scenario_num = 4
        self.task_num = output
        self.scenario_embedding = nn.Embedding(1, dimensions)
        self.task_embedding = nn.Embedding(output, dimensions)

        self.scenario_task_weights = nn.Parameter(torch.empty((self.task_num, self.scenario_num, 1), dtype=torch.float32))
        nn.init.xavier_normal_(self.scenario_task_weights)

        self.interact_weights = nn.Parameter(torch.empty((1, self.interact_dim, 1), dtype=torch.float32))
        nn.init.xavier_normal_(self.interact_weights)

        # self.embeddings_linear = EmbeddingModule(
        #     datatypes, use_se_net = "false", dimensions=dimensions)
        # self.scenario_linear = LinearLayer([self.scenario_num], output_dim=dimensions)
        # self.task_linear = LinearLayer([self.task_num], output_dim=dimensions)

        self.product_layer = InnerProductLayer(num_fields=int(categorical_field_dims), output='elementwise_product')

        # self.linear_layer = EmbeddingLayer(np.delete(categorical_field_dims, [0]), embed_dim)
        self.bias = torch.nn.Parameter(torch.zeros((self.task_num, self.scenario_num, self.embed_dim), requires_grad=True))

        self.hard_sparse = False
        self.mid_size = 4
        # print("use hard_sparse")
        self.share_bottom = nn.Linear(self.embed_dim, 14)
        self.expert_num = 4
        self.linear_share_bottom = nn.Linear(self.embed_dim, 14)
        self.share_experts = nn.ModuleList([nn.Linear(14, self.mid_size) for _ in range(self.expert_num)])
        self.specific_experts = nn.ModuleList(
            [nn.ModuleList([nn.ModuleList([nn.Linear(14, self.mid_size) for _ in range(int(self.expert_num / 2))])
                            for _ in range(self.task_num)]) for _ in range(self.scenario_num)])
        self.linear_share_experts = nn.ModuleList([nn.Linear(14, self.mid_size) for _ in range(self.expert_num)])
        self.linear_specific_experts = nn.ModuleList([nn.ModuleList([nn.ModuleList([nn.Linear(14, self.mid_size)
                                                       for _ in range(int(self.expert_num / 2))])
                                                       for _ in range(self.task_num)]) for _ in range(self.scenario_num)])

        self.gates = nn.ModuleList([nn.ModuleList([nn.Sequential(
            nn.Linear(self.embed_dim + self.embed_dim, int(3 * self.expert_num / 2)),
            nn.Softmax(dim=2)) for _ in range(self.task_num)]) for _ in range(self.scenario_num)])
        self.linear_gates = nn.ModuleList([nn.ModuleList([nn.Sequential(
            nn.Linear(self.embed_dim + self.embed_dim, int(3 * self.expert_num / 2)),
            nn.Softmax(dim=2)) for _ in range(self.task_num)]) for _ in range(self.scenario_num)])
        self.linear_tower = nn.ModuleList([nn.ModuleList(
            [nn.Sequential(nn.Linear(self.mid_size * self.num_fields, self.num_fields), nn.BatchNorm1d(self.num_fields),
                           temperature(1), nn.Softmax(dim=1)) for _ in range(self.task_num)]) for _ in range(self.scenario_num)])


        if self.hard_sparse:
            self.masks = nn.ModuleList([nn.ModuleList([nn.Sequential(nn.Linear(self.mid_size * self.interact_dim, self.interact_dim),
                                                                     nn.BatchNorm1d(self.interact_dim), nn.Tanh(),
                                                                     nn.ReLU()) for _ in range(self.task_num)]) for _ in range(self.scenario_num)])
            self.attention_tower = nn.ModuleList([nn.ModuleList([nn.Sequential(nn.Linear(self.interact_dim, 1, bias=False))
                                                                 for _ in range(self.task_num)]) for _ in range(self.scenario_num)])
        else:
            self.attention_tower = nn.ModuleList([nn.ModuleList([nn.Sequential(nn.Linear(self.mid_size * self.interact_dim, self.interact_dim),
                                                                               nn.BatchNorm1d(self.interact_dim),
                               temperature(1), nn.Softmax(dim=1)) for _ in range(self.task_num)]) for _ in range(self.scenario_num)])

        print(f"hard space:{self.hard_sparse},activation:{activation}, expert:{self.expert_num}, mmoe_hidden_dim:{self.mid_size}, hid_dim:{dnn_size}, dropout:{dropout}")
        # self.last = nn.ModuleList([nn.Sequential(nn.Linear(self.embed_output_dim, 40), nn.BatchNorm1d(40), nn.ReLU(),
        #                                          nn.Linear(40, 32), nn.BatchNorm1d(32), nn.ReLU(),
        #                                          nn.Linear(32, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8, 1),
        #                                          nn.Sigmoid()) for _ in range(self.task_num)])


    def forward(self, x):
        # liner
        # hidden = self.embeddings(x)
        batch_size = x.shape[0]
        task_indicators = []
        scenario_task_interactions = []
        task_ids = []
        device = x.device
        scenario_ids = torch.zeros([batch_size, 1], dtype=torch.long).to(device)
        # scenario_ids = categorical_x[:, 0].unsqueeze(1).to(device)
        scenario_idxs = [nn.functional.one_hot(scenario_ids.clone().detach(), self.scenario_num).to(device)]

        scenario_indicator = self.scenario_embedding(scenario_ids).squeeze()

        for t in range(self.task_num):
            task_id = torch.ones([batch_size, 1], dtype=torch.long) * t
            task_id = task_id.to(device)
            task_ids.append(task_id)

            task_indicator = self.task_embedding(task_id).squeeze()
            task_indicators.append(task_indicator)
            scenario_task_weights = torch.einsum("ij, jk->ik",  [torch.squeeze(scenario_idxs[0].float()), self.scenario_task_weights[t]])
            scenario_task_interactions.append(scenario_indicator * task_indicator * scenario_task_weights)

        categorical_emb = self.embeddings(x)
        categorical_emb = categorical_emb.view(batch_size, -1, self.embed_dim)
        categorical_emb = self.product_layer(categorical_emb) * self.interact_weights

        attention_inputs = [torch.cat([(scenario_task_interactions[t]).unsqueeze(1).expand(-1, self.interact_dim, -1),
                                       categorical_emb], dim=2) for t in range(self.task_num)]

        # interaction
        experts_in = self.share_bottom(categorical_emb)
        share_out = torch.stack([self.share_experts[e](experts_in) for e in range(int(self.expert_num))])
        specific_out = [torch.stack([torch.stack([self.specific_experts[s][t][e](experts_in) for e in range(int(self.expert_num / 2))])
                                    for t in range(self.task_num)]) for s in range(self.scenario_num)]
        attention_weights = [[torch.cat([share_out, specific_out[s][t]], dim=0)
                              for t in range(self.task_num)] for s in range(self.scenario_num)]
        attention_gates = [torch.stack([self.gates[s][t](attention_inputs[t]) for t in range(self.task_num)])
                           for s in range(self.scenario_num)]
        attention_gates = [[attention_gates[s][t].permute(2, 0, 1) for t in range(self.task_num)]
                           for s in range(self.scenario_num)]
        attention_weights = [[torch.sum(attention_weights[s][t] * attention_gates[s][t].unsqueeze(3), dim=0)
                              for t in range(self.task_num)] for s in range(self.scenario_num)]

        if self.hard_sparse:
            attention_weights = [[self.masks[s][t](attention_weights[s][t].view(-1, self.mid_size * self.interact_dim))
                                  for t in range(self.task_num)] for s in range(self.scenario_num)]
            attention_weights = [[(attention_weights[s][t].unsqueeze(2) * categorical_emb).transpose(1, 2) for t in range(self.task_num)]
                                   for s in range(self.scenario_num)]
            attention_weights = [torch.stack([self.attention_tower[s][t](attention_weights[s][t]).squeeze()
                                              for s in range(self.scenario_num)]) for t in range(self.task_num)]
            interaction_out = [torch.einsum('ji, ijk->jk', [scenario_idxs[0].squeeze().float(), attention_weights[t]])
                             for t in range(self.task_num)]

        else:
            attention_weights = [torch.stack([self.attention_tower[s][t](attention_weights[s][t].view(-1, self.mid_size * self.interact_dim))
                                 for s in range(self.scenario_num)]) for t in range(self.task_num)]
            attention_weights = [torch.einsum('ji, ijk->jk', [scenario_idxs[0].squeeze().float(), attention_weights[t]])
                                 for t in range(self.task_num)]
            interaction_out = [torch.sum(attention_weights[t].unsqueeze(2) * categorical_emb, dim=1) for t in  range(self.task_num)]

        linear_out = self.embeddings_linear(x)
        linear_out = linear_out.view(batch_size, -1, self.embed_dim)
        # linear_out = self.linear_layer(categorical_x[:, 1:])
        # user_numerical_linear = self.user_numerical_linear(numerical_x[:, :self.user_numerical_dim]).unsqueeze(1)
        # item_numerical_linear = self.item_numerical_linear(numerical_x[:, self.user_numerical_dim:]).unsqueeze(1)
        # linear_out = torch.cat([linear_out, user_numerical_linear, item_numerical_linear], dim=1)

        attention_inputs_ = [torch.cat([(scenario_task_interactions[t]).unsqueeze(1).expand(-1, self.num_fields, -1),
                                        linear_out], dim=2) for t in range(self.task_num)]

        # first order
        experts_in_ = self.linear_share_bottom(linear_out)
        share_out_ = torch.stack([self.linear_share_experts[e](experts_in_) for e in
                                  range(int(self.expert_num))])
        specific_out_ = [torch.stack([torch.stack([self.linear_specific_experts[s][t][e](experts_in_) for e in
                                                   range(int(self.expert_num / 2))]) for t in range(self.task_num)]) for
                                                    s in range(self.scenario_num)]
        attention_weights_ = [[torch.cat([share_out_, specific_out_[s][t]], dim=0) for t in range(self.task_num)]
                              for s in range(self.scenario_num)]
        attention_gates_ = [torch.stack([self.gates[s][t](attention_inputs_[t]) for t in range(self.task_num)])
                            for s in range(self.scenario_num)]
        attention_gates_ = [[attention_gates_[s][t].permute(2, 0, 1) for t in range(self.task_num)]
                            for s in range(self.scenario_num)]
        attention_weights_ = [[torch.sum(attention_weights_[s][t] * attention_gates_[s][t].unsqueeze(3), dim=0)
                               for t in range(self.task_num)] for s in range(self.scenario_num)]
        attention_weights_ = [torch.stack([self.linear_tower[s][t](attention_weights_[s][t].view(-1, self.num_fields * self.mid_size))
                                for s in range(self.scenario_num)]) for t in range(self.task_num)]
        attention_weights_ = [torch.einsum('ji, ijk->jk', [scenario_idxs[0].squeeze().float(), attention_weights_[t]])
                              for t in range(self.task_num)]
        linear_out = [torch.sum(attention_weights_[t].unsqueeze(2) * linear_out, dim=1) for t in range(self.task_num)]

        bias = [torch.einsum("ij, jk->ik", [torch.squeeze(scenario_idxs[0].float()), self.bias[t]])
                for t in range(self.task_num)]

        linear_out = [linear_out[t] + bias[t] for t in range(self.task_num)]
        embeddings = [torch.cat([interaction_out[t], linear_out[t]], 1) for t in range(self.task_num)]
        # outputs = [t(ti).squeeze() for (t, ti) in zip(self.last, embeddings)]
        task_outputs = torch.stack(embeddings, dim=1)

        return task_outputs


class squeeze(nn.Module):
    def __init__(self):
        super(squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze()

class temperature(nn.Module):
    def __init__(self, tem):
        super(temperature, self).__init__()
        self.tem = tem

    def forward(self, x):
        return x/self.tem