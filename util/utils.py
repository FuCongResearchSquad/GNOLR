import datetime
import json
import os
import pickle
import sys
import torch
import random
import pandas as pd
import numpy as np
from collections import defaultdict

from sklearn import preprocessing
from torch import nn
from tqdm import tqdm

sys.path.append("../")


def get_activation(activation):
    if isinstance(activation, str):
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "leakyrelu":
            return nn.LeakyReLU()
        else:
            return getattr(nn, activation)()
    else:
        return


def get_optimizer(optimizer, params, lr):
    if isinstance(optimizer, str):
        if optimizer.lower() == "adam":
            optimizer = "Adam"
    try:
        optimizer = getattr(torch.optim, optimizer)(params, lr=lr)
    except:
        raise NotImplementedError("optimizer={} is not supported.".format(optimizer))
    return optimizer


def list_to_slate(true, pred, groups):
    # Convert a tensor with shape (len) or (len, 1) to shape (usernum, max_len).
    # groups is a list where each value represents the length of items a user has interacted with
    max_len = max(groups)
    pred = pred.squeeze()
    true = true.squeeze()
    grouped_true = torch.full((len(groups), max_len), -1.0)
    grouped_pred = torch.full((len(groups), max_len), -1.0)

    start_idx = 0

    for i, length in enumerate(groups):
        end_idx = start_idx + length
        grouped_true[i, :length] = true[start_idx:end_idx].view(-1)
        grouped_pred[i, :length] = pred[start_idx:end_idx].view(-1)
        start_idx = end_idx

    return grouped_true, grouped_pred


def embedding_list_to_slate(user_embedding, item_embedding, groups):
    # u,i: [batch*itemnum,dim]
    # return u[batch,dim] i[batch,itemnum,dim]
    max_itemnum = max(groups)
    batch = len(groups)
    dim = user_embedding.shape[1]
    new_user_embedding = torch.zeros(batch, dim).to(user_embedding.device)
    new_item_embedding = torch.zeros(batch, max_itemnum, dim).to(item_embedding.device)

    start_idx = 0
    for user_id in range(batch):
        end_idx = start_idx + groups[user_id]
        new_user_embedding[user_id] = user_embedding[start_idx]
        new_item_embedding[user_id, : groups[user_id]] = item_embedding[
            start_idx:end_idx
        ]
        start_idx = end_idx
    return new_user_embedding, new_item_embedding


def slate_to_list(pred, group):
    embeddings_list = []
    for id in range(len(group)):
        itemnum = group[id]

        actual_item_embedding = pred[id, :itemnum]

        embeddings_list.append(actual_item_embedding)

    restored_item_embedding = torch.cat(embeddings_list, dim=0)
    return restored_item_embedding
