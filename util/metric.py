import torch
import numpy as np
from util.utils import list_to_slate
from torcheval.metrics.functional import binary_auroc

PADDED_Y_VALUE = -1


def get_metric(metrics, true, pred, group, ats=[5]):
    if pred.dim() > 1:
        pred = pred[:, 0]
    if true.dim() > 1:
        true = true[:, 0]
    true, pred = list_to_slate(true, pred, group)

    ndcg_n = ndcg(pred, true, ats)
    mrr_n = mrr(pred, true)
    g_auc = group_auc(pred, true)

    if metrics is None:
        metrics = {}
        for idx, at in enumerate(ats):
            metrics["ndcg_{at}".format(at=at)] = ndcg_n[:, idx]
        metrics["mrr"] = mrr_n[:, 0]
        metrics["group_auc"] = g_auc
    else:
        for idx, at in enumerate(ats):
            metrics["ndcg_{at}".format(at=at)] = torch.cat(
                (metrics["ndcg_{at}".format(at=at)], ndcg_n[:, idx]), dim=0
            )
        metrics["mrr"] = torch.cat((metrics["mrr"], mrr_n[:, 0]), dim=0)
        metrics["group_auc"] = torch.cat((metrics["group_auc"], g_auc), dim=0)

    return metrics


def group_auc(pred, true, padding_indicator=PADDED_Y_VALUE):
    mask = true != padding_indicator

    # result = binary_auroc(
    #     pred.squeeze(), true.squeeze(), num_tasks=pred.shape[0], weight=mask.squeeze()
    # )
    result = binary_auroc(pred, true, num_tasks=pred.shape[0], weight=mask)

    if pred.shape[0] == 1:
        result = result.unsqueeze(0)
    return result


def ndcg(
    y_pred,
    y_true,
    ats=None,
    gain_function=lambda x: torch.pow(2, x) - 1,
    padding_indicator=PADDED_Y_VALUE,
    filler_value=1.0,
):
    """
    Normalized Discounted Cumulative Gain at k.

    Compute NDCG at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for NDCG evaluation, if None, maximum rank is used
    :param gain_function: callable, gain function for the ground truth labels, e.g. torch.pow(2, x) - 1
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param filler_value: a filler NDCG value to use when there are no relevant items in listing
    :return: NDCG values for each slate and rank passed, shape [batch_size, len(ats)]
    """
    idcg = dcg(y_true, y_true, ats, gain_function, padding_indicator)
    ndcg_ = dcg(y_pred, y_true, ats, gain_function, padding_indicator) / idcg
    idcg_mask = idcg == 0
    ndcg_[idcg_mask] = filler_value  # if idcg == 0 , set ndcg to filler_value

    assert (ndcg_ < 0.0).sum() >= 0, "every ndcg should be non-negative"

    return ndcg_


def __apply_mask_and_get_true_sorted_by_preds(
    y_pred, y_true, padding_indicator=PADDED_Y_VALUE
):
    mask = y_true == padding_indicator

    y_pred[mask] = float("-inf")
    y_true[mask] = 0.0

    _, indices = y_pred.sort(descending=True, dim=-1)
    return torch.gather(y_true, dim=1, index=indices)


def dcg(
    y_pred,
    y_true,
    ats=None,
    gain_function=lambda x: torch.pow(2, x) - 1,
    padding_indicator=PADDED_Y_VALUE,
):
    """
    Discounted Cumulative Gain at k.

    Compute DCG at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for DCG evaluation, if None, maximum rank is used
    :param gain_function: callable, gain function for the ground truth labels, e.g. torch.pow(2, x) - 1
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: DCG values for each slate and evaluation position, shape [batch_size, len(ats)]
    """
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    actual_length = y_true.shape[1]

    if ats is None:
        ats = [actual_length]
    ats = [min(at, actual_length) for at in ats]

    true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(
        y_pred, y_true, padding_indicator
    )

    discounts = (
        torch.tensor(1)
        / torch.log2(
            torch.arange(true_sorted_by_preds.shape[1], dtype=torch.float) + 2.0
        )
    ).to(device=true_sorted_by_preds.device)

    gains = gain_function(true_sorted_by_preds)

    discounted_gains = (gains * discounts)[:, : np.max(ats)]

    cum_dcg = torch.cumsum(discounted_gains, dim=1)

    ats_tensor = torch.tensor(ats, dtype=torch.long) - torch.tensor(1)

    dcg = cum_dcg[:, ats_tensor]

    return dcg


def mrr(y_pred, y_true, ats=None, padding_indicator=PADDED_Y_VALUE):
    """
    Mean Reciprocal Rank at k.

    Compute MRR at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for MRR evaluation, if None, maximum rank is used
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: MRR values for each slate and evaluation position, shape [batch_size, len(ats)]
    """
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    if ats is None:
        ats = [y_true.shape[1]]

    true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(
        y_pred, y_true, padding_indicator
    )

    values, indices = torch.max(true_sorted_by_preds, dim=1)
    indices = indices.type_as(values).unsqueeze(dim=0).t().expand(len(y_true), len(ats))

    ats_rep = torch.tensor(data=ats, device=indices.device, dtype=torch.float32).expand(
        len(y_true), len(ats)
    )

    within_at_mask = (indices < ats_rep).type(torch.float32)

    result = torch.tensor(1.0) / (indices + torch.tensor(1.0))

    zero_sum_mask = torch.sum(values) == 0.0
    result[zero_sum_mask] = 0.0

    result = result * within_at_mask

    return result


def MAP(y_pred, y_true, ats=None, padding_indicator=PADDED_Y_VALUE):
    """
    Computes mean average precision (MAP).

    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for MAP evaluation, if None, maximum rank is used
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: MAP values for each slate and evaluation position, shape [batch_size, len(ats)]

    """
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    if ats is None:
        ats = [y_true.shape[1]]

    true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(
        y_pred, y_true, padding_indicator
    )
    result = torch.zeros(true_sorted_by_preds.size(0), len(ats))
    for j, at in enumerate(ats):
        for i, slate in enumerate(true_sorted_by_preds):
            hits = 0
            sum_precs = 0
            for n in range(at):
                if slate[n] > 0:
                    hits += 1
                    sum_precs += hits / (n + 1.0)
            result[i][j] = sum_precs / at
    return result
