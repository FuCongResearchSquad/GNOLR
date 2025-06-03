import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
from util.utils import list_to_slate

DEFAULT_EPS = 1e-10
PADDED_Y_VALUE = -1
PADDED_INDEX_VALUE = -1


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=1.0, reduction="mean"):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, outputs, targets):
        outputs = torch.clamp(outputs, min=1e-7, max=1 - 1e-7)
        bce_loss = -(
            self.pos_weight * targets * torch.log(outputs)
            + (1 - targets) * torch.log(1 - outputs)
        )
        if self.reduction == "none":
            return bce_loss
        elif self.reduction == "sum":
            return bce_loss.sum()
        else:
            return bce_loss.mean()


def get_loss(
    loss: str,
    true: torch.Tensor,
    pred: torch.Tensor,
    group=None,
    S1=1.5,
    S2=1.5,
    m=0.2,
    pos_weight=[1, 1],
    a_param=None,
):
    if loss in ["setrank", "listnet"]:
        pred = torch.sigmoid(pred)

    if isinstance(loss, str):
        if loss.lower() == "bceloss":
            pos_weight_tensor = torch.tensor(pos_weight[0], dtype=torch.float32)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)(
                pred.squeeze(), true.squeeze()
            )
        elif loss.lower() == "setrank":
            return batch_setrank(true, pred, group)
        elif loss.lower() == "set2setrank":
            return batch_set2setrank(true.squeeze(), pred.squeeze(), group, 1)
        elif loss.lower() == "jrc":
            return jrc(true, pred, group, pos_weight=pos_weight[0])
        elif loss.lower() == "single_gnolr":
            return pointwise(true, pred, S1, m)
        elif loss.lower() == "single_gnolr_l":
            return point_list(true, pred, group, S1, S2, m)

        # multi-task
        elif loss.lower() == "multi_bceloss":
            return multi_bceloss(pred, true, pos_weight=pos_weight)
        elif loss.lower() == "esmm_bceloss":
            return esmm_bceloss(pred, true, pos_weight=pos_weight)
        elif loss.lower() == "ipw_bceloss":
            return ipw_bceloss(pred, true, pos_weight=pos_weight)
        elif loss.lower() == "dr_bceloss":
            return dr_bceloss(pred, true, pos_weight=pos_weight)
        elif loss.lower() == "dcmt_bceloss":
            return dcmt_bceloss(pred, true, pos_weight=pos_weight)
        elif loss.lower() == "nise_bceloss":
            return nise_bceloss(pred, true, pos_weight=pos_weight)
        elif loss.lower() == "tafe_bceloss":
            return tafe_bceloss(pred, true, pos_weight=pos_weight)

        # naive olr (Neural OLR)
        elif loss.lower() == "multi_naive_olr":
            if not isinstance(m, list):
                raise TypeError("m should be a list")
            if not isinstance(S1, list):
                raise TypeError("S1 should be a list")
            return ordinal_multi_tasks_naive_olr(pred, true, m, S1)
        # naive olr with multi-tower (GNOLR V0)
        elif loss.lower() == "multi_naive_olr_with_multi_embd":
            if not isinstance(m, list):
                raise TypeError("m should be a list")
            if not isinstance(S1, list):
                raise TypeError("S1 should be a list")
            return ordinal_multi_tasks_naive_olr_with_multi_embd(pred, true, m, S1)
        # gnolr without nested optimization (GNOLR V1)
        elif loss.lower() == "multi_gnolr_without_nested_opt":
            if not isinstance(m, list):
                raise TypeError("m should be a list")
            if not isinstance(S1, list):
                raise TypeError("S1 should be a list")
            return ordinal_multi_tasks_gnolr_without_nested_opt(pred, true, m, S1)
        # gnolr with two feedback (GNOLR)
        elif loss.lower() == "multi_gnolr":
            if not isinstance(m, list):
                raise TypeError("m should be a list")
            if not isinstance(S1, list):
                raise TypeError("S1 should be a list")
            return ordinal_multi_tasks_gnolr(pred, true, m, S1)
        # gnolr with multiple feedback
        elif loss.lower() == "multi_gnolr_with_multi_feedback":
            if not isinstance(m, list):
                raise TypeError("m should be a list")
            if not isinstance(S1, list):
                raise TypeError("S1 should be a list")
            return ordinal_multi_tasks_gnolr_with_multi_feedback(pred, true, m, S1)
        # gnolr with a learnable hyperparameter 'a'
        elif loss.lower() == "multi_gnolr_with_learnable_a":
            if not isinstance(m, list):
                raise TypeError("m should be a list")
            if not isinstance(S1, list):
                raise TypeError("S1 should be a list")
            assert (
                a_param is not None
            ), "a_param should not be not none when using `multi_gnolr_with_learnable_a`"
            return ordinal_multi_tasks_gnolr_with_learnable_a(
                pred, true, m, S1, a_param
            )
        # gnolr with parallel positive and negative feedback
        elif loss.lower() == "multi_gnolr_with_pos_neg":
            if not isinstance(m, list):
                raise TypeError("m should be a list")
            if not isinstance(S1, list):
                raise TypeError("S1 should be a list")
            return ordinal_multi_tasks_gnolr_with_pos_neg(pred, true, m, S1)

        else:
            true, pred = list_to_slate(true, pred, group)
            if loss.lower() == "listnet":
                return listNet(pred, true)
            elif loss.lower() == "ranknet":
                return rankNet(pred, true)
            elif loss.lower() == "lambdarank":
                return lambdaLoss(pred, true, weighing_scheme="lambdaRank_scheme")
    return


def sigmoid_m(x, m):
    return m / (m + torch.exp(-x))


def pointwise(true, pred, S1, m):
    pred = sigmoid_m(S1 * pred, m)
    return nn.BCELoss()(pred.squeeze(), true.squeeze())


def listwise(true, pred, total_len, S2, padded_value_indicator=PADDED_Y_VALUE):
    mask = true == padded_value_indicator
    pred = pred * S2
    pred[mask] = float("-inf")
    true[mask] = float(0)

    pred = F.softmax(pred, dim=1) + DEFAULT_EPS
    pred = torch.log(pred)

    return (-torch.sum(true * pred)) / total_len


def point_list(true, pred, group, S1, S2, m):
    true_l, pred_l = list_to_slate(true, pred, group)
    total_len = sum(group)

    loss_1 = pointwise(true, pred, S1, m)

    loss_2 = listwise(true_l, pred_l, total_len, S2)

    return loss_1 + loss_2


def ordinal_multi_tasks_naive_olr(logit, label, m, s):
    """
    Naive Neural OLR
    logit: [bs, 1]
    label: [bs, num_tasks]
    """
    a_i = torch.tensor([-math.log(mi) for mi in m]).to(logit.device)
    s_i = torch.tensor(s).to(logit.device)

    logits = logit * s_i - a_i

    probs = torch.sigmoid(logits)

    # loss
    total_loss = -(
        (1 - label[:, 0]) * torch.log(1 - probs[:, 0])
        + (label[:, 0] - label[:, 1]) * torch.log((probs[:, 0] - probs[:, 1]))
        + label[:, 1] * torch.log(probs[:, 1])
    )
    total_loss = total_loss.mean()

    return total_loss


def ordinal_multi_tasks_naive_olr_with_multi_embd(logit, label, m, s):
    """
    Naive Neural OLR With Multi-Embedding
    logit: [bs, num_tasks]
    label: [bs, num_tasks]
    """
    a_i = torch.tensor([-math.log(mi) for mi in m]).to(logit.device)
    s_i = torch.tensor(s).to(logit.device)

    logits = logit  # [batch_size, 2]
    updated_logits = logits.clone()
    updated_logits = updated_logits * s_i - a_i

    probs = torch.sigmoid(updated_logits)

    # loss
    total_loss = -(
        (1 - label[:, 0]) * torch.log(1 - probs[:, 0])
        + (label[:, 0] - label[:, 1]) * torch.log((probs[:, 0] - probs[:, 1]))
        + label[:, 1] * torch.log(probs[:, 1])
    )
    total_loss = total_loss.mean()

    return total_loss


def ordinal_multi_tasks_gnolr_without_nested_opt(logit, label, m, s):
    """
    OLR V1 (Two-Tower)
    logit: [bs, num_tasks]
    label: [bs, num_tasks]
    """
    a_i = torch.tensor([-math.log(mi) for mi in m]).to(logit.device)
    s_i = torch.tensor(s).to(logit.device)

    logits = logit  # [batch_size, 2]
    updated_logits = logits.clone()
    updated_logits[:, 1] += logits[:, 0]
    updated_logits = updated_logits * s_i - a_i

    probs = torch.sigmoid(updated_logits)

    total_loss = -(
        (1 - label[:, 0]) * torch.log(1 - probs[:, 0])
        + (label[:, 0] - label[:, 1]) * torch.log((probs[:, 0] - probs[:, 1]))
        + label[:, 1] * torch.log(probs[:, 1])
    )

    total_loss = total_loss.mean()

    return total_loss


def ordinal_multi_tasks_gnolr(logit, label, m, s):
    """
    GNOLR
    logit: [bs, 2]
    label: [bs, 2]
    """
    a_i = torch.tensor([-math.log(mi) for mi in m]).to(logit.device)
    s_i = torch.tensor(s).to(logit.device)

    logit = logit * s_i  # [batch_size, 2]

    updated_logits = logit.clone()
    updated_logits[:, 1] += logit[:, 0]
    updated_logits = updated_logits - a_i

    probs = torch.sigmoid(updated_logits)

    # OLR Step1
    loss_v1 = -(
        (1 - label[:, 0]) * torch.log(1 - probs[:, 0])
        + label[:, 0] * torch.log(probs[:, 0])
    )

    # OLR Step2
    k_label = label[:, 0] - label[:, 1]
    k_label[k_label < 0] = 0
    loss_v2 = -(
        (1 - label[:, 0]) * torch.log(1 - probs[:, 0])
        + k_label * torch.log((probs[:, 0] - probs[:, 1]))
        + label[:, 1] * torch.log(probs[:, 1])
    )

    # Total Loss
    total_loss = loss_v1 + loss_v2
    total_loss = total_loss.mean()

    return total_loss


def ordinal_multi_tasks_gnolr_with_multi_feedback(logit, label, m, s):
    """
    GNOLR
    logit: [bs, num_tasks]
    label: [bs, num_tasks]
    """
    a_i = torch.tensor([-math.log(mi) for mi in m]).to(logit.device)
    s_i = torch.tensor(s).to(logit.device)

    logit = logit * s_i  # [batch_size, 2]

    updated_logits = logit.clone()
    for i in range(1, logit.shape[1]):
        updated_logits[:, i] += updated_logits[:, i - 1]
    updated_logits = updated_logits - a_i

    probs = torch.sigmoid(updated_logits)

    # OLR Step1
    total_loss = -(
        (1 - label[:, 0]) * torch.log(1 - probs[:, 0])
        + label[:, 0] * torch.log(probs[:, 0])
    )

    # Total Loss
    for i in range(1, logit.shape[1]):
        loss = (1 - label[:, 0]) * torch.log(1 - probs[:, 0])

        for j in range(i):
            is_positive = label[:, j] == 1
            no_higher_positive = (label[:, j + 1 :] == 0).all(dim=1)
            new_label = (is_positive & no_higher_positive).float()  # [batch_size]
            loss += new_label * torch.log((probs[:, j] - probs[:, j + 1]))

        is_positive = label[:, i:].sum(dim=1) > 0
        new_label = (is_positive).float()
        loss += new_label * torch.log(probs[:, i])

        total_loss += -1 * loss

    total_loss = total_loss.mean()

    return total_loss


def ordinal_multi_tasks_gnolr_with_learnable_a(
    logit, label, m, s, a_param, regularization=True, beta=1e-3
):
    """
    logit: [bs, 2]
    label: [bs, 2]
    a: [2]
    """
    # print(f"a_param: {a_param}, m_param: {torch.exp(-a_param)}, real_m: {m}")
    s_i = torch.tensor(s).to(logit.device)

    logit = logit * s_i  # [batch_size, 2]

    updated_logits = logit.clone()
    updated_logits[:, 1] += logit[:, 0]

    a_actual = torch.cumsum(torch.exp(a_param), dim=0)
    updated_logits = updated_logits - a_actual
    # updated_logits = updated_logits - a_param

    probs = torch.sigmoid(updated_logits)

    # OLR Step1
    loss_v1 = -(
        (1 - label[:, 0]) * torch.log(1 - probs[:, 0])
        + label[:, 0] * torch.log(probs[:, 0])
    )

    # OLR Step2
    k_label = label[:, 0] - label[:, 1]
    k_label[k_label < 0] = 0
    loss_v2 = -(
        (1 - label[:, 0]) * torch.log(1 - probs[:, 0])
        + k_label * torch.log((probs[:, 0] - probs[:, 1]))
        + label[:, 1] * torch.log(probs[:, 1])
    )

    # Total Loss
    total_loss = loss_v1 + loss_v2
    total_loss = total_loss.mean()

    if regularization:
        loss_reg = nn.MSELoss()(
            a_actual,
            -torch.log(torch.tensor(m, requires_grad=False).to(a_actual.device)),
        )
        # loss_reg = nn.MSELoss()(
        #     a_param, -torch.log(torch.tensor(m, requires_grad=False).to(a_param.device))
        # )

        total_loss = total_loss + beta * loss_reg

    return total_loss


def ordinal_multi_tasks_gnolr_with_pos_neg(logit, label, m, s):
    """
    GNOLR
    logit: [bs, 3]
    label: [bs, 3]
    """
    a_i = torch.tensor([-math.log(mi) for mi in m]).to(logit.device)
    s_i = torch.tensor(s).to(logit.device)

    logit = logit * s_i  # [batch_size, 3]

    updated_logits = logit.clone()
    updated_logits[:, 1] += logit[:, 0]
    updated_logits[:, 2] += logit[:, 0]
    updated_logits = updated_logits - a_i

    probs = torch.sigmoid(updated_logits)

    # OLR Step1
    loss_v1 = -(
        (1 - label[:, 0]) * torch.log(1 - probs[:, 0])
        + label[:, 0] * torch.log(probs[:, 0])
    )

    # OLR Step2
    k_label = label[:, 0] - label[:, 1]
    k_label[k_label < 0] = 0
    j_label = label[:, 0] - label[:, 2]
    j_label[j_label < 0] = 0
    loss_v2 = -(
        (1 - label[:, 0]) * torch.log(1 - probs[:, 0])
        + k_label * torch.log((probs[:, 0] - probs[:, 1]))
        + label[:, 1] * torch.log(probs[:, 1])
        + j_label * torch.log((probs[:, 0] - probs[:, 2]))
        + label[:, 2] * torch.log(probs[:, 2])
    )

    # Total Loss
    total_loss = loss_v1 + loss_v2
    total_loss = total_loss.mean()

    return total_loss


def nise_bceloss(logit, label, ablation_weight=1, weight=50, pos_weight=[1, 1]):
    """
    ctr, cvr
    logit: [bs, num_tasks]

    ctr, cvr/ctvr
    label: [bs, num_tasks]
    """
    pos_weight = torch.tensor(pos_weight)
    # simoid & clip
    ctr_prob = torch.sigmoid(logit[:, 0])
    cvr_prob = torch.sigmoid(logit[:, 1])

    # ctcvr
    # ctcvr_prob = ctr_prob * cvr_prob
    clipped_ctr_pred = torch.clamp(ctr_prob, min=1e-15, max=1 - 1e-15)
    clipped_cvr_pred = torch.clamp(cvr_prob, min=1e-15, max=1 - 1e-15)

    # Loss
    # 1. CTR
    loss_ctr = WeightedBCELoss(pos_weight=pos_weight[0])(clipped_ctr_pred, label[:, 0])
    # 2. CVR
    # 2.1 click
    click_cvr_weight = (label[:, 0]).detach()
    loss_click_cvr = WeightedBCELoss(pos_weight=pos_weight[0], reduction="none")(
        clipped_cvr_pred, label[:, 1]
    )
    loss_click_cvr = loss_click_cvr * click_cvr_weight
    # 2.2 unclick
    unclicked_cvr_weight = (1 - label[:, 0]).detach()
    loss_unclick_cvr = WeightedBCELoss(pos_weight=pos_weight[0], reduction="none")(
        clipped_cvr_pred, clipped_cvr_pred
    )
    loss_unclick_cvr = loss_unclick_cvr * unclicked_cvr_weight

    loss_cvr = torch.mean(loss_click_cvr) + torch.mean(loss_unclick_cvr)
    # 3. dynamic weight
    # cvr_weight_per_batch = loss_ctr.item() / (loss_cvr.item() + 1e-6)
    # cvr_weight_per_batch = ablation_weight * min(cvr_weight_per_batch, weight)

    # Total Loss
    total_loss = loss_ctr + loss_cvr
    # total_loss = loss_ctr + cvr_weight_per_batch * loss_cvr

    return total_loss


def dcmt_bceloss(logit, label, pos_weight=[1, 1]):
    """
    ctr, cvr
    logit: [bs, num_tasks(ctr, cvr, cvr_n)]

    ctr, cvr/ctvr
    label: [bs, num_tasks]
    """
    pos_weight = torch.tensor(pos_weight)
    # simoid & clip
    ctr_prob = torch.sigmoid(logit[:, 0])
    cvr_prob = torch.sigmoid(logit[:, 1])
    cvr_n_prob = torch.sigmoid(logit[:, 2])

    # ctcvr
    ctcvr_prob = ctr_prob * cvr_prob

    clipped_ctr_pred = torch.clamp(ctr_prob, min=1e-15, max=1 - 1e-15)
    clipped_cvr_pred = torch.clamp(cvr_prob, min=1e-15, max=1 - 1e-15)
    clipped_cvr_n_pred = torch.clamp(cvr_n_prob, min=1e-15, max=1 - 1e-15)

    # Loss
    # 1. CTR
    loss_ctr = WeightedBCELoss(pos_weight=pos_weight[0])(clipped_ctr_pred, label[:, 0])

    # 2. CVR
    # 2.1 actual Space
    actual_weight = (
        label[:, 0]
        / torch.maximum(
            clipped_ctr_pred, torch.tensor(1e-6, device=clipped_ctr_pred.device)
        )
    ).detach()
    loss_cvr = WeightedBCELoss(pos_weight=pos_weight[0], reduction="none")(
        clipped_cvr_pred, label[:, 1]
    )
    loss_cvr = loss_cvr * actual_weight
    # 2.2 Counterfactual Space
    counterfactual_weight = (
        (1 - label[:, 0])
        / torch.maximum(
            (1 - clipped_ctr_pred), torch.tensor(1e-6, device=clipped_ctr_pred.device)
        )
    ).detach()
    loss_counterfactual_cvr = WeightedBCELoss(
        pos_weight=pos_weight[0], reduction="none"
    )(clipped_cvr_n_pred, (1 - label[:, 1]))
    loss_counterfactual_cvr = loss_counterfactual_cvr * counterfactual_weight
    # 2.3 soft constrain
    loss_cvr_constraint = torch.abs(1.0 - (clipped_cvr_pred + clipped_cvr_n_pred))
    total_dcmt = (
        torch.mean(loss_cvr)
        + torch.mean(loss_counterfactual_cvr)
        + 0.001 * torch.mean(loss_cvr_constraint)
    )

    # 3. CTCVR
    loss_ctcvr = WeightedBCELoss(pos_weight=pos_weight[1])(ctcvr_prob, label[:, 1])

    # Total Loss
    total_loss = loss_ctr + loss_ctcvr + 0.5 * total_dcmt

    return total_loss


def dr_bceloss(logit, label, pos_weight=[1, 1]):
    """
    ctr, cvr
    logit: [bs, num_tasks(ctr, cvr, imputation_tower)]

    ctr, cvr/ctvr
    label: [bs, num_tasks]
    """
    pos_weight = torch.tensor(pos_weight)
    # simoid & clip
    ctr_prob = torch.sigmoid(logit[:, 0])
    cvr_prob = torch.sigmoid(logit[:, 1])
    imp_prob = torch.sigmoid(logit[:, 2])

    # ctcvr
    ctcvr_prob = ctr_prob * cvr_prob

    clipped_ctr_pred = torch.clamp(ctr_prob, min=1e-15, max=1 - 1e-15)
    clipped_cvr_pred = torch.clamp(cvr_prob, min=1e-15, max=1 - 1e-15)
    clipped_imp_pred = torch.clamp(imp_prob, min=1e-15, max=1 - 1e-15)

    # Loss
    # 1. CTR
    loss_ctr = WeightedBCELoss(pos_weight=pos_weight[0])(clipped_ctr_pred, label[:, 0])
    # 2. CVR(DR)
    loss_cvr = WeightedBCELoss(pos_weight=pos_weight[0], reduction="none")(
        clipped_cvr_pred, label[:, 1]
    )
    # DR: error part
    err = loss_cvr - clipped_imp_pred
    ips = (label[:, 0] / clipped_ctr_pred).detach()
    ips = torch.clamp(ips, min=-15, max=15)
    loss_error_second = err * ips
    loss_error = clipped_imp_pred + loss_error_second
    # DR: imp part
    loss_imp = torch.square(err)
    loss_imp = loss_imp * ips
    loss_dr = torch.mean(loss_error) + torch.mean(loss_imp)
    # 3. CTCVR
    loss_ctcvr = WeightedBCELoss(pos_weight=pos_weight[1])(ctcvr_prob, label[:, 1])

    # Total Loss
    total_loss = loss_ctr + loss_ctcvr + 0.5 * loss_dr

    return total_loss


def ipw_bceloss(logit, label, pos_weight=[1, 1]):
    """
    ctr, cvr
    logit: [bs, num_tasks]

    ctr, cvr/ctvr
    label: [bs, num_tasks]
    """
    # simoid & clip
    ctr_prob = torch.sigmoid(logit[:, 0])
    cvr_prob = torch.sigmoid(logit[:, 1])
    pos_weight = torch.tensor(pos_weight)
    # ctcvr
    ctcvr_prob = ctr_prob * cvr_prob

    clipped_ctr_pred = torch.clamp(ctr_prob, min=1e-15, max=1 - 1e-15)
    clipped_cvr_pred = torch.clamp(cvr_prob, min=1e-15, max=1 - 1e-15)

    # loss
    # 1. CTR
    loss_ctr = WeightedBCELoss(pos_weight=pos_weight[0])(clipped_ctr_pred, label[:, 0])
    # 2. CVR(IPW)
    ips = (
        label[:, 0]
        / torch.maximum(
            clipped_ctr_pred, torch.tensor(1e-6, device=clipped_ctr_pred.device)
        )
    ).detach()
    # ips = torch.clamp(ips, min=-15, max=15)
    # loss_cvr = nn.BCELoss(reduction="none")(clipped_cvr_pred, label[:, 1])
    loss_cvr = WeightedBCELoss(pos_weight=pos_weight[0], reduction="none")(
        clipped_cvr_pred, label[:, 1]
    )
    loss_ips = loss_cvr * ips

    # 3. CTCVR
    loss_ctcvr = WeightedBCELoss(pos_weight=pos_weight[1])(ctcvr_prob, label[:, 1])

    total_loss = loss_ctr + loss_ctcvr + 0.5 * torch.mean(loss_ips)

    return total_loss


def tafe_bceloss(logit, label, pos_weight=[1, 1]):
    """
    logit: [bs, num_tasks]
    label: [bs, num_tasks]
    """
    # ctr, ctcvr
    # # simoid
    # prob = torch.sigmoid(logit)
    # pos_weight = torch.tensor(pos_weight)
    # # m_loss
    # loss_ctr = WeightedBCELoss(pos_weight=pos_weight[0])(prob[:, 0], label[:, 0])
    # loss_ctcvr = WeightedBCELoss(pos_weight=pos_weight[1])(prob[:, 1], label[:, 1])
    # # d_loss
    # d_loss = torch.nn.MSELoss()((prob[:, 0] - prob[:, 1]), (label[:, 0] - label[:, 1]))

    # total_loss = loss_ctr + loss_ctcvr + d_loss

    # multi-feedback
    # simoid
    prob = torch.sigmoid(logit)
    pos_weight = torch.tensor(pos_weight)
    # m_loss
    loss = 0.0
    for i in range(logit.shape[1]):
        loss += WeightedBCELoss(pos_weight=pos_weight[i])(prob[:, i], label[:, i])

    d_loss = 0.0
    for i in range(logit.shape[1] - 1):
        d_loss += nn.MSELoss()(
            (prob[:, i] - prob[:, i + 1]), (label[:, i] - label[:, i + 1])
        )

    total_loss = loss + d_loss

    return total_loss


def esmm_bceloss(logit, label, pos_weight=[1, 1]):
    """
    logit: [bs, num_tasks]
    label: [bs, num_tasks]
    """
    # ctr_prob = torch.sigmoid(logit[:, 0])
    # cvr_prob = torch.sigmoid(logit[:, 1])
    # pos_weight = torch.tensor(pos_weight)
    # # ctcvr
    # ctcvr_prob = ctr_prob * cvr_prob

    # # loss
    # loss_ctr = WeightedBCELoss(pos_weight=pos_weight[0])(ctr_prob, label[:, 0])
    # loss_ctcvr = WeightedBCELoss(pos_weight=pos_weight[1])(ctcvr_prob, label[:, 1])

    # total_loss = loss_ctr + loss_ctcvr

    prob = torch.sigmoid(logit)
    pos_weight = torch.tensor(pos_weight)

    prob = torch.cumprod(prob, dim=1)
    # for i in range(1, logit.shape[1]):
    #     prob[:, i] = prob[:, i] * prob[:, i - 1]

    # loss
    total_loss = 0.0

    for i in range(logit.shape[1]):
        pos = (label[:, 0 : i + 1] == 1).all(dim=1)
        new_label = pos.float()
        total_loss += WeightedBCELoss(pos_weight=pos_weight[i])(prob[:, i], new_label)

    return total_loss


def multi_bceloss(logit, label, pos_weight=[1, 1]):
    """
    logit: [bs, num_tasks]
    label: [bs, num_tasks]
    """
    # simoid
    # prob = torch.sigmoid(logit)
    # pos_weight = torch.tensor(pos_weight)
    # loss
    # loss_ctr = nn.BCEWithLogitsLoss(pos_weight=pos_weight[0])(logit[:, 0], label[:, 0])
    # loss_ctcvr = nn.BCEWithLogitsLoss(pos_weight=pos_weight[1])(
    #     logit[:, 1], label[:, 1]
    # )
    # total_loss = loss_ctr + loss_ctcvr

    prob = torch.sigmoid(logit)
    pos_weight = torch.tensor(pos_weight)
    total_loss = 0.0

    for i in range(logit.shape[1]):
        total_loss += WeightedBCELoss(pos_weight=pos_weight[i])(prob[:, i], label[:, i])

    return total_loss


def setrank(true, pred):
    pos_indices = true.squeeze() == 1
    neg_indices = true.squeeze() == 0

    pos_pred = pred[pos_indices]
    neg_pred = pred[neg_indices]

    if pos_pred.size(0) == 0 or neg_pred.size(0) == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    neg_sum = torch.sum(neg_pred)
    loss = torch.sum(-torch.log(pos_pred / (pos_pred + neg_sum)))

    return loss


def batch_setrank(true, pred, groups):
    sum_group = 0
    total_loss = 0
    for group in groups:
        true_group = true[sum_group : sum_group + group]
        pred_group = pred[sum_group : sum_group + group]
        sum_group += group
        total_loss += setrank(true_group, pred_group)

    return total_loss / len(groups)


def jrc(true, pred, group, pos_weight=1):
    groupoint_listen = len(group)
    alpha = 0.5
    label = true.squeeze().long()
    logits = pred.squeeze()
    batch = true.shape[0]
    mask = torch.repeat_interleave(
        torch.arange(groupoint_listen), torch.tensor(group)
    ).to(true.device)
    mask = mask.unsqueeze(-1).expand(batch, groupoint_listen)
    mask_m = torch.arange(groupoint_listen).repeat(batch, 1).to(true.device)
    mask = (mask == mask_m).int()
    weight = torch.tensor([1.0, pos_weight], device=pred.device)
    ce_loss = F.cross_entropy(logits, label, weight=weight)

    logits = logits.unsqueeze(1).expand(batch, groupoint_listen, 2)
    y = label.unsqueeze(1).expand(batch, groupoint_listen)
    y_neg, y_pos = 1 - y, y
    y_neg = y_neg * mask
    y_pos = y_pos * mask
    logits = logits + (1 - mask.unsqueeze(2)) * -1e9

    l_neg, l_pos = logits[:, :, 0], logits[:, :, 1]

    loss_pos = -torch.sum(y_pos * F.log_softmax(l_pos, dim=0), dim=0) * pos_weight
    loss_neg = -torch.sum(y_neg * F.log_softmax(l_neg, dim=0), dim=0)
    ge_loss = torch.mean((loss_pos + loss_neg) / torch.sum(mask, dim=0))

    loss = alpha * ce_loss + (1 - alpha) * ge_loss
    return loss


def set2setrank(true, pred, w):
    pos_indices = true.squeeze() == 1
    neg_indices = true.squeeze() == 0

    pos_pred = pred[pos_indices]
    neg_pred = pred[neg_indices]

    if pos_pred.size(0) == 0 or neg_pred.size(0) == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    pos_pred_exp = pos_pred.unsqueeze(0)  # (1, num_pos)
    neg_pred_exp = neg_pred.unsqueeze(1)  # (num_neg, 1)
    log_sigmoid_diff = -torch.log(torch.sigmoid(pos_pred_exp - neg_pred_exp))
    loss_2 = torch.mean(log_sigmoid_diff, dim=1)
    fneg = torch.min(loss_2)
    loss_2 = torch.sum(loss_2)

    pos_pred_exp = pos_pred.unsqueeze(0)  # (1, num_pos)
    pos_pred_diff = torch.abs(
        pos_pred.unsqueeze(1) - pos_pred_exp
    )  # (num_pos, num_pos)
    log_sigmoid_diff = torch.where(
        pos_pred_diff > 0.5,
        torch.full_like(pos_pred_diff, 0.5, device=pos_pred_diff.device),
        pos_pred_diff,
    )
    # log_sigmoid_diff = -torch.log(torch.sigmoid(pos_pred_diff))
    # fpos = torch.sum(log_sigmoid_diff, dim=1)
    fpos = torch.mean(log_sigmoid_diff)

    loss_3 = -torch.log(torch.sigmoid(fpos - fneg))
    return loss_2 + w * loss_3


def batch_set2setrank(true, pred, groups, w=1):
    sum_group = 0
    loss = 0
    for group in groups:
        true_group = true[sum_group : sum_group + group]
        pred_group = pred[sum_group : sum_group + group]
        sum_group += group
        loss += set2setrank(true_group, pred_group, w)

    return loss / len(groups)


def listNet(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE):
    """
    ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    y_pred[mask] = float("-inf")
    y_true[mask] = float("-inf")

    preds_smax = F.softmax(y_pred, dim=1)
    true_smax = F.softmax(y_true, dim=1)

    preds_smax = preds_smax + eps
    preds_log = torch.log(preds_smax)

    return torch.mean(-torch.sum(true_smax * preds_log, dim=1))


def rankNet(
    y_pred,
    y_true,
    padded_value_indicator=PADDED_Y_VALUE,
    weight_by_diff=False,
    weight_by_diff_powed=False,
):
    """
    RankNet loss introduced in "Learning to Rank using Gradient Descent".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param weight_by_diff: flag indicating whether to weight the score differences by ground truth differences.
    :param weight_by_diff_powed: flag indicating whether to weight the score differences by the squared ground truth differences.
    :return: loss value, a torch.Tensor
    """
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    y_pred[mask] = float("-inf")
    y_true[mask] = float("-inf")

    # here we generate every pair of indices from the range of document length in the batch
    document_pairs_candidates = list(product(range(y_true.shape[1]), repeat=2))

    pairs_true = y_true[:, document_pairs_candidates]
    selected_pred = y_pred[:, document_pairs_candidates]

    # here we calculate the relative true relevance of every candidate pair
    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
    pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]

    # here we filter just the pairs that are 'positive' and did not involve a padded instance
    # we can do that since in the candidate pairs we had symetric pairs so we can stick with
    # positive ones for a simpler loss function formulation
    the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))
    pred_diffs = pred_diffs[the_mask]

    weight = None
    if weight_by_diff:
        abs_diff = torch.abs(true_diffs)
        weight = abs_diff[the_mask]
    elif weight_by_diff_powed:
        true_pow_diffs = torch.pow(pairs_true[:, :, 0], 2) - torch.pow(
            pairs_true[:, :, 1], 2
        )
        abs_diff = torch.abs(true_pow_diffs)
        weight = abs_diff[the_mask]

    # here we 'binarize' true relevancy diffs since for a pairwise loss we just need to know
    # whether one document is better than the other and not about the actual difference in
    # their relevancy levels
    true_diffs = (true_diffs > 0).type(torch.float32)
    true_diffs = true_diffs[the_mask]

    return nn.BCEWithLogitsLoss(weight=weight)(pred_diffs, true_diffs)


def lambdaLoss(
    y_pred,
    y_true,
    eps=DEFAULT_EPS,
    padded_value_indicator=PADDED_Y_VALUE,
    weighing_scheme=None,
    k=None,
    sigma=1.0,
    mu=10.0,
    reduction="mean",
    reduction_log="binary",
):
    """
    LambdaLoss framework for LTR losses implementations, introduced in "The LambdaLoss Framework for Ranking Metric Optimization".
    Contains implementations of different weighing schemes corresponding to e.g. LambdaRank or RankNet.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param weighing_scheme: a string corresponding to a name of one of the weighing schemes
    :param k: rank at which the loss is truncated
    :param sigma: score difference weight used in the sigmoid function
    :param mu: optional weight used in NDCGLoss2++ weighing scheme
    :param reduction: losses reduction method, could be either a sum or a mean
    :param reduction_log: logarithm variant used prior to masking and loss reduction, either binary or natural
    :return: loss value, a torch.Tensor
    """

    device = y_pred.device
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    padded_mask = y_true == padded_value_indicator
    y_pred[padded_mask] = float("-inf")
    y_true[padded_mask] = float("-inf")

    # Here we sort the true and predicted relevancy scores.
    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

    # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs)

    if weighing_scheme != "ndcgLoss1_scheme":
        padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

    ndcg_at_k_mask = torch.zeros(
        (y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device
    )
    ndcg_at_k_mask[:k, :k] = 1

    # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    true_sorted_by_preds.clamp_(min=0.0)
    y_true_sorted.clamp_(min=0.0)

    # Here we find the gains, discounts and ideal DCGs per slate.
    pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
    D = torch.log2(1.0 + pos_idxs.float())[None, :]
    maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(
        min=eps
    )
    G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

    # Here we apply appropriate weighing scheme - ndcgLoss1, ndcgLoss2, ndcgLoss2++ or no weights (=1.0)
    if weighing_scheme is None:
        weights = 1.0
    else:
        weights = globals()[weighing_scheme](
            G, D, mu, true_sorted_by_preds
        )  # type: ignore

    # We are clamping the array entries to maintain correct backprop (log(0) and division by 0)
    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(
        min=-1e8, max=1e8
    )
    scores_diffs.masked_fill(torch.isnan(scores_diffs), 0.0)
    weighted_probas = (
        torch.sigmoid(sigma * scores_diffs).clamp(min=eps) ** weights
    ).clamp(min=eps)
    if reduction_log == "natural":
        losses = torch.log(weighted_probas)
    elif reduction_log == "binary":
        losses = torch.log2(weighted_probas)
    else:
        raise ValueError("Reduction logarithm base can be either natural or binary")

    if reduction == "sum":
        loss = -torch.sum(losses[padded_pairs_mask & ndcg_at_k_mask])
    elif reduction == "mean":
        loss = -torch.mean(losses[padded_pairs_mask & ndcg_at_k_mask])
    else:
        raise ValueError("Reduction method can be either sum or mean")

    return loss


def ndcgLoss1_scheme(G, D, *args):
    return (G / D)[:, :, None]


def lambdaRank_scheme(G, D, *args):
    return torch.abs(
        torch.pow(D[:, :, None], -1.0) - torch.pow(D[:, None, :], -1.0)
    ) * torch.abs(G[:, :, None] - G[:, None, :])
