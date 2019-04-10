# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     losses
   Description :
   Author :       haxu
   date：          2019/3/29
-------------------------------------------------
   Change Activity:
                   2019/3/29:
-------------------------------------------------
"""
__author__ = 'haxu'

from torch import nn

"""
y: (bs,num_obj, 5) # (cx,cy,w,h,1) #last idx is the label
"""

import torch
import torch.nn.functional as F


# num_classes, 0.35, True, 0, True, 7, 0.35, False
# num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.   (cx,cy,w,h)
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                      boxes[:, :2] + boxes[:, 2:] / 2), 1)


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):  # 交并比
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


# [num_priors,4], [n_priors,4],  [0.1, 0.2]
def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index  # 交并比 (A,B)
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)  # 每一行的最大值，并保持维度

    # ignore hard gt
    valid_gt_idx = best_prior_overlap[:, 0] >= 0.2
    best_prior_idx_filter = best_prior_idx[valid_gt_idx, :]
    if best_prior_idx_filter.shape[0] <= 0:
        loc_t[idx] = 0
        conf_t[idx] = 0
        return

    # [1,num_priors] best ground truth for each prior
    # 对每一个先验框最大的真实的框
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)

    best_truth_idx.squeeze_(0)  # (n,) idx 对每一个先验框最大的真实的框
    best_truth_overlap.squeeze_(0)  # (n,) value  对每一个先验框最大的真实的框
    best_prior_idx.squeeze_(1)  # idx (n,) 对每一个真实的框最大的先验框
    best_prior_idx_filter.squeeze_(1)  # (m,) idx 概率超过0.2的对每一个真实的框最大的先验框

    best_prior_overlap.squeeze_(1)  # 对每一个真实的框最大的先验框
    best_truth_overlap.index_fill_(0, best_prior_idx_filter, 2)  # ensure best prior

    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j  # 保证这个式子是完美的 ensure every gt matches with its prior of max overlap

    # (num_obj, 4)
    # (num_obj, 1)
    # 找每一个gt对应的最大的overlap
    matches = truths[best_truth_idx]  # Shape: [num_obj,4]
    # 找到每一个gt对应的标签
    conf = labels[best_truth_idx]  # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background

    # [num_priors,4], [n_priors,4],  [0.1, 0.2]
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc  # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


class MultiBoxLoss(nn.Module):
    def __init__(self, device, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap,
                 encode_target):
        super(MultiBoxLoss, self).__init__()
        self.device = device

        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets):
        loc_data, conf_data, _ = predictions  # 预测的框以及类别概率(bs,-1,4) (bs,-1,2)
        priors = priors
        num = loc_data.size(0)  # bs
        num_priors = priors.size(0)

        # (bs, 21824, 4)
        loc_t = torch.Tensor(num, num_priors, 4)
        # (bs, 21824)
        conf_t = torch.LongTensor(num, num_priors)
        # (bs,num_obj, 5)
        for idx in range(num):
            truths = targets[idx][:, :-1].data  # cx,cy,w,h
            labels = targets[idx][:, -1].data  # 1 or 0
            defaults = priors.data  # default boxes
            match(0.35, truths, defaults, [0.1, 0.2], labels, loc_t, conf_t, idx)

        if self.device.type == 'cuda':
            loc_t = loc_t.to(self.device)
            conf_t = conf_t.to(self.device)

        # 得到概率 >0 的idx
        pos = conf_t > 0
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)

        # 全部展开 计算loss
        loc_p = loc_data[pos_idx].view(-1, 4)  # predict
        loc_t = loc_t[pos_idx].view(-1, 4)  # label
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        batch_conf = conf_data.view(-1, 2)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, 2)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c
