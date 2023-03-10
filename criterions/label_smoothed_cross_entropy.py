# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import math
import pdb
from turtle import pd
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
from utils.eval_utils import decode_fn
from data.mm_data.sgg_VG_dataset import VGDatasetReader
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou


@dataclass
class AdjustLabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    ignore_eos: bool = field(
        default=False,
        metadata={"help": "Ignore eos token"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    drop_worst_ratio: float = field(
        default=0.0,
        metadata={"help": "ratio for discarding bad samples"},
    )
    drop_worst_after: int = field(
        default=0,
        metadata={"help": "steps for discarding bad samples"},
    )
    use_rdrop: bool = field(
        default=False, metadata={"help": "use R-Drop"}
    )
    reg_alpha: float = field(
        default=1.0, metadata={"help": "weight for R-Drop"}
    )
    sample_patch_num: int = field(
        default=196, metadata={"help": "sample patches for v1"}
    )
    constraint_range: Optional[str] = field(
        default=None,
        metadata={"help": "constraint range"}
    )


def construct_rdrop_sample(x):
    if isinstance(x, dict):
        for key in x:
            x[key] = construct_rdrop_sample(x[key])
        return x
    elif isinstance(x, torch.Tensor):
        return x.repeat(2, *([1] * (x.dim()-1)))
    elif isinstance(x, int):
        return x * 2
    elif isinstance(x, np.ndarray):
        return x.repeat(2)
    else:
        raise NotImplementedError


def kl_loss(p, q):
    p_loss = F.kl_div(p, torch.exp(q), reduction='sum')
    q_loss = F.kl_div(q, torch.exp(p), reduction='sum')
    loss = (p_loss + q_loss) / 2
    return loss

def bbox_iou(boxes_pred, boxes_gt):
    num_pred = len(boxes_pred)
    num_gt = len(boxes_gt)
    iou_matrix = np.zeros((num_pred, num_gt))
    
    # Compute IoU matrix
    for i in range(num_pred):
        for j in range(num_gt):
            intersection = compute_intersection(boxes_pred[i], boxes_gt[j])
            union = compute_union(boxes_pred[i], boxes_gt[j], intersection)
            iou_matrix[i][j] = intersection / union
    
    # Perform matching using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    
    # Compute total IoU for matched pairs
    total_iou = 0
    for i, j in zip(row_ind, col_ind):
        total_iou += iou_matrix[i][j]
    total_iou /= len(row_ind)
    
    return total_iou, iou_matrix
    # total_iou, 
    
def compute_intersection(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0
    else:
        return (x2 - x1) * (y2 - y1)

def compute_union(box1, box2, intersection):
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return area1 + area2 - intersection


def label_smoothed_nll_loss(
        lprobs, target, epsilon, update_num, reduce=True,
        drop_worst_ratio=0.0, drop_worst_after=0, use_rdrop=False, reg_alpha=1.0,
        constraint_masks=None, constraint_start=None, constraint_end=None
):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    target = target.type(torch.int64)
    nll_loss = -lprobs.gather(dim=-1, index=target).squeeze(-1) # out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
    if constraint_masks is not None:
        smooth_loss = -lprobs.masked_fill(~constraint_masks, 0).sum(dim=-1, keepdim=True).squeeze(-1)
        eps_i = epsilon / (constraint_masks.sum(1) - 1 + 1e-6)
    elif constraint_start is not None and constraint_end is not None:
        constraint_range = [0, 1, 2, 3] + list(range(constraint_start, constraint_end))
        smooth_loss = -lprobs[:, constraint_range].sum(dim=-1, keepdim=True).squeeze(-1)
        eps_i = epsilon / (len(constraint_range) - 1 + 1e-6)
    else:
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True).squeeze(-1)
        eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    if drop_worst_ratio > 0 and update_num > drop_worst_after:
        if use_rdrop:
            true_batch_size = loss.size(0) // 2
            _, indices = torch.topk(loss[:true_batch_size], k=int(true_batch_size * (1 - drop_worst_ratio)), largest=False)
            loss = torch.cat([loss[indices], loss[indices+true_batch_size]])
            nll_loss = torch.cat([nll_loss[indices], nll_loss[indices+true_batch_size]])
            lprobs = torch.cat([lprobs[indices], lprobs[indices+true_batch_size]])
        else:
            loss, indices = torch.topk(loss, k=int(loss.shape[0] * (1 - drop_worst_ratio)), largest=False)
            nll_loss = nll_loss[indices]
            lprobs = lprobs[indices]

    ntokens = loss.numel()
    nll_loss = nll_loss.sum()
    loss = loss.sum()
    if use_rdrop:
        true_batch_size = lprobs.size(0) // 2
        p = lprobs[:true_batch_size]
        q = lprobs[true_batch_size:]
        if constraint_start is not None and constraint_end is not None:
            constraint_range = [0, 1, 2, 3] + list(range(constraint_start, constraint_end))
            p = p[:, constraint_range]
            q = q[:, constraint_range]
        loss += kl_loss(p, q) * reg_alpha

    return loss, nll_loss, ntokens


@register_criterion(
    "adjust_label_smoothed_cross_entropy", dataclass=AdjustLabelSmoothedCrossEntropyCriterionConfig
)
class AdjustLabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        ignore_eos=False,
        report_accuracy=False,
        drop_worst_ratio=0,
        drop_worst_after=0,
        use_rdrop=False,
        reg_alpha=1.0,
        sample_patch_num=196,
        constraint_range=None
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.ignore_eos = ignore_eos
        self.report_accuracy = report_accuracy
        self.drop_worst_ratio = drop_worst_ratio
        self.drop_worst_after = drop_worst_after
        self.use_rdrop = use_rdrop
        self.reg_alpha = reg_alpha
        self.sample_patch_num = sample_patch_num

        self.constraint_start = None
        self.constraint_end = None
        if constraint_range is not None:
            constraint_start, constraint_end = constraint_range.split(',')
            self.constraint_start = int(constraint_start)
            self.constraint_end = int(constraint_end)

    def forward(self, generator, model, sample, update_num=0, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # sample is dict
        if isinstance(sample, list):
            if self.sample_patch_num > 0:
                sample[0]['net_input']['sample_patch_num'] = self.sample_patch_num
            loss_v1, sample_size_v1, logging_output_v1 = self.forward(model, sample[0], update_num, reduce)
            loss_v2, sample_size_v2, logging_output_v2 = self.forward(model, sample[1], update_num, reduce)
            loss = loss_v1 / sample_size_v1 + loss_v2 / sample_size_v2
            sample_size = 1
            logging_output = {
                "loss": loss.data,
                "loss_v1": loss_v1.data,
                "loss_v2": loss_v2.data,
                "nll_loss": logging_output_v1["nll_loss"].data / sample_size_v1 + logging_output_v2["nll_loss"].data / sample_size_v2,
                "ntokens": logging_output_v1["ntokens"] + logging_output_v2["ntokens"],
                "nsentences": logging_output_v1["nsentences"] + logging_output_v2["nsentences"],
                "sample_size": 1,
                "sample_size_v1": sample_size_v1,
                "sample_size_v2": sample_size_v2,
            }
            return loss, sample_size, logging_output

        if self.use_rdrop:
            construct_rdrop_sample(sample)

        net_output = model(**sample["net_input"]) 
        '''
        sample["net_input"]: dict, 
            keys: ['src_tokens', 'src_lengths', 'patch_images', 'patch_masks', 'prev_output_tokens']
        sample["target"]:                   [4, 628]     or     [4, 755]    or    [4, 576]...


        net_output: tuple, len=2
        net_output[0]:                      [4, 628, 59457] or [4, 755, 59457] or [4, 576, 59457]...
        net_output[1]: dict, keys: ['attn', 'inner_states']
        net_output[1]['attn']: list, len=1, [4, 628, 1029] or [4, 755, 1029] or [4, 576, 1029]...
        net_output[1]['inner_states']: list, len=5
                                       [0]: [628, 4, 256] or [755, 4, 256] or [576, 4, 256]...
                                       [1]: same as [0]
                                       [2]: same as [0]
                                       [3]: same as [0]
                                       [4]: same as [0]
        '''
        # print("target shape just before input to model: ", sample["target"].shape)
        
        loss, nll_loss, ntokens = self.compute_loss(generator, model, net_output, sample, update_num, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else ntokens
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        conf = sample['conf'][:, None, None] if 'conf' in sample and sample['conf'] is not None else 1
        constraint_masks = None
        if "constraint_masks" in sample and sample["constraint_masks"] is not None:
            constraint_masks = sample["constraint_masks"]
            net_output[0].masked_fill_(~constraint_masks, -math.inf)
        if self.constraint_start is not None and self.constraint_end is not None:
            net_output[0][:, :, 4:self.constraint_start] = -math.inf
            net_output[0][:, :, self.constraint_end:] = -math.inf
        lprobs = model.get_normalized_probs(net_output, log_probs=True) * conf # log softmax to the -1 dim of net_output[0]
        target = model.get_targets(sample, net_output) # sample["target"]
        if self.ignore_prefix_size > 0:
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
            if constraint_masks is not None:
                constraint_masks = constraint_masks[:, self.ignore_prefix_size :, :].contiguous()
        if self.ignore_eos:
            bsz, seq_len, embed_dim = lprobs.size()
            # print("bsz, seq_len, embed_dim: ", lprobs.shape)
            eos_indices = target.eq(self.task.tgt_dict.eos())
            lprobs = lprobs[~eos_indices].reshape(bsz, seq_len-1, embed_dim)
            target = target[~eos_indices].reshape(bsz, seq_len-1)
            if constraint_masks is not None:
                constraint_masks = constraint_masks[~eos_indices].reshape(bsz, seq_len-1, embed_dim)
        if constraint_masks is not None:
            constraint_masks = constraint_masks.view(-1, constraint_masks.size(-1))
        return lprobs.view(-1, lprobs.size(-1)), lprobs, target.view(-1), target, constraint_masks

    def compute_loss(self, generator, model, net_output, sample, update_num, reduce=True):
        # sample['target']: [4, 14]
        lprobs, lprobs_raw, target, target_raw, constraint_masks = self.get_lprobs_and_target(model, net_output, sample)
        # target [56]
        if constraint_masks is not None:
            constraint_masks = constraint_masks[target != self.padding_idx]
        
        bbox_loss, loss_obj_num = self.compute_bbox_loss(generator, lprobs_raw, target_raw, sample)
        # print("bbox_loss: ", bbox_loss)
        lprobs = lprobs[target != self.padding_idx]
        target = target[target != self.padding_idx] # [26]
        # print("lprobs shape: ", lprobs.shape)
        # print("target shape: ", target.shape)
        loss, nll_loss, ntokens = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            update_num,
            reduce=reduce,
            drop_worst_ratio=self.drop_worst_ratio,
            drop_worst_after=self.drop_worst_after,
            use_rdrop=self.use_rdrop,
            reg_alpha=self.reg_alpha,
            constraint_masks=constraint_masks,
            constraint_start=self.constraint_start,
            constraint_end=self.constraint_end
        )
        # print("loss: ", loss)
        bbox_weight_l2 = 30.0
        bbox_weight_l1 = 3.0
        loss += bbox_weight_l2 * (10.0 * bbox_loss + 1.0 * loss_obj_num)
        # print("loss with bbox: ", loss)
        return loss, nll_loss, ntokens
    
    def compute_bbox_loss(self, generator, lprobs, target, sample):
        '''
        lprobs shape: [26, 59457]
        target shape: [26, 1]

        lprobs:  tensor([[ -9.7679, -11.5233, -10.5523,  ..., -11.0178, -11.4471, -11.2739],
        [-10.8818, -11.2606, -10.6506,  ..., -11.1683, -11.4382, -11.4560],
        [-11.2819, -11.5219, -10.7650,  ..., -11.1123, -11.1373, -11.4442],
        ...,
        [ -9.6215, -11.2327, -10.6339,  ..., -10.9131, -11.4379, -11.2017],
        [-11.2439, -11.4125, -11.2076,  ..., -11.1913, -11.1589, -11.1864],
        [-11.1555, -10.7742, -10.7993,  ..., -10.9660, -11.1646, -11.0310]])

        target:  tensor([[    3], [17531],[    3],[17531],[    3],[    2],[    3],[    3],[    2],[    3],[ 1657],[    3],[    3],[    3],[    3],
                        [ 1657],[    3],[    3],[    3],[    3],[    3],[    3],[    2],[  462],[    3],[    2]])
        
        '''

        indexes = sample["idx"].tolist()
        bpe = None
        lprobs = lprobs.argmin(dim=-1) # THIS is very important! don't comments off it!
        bbox_loss_fn = nn.MSELoss()
        loss_total = 0

        # print("lprobs shape: ", lprobs.shape)
        # print("target shape: ", target.shape)

        # pdb.set_trace()

        # lprobs_obj_num_total = 0
        # target_obj_num_total = 0

        bbox_iou_total = 0


        for i in range(len(indexes)):

            lprobs_i = lprobs[i,:]
            target_i = target[i,:]

            lprobs_i = lprobs_i[target_i != self.padding_idx]
            target_i = target_i[target_i != self.padding_idx]
            
            lprobs_i_decode = decode_fn(lprobs_i, self.task.tgt_dict, bpe, generator)
            target_i_decode = decode_fn(target_i, self.task.tgt_dict, bpe, generator)

            lprobs_i_decode = lprobs_i_decode.split()
            target_i_decode = target_i_decode.split()

            # print("lprobs_i_decode: ", lprobs_i_decode)
            # print("target_i_decode: ", target_i_decode)

            
            if len(lprobs_i_decode) == 0 or len(target_i_decode) == 0:
                continue

            lprobs_bbox_i = []
            target_bbox_i = []
            for j in range(len(lprobs_i_decode)):
                if '<' in lprobs_i_decode[j] and lprobs_i_decode[j] != '<unk>' and lprobs_i_decode[j] != '<pad>':
                    try:
                        bbox_half_lprobs = lprobs_i_decode[j].split('_')[1]
                        bbox_val_lprobs = int(bbox_half_lprobs.strip('>'))
                        lprobs_bbox_i.append(bbox_val_lprobs)
                    except:
                        # continue
                        print("Unexpected bbox from lropbs: ", lprobs_i_decode[j])
            
            for k in range(len(target_i_decode)):
                if '<' in target_i_decode[k] and target_i_decode[k] != '<unk>':
                    try:
                        bbox_half_target = target_i_decode[k].split('_')[1]
                        bbox_val_target = int(bbox_half_target.strip('>'))
                        target_bbox_i.append(bbox_val_target)
                    except:
                        # continue
                        print("Unexpected bbox from target: ", target_i_decode[k])

            # print("lprobs_bbox_i: ", lprobs_bbox_i)
            # print("target_bbox_i: ", target_bbox_i)

            lprobs_bbox_in_list = []
            target_bbox_in_list = []

            if len(lprobs_bbox_i) < 4:
                return 0, 100
            else:
                for j in range(len(lprobs_bbox_i) // 4):
                    bbox_j = lprobs_bbox_i[4*j:4*j+4]
                    lprobs_bbox_in_list.append(bbox_j)
            
            for k in range(len(target_bbox_i) // 4):
                bbox_k = target_bbox_i[4*k:4*k+4]
                target_bbox_in_list.append(bbox_k)

            # pdb.set_trace()

            bbox_iou_i, iou_matrix = bbox_iou(lprobs_bbox_in_list, target_bbox_in_list)
            bbox_iou_total += bbox_iou_i

            # Compute the set loss term
            # num_pred = len(lprobs_bbox_in_list)
            # num_target = len(target_bbox_in_list)
            # cost = torch.zeros((num_pred, num_target))  # Initialize the cost matrix
            # if num_pred > 0 and num_target > 0:
            #     iou = box_iou(torch.Tensor(lprobs_bbox_in_list), torch.Tensor(target_bbox_in_list))
            #     cost = -iou  # Negative IoU represents the matching cost

            # # Solve the optimal matching using the Hungarian algorithm
            # row_ind, col_ind = torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)
            # if num_pred > 0 and num_target > 0:
            #     # try:
            #     row_ind, col_ind = torch.linalg.solve(torch.Tensor(cost), torch.zeros(num_pred, dtype=torch.float32))
            #     row_ind, col_ind = row_ind.round().to(torch.int64), col_ind.round().to(torch.int64)
            #     # except:
            #     #     print("cost type: ", type(cost))
            #     #     print("second type: ", type(torch.zeros(num_pred, dtype=torch.float32)))
            #     #     print("num pred type: ", num_pred)
            #     #     # pdb.set_trace()
            
            # # Compute the set loss term as the number of false positives and false negatives
            # num_false_pos = (row_ind == -1).sum()
            # num_false_neg = (col_ind == -1).sum()
            # loss_obj_num = num_false_pos + num_false_neg

            # Compute the pairwise box similarity scores
            similarity = torch.cdist(torch.Tensor(lprobs_bbox_in_list), torch.Tensor(target_bbox_in_list), p=1)

            # Convert the similarity scores to a cost matrix
            cost = similarity.numpy()

            # Use the Hungarian algorithm to find the minimum-cost matching
            row_ind, col_ind = linear_sum_assignment(cost)

            # Extract the matched boxes
            matched_indices = np.stack([row_ind, col_ind], axis=1)
            matched_boxes = torch.zeros(len(lprobs_bbox_in_list), 4)
            ground_truth_boxes = torch.Tensor(target_bbox_in_list)
            matched_boxes[row_ind] = ground_truth_boxes[col_ind]

            # Compute the foreground/background binary mask
            fg_mask = torch.zeros(len(lprobs_bbox_in_list), dtype=torch.bool)
            fg_mask[row_ind] = True

            # Compute the matching cost
            matching_cost = cost[row_ind, col_ind].sum()
            

            # Calculate MSE loss
            if len(lprobs_bbox_i) < len(target_bbox_i):
                for l in range(len(target_bbox_i)-len(lprobs_bbox_i)):
                    lprobs_bbox_i.append(0)
            else:
                for l in range(len(lprobs_bbox_i)-len(target_bbox_i)):
                    target_bbox_i.append(0)
            
            lprobs_bbox_i = torch.Tensor(lprobs_bbox_i)
            target_bbox_i = torch.Tensor(target_bbox_i)

            assert lprobs_bbox_i.shape == target_bbox_i.shape

            # print("lprobs_bbox_i: ", lprobs_bbox_i)
            # print("target_bbox_i: ", target_bbox_i)


            # loss = F.l1_loss(lprobs_bbox_i, target_bbox_i)
            loss = bbox_loss_fn(lprobs_bbox_i, target_bbox_i)
            loss_total += loss

        average_loss = loss_total / len(indexes)
        average_iou = bbox_iou_total / len(indexes)
        
        return (1 - average_iou), matching_cost


    def compute_accuracy(self, model, net_output, sample):
        lprobs, lprobs_raw, target, target_raw, constraint_masks = self.get_lprobs_and_target(model, net_output, sample)
        # lprobs shape:  [1852, 59457]
        # target shape:  [1852]
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        loss_sum_v1 = sum(log.get("loss_v1", 0) for log in logging_outputs)
        loss_sum_v2 = sum(log.get("loss_v2", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        sample_size_v1 = sum(log.get("sample_size_v1", 0) for log in logging_outputs)
        sample_size_v2 = sum(log.get("sample_size_v2", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "loss_v1", loss_sum_v1 / max(sample_size_v1, 1), max(sample_size_v1, 1), round=3
        )
        metrics.log_scalar(
            "loss_v2", loss_sum_v2 / max(sample_size_v2, 1), max(sample_size_v2, 1), round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / sample_size, ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        metrics.log_scalar(
            "ntokens", ntokens, 1, round=3
        )
        metrics.log_scalar(
            "nsentences", nsentences, 1, round=3
        )
        metrics.log_scalar(
            "sample_size", sample_size, 1, round=3
        )
        metrics.log_scalar(
            "sample_size_v1", sample_size_v1, 1, round=3
        )
        metrics.log_scalar(
            "sample_size_v2", sample_size_v2, 1, round=3
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
