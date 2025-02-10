import os
import shutil
import time
import pickle

import numpy as np
import random
from copy import deepcopy
import math

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from .lr_schedulers import LinearWarmupMultiStepLR, LinearWarmupCosineAnnealingLR
from .postprocessing import postprocess_results
from ..modeling import MaskedConv1D, Scale, AffineDropPath, LayerNorm
from ..modeling.losses import ctr_diou_loss_1d, sigmoid_focal_loss
from libs.modeling import make_meta_arch
import torch.nn as nn
from libs.utils.meta import *
import torch.nn.functional as F

################################################################################
def fix_random_seed(seed, include_cuda=True):
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_cuda:
        # training: disable cudnn benchmark to ensure the reproducibility
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # this is needed for CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
    return rng_generator


def save_checkpoint(state, is_best, file_folder,
                    file_name='checkpoint.pth.tar'):
    """save checkpoint to file"""
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    torch.save(state, os.path.join(file_folder, file_name))
    if is_best:
        # skip the optimization / scheduler state
        state.pop('optimizer', None)
        state.pop('scheduler', None)
        torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))

def print_grad(grad):
    if grad is not None:
        print("Gradient is not None")
    else:
        print("Gradient is None")

def print_model_params(model):
    for name, param in model.named_parameters():
        print(name, param.min().item(), param.max().item(), param.mean().item())
        if param.requires_grad:
            param.register_hook(print_grad)
    return


def make_optimizer(model, optimizer_config):
    """create optimizer
    return a supported optimizer
    """
    # separate out all parameters that with / without weight decay
    # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, MaskedConv1D)
    blacklist_weight_modules = (LayerNorm, torch.nn.GroupNorm)

    # loop over all modules / params
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn.endswith('scale') and isinstance(m, (Scale, AffineDropPath)):
                # corner case of our scale layer
                no_decay.add(fpn)
            elif pn.endswith('rel_pe'):
                # corner case for relative position encoding
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, \
        "parameters %s were not separated into either decay/no_decay set!" \
        % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": optimizer_config['weight_decay']},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(
            optim_groups,
            lr=optimizer_config["learning_rate"],
            momentum=optimizer_config["momentum"],
            nesterov=True
        )
    elif optimizer_config["type"] == "AdamW":
        optimizer = optim.AdamW(
            optim_groups,
            lr=optimizer_config["learning_rate"]
        )
    else:
        raise TypeError("Unsupported optimizer!")

    return optimizer

def make_optimizer_pseudo(model, optimizer_config):
    """create optimizer
    return a supported optimizer
    """
    # separate out all parameters that with / without weight decay
    # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, MaskedConv1D)
    blacklist_weight_modules = (LayerNorm, torch.nn.GroupNorm)

    # loop over all modules / params
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn.endswith('scale') and isinstance(m, (Scale, AffineDropPath)):
                # corner case of our scale layer
                no_decay.add(fpn)
            elif pn.endswith('rel_pe'):
                # corner case for relative position encoding
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, \
        "parameters %s were not separated into either decay/no_decay set!" \
        % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": optimizer_config['weight_decay']},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    #pseudo_optimizer = MetaSGD(pseudo_net, pseudo_net.parameters(), lr=lr)
    if optimizer_config["type"] == "SGD":
        optimizer = MetaSGD(
            model,
            optim_groups,
            lr=optimizer_config["learning_rate"]
        )
    elif optimizer_config["type"] == "AdamW":
        optimizer = MetaAdamW(
            model,
            optim_groups,
            lr=optimizer_config["learning_rate"]
        )
        # optimizer = optim.AdamW(
        #     optim_groups,
        #     lr=optimizer_config["learning_rate"]
        # )
    else:
        raise TypeError("Unsupported optimizer!")

    return optimizer


def make_scheduler(
    optimizer,
    optimizer_config,
    num_iters_per_epoch,
    last_epoch=-1
):
    """create scheduler
    return a supported scheduler
    All scheduler returned by this function should step every iteration
    """
    if optimizer_config["warmup"]:
        max_epochs = optimizer_config["epochs"] + optimizer_config["warmup_epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # get warmup params
        warmup_epochs = optimizer_config["warmup_epochs"]
        warmup_steps = warmup_epochs * num_iters_per_epoch

        # with linear warmup: call our custom schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # Cosine
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_steps,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # Multi step
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = LinearWarmupMultiStepLR(
                optimizer,
                warmup_steps,
                steps,
                gamma=optimizer_config["schedule_gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    else:
        max_epochs = optimizer_config["epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # without warmup: call default schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # step per iteration
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # step every some epochs
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                steps,
                gamma=schedule_config["gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    return scheduler


class AverageMeter(object):
    """Computes and stores the average and current value.
    Used to compute dataset stats from mini-batches
    """
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.999, device=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


################################################################################
def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])



def losses_vector(cfg, fpn_masks,
                 out_cls_logits, out_offsets,
                 gt_cls_labels, gt_offsets):
    #init
    loss_normalizer_momentum = 0.9
    loss_normalizer = cfg['train_cfg']['init_loss_norm']
    train_label_smoothing = cfg['train_cfg']['label_smoothing']
    num_classes = cfg['dataset']['num_classes']
    train_loss_weight = cfg['train_cfg']['loss_weight']
    # fpn_masks, out_*: F (List) [B, T_i, C]
    # gt_* : B (list) [F T, C]
    # fpn_masks -> (B, FT)
    valid_mask = torch.cat(fpn_masks, dim=1)

    # 1. classification loss
    # stack the list -> (B, FT) -> (# Valid, )
    gt_cls = torch.stack(gt_cls_labels)
    pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)
    # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))

    pred_offsets = out_offsets
    gt_offsets = gt_offsets

    # update the loss normalizer
    num_pos = pos_mask.sum().item()
    loss_normalizer = loss_normalizer_momentum * loss_normalizer + (
        1 - loss_normalizer_momentum
    ) * max(num_pos, 1)

    # gt_cls is already one hot encoded now, simply masking out
    gt_target = gt_cls[valid_mask]

    # optinal label smoothing
    gt_target *= 1 - train_label_smoothing
    gt_target += train_label_smoothing / (num_classes + 1)

    # focal loss
    cls_loss = sigmoid_focal_loss(
        torch.cat(out_cls_logits, dim=1)[valid_mask],
        gt_target,
        reduction='sum'
    )
    cls_loss /= loss_normalizer

    # 2. regression using IoU/GIoU loss (defined on positive samples)
    if num_pos == 0:
        reg_loss = 0 * pred_offsets.sum()
    else:
        # giou loss defined on positive samples
        reg_loss = ctr_diou_loss_1d(
            pred_offsets,
            gt_offsets,
            reduction='none'
        )
        reg_loss /= loss_normalizer

    if train_loss_weight > 0:
        loss_weight = train_loss_weight
    else:
        loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

    # return a dict of losses
    return {'cls_loss': cls_loss,
            'reg_loss': reg_loss}

def losses_train(cfg, fpn_masks,
                 out_cls_logits, out_offsets,
                 gt_cls_labels, gt_offsets,
                 confidence_scores):
    #init
    loss_normalizer_momentum = 0.9
    loss_normalizer = cfg['train_cfg']['init_loss_norm']
    train_label_smoothing = cfg['train_cfg']['label_smoothing']
    num_classes = cfg['dataset']['num_classes']
    train_loss_weight = cfg['train_cfg']['loss_weight']
    # fpn_masks, out_*: F (List) [B, T_i, C]
    # gt_* : B (list) [F T, C]
    # fpn_masks -> (B, FT)
    valid_mask = torch.cat(fpn_masks, dim=1)

    # 1. classification loss
    # stack the list -> (B, FT) -> (# Valid, )
    gt_cls = torch.stack(gt_cls_labels)
    pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)
    # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))

    pred_offsets = out_offsets
    gt_offsets = gt_offsets

    # update the loss normalizer
    num_pos = pos_mask.sum().item()
    loss_normalizer = loss_normalizer_momentum * loss_normalizer + (
        1 - loss_normalizer_momentum
    ) * max(num_pos, 1)

    # gt_cls is already one hot encoded now, simply masking out
    gt_target = gt_cls[valid_mask]

    # optinal label smoothing
    gt_target *= 1 - train_label_smoothing
    gt_target += train_label_smoothing / (num_classes + 1)

    # focal loss
    cls_loss = sigmoid_focal_loss(
        torch.cat(out_cls_logits, dim=1)[valid_mask],
        gt_target,
        reduction='sum'
    )
    cls_loss /= loss_normalizer

    # 2. regression using IoU/GIoU loss (defined on positive samples)
    if num_pos == 0:
        reg_loss = 0 * pred_offsets.sum()
    else:
        # def replace_nan_with_zero(tensor):
        #     # 使用 torch.isnan 检查 nan，然后用 0 替换这些位置
        #     tensor[torch.isnan(tensor)] = 0
        #     return tensor
        # if torch.isnan(gt_offsets).any():
        #     gt_offsets[torch.isnan(gt_offsets)] = 1.000


        # giou loss defined on positive samples
        # print("gt_offsets", gt_offsets)
        reg_loss = ctr_diou_loss_1d(
            pred_offsets,
            gt_offsets,
            reduction='sum',
        )
        '''reg_loss = reg_loss * confidence_scores
        reg_loss = reg_loss.sum()'''
        reg_loss /= loss_normalizer

    if train_loss_weight > 0:
        loss_weight = train_loss_weight
    else:
        loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

    # return a dict of losses
    final_loss = cls_loss + reg_loss * loss_weight
    return {'cls_loss'   : cls_loss,
            'reg_loss'   : reg_loss,
            'final_loss' : final_loss}

################################################################################
def group_adjacent_positives(mask_tensor):
    batch_size, num_samples = mask_tensor.shape
    groups = []
    for i in range(batch_size):
        positives = (mask_tensor[i] == True).nonzero()  # 获取正样本的位置索引
        num_positives = positives.size(0)

        if num_positives > 0:
            group = [torch.tensor([i, positives[0]])]  # 初始化当前组的起始位置
            for j in range(1, num_positives):
                if positives[j].item() == positives[j-1].item() + 1:
                    group.append(torch.tensor([i, positives[j]]))  # 相邻正样本，将其加入当前组
                else:
                    groups.append(group)  # 不相邻正样本，将当前组加入 groups
                    group = [torch.tensor([i, positives[j]])]  # 开启新组

            groups.append(group)  # 将最后一个组加入 groups

    return tuple(groups)

def split_offsets(offsets, num_split):
    split_lists = []
    start_index = 0

    for split_size in num_split:
        end_index = start_index + split_size
        split_list = offsets[start_index:end_index]
        split_lists.append(split_list)
        start_index = end_index

    return split_lists

def get_confidence_score(tensor):
    n = tensor.size(0)
    if n == 1:
        return torch.tensor([1], dtype=float)
    else:
        distances = torch.zeros(n, n)  # 创建一个大小为 (N, N) 的零张量来存储距离

        # 计算每对元素之间的欧几里得距离
        for i in range(n):
            for j in range(n):
                distances[i, j] = torch.dist(tensor[i], tensor[j])

        diagonal_mask = torch.eye(n, dtype=torch.bool)
        distances = distances[~diagonal_mask]
        # 将距离转换为置信度得分
        max_distance = torch.max(distances)  # 获取最大距离
        min_distance = torch.min(distances)  # 获取最距离
        range_distance = max_distance - min_distance  # 距离范围

        # 计算置信度得分
        confidence_scores = 0.05 / range_distance
        confidence_scores = confidence_scores.expand(n)
        return torch.clamp(confidence_scores, 0, 1)

def mark_offsets_boundary(offsets, pos_mask):
    N = 1
    # 获取张量的形状和时间步数
    batch_size = offsets.shape[0]
    T = offsets.shape[1]
    # 创建用于存储边界位置的张量，默认为 False
    action_mask = torch.zeros(batch_size, T, dtype=torch.bool)

    # 遍历每个样本和时间步
    for i in range(batch_size):
        for j in range(T):
            # 如果 pos_mask 为 True，即该位置的 offset 有意义
            if pos_mask[i, j]:
                # 计算边界位置的索引
                start = max(0, int(j - offsets[i, j, 0]))
                end = min(T, int(j + offsets[i, j, 1]) + 1)

                # 将动作位置设置为 True
                action_mask[i, start:end] = True

    boundary_mask_start = torch.zeros(batch_size, T, dtype=torch.bool)
    boundary_mask_end = torch.zeros(batch_size, T, dtype=torch.bool)
    in_action = False
    for i in range(batch_size):
        for j in range(T):
            if action_mask[i, j] == True and in_action == False:
                boundary_mask_start[i, j] = True
                in_action = True
            if action_mask[i,j] == False and in_action == True:
                boundary_mask_end[i, j-1] = True
                in_action = False

    boundary_mask_in = torch.zeros(batch_size, T, dtype=torch.bool)
    boundary_mask_out = torch.zeros(batch_size, T, dtype=torch.bool)
    for i in range(batch_size):
        for j in range(T):
            if boundary_mask_start[i, j] == True:
                boundary_mask_in[i, j:j+N] = True
                boundary_mask_out[i, j-N:j] = True
            if boundary_mask_end[i, j] == True:
                boundary_mask_in[i, j-N:j] = True
                boundary_mask_out[i, j:j+N] = True

    return boundary_mask_in, boundary_mask_out

def compute_energy(out_cls_logits, gt_offsets_s, pos_mask):
    boundary_mask_in, boundary_mask_out = mark_offsets_boundary(gt_offsets_s, pos_mask)
    out_cls_logits = torch.cat(out_cls_logits, dim=1)
    E_in = (-torch.logsumexp(out_cls_logits[boundary_mask_in], dim=1)).mean()
    E_out = (-torch.logsumexp(out_cls_logits[boundary_mask_out], dim=1)).mean()
    return E_in, E_out

def compute_energy_loss(E_in_s, E_out_s, E_in_pseudo, E_out_pseudo):
    m_in_inter = 0
    m_out_inter = 0.01
    m_in_intra = 0
    m_out_intra = 2
    inter_energy_loss = 0.0001 * (torch.pow(F.relu((E_in_s - E_out_s) - m_in_inter), 2).mean() + torch.pow(F.relu(m_out_inter - (E_in_pseudo - E_out_pseudo)), 2).mean())
    intra_energy_loss = 0.0001 * (torch.pow(F.relu(E_out_pseudo - m_in_intra), 2).mean() + torch.pow(F.relu(m_out_intra - E_in_pseudo), 2).mean())
    energy_loss = intra_energy_loss + intra_energy_loss
    return energy_loss

def train_one_epoch(
    train_loader,
    model,
    optimizer,
    scheduler,
    curr_epoch,
    model_ema=None,
    clip_grad_l2norm=-1,
    tb_writer=None,
    print_freq=20,
    train_loader_meta=None,
    model_meta=None,
    cfg=None,
    optimizer_meta=None,
    args=None
):
    """Training the model for one epoch"""
    # set up meters
    batch_time = AverageMeter()
    losses_tracker = {}
    # number of iterations per epoch
    num_iters = len(train_loader)
    # switch to train mode
    model.train()

    print("\n[Train]: Epoch {:d} started".format(curr_epoch))
    start = time.time()
    meta_dataloader_iter = iter(train_loader_meta)

    for iter_idx, video_list in enumerate(train_loader, 0):
        #Meta learning
        pseudo_net = make_meta_arch(cfg['model_name'], **cfg['model'])
        pseudo_net = nn.DataParallel(pseudo_net, device_ids=cfg['devices'])
        pseudo_net.load_state_dict(model.state_dict())
        pseudo_net.train()
        # given current meta net, get corrected label
        pred_offsets_s, fpn_feats, fpn_masks, out_cls_logits, gt_cls_labels, gt_offsets_s = pseudo_net(video_list, return_h=True)
        fpn_feats_detached = [item.detach() for item in fpn_feats]
        #########only use positive action
        valid_mask = torch.cat(fpn_masks, dim=1)
        gt_cls = torch.stack(gt_cls_labels)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)   #(2,4536)

        #E_in_s, E_out_s = compute_energy(out_cls_logits, gt_offsets_s, pos_mask)

        positives_group = group_adjacent_positives(pos_mask)
        num_split = []
        for i in positives_group:
            num_split.append(len(i))

        pred_offsets_s_full = pred_offsets_s
        pred_offsets_s = pred_offsets_s[pos_mask]
        gt_offsets_s = gt_offsets_s[pos_mask]      #(155,2)
        # Concatenate the features of different scales
        masked_hx_cat = torch.cat(fpn_feats_detached, dim=-1)
        masked_hx_cat = masked_hx_cat.permute(0, 2, 1)  # (2,4536,512)
        masked_hx_cat = masked_hx_cat[pos_mask]    #(155,512)
        #########
        pseudo_loss_vector = losses_vector(cfg, fpn_masks,
                                   out_cls_logits, pred_offsets_s,
                                   gt_cls_labels, gt_offsets_s)

        pseudo_reg_loss_vector = pseudo_loss_vector['reg_loss']
        pseudo_reg_loss_vector_reshape = torch.reshape(pseudo_reg_loss_vector, (-1, 1))
        pseudo_target_s, offsets = model_meta(pseudo_reg_loss_vector_reshape, masked_hx_cat, gt_offsets_s)   #(155,2)


        # for name, param in model_meta.named_parameters():
        #     if param.grad is not None:
        #         print(f"Before Gradient of {name} is not None and has norm: {param.grad.norm()}")
        #     else:
        #         print(f"Before Gradient of {name} is None")
        #E_in_pseudo, E_out_pseudo = compute_energy(out_cls_logits, pred_offsets_s_full, pos_mask)

        #pseudo_energy_loss = compute_energy_loss(E_in_s, E_out_s, E_in_pseudo, E_out_pseudo)

        split_list = split_offsets(offsets, num_split)
        scores_list = []
        for i in split_list:
            scores_list.append(get_confidence_score(i))
        confidence_scores = torch.cat(scores_list)
        confidence_scores = confidence_scores.cuda()
        confidence_scores = confidence_scores.detach()

        pseudo_target_s[pseudo_target_s < 0] = 0

        pseudo_loss = losses_train(cfg, fpn_masks,
                              out_cls_logits, pred_offsets_s,
                              gt_cls_labels, pseudo_target_s,
                                   confidence_scores)

        pseudo_grads = torch.autograd.grad(pseudo_loss['final_loss'], pseudo_net.parameters(), create_graph=True)

        pseudo_optimizer = make_optimizer_pseudo(pseudo_net, cfg['opt'])

        pseudo_optimizer.load_state_dict(optimizer.state_dict())
        pseudo_optimizer.meta_step(pseudo_grads)

        pseudo_optimizer.zero_grad()
        # pseudo_optimizer.step()


        del pseudo_grads

        try:
            video_list_meta = next(meta_dataloader_iter)
        except StopIteration:
            meta_dataloader_iter = iter(train_loader_meta)
            video_list_meta = next(meta_dataloader_iter)

        meta_loss = pseudo_net(video_list_meta)
        optimizer_meta.zero_grad()
        meta_loss['final_loss'].backward()
        # print("Requires grad:", meta_loss['final_loss'].requires_grad)

        prev_weights = {}
        for name, param in model_meta.named_parameters():
            prev_weights[name] = param.clone()

        # 输出backbone参数信息
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Gradient of {name} is not None and has norm: {param.grad.norm()}")
        #     else:
        #         print(f"Gradient of {name} is None")
        tracked_param_name = 'module.backbone.branch.3.attn.proj.weight'
        tracked_param_before = None
        tracked_param_after = None

        for name, param in pseudo_net.named_parameters():
            if name == tracked_param_name:
                tracked_param_before = param.clone()

#########################################
        # pseudo_grads = torch.autograd.grad(pseudo_loss['final_loss'], pseudo_net.parameters(), create_graph=True,
        #                                    allow_unused=True)
        # for i, grad in enumerate(pseudo_grads):
        #     if grad is not None:
        #         print(f'Gradient for parameter {i}: {grad}')

        optimizer_meta.step()

        has_updated = False
        for name, param in pseudo_net.named_parameters():
            if name == tracked_param_name:
                tracked_param_after = param

        # print("Before update:", tracked_param_before[0])
        # print("After update:", tracked_param_after[0])
        def print_gradients(model):
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad = param.grad.data
                    # 打印梯度的统计信息
                    print(f"Gradient of {name} is not None.")
                    print(f" - Max: {grad.max()}")
                    print(f" - Min: {grad.min()}")
                    print(f" - Mean: {grad.mean()}")
                    print(f" - Std: {grad.std()}")
                    # 如果需要查看具体数值，可以选择打印完整梯度或其部分
                    print(f" - Values: {grad.view(-1)[:10]}")  # 仅展示前10个梯度值
                else:
                    print(f"Gradient of {name} is None.")

        # 打印模型的梯度参数
        # print_gradients(model_meta)
        # for name, param in model_meta.named_parameters():
        #     if param.grad is not None:
        #         print(f"Gradient of {name} is not None and has norm: {param.grad.norm()}")
        #     else:
        #         print(f"Gradient of {name} is None")

        for name, param in model_meta.named_parameters():
            if not torch.equal(param, prev_weights[name]):
                has_updated = True
                break
        if has_updated:
            print("Model_meta parameters have been updated!")
        else:
            print("Model_meta parameters have not been updated.")

        # zero out optim
        optimizer.zero_grad(set_to_none=True)
        # forward / backward the model
        pred_offsets_s, fpn_feats, fpn_masks, out_cls_logits, gt_cls_labels, gt_offsets_s = model(video_list, return_h=True)
        fpn_feats_detached = [item.detach() for item in fpn_feats]
        #########only use positive action
        valid_mask = torch.cat(fpn_masks, dim=1)
        gt_cls = torch.stack(gt_cls_labels)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)  # (2,4536)

        #E_in_s, E_out_s = compute_energy(out_cls_logits, gt_offsets_s, pos_mask)

        positives_group = group_adjacent_positives(pos_mask)
        num_split = []
        for i in positives_group:
            num_split.append(len(i))

        pred_offsets_s_full = pred_offsets_s
        pred_offsets_s = pred_offsets_s[pos_mask]
        gt_offsets_s = gt_offsets_s[pos_mask]  # (62,2)
        # Concatenate the features of different scales
        masked_hx_cat = torch.cat(fpn_feats_detached, dim=-1)
        masked_hx_cat = masked_hx_cat.permute(0, 2, 1)  # (2,4536,512)
        masked_hx_cat = masked_hx_cat[pos_mask]  # (62,512)
        #########
        pseudo_loss_vector = losses_vector(cfg, fpn_masks,
                                           out_cls_logits, pred_offsets_s,
                                           gt_cls_labels, gt_offsets_s)

        pseudo_reg_loss_vector = pseudo_loss_vector['reg_loss']
        pseudo_reg_loss_vector_reshape = torch.reshape(pseudo_reg_loss_vector, (-1, 1))
        with torch.no_grad():
            pseudo_target_s, _ = model_meta(pseudo_reg_loss_vector_reshape, masked_hx_cat, gt_offsets_s)   #(62,2)

        #E_in_pseudo, E_out_pseudo = compute_energy(out_cls_logits, pred_offsets_s_full, pos_mask)

        #pseudo_energy_loss = compute_energy_loss(E_in_s, E_out_s, E_in_pseudo, E_out_pseudo)

        split_list = split_offsets(offsets, num_split)
        scores_list = []
        for i in split_list:
            scores_list.append(get_confidence_score(i))
        confidence_scores = torch.cat(scores_list)
        confidence_scores = confidence_scores.cuda()
        confidence_scores = confidence_scores.detach()
        # tensor
        # print(pseudo_target_s - gt_offsets_s)
        pseudo_target_s[pseudo_target_s < 0] = 0

        losses = losses_train(cfg, fpn_masks,
                              out_cls_logits, pred_offsets_s,
                              gt_cls_labels, pseudo_target_s,
                              confidence_scores)

        losses['final_loss'].backward()
        # gradient cliping (to stabilize training if necessary)
        if clip_grad_l2norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                clip_grad_l2norm
            )
        # step optimizer / scheduler
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Gradient of {name} is not None and has norm: {param.grad.norm()}")
        #     else:
        #         print(f"Gradient of {name} is None")

        optimizer.step()
        scheduler.step()

        if model_ema is not None:
            model_ema.update(model)

        # printing (only check the stats when necessary to avoid extra cost)
        if (iter_idx != 0) and (iter_idx % print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # track all losses
            for key, value in losses.items():
                # init meter if necessary
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                # update
                losses_tracker[key].update(value.item())

            # log to tensor board
            lr = scheduler.get_last_lr()[0]
            global_step = curr_epoch * num_iters + iter_idx
            if tb_writer is not None:
                # learning rate (after stepping)
                tb_writer.add_scalar(
                    'train/learning_rate',
                    lr,
                    global_step
                )
                # all losses
                tag_dict = {}
                for key, value in losses_tracker.items():
                    if key != "final_loss":
                        tag_dict[key] = value.val
                tb_writer.add_scalars(
                    'train/all_losses',
                    tag_dict,
                    global_step
                )
                # final loss
                tb_writer.add_scalar(
                    'train/final_loss',
                    losses_tracker['final_loss'].val,
                    global_step
                )

            # print to terminal
            block1 = 'Epoch: [{:03d}][{:05d}/{:05d}]'.format(
                curr_epoch, iter_idx, num_iters
            )
            block2 = 'Time {:.2f} ({:.2f})'.format(
                batch_time.val, batch_time.avg
            )
            block3 = 'Loss {:.2f} ({:.2f})\n'.format(
                losses_tracker['final_loss'].val,
                losses_tracker['final_loss'].avg
            )
            block4 = ''
            for key, value in losses_tracker.items():
                if key != "final_loss":
                    block4 += '\t{:s} {:.2f} ({:.2f})'.format(
                        key, value.val, value.avg
                    )

            print('\t'.join([block1, block2, block3, block4]))

        # finish up and print
    lr = scheduler.get_last_lr()[0]
    print("[Train]: Epoch {:d} finished with lr={:.8f}\n".format(curr_epoch, lr))
    return


def valid_one_epoch(
    val_loader,
    model,
    curr_epoch,
    ext_score_file = None,
    evaluator = None,
    output_file = None,
    tb_writer = None,
    print_freq = 20
):
    """Test the model on the validation set"""
    # either evaluate the results or save the results
    assert (evaluator is not None) or (output_file is not None)

    # set up meters
    batch_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    # dict for results (for our evaluation code)
    results = {
        'video-id': [],
        't-start' : [],
        't-end': [],
        'label': [],
        'score': []
    }

    # loop over validation set
    start = time.time()
    for iter_idx, video_list in enumerate(val_loader, 0):
        # forward the model (wo. grad)
        with torch.no_grad():
            output = model(video_list)
            # unpack the results into ANet format
            num_vids = len(output)
            for vid_idx in range(num_vids):
                if output[vid_idx]['segments'].shape[0] > 0:
                    results['video-id'].extend(
                        [output[vid_idx]['video_id']] *
                        output[vid_idx]['segments'].shape[0]
                    )
                    results['t-start'].append(output[vid_idx]['segments'][:, 0])
                    results['t-end'].append(output[vid_idx]['segments'][:, 1])
                    results['label'].append(output[vid_idx]['labels'])
                    results['score'].append(output[vid_idx]['scores'])

        # printing
        if (iter_idx != 0) and iter_idx % (print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # print timing
            print('Test: [{0:05d}/{1:05d}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                  iter_idx, len(val_loader), batch_time=batch_time))

    # gather all stats and evaluate
    results['t-start'] = torch.cat(results['t-start']).numpy()
    results['t-end'] = torch.cat(results['t-end']).numpy()
    results['label'] = torch.cat(results['label']).numpy()
    results['score'] = torch.cat(results['score']).numpy()

    if evaluator is not None:
        if ext_score_file is not None and isinstance(ext_score_file, str):
            results = postprocess_results(results, ext_score_file)
        # call the evaluator
        _, mAP, _ = evaluator.evaluate(results, verbose=True)
    else:
        # dump to a pickle file that can be directly used for evaluation
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
        mAP = 0.0

    # log mAP to tb_writer
    if tb_writer is not None:
        tb_writer.add_scalar('validation/mAP', mAP, curr_epoch)

    return mAP

################################################################################
