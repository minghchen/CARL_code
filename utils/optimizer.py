#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# copy from slowfast.models.optimizer
"""Optimizer."""

import torch
import numpy as np
import math

def construct_optimizer(model, cfg):
    """
    Construct a stochastic gradient descent or ADAM optimizer with momentum.
    Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."

    Args:
        model (model): model to perform stochastic gradient descent
        optimization or ADAM optimization.
        cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
        learning rate,  momentum, weight_decay, dampening, and etc.
    """
    # Batchnorm parameters.
    bn_params = []
    # Non-batchnorm parameters.
    non_bn_parameters = []
    for n, m in model.named_modules():
        is_bn = isinstance(m, torch.nn.modules.batchnorm._NormBase)
        for p in m.parameters(recurse=False):
            if 'backbone' in n and cfg.MODEL.TRAIN_BASE != 'train_all':
                if cfg.MODEL.TRAIN_BASE == 'frozen':
                    continue
                elif cfg.MODEL.TRAIN_BASE == 'only_bn':
                    if is_bn:
                        bn_params.append(p)
            else:
                if is_bn:
                    bn_params.append(p)
                else:
                    non_bn_parameters.append(p)

    # Apply different weight decay to Batchnorm and non-batchnorm parameters.
    # In Caffe2 classification codebase the weight decay for batchnorm is 0.0.
    # Having a different weight decay on batchnorm might cause a performance
    # drop.
    optim_params = [
        {"params": bn_params, "weight_decay": cfg.OPTIMIZER.WEIGHT_DECAY},
        {"params": non_bn_parameters, "weight_decay": cfg.OPTIMIZER.WEIGHT_DECAY},
    ]

    if cfg.OPTIMIZER.TYPE == "MomentumOptimizer":
        return torch.optim.SGD(
            optim_params,
            lr=cfg.OPTIMIZER.LR.INITIAL_LR,
            momentum=0.9,
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
        )
    elif cfg.OPTIMIZER.TYPE == "AdamOptimizer":
        return torch.optim.Adam(
            optim_params,
            lr=cfg.OPTIMIZER.LR.INITIAL_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
        )
    elif cfg.OPTIMIZER.TYPE == "AdamWOptimizer":
        return torch.optim.AdamW(
            optim_params,
            lr=cfg.OPTIMIZER.LR.INITIAL_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.OPTIMIZER.TYPE)
        )

def construct_scheduler(optimizer, cfg):
    if cfg.OPTIMIZER.LR.DECAY_TYPE == "fixed":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    elif cfg.OPTIMIZER.LR.DECAY_TYPE == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.TRAIN.MAX_EPOCHS + 1, 
            eta_min=0,
            last_epoch=-1
        )
    elif cfg.OPTIMIZER.LR.DECAY_TYPE == "cosinewarmup":
        base_lr = cfg.OPTIMIZER.LR.INITIAL_LR
        warmup_lr_schedule = np.linspace(cfg.OPTIMIZER.LR.WARMUP_LR / base_lr, 1, cfg.OPTIMIZER.LR.NUM_WARMUP_STEPS)
        iters = np.arange(cfg.TRAIN.MAX_EPOCHS + 1 - cfg.OPTIMIZER.LR.NUM_WARMUP_STEPS)
        cosine_lr_schedule = np.array([cfg.OPTIMIZER.LR.FINAL_LR / base_lr + 0.5 * (1 - cfg.OPTIMIZER.LR.FINAL_LR / base_lr) * \
                                (1 + math.cos(math.pi * t / len(iters))) for t in iters])
        lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_schedule[epoch])
    else:
        raise NotImplementedError(
            "Does not support {} scheduler".format(cfg.OPTIMIZER.LR.DECAY_TYPE)
        )

def get_lr(optimizer):
    return [param_group["lr"] for param_group in optimizer.param_groups]

def set_lr(optimizer, new_lr):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
