import torch
import torch.nn as nn
import torch.nn.functional as F
import models
from typing import Union


def init_param(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.sigma_weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(m.phi_weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
    return m


def make_batchnorm(m, momentum: float, track_running_stats: bool):
    if isinstance(m, nn.BatchNorm2d):
        m.momentum = momentum
        m.track_running_stats = track_running_stats
        if track_running_stats:
            m.register_buffer("running_mean", torch.zeros(m.num_features, device=m.weight.device))
            m.register_buffer("running_var", torch.ones(m.num_features, device=m.weight.device))
            m.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long, device=m.weight.device))
        else:
            m.running_mean = None
            m.running_var = None
            m.num_batches_tracked = None
    return m


def loss_fn(output, target, reduction="mean"):
    if target.dtype == torch.int64:
        loss = F.cross_entropy(output, target, reduction=reduction)
    else:
        loss = F.mse_loss(output, target, reduction=reduction)
    return loss
