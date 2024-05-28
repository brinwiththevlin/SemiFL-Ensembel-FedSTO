"""inspired by: https://github.com/diaoenmao/SemiFL-Semi-Supervised-Federated-Learning-for-Unlabeled-Clients-with-Alternate-Training"""

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


# TODO: change this based on final loss calculation
def loss_fn(output: torch.Tensor, target: torch.Tensor, reduction="mean") -> torch.Tensor:
    """loss function

    Args:
        output (torch.Tensor): inference output
        target (torch.Tensor): target
        reduction (str, optional): reduction method. Defaults to "mean".

    Returns:
        torch.Tensor: loss
    """
    if target.dtype == torch.int64:
        loss = F.cross_entropy(output, target, reduction=reduction)
    else:
        loss = F.mse_loss(output, target, reduction=reduction)
    return loss
