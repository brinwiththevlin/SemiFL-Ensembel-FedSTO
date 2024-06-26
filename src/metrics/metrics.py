"""inspired by: https://github.com/diaoenmao/SemiFL-Semi-Supervised-Federated-Learning-for-Unlabeled-Clients-with-Alternate-Training"""

import torch

import datasets
from config import cfg
from utils import recur


def Accuracy(output, target, topk=1):
    with torch.no_grad():
        if target.dtype != torch.int64:
            target = (target.topk(1, 1, True, True)[1]).view(-1)
        batch_size = target.size(0)
        pred_k = output.topk(topk, 1, True, True)[1]
        correct_k = pred_k.eq(target.view(-1, 1).expand_as(pred_k)).float().sum()
        acc = (correct_k * (100.0 / batch_size)).item()
    return acc


def MAccuracy(output, target, mask, topk=1):
    if torch.any(mask):
        output = output[mask]
        target = target[mask]
        acc = Accuracy(output, target, topk)
    else:
        acc = 0
    return acc


def LabelRatio(mask):
    with torch.no_grad():
        lr = mask.float().mean().item()
    return lr


class Metric(object):
    def __init__(self, metric_name):
        self.metric_name = metric_name
        self.pivot, self.pivot_name, self.pivot_direction = self.make_pivot()
        self.metric = {
            "Loss": (lambda input, output: output["loss"].item()),
            "Accuracy": (lambda input, output: recur(Accuracy, output["target"], input["target"])),
            "PAccuracy": (lambda input, output: recur(Accuracy, output["target"], input["target"])),
            "MAccuracy": (lambda input, output: recur(MAccuracy, output["target"], input["target"], output["mask"])),
            "LabelRatio": (lambda input, output: recur(LabelRatio, output["mask"])),
        }

    def make_pivot(self):
        assert cfg["data_name"] in datasets.__all__, "Not valid data name"
        pivot = -float("inf")
        pivot_direction = "up"
        pivot_name = "Accuracy"
        return pivot, pivot_name, pivot_direction

    def evaluate(self, metric_names, input, output):
        evaluation = {}
        for metric_name in metric_names:
            evaluation[metric_name] = self.metric[metric_name](input, output)
        return evaluation

    def compare(self, val):
        if self.pivot_direction == "down":
            compared = self.pivot > val
        elif self.pivot_direction == "up":
            compared = self.pivot < val
        else:
            raise ValueError("Not valid pivot direction")
        return compared

    def update(self, val):
        self.pivot = val
        return
