"""inspired by: https://github.com/diaoenmao/SemiFL-Semi-Supervised-Federated-Learning-for-Unlabeled-Clients-with-Alternate-Training"""

import copy
from typing import List, OrderedDict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import models
from config import cfg
from data import make_data_loader
from logger import Logger
from metrics import Metric
from utils import collate, make_optimizer, to_device


class Client:
    """Client class"""

    def __init__(self, client_id: int, model: nn.Module, data_split: dict[str, Dataset]):
        """Client class

        Args:
            client_id (int): client id
            model (nn.Module): model to train
            data_split (dict[str, Dataset]): data split
        """
        self.client_id = client_id
        self.data_split = data_split  # {"train": dataset, "test": dataset}
        self.model_state_dict = save_model_state_dict(model.state_dict())
        optimizer = make_optimizer(model.parameters(), "local")
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.active = False
        self.beta = torch.distributions.beta.Beta(torch.tensor([cfg["alpha"]]), torch.tensor([cfg["alpha"]]))
        self.verbose = cfg["verbose"]

    def make_hard_pseudo_label(self, soft_pseudo_label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """given a soft label, return a hard label

        Args:
            soft_pseudo_label (torch.Tensor): soft pseudo label

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: hard pseudo label, mask
        """
        max_p, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
        mask = max_p.ge(cfg["threshold"])
        return hard_pseudo_label, mask

    def train(
        self,
        dataset: Dataset,
        lr: float,
        metric: Metric,
        logger: Logger,
        selective: bool = False,
        orthogonal: bool = False,
    ):
        assert selective + orthogonal == 1, "selective and orthogonal are mutually exclusive"
        model: nn.Module = getattr(models, cfg["model_name"])()
        model = model.to(cfg["device"])
        model.load_state_dict(self.model_state_dict, strict=False)
        self.optimizer_state_dict["param_groups"][0]["lr"] = lr
        optimizer = make_optimizer(model.parameters(), "local")
        optimizer.load_state_dict(self.optimizer_state_dict)

        self.efficient_teacher(model, dataset, optimizer, metric, logger, selective, orthogonal)

        # if cfg["client"]["num_epochs"] == 1:
        #     num_batches = int(np.ceil(len(fix_data_loader) * float(cfg["local_epoch"][0])))
        # else:
        #     num_batches = None
        # for epoch in range(1, cfg["client"]["num_epochs"] + 1):
        #     for i, (fix_input, mix_input) in enumerate(zip(fix_data_loader, mix_data_loader)):
        #         input = {
        #             "data": fix_input["data"],
        #             "target": fix_input["target"],
        #             "aug": fix_input["aug"],
        #             "mix_data": mix_input["data"],
        #             "mix_target": mix_input["target"],
        #         }
        #         input = collate(input)
        #         input_size = input["data"].size(0)
        #         input["lam"] = self.beta.sample()[0]
        #         input["mix_data"] = (input["lam"] * input["data"] + (1 - input["lam"]) * input["mix_data"]).detach()
        #         input["mix_target"] = torch.stack([input["target"], input["mix_target"]], dim=-1)
        #         input["loss_mode"] = cfg["loss_mode"]
        #         input = to_device(input, cfg["device"])
        #         optimizer.zero_grad()
        #         output = model(input)
        #         output["loss"].backward()
        #         torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        #         optimizer.step()
        #         evaluation = metric.evaluate(["Loss", "Accuracy"], input, output)
        #         logger.append(evaluation, "train", n=input_size)
        #         if num_batches is not None and i == num_batches - 1:
        #             break
        # self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        # self.model_state_dict = save_model_state_dict(model.state_dict())
        # return

    def efficient_teacher(self, model, dataset, optimizer, metric, logger, selective, orthogonal):
        model.train(True)


class Server:
    def __init__(self, model: nn.Module):
        self.model_state_dict = save_model_state_dict(model.state_dict())
        optimizer = make_optimizer(model.parameters(), "local")
        global_optimizer = make_optimizer(model.parameters(), "global")
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())

    def distribute(self, client: List[Client]):
        """distribute the model state dict to the clients

        Args:
            client (List[Client]): list of clients
        """
        for m in range(len(client)):
            if client[m].active:
                client[m].model_state_dict = copy.deepcopy(self.model_state_dict)
        return

    def update(self, client):
        with torch.no_grad():
            valid_client = [client[i] for i in range(len(client)) if client[i].active]
            if len(valid_client) > 0:
                model: nn.Module = getattr(models, cfg["model_name"])()
                model = model.to(cfg["device"])
                model.load_state_dict(self.model_state_dict)
                global_optimizer = make_optimizer(model.parameters(), "global")
                global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                global_optimizer.zero_grad()
                weight = torch.ones(len(valid_client))
                weight = weight / weight.sum()
                for k, v in model.named_parameters():
                    parameter_type = k.split(".")[-1]
                    if "weight" in parameter_type or "bias" in parameter_type:
                        tmp_v = v.data.new_zeros(v.size())
                        for m in range(len(valid_client)):
                            tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                        v.grad = (v.data - tmp_v).detach()
                global_optimizer.step()
                self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                self.model_state_dict = save_model_state_dict(model.state_dict())
        for i in range(len(client)):
            client[i].active = False
        return

    def train(self, dataset, lr, metric, logger, epoch, ortohgonal):
        data_loader = make_data_loader({"train": dataset}, "server")["train"]
        model: nn.Module = getattr(models, cfg["model_name"])()
        model = model.to(cfg["device"])
        model.load_state_dict(self.model_state_dict)
        self.optimizer_state_dict["param_groups"][0]["lr"] = lr
        optimizer = make_optimizer(model.parameters(), "local")
        optimizer.load_state_dict(self.optimizer_state_dict)
        model.train(True)
        if cfg["server"]["num_epochs"] == 1:
            num_batches = int(np.ceil(len(data_loader) * float(cfg["local_epoch"][1])))
        else:
            num_batches = None
        for epoch in range(1, cfg["server"]["num_epochs"] + 1):
            for i, input in enumerate(data_loader):
                input = collate(input)
                input_size = input["data"].size(0)
                input = to_device(input, cfg["device"])
                optimizer.zero_grad()
                output = model(input)
                output["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                evaluation = metric.evaluate(["Loss", "Accuracy"], input, output)
                logger.append(evaluation, "train", n=input_size)
                if num_batches is not None and i == num_batches - 1:
                    break
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.model_state_dict = save_model_state_dict(model.state_dict())
        return


def save_model_state_dict(model_state_dict: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    """sends the model state dict to the cpu

    Args:
        model_state_dict (OrderedDict[str, torch.Tensor]): saved model state dict

    Returns:
        OrderedDict[str, torch.Tensor]: model state dict on the cpu
    """
    return {k: v.cpu() for k, v in model_state_dict.items()}


def save_optimizer_state_dict(optimizer_state_dict: dict) -> dict:
    """sends the optimizer state dict to the cpu

    Args:
        optimizer_state_dict (dict): saved optimizer state dict

    Returns:
        dict: optimizer state dict on the cpu
    """
    optimizer_state_dict_ = {}
    for k, v in optimizer_state_dict.items():
        if k == "state":
            optimizer_state_dict_[k] = to_device(optimizer_state_dict[k], "cpu")
        else:
            optimizer_state_dict_[k] = copy.deepcopy(optimizer_state_dict[k])
    return optimizer_state_dict_
