import copy
import datetime
import numpy as np
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import models
from itertools import compress
from config import cfg
from data import make_data_loader, FixTransform, MixDataset
from utils import to_device, make_optimizer, collate, to_device
from metrics import Accuracy
from torch.utils.data import Dataset
from typing import OrderedDict, List


class Client:
    def __init__(self, client_id: int, model: nn.Module, data_split: dict[str, Dataset]):
        assert all(match in cfg["loss_mode"] for match in ["fix", "mix"]), "Not valid client loss mode"
        self.client_id = client_id
        self.data_split = data_split  # {"train": dataset, "test": dataset}
        self.model_state_dict = save_model_state_dict(model.state_dict())
        optimizer = make_optimizer(model.parameters(), "local")
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.active = False
        self.beta = torch.distributions.beta.Beta(torch.tensor([cfg["alpha"]]), torch.tensor([cfg["alpha"]]))
        self.verbose = cfg["verbose"]

    def make_hard_pseudo_label(self, soft_pseudo_label: torch.Tensor):
        max_p, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
        mask = max_p.ge(cfg["threshold"])
        return hard_pseudo_label, mask

    def make_dataset(self, dataset, metric, logger):
        assert all(match in cfg["loss_mode"] for match in ["fix", "mix"]), "Not valid client loss mode"
        with torch.no_grad():
            data_loader = make_data_loader({"train": dataset}, "global", shuffle={"train": False})["train"]
            model: nn.Module = eval('models.{}(track=True).to(cfg["device"])'.format(cfg["model_name"]))
            model.load_state_dict(self.model_state_dict)
            model.train(False)
            output = []
            target = []
            for i, input in enumerate(data_loader):
                input = collate(input)
                input = to_device(input, cfg["device"])
                output_ = model(input)
                output_i = output_["target"]
                target_i = input["target"]
                output.append(output_i.cpu())
                target.append(target_i.cpu())
            output_, input_ = {}, {}
            output_["target"] = torch.cat(output, dim=0)
            input_["target"] = torch.cat(target, dim=0)
            output_["target"] = F.softmax(output_["target"], dim=-1)
            new_target, mask = self.make_hard_pseudo_label(output_["target"])
            output_["mask"] = mask
            evaluation = metric.evaluate(["PAccuracy", "MAccuracy", "LabelRatio"], input_, output_)
            logger.append(evaluation, "train", n=len(input_["target"]))
            if torch.any(mask):
                fix_dataset = copy.deepcopy(dataset)
                fix_dataset.target = new_target.tolist()
                mask = mask.tolist()
                fix_dataset.data = list(compress(fix_dataset.data, mask))
                fix_dataset.target = list(compress(fix_dataset.target, mask))
                fix_dataset.other = {"id": list(range(len(fix_dataset.data)))}
                if "mix" in cfg["loss_mode"]:
                    mix_dataset = copy.deepcopy(dataset)
                    mix_dataset.target = new_target.tolist()
                    mix_dataset = MixDataset(len(fix_dataset), mix_dataset)
                else:
                    mix_dataset = None
                return fix_dataset, mix_dataset
            else:
                return None

    def train(self, dataset, lr, metric, logger):
        assert all(match in cfg["loss_mode"] for match in ["fix", "mix"]), "Not valid client loss mode"
        assert not any(
            match in cfg["loss_mode"] for match in ["batch", "frgd", "fmatch", "sup"]
        ), "Not valid client loss mode"

        fix_dataset, mix_dataset = dataset
        fix_data_loader = make_data_loader({"train": fix_dataset}, "client")["train"]
        mix_data_loader = make_data_loader({"train": mix_dataset}, "client")["train"]
        model = eval('models.{}().to(cfg["device"])'.format(cfg["model_name"]))
        model.load_state_dict(self.model_state_dict, strict=False)
        self.optimizer_state_dict["param_groups"][0]["lr"] = lr
        optimizer = make_optimizer(model.parameters(), "local")
        optimizer.load_state_dict(self.optimizer_state_dict)
        model.train(True)
        if cfg["client"]["num_epochs"] == 1:
            num_batches = int(np.ceil(len(fix_data_loader) * float(cfg["local_epoch"][0])))
        else:
            num_batches = None
        for epoch in range(1, cfg["client"]["num_epochs"] + 1):
            for i, (fix_input, mix_input) in enumerate(zip(fix_data_loader, mix_data_loader)):
                input = {
                    "data": fix_input["data"],
                    "target": fix_input["target"],
                    "aug": fix_input["aug"],
                    "mix_data": mix_input["data"],
                    "mix_target": mix_input["target"],
                }
                input = collate(input)
                input_size = input["data"].size(0)
                input["lam"] = self.beta.sample()[0]
                input["mix_data"] = (input["lam"] * input["data"] + (1 - input["lam"]) * input["mix_data"]).detach()
                input["mix_target"] = torch.stack([input["target"], input["mix_target"]], dim=-1)
                input["loss_mode"] = cfg["loss_mode"]
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


class Server:
    def __init__(self, model: nn.Module):
        assert all(match in cfg["loss_mode"] for match in ["fix", "mix"]), "Not valid server loss mode"
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
        if "fmatch" not in cfg["loss_mode"]:
            with torch.no_grad():
                valid_client = [client[i] for i in range(len(client)) if client[i].active]
                if len(valid_client) > 0:
                    model = eval("models.{}()".format(cfg["model_name"]))
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
        elif "fmatch" in cfg["loss_mode"]:
            with torch.no_grad():
                valid_client = [client[i] for i in range(len(client)) if client[i].active]
                if len(valid_client) > 0:
                    model = eval("models.{}()".format(cfg["model_name"]))
                    model.load_state_dict(self.model_state_dict)
                    global_optimizer = make_optimizer(model.make_phi_parameters(), "global")
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

    def update_parallel(self, client):
        if "frgd" not in cfg["loss_mode"]:
            with torch.no_grad():
                valid_client_server = [self] + [client[i] for i in range(len(client)) if client[i].active]
                model = eval("models.{}()".format(cfg["model_name"]))
                model.load_state_dict(self.model_state_dict)
                global_optimizer = make_optimizer(model.parameters(), "global")
                global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                global_optimizer.zero_grad()
                weight = torch.ones(len(valid_client_server))
                weight = weight / (2 * (weight.sum() - 1))
                weight[0] = 1 / 2 if len(valid_client_server) > 1 else 1
                for k, v in model.named_parameters():
                    parameter_type = k.split(".")[-1]
                    if "weight" in parameter_type or "bias" in parameter_type:
                        tmp_v = v.data.new_zeros(v.size())
                        for m in range(len(valid_client_server)):
                            tmp_v += weight[m] * valid_client_server[m].model_state_dict[k]
                        v.grad = (v.data - tmp_v).detach()
                global_optimizer.step()
                self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                self.model_state_dict = save_model_state_dict(model.state_dict())
        elif "frgd" in cfg["loss_mode"]:
            with torch.no_grad():
                valid_client_server = [self] + [client[i] for i in range(len(client)) if client[i].active]
                num_valid_client = len(valid_client_server) - 1
                if len(valid_client_server) > 0:
                    model = eval("models.{}()".format(cfg["model_name"]))
                    model.load_state_dict(self.model_state_dict)
                    global_optimizer = make_optimizer(model.parameters(), "global")
                    global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                    global_optimizer.zero_grad()
                    weight = torch.ones(len(valid_client_server)) / (num_valid_client // 2 + 1)
                    for k, v in model.named_parameters():
                        parameter_type = k.split(".")[-1]
                        if "weight" in parameter_type or "bias" in parameter_type:
                            tmp_v_1 = v.data.new_zeros(v.size())
                            tmp_v_1 += weight[0] * valid_client_server[0].model_state_dict[k]
                            for m in range(1, num_valid_client // 2 + 1):
                                tmp_v_1 += weight[m] * valid_client_server[m].model_state_dict[k]
                            tmp_v_2 = v.data.new_zeros(v.size())
                            tmp_v_2 += weight[0] * valid_client_server[0].model_state_dict[k]
                            for m in range(num_valid_client // 2 + 1, len(valid_client_server)):
                                tmp_v_2 += weight[m] * valid_client_server[m].model_state_dict[k]
                            tmp_v = (tmp_v_1 + tmp_v_2) / 2
                            v.grad = (v.data - tmp_v).detach()
                    global_optimizer.step()
                    self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                    self.model_state_dict = save_model_state_dict(model.state_dict())
        for i in range(len(client)):
            client[i].active = False
        return

    def train(self, dataset, lr, metric, logger):
        data_loader = make_data_loader({"train": dataset}, "server")["train"]
        model: nn.Module = eval('models.{}().to(cfg["device"])'.format(cfg["model_name"]))
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
