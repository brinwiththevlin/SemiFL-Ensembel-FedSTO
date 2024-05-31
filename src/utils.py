import os
import pickle
from typing import Any, Callable, Dict, Generator, List, Union

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset

from config import cfg


def to_device(input: Any, device: torch.device) -> Any:
    """recursive function to move input to device

    Args:
        input (Any): input to move
        device (torch.device): target device

    Returns:
        Any: moved input
    """
    output = recur(lambda x, y: x.to(y), input, device)
    return output


def collate(input: Dict[str, List[Union[torch.Tensor, np.ndarray]]]) -> Dict[str, torch.Tensor]:
    """collate function for DataLoader

    Args:
        input (Dict[str, List[Union[torch.Tensor, np.ndarray]]]): dictionary of list of tensors

    Returns:
        Dict[str, torch.Tensor]: dictionary of stacked tensors
    """
    for k in input:
        input[k] = torch.stack(input[k], 0)
    return input


def make_optimizer(parameters: Generator, tag: str) -> optim.Optimizer:
    """set optimizer

    Args:
        parameters (Generator): model parameter generator
        tag (str): local or global

    Returns:
        optim.Optimizer: model optimizer
    """
    assert tag in ["local", "global"], "Not valid tag"
    assert cfg[tag]["optimizer_name"] == "SGD", "Not valid optimizer"
    optimizer = optim.SGD(
        parameters,
        lr=cfg[tag]["lr"],
        momentum=cfg[tag]["momentum"],
        weight_decay=cfg[tag]["weight_decay"],
        nesterov=cfg[tag]["nesterov"],
    )
    return optimizer


def make_scheduler(optimizer: optim.Optimizer, tag: str) -> optim.lr_scheduler._LRScheduler:
    """set scheduler

    Args:
        optimizer (optim.Optimizer): model optimizer
        tag (str): local or global

    Returns:
        optim.lr_scheduler._LRScheduler: model scheduler
    """
    assert tag in ["local", "global"], "Not valid tag"
    assert cfg[tag]["scheduler_name"] == "CosineAnnealingLR", "Not valid scheduler"
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg[tag]["num_epochs"], eta_min=0)
    return scheduler


def process_dataset(dataset: Dict[str, Dataset]) -> None:
    """add data size and target size to cfg

    Args:
        dataset (Dict[str,Dataset]): dictionary of dataset. keys: "train", "test"
    """
    cfg["data_size"] = {"train": len(dataset["train"]), "test": len(dataset["test"])}
    cfg["target_size"] = dataset["train"].target_size
    return


def recur(
    fn: Callable, input: Union[torch.Tensor, np.ndarray, list, tuple, dict, str, None], *args: Any
) -> Union[torch.Tensor, np.ndarray, list, tuple, dict, str, None]:
    """recusive function to apply fn to input

    Args:
        fn (function): function to apply
        input (Any): input to apply fn

    Returns:
        Any: transformed object, same type as input
    """
    assert callable(fn), "Not valid function"
    assert any(
        [
            isinstance(input, torch.Tensor),
            isinstance(input, np.ndarray),
            isinstance(input, list),
            isinstance(input, tuple),
            isinstance(input, dict),
            isinstance(input, str),
            input is None,
        ]
    ), "Not valid input type"

    if isinstance(input, torch.Tensor) or isinstance(input, np.ndarray):
        output = fn(input, *args)
    elif isinstance(input, list):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
    elif isinstance(input, tuple):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
        output = tuple(output)
    elif isinstance(input, dict):
        output = {}
        for key in input:
            output[key] = recur(fn, input[key], *args)
    elif isinstance(input, str):
        output = input
    elif input is None:
        output = None
    else:
        raise ValueError("Not valid input type")
    return output


def process_control():
    """process control parameters"""
    assert "num_clients" in cfg["control"], "Not valid control"

    cfg["num_supervised"] = int(cfg["control"]["num_supervised"])
    data_shape = {
        # "MNIST": [1, 28, 28],
        # "FashionMNIST": [1, 28, 28],
        # "CIFAR10": [3, 32, 32],
        # "CIFAR100": [3, 32, 32],
        # "SVHN": [3, 32, 32],
    }
    cfg["data_shape"] = data_shape[cfg["data_name"]]
    cfg["conv"] = {"hidden_size": [32, 64]}
    # cfg["resnet9"] = {"hidden_size": [64, 128, 256, 512]}
    # cfg["resnet18"] = {"hidden_size": [64, 128, 256, 512]}
    # cfg["wresnet28x2"] = {"depth": 28, "widen_factor": 2, "drop_rate": 0.0}
    # cfg["wresnet28x8"] = {"depth": 28, "widen_factor": 8, "drop_rate": 0.0}
    cfg["unsup_ratio"] = 1
    # if "loss_mode" in cfg["control"]:
    #     cfg["loss_mode"] = cfg["control"]["loss_mode"]
    #     cfg["threshold"] = float(cfg["control"]["loss_mode"].split("-")[0].split("@")[1])

    cfg["num_clients"] = int(cfg["control"]["num_clients"])
    cfg["active_rate"] = float(cfg["control"]["active_rate"])
    cfg["data_split_mode"] = cfg["control"]["data_split_mode"]
    cfg["local_epoch"] = cfg["control"]["local_epoch"].split("-")
    cfg["gm"] = float(cfg["control"]["gm"])
    cfg["sbn"] = int(cfg["control"]["sbn"])
    cfg["ft"] = int(cfg["control"]["ft"])
    cfg["server"] = {}
    cfg["server"]["shuffle"] = {"train": True, "test": False}
    if cfg["num_supervised"] > 1000:
        cfg["server"]["batch_size"] = {"train": 250, "test": 500}
    else:
        cfg["server"]["batch_size"] = {"train": 10, "test": 500}
    cfg["server"]["num_epochs"] = int(np.ceil(float(cfg["local_epoch"][1])))
    cfg["client"] = {}
    cfg["client"]["shuffle"] = {"train": True, "test": False}
    cfg["client"]["batch_size"] = {"train": 10, "test": 250}
    cfg["client"]["num_epochs"] = int(np.ceil(float(cfg["local_epoch"][0])))
    cfg["local"] = {}
    cfg["local"]["optimizer_name"] = "SGD"
    cfg["local"]["lr"] = 3e-2
    cfg["local"]["momentum"] = 0.9
    cfg["local"]["weight_decay"] = 5e-4
    cfg["local"]["nesterov"] = True
    cfg["global"] = {}
    cfg["global"]["batch_size"] = {"train": 250, "test": 250}
    cfg["global"]["shuffle"] = {"train": True, "test": False}
    cfg["global"]["num_epochs"] = 800
    cfg["global"]["optimizer_name"] = "SGD"
    cfg["global"]["lr"] = 1
    cfg["global"]["momentum"] = cfg["gm"]
    cfg["global"]["weight_decay"] = 0
    cfg["global"]["nesterov"] = False
    cfg["global"]["scheduler_name"] = "CosineAnnealingLR"
    cfg["alpha"] = 0.75
    return


def save(input, path, mode="torch"):
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    if mode == "torch":
        torch.save(input, path)
    elif mode == "np":
        np.save(path, input, allow_pickle=True)
    elif mode == "pickle":
        pickle.dump(input, open(path, "wb"))
    else:
        raise ValueError("Not valid save mode")
    return


def load(path, mode="torch"):
    if mode == "torch":
        return torch.load(path, map_location=lambda storage, loc: storage)
    elif mode == "np":
        return np.load(path, allow_pickle=True)
    elif mode == "pickle":
        return pickle.load(open(path, "rb"))
    else:
        raise ValueError("Not valid save mode")
