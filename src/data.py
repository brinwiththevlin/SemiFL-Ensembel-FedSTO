"""Data module"""

import copy
from typing import Dict, Tuple, List, Any
import torch
import numpy as np
from config import cfg
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from utils import collate, to_device
import datasets

data_stats = {
    "MNIST": ((0.1307,), (0.3081,)),
    "FashionMNIST": ((0.2860,), (0.3530,)),
    "CIFAR10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "CIFAR100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "SVHN": ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    "STL10": ((0.4409, 0.4279, 0.3868), (0.2683, 0.2610, 0.2687)),
}


def fetch_dataset(data_name: str) -> Dict[str, Dataset]:
    """fetches dataset from official torchvision datasets

    Args:
        data_name (str): name of the dataset

    Returns:
        Dict[str, Dataset]: dictoinary of dataset. keys: "train", "test"
    """
    assert data_name in datasets.__all__, "Not valid dataset name"
    dataset: Dict[str, Dataset] = {}
    print(f"fetching data {data_name}...")
    root = f"./data/{data_name}"

    dataset_class = getattr(datasets, data_name)
    dataset["train"] = dataset_class(root=root, split="train", transform=transforms.ToTensor())
    dataset["test"] = dataset_class(root=root, split="test", transform=transforms.ToTensor())

    if data_name in ["MNIST", "FashionMNIST"]:
        dataset["train"].transform = datasets.Compose(
            [transforms.ToTensor(), transforms.Normalize(*data_stats[data_name])]
        )
        dataset["test"].transform = datasets.Compose(
            [transforms.ToTensor(), transforms.Normalize(*data_stats[data_name])]
        )
    elif data_name in ["CIFAR10", "CIFAR100"]:
        dataset["train"].transform = datasets.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name]),
            ]
        )
        dataset["test"].transform = datasets.Compose(
            [transforms.ToTensor(), transforms.Normalize(*data_stats[data_name])]
        )
    elif data_name in ["SVHN"]:
        dataset["train"].transform = datasets.Compose(
            [
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name]),
            ]
        )
        dataset["test"].transform = datasets.Compose(
            [transforms.ToTensor(), transforms.Normalize(*data_stats[data_name])]
        )
    elif data_name in ["STL10"]:
        dataset["train"].transform = datasets.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(96, padding=12, padding_mode="reflect"),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name]),
            ]
        )
        dataset["test"].transform = datasets.Compose(
            [transforms.ToTensor(), transforms.Normalize(*data_stats[data_name])]
        )
    print("data ready")
    return dataset


def separate_dataset(dataset: Dataset, idx: List[int]) -> Dataset:
    """selects data points at idx from dataset

    Args:
        dataset (Dataset): dataset
        idx (List[int]): indexs to select

    Returns:
        Dataset: selected dataset
    """
    separated_dataset = copy.deepcopy(dataset)
    separated_dataset.data = [dataset.data[s] for s in idx]
    separated_dataset.target = [dataset.target[s] for s in idx]
    separated_dataset.other["id"] = list(range(len(separated_dataset.data)))
    return separated_dataset


def separate_dataset_su(
    server_dataset: Dataset, client_dataset: Dataset = None, supervised_idx=None
) -> Tuple[Dataset, Dataset, list]:
    """separate dataset into supervised and unsupervised

    Args:
        server_dataset (Dataset): server dataset
        client_dataset (Dataset): client dataset. Defaults to None.
        supervised_idx (List, optional): indexs selected for supervised training. Defaults to None.

    Returns:
        Tuple[Dataset, Dataset, list]: server dataset, client dataset, supervised index
    """
    if cfg["data_name"] in ["STL10"]:
        if cfg["num_supervised"] == -1:
            supervised_idx = torch.arange(5000).tolist()
        else:
            target = torch.tensor(server_dataset.target)[:5000]
            num_supervised_per_class = cfg["num_supervised"] // cfg["target_size"]
            supervised_idx = []
            for i in range(cfg["target_size"]):
                idx = torch.where(target == i)[0]
                idx = idx[torch.randperm(len(idx))[:num_supervised_per_class]].tolist()
                supervised_idx.extend(idx)
    else:
        if cfg["num_supervised"] == -1:
            supervised_idx = list(range(len(server_dataset)))
        else:
            target = torch.tensor(server_dataset.target)
            num_supervised_per_class = cfg["num_supervised"] // cfg["target_size"]
            supervised_idx = []
            for i in range(cfg["target_size"]):
                idx = torch.where(target == i)[0]
                idx = idx[torch.randperm(len(idx))[:num_supervised_per_class]].tolist()
                supervised_idx.extend(idx)
    idx = list(range(len(server_dataset)))
    unsupervised_idx = list(set(idx) - set(supervised_idx))
    _server_dataset: Dataset = separate_dataset(server_dataset, supervised_idx)  # sampled dataset for server
    _client_dataset: Dataset = separate_dataset(client_dataset, unsupervised_idx)  # the rest for client
    return _server_dataset, _client_dataset, supervised_idx


def make_dataset_normal(dataset: Dataset) -> Tuple[Dataset, transforms.Compose]:
    """normalizes a dataset

    Args:
        dataset (Dataset): dataset

    Returns:
        Tuple[Dataset, transforms.Compose]: normalized dataset, transform
    """
    _transform = dataset.transform
    transform = datasets.Compose([transforms.ToTensor(), transforms.Normalize(*data_stats[cfg["data_name"]])])
    dataset.transform = transform
    return dataset, _transform


def input_collate(batch: Any):
    if isinstance(batch[0], dict):
        output = {key: [] for key in batch[0].keys()}
        for b in batch:
            for key in b:
                output[key].append(b[key])
        return output
    else:
        return default_collate(batch)


def make_data_loader(dataset: Dict[str, Dataset], tag: str, shuffle: Dict[str, bool] = None) -> Dict[str, DataLoader]:
    """creates a dictionary of DataLoader objects

    Args:
        dataset (Dict[str, Dataset]): dictionary of dataset. keys: "train", "test"
        tag (str): tag for configuration
        shuffle (Dict[str, bol], optional): set values to true if you want to shuffle datasets. Defaults to None.

    Returns:
        Dict[str, DataLoader]: dictionary of dataloaders

    """
    data_loader = {}
    for k in dataset:
        _batch_size = cfg[tag]["batch_size"][k]
        _shuffle = cfg[tag]["shuffle"][k] if shuffle is None else shuffle[k]
        data_loader[k] = DataLoader(
            dataset=dataset[k],
            batch_size=_batch_size,
            shuffle=_shuffle,
            pin_memory=True,
            num_workers=cfg["num_workers"],
            collate_fn=input_collate,
            worker_init_fn=np.random.seed(cfg["seed"]),
        )

    return data_loader


def split_dataset(dataset: Dataset, num_users: int, data_split_mode: str) -> Dict[int, List[int]]:
    """
    Split the dataset into train and test sets based on the specified data split mode.

    Args:
        dataset (Dataset): The dataset to be split.
        num_users (int): The number of users to split the dataset among.
        data_split_mode (str): The mode for splitting the data. Must be either "iid" or "non-iid".

    Returns:
        Dict[int, List[int]]: A dictionary containing the split dataset, with keys "train" and "test".
            The values are lists of indices representing the data samples for each user.
    """
    assert cfg["data_split_mode"] == "iid" or "non-iid" in cfg["data_split_mode"], "Not valid data split mode"
    data_split = {}
    if data_split_mode == "iid":
        data_split["train"] = iid(dataset["train"], num_users)
        data_split["test"] = iid(dataset["test"], num_users)
    elif "non-iid" in cfg["data_split_mode"]:
        data_split["train"] = non_iid(dataset["train"], num_users)
        data_split["test"] = non_iid(dataset["test"], num_users)
    return data_split


def iid(dataset: Dataset, num_users: int) -> Dict[int, List[int]]:
    """splits dataset into num_users iid parts

    Args:
        dataset (Dataset): dataset to be split
        num_users (int): number of users

    Returns:
        Dict[int, List[int]]: dictionary of data split. keys: user index, values: list of data index
    """
    num_items = int(len(dataset) / num_users)
    data_split, idx = {}, list(range(len(dataset)))
    for i in range(num_users):
        num_items_i = min(len(idx), num_items)
        data_split[i] = torch.tensor(idx)[torch.randperm(len(idx))[:num_items_i]].tolist()
        idx = list(set(idx) - set(data_split[i]))
    return data_split


def non_iid(dataset: Dataset, num_users: int) -> Dict[int, List[int]]:
    """splits dataset into num_users non-iid parts

    Args:
        dataset (Dataset): dataset to split
        num_users (int): number of users

    Returns:
        Dict[int, List[int]]: dictionary of data split. keys: user index, values: list of data index
    """
    target = torch.tensor(dataset.target)
    data_split_mode_list = cfg["data_split_mode"].split("-")
    data_split_mode_tag = data_split_mode_list[-2]
    assert data_split_mode_tag in ["l", "d"], "Not valid data split mode tag"
    if data_split_mode_tag == "l":  # balanced non-iid with at most l classes per user
        data_split = {i: [] for i in range(num_users)}
        shard_per_user = int(data_split_mode_list[-1])
        target_idx_split = {}
        shard_per_class = int(shard_per_user * num_users / cfg["target_size"])
        for target_i in range(cfg["target_size"]):
            target_idx = torch.where(target == target_i)[0]
            num_leftover = len(target_idx) % shard_per_class
            leftover = target_idx[-num_leftover:] if num_leftover > 0 else []
            new_target_idx = target_idx[:-num_leftover] if num_leftover > 0 else target_idx
            new_target_idx = new_target_idx.reshape((shard_per_class, -1)).tolist()
            for i, leftover_target_idx in enumerate(leftover):
                new_target_idx[i] = new_target_idx[i] + [leftover_target_idx.item()]
            target_idx_split[target_i] = new_target_idx
        target_split = list(range(cfg["target_size"])) * shard_per_class
        target_split = torch.tensor(target_split)[torch.randperm(len(target_split))].tolist()
        target_split = torch.tensor(target_split).reshape((num_users, -1)).tolist()
        for i in range(num_users):
            for target_i in target_split[i]:
                idx = torch.randint(len(target_idx_split[target_i]), (1,)).item()
                data_split[i].extend(target_idx_split[target_i].pop(idx))
    else:  # dirichlet non-iid with beta parameter
        beta = float(data_split_mode_list[-1])
        dir = torch.distributions.dirichlet.Dirichlet(torch.tensor(beta).repeat(num_users))
        min_size = 0
        required_min_size = 10
        N = target.size(0)
        while min_size < required_min_size:
            data_split = [[] for _ in range(num_users)]
            for target_i in range(cfg["target_size"]):
                target_idx = torch.where(target == target_i)[0]
                proportions = dir.sample()
                proportions = torch.tensor(
                    [p * (len(data_split_idx) < (N / num_users)) for p, data_split_idx in zip(proportions, data_split)]
                )
                proportions = proportions / proportions.sum()
                split_idx = (torch.cumsum(proportions, dim=-1) * len(target_idx)).long().tolist()[:-1]
                split_idx = torch.tensor_split(target_idx, split_idx)
                data_split = [data_split_idx + idx.tolist() for data_split_idx, idx in zip(data_split, split_idx)]
            min_size = min([len(data_split_idx) for data_split_idx in data_split])
        data_split = {i: data_split[i] for i in range(num_users)}
    return data_split
