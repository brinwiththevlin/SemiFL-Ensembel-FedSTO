import copy
from typing import Dict, Tuple
import torch
import numpy as np
import models
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
    dataset = {}
    print("fetching data {}...".format(data_name))
    root = "./data/{}".format(data_name)

    dataset["train"] = eval(
        "datasets.{}(root=root, split='train', "
        "transform=datasets.Compose([transforms.ToTensor()]))".format(data_name)
    )
    dataset["test"] = eval(
        "datasets.{}(root=root, split='test', " "transform=datasets.Compose([transforms.ToTensor()]))".format(data_name)
    )

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


def separate_dataset(dataset, idx):
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
    transform = FixTransform(cfg["data_name"])
    _client_dataset.transform = transform
    return _server_dataset, _client_dataset, supervised_idx


def input_collate(batch):
    if isinstance(batch[0], dict):
        output = {key: [] for key in batch[0].keys()}
        for b in batch:
            for key in b:
                output[key].append(b[key])
        return output
    else:
        return default_collate(batch)


def make_data_loader(
    dataset: Dict[str, Dataset], tag: str, batch_size=None, shuffle=None, sampler=None, batch_sampler=None
) -> Dict[str, DataLoader]:
    data_loader = {}
    for k in dataset:
        _batch_size = cfg[tag]["batch_size"][k] if batch_size is None else batch_size[k]
        _shuffle = cfg[tag]["shuffle"][k] if shuffle is None else shuffle[k]
        if sampler is not None:
            data_loader[k] = DataLoader(
                dataset=dataset[k],
                batch_size=_batch_size,
                sampler=sampler[k],
                pin_memory=True,
                num_workers=cfg["num_workers"],
                collate_fn=input_collate,
                worker_init_fn=np.random.seed(cfg["seed"]),
            )
        elif batch_sampler is not None:
            data_loader[k] = DataLoader(
                dataset=dataset[k],
                batch_sampler=batch_sampler[k],
                pin_memory=True,
                num_workers=cfg["num_workers"],
                collate_fn=input_collate,
                worker_init_fn=np.random.seed(cfg["seed"]),
            )
        else:
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


class FixTransform(object):
    def __init__(self, data_name):
        assert data_name in datasets.__all__, "Not valid dataset name"
        if data_name in ["CIFAR10", "CIFAR100"]:
            self.weak = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                    transforms.ToTensor(),
                    transforms.Normalize(*data_stats[data_name]),
                ]
            )
            self.strong = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                    datasets.RandAugment(n=2, m=10),
                    transforms.ToTensor(),
                    transforms.Normalize(*data_stats[data_name]),
                ]
            )
        elif data_name in ["SVHN"]:
            self.weak = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                    transforms.ToTensor(),
                    transforms.Normalize(*data_stats[data_name]),
                ]
            )
            self.strong = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                    datasets.RandAugment(n=2, m=10),
                    transforms.ToTensor(),
                    transforms.Normalize(*data_stats[data_name]),
                ]
            )
        elif data_name in ["STL10"]:
            self.weak = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(96, padding=12, padding_mode="reflect"),
                    transforms.ToTensor(),
                    transforms.Normalize(*data_stats[data_name]),
                ]
            )
            self.strong = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(96, padding=12, padding_mode="reflect"),
                    datasets.RandAugment(n=2, m=10),
                    transforms.ToTensor(),
                    transforms.Normalize(*data_stats[data_name]),
                ]
            )
        else:
            raise ValueError("Not valid dataset")

    def __call__(self, input):
        data = self.weak(input["data"])
        aug = self.strong(input["data"])
        input = {**input, "data": data, "aug": aug}
        return input
