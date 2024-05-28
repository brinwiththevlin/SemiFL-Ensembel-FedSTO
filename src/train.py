import argparse
import torch.optim as optim
import datetime
import models
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from config import cfg, process_args
from data import (
    fetch_dataset,
    split_dataset,
    make_data_loader,
    separate_dataset,
    separate_dataset_su,
)
from metrics import Metric
from modules import Server, Client
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, collate
from logger import make_logger, Logger
from typing import Dict, List
import torch.nn as nn

cudnn.benchmark = True
parser = argparse.ArgumentParser(description="cfg")
for k in cfg:
    parser.add_argument(f"--{k}", default=cfg[k], type=type(cfg[k]))
parser.add_argument("--control_name", default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    process_control()
    seeds = list(range(cfg["init_seed"], cfg["init_seed"] + cfg["num_experiments"]))
    for i in range(cfg["num_experiments"]):
        model_tag_list = [str(seeds[i]), cfg["data_name"], cfg["model_name"], cfg["control_name"]]
        cfg["model_tag"] = "_".join([x for x in model_tag_list if x])
        print("Experiment: {}".format(cfg["model_tag"]))
        runExperiment()
    return


def runExperiment():
    cfg["seed"] = int(cfg["model_tag"].split("_")[0])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])

    server_dataset = fetch_dataset(cfg["data_name"])
    client_dataset = fetch_dataset(cfg["data_name"])
    process_dataset(server_dataset)

    server_dataset["train"], client_dataset["train"], supervised_idx = separate_dataset_su(
        server_dataset["train"], client_dataset["train"]
    )
    data_loader = make_data_loader(server_dataset, "global")

    model: nn.Module = getattr(models, cfg["model_name"])()
    model = model.to(cfg["device"])
    optimizer = make_optimizer(model.parameters(), "local")
    scheduler = make_scheduler(optimizer, "global")

    data_split = split_dataset(client_dataset, cfg["num_clients"], cfg["data_split_mode"])

    # TODO: change this based on final loss mode
    metric = Metric(
        {"train": ["Loss", "Accuracy", "PAccuracy", "MAccuracy", "LabelRatio"], "test": ["Loss", "Accuracy"]}
    )

    # if cfg["resume_mode"] == 1:
    #     result = resume(cfg["model_tag"])
    #     last_epoch = result["epoch"]
    #     if last_epoch > 1:
    #         data_split = result["data_split"]
    #         supervised_idx = result["supervised_idx"]
    #         server = result["server"]
    #         client = result["client"]
    #         optimizer.load_state_dict(result["optimizer_state_dict"])
    #         scheduler.load_state_dict(result["scheduler_state_dict"])
    #         logger = result["logger"]
    #     else:
    #         server = make_server(model)
    #         client = make_client(model, data_split)
    #         logger = make_logger(os.path.join("output", "runs", "train_{}".format(cfg["model_tag"])))
    # else:

    last_epoch = 0
    server: Server = make_server(model)
    client: list[Client] = make_client(model, data_split)
    logger = make_logger(os.path.join("output", "runs", "train_{}".format(cfg["model_tag"])))

    # Training
    Warmup(server_dataset["train"], server, optimizer, metric, logger, cfg["T0"])
    for epoch in range(last_epoch, cfg["T1"]):
        train_client(
            client_dataset["train"], server, client, optimizer, metric, logger, epoch, selective=True, orthogonal=False
        )

        logger.reset()
        server.update(client)
        train_server(server_dataset["train"], server, optimizer, metric, logger, epoch)

        record_result(
            model, server, client, optimizer, scheduler, supervised_idx, data_loader, data_split, logger, metric, epoch
        )

    for epoch in range(last_epoch, cfg["T2"]):
        train_client(
            client_dataset["train"], server, client, optimizer, metric, logger, epoch, selective=False, orthogonal=True
        )

        logger.reset()
        server.update(client)
        train_server(server_dataset["train"], server, optimizer, metric, logger, epoch)

        record_result(
            model, server, client, optimizer, scheduler, supervised_idx, data_loader, data_split, logger, metric, epoch
        )
    return


def make_server(model: nn.Module) -> Server:
    """returns a server object

    Args:
        model (nn.Module): machine learning model

    Returns:
        Server: server object
    """
    server = Server(model)
    return server


def make_client(model: nn.Module, data_split: Dict[str, Dict[int, List[int]]]) -> List[Client]:
    """creates a list of client objects

    Args:
        model (nn.Module): machine learning model
        data_split (Dict[str, Dict[int, List[int]]]): dictionary of data split. keys: "train", "test"

    Returns:
        List[Client]: list of client objects
    """
    client_id = torch.arange(cfg["num_clients"])
    client = [None for _ in range(cfg["num_clients"])]
    for m in range(len(client)):
        client[m] = Client(client_id[m], model, {"train": data_split["train"][m], "test": data_split["test"][m]})
    return client


def train_client(
    client_dataset: Dataset,
    server: Server,
    client: List[Client],
    optimizer: torch.optim.Optimizer,
    metric: Metric,
    logger: Logger,
    epoch: int,
    selective: bool = False,
    orthogonal: bool = False,
):
    # TODO: might need to change logger info
    logger.safe(True)
    num_active_clients = int(np.ceil(cfg["active_rate"] * cfg["num_clients"]))
    start_time = time.time()
    client_id: list = torch.randperm(cfg["num_clients"])[:num_active_clients].tolist()
    for i in range(num_active_clients):
        client[client_id[i]].active = True
    server.distribute(client)
    num_activae_clients = len(client_id)
    lr = optimizer.param_groups[0]["lr"]
    for i, m in enumerate(client_id):
        dataset_m = separate_dataset(client_dataset, client[m].data_split["train"])
        client[m].active = True
        client[m].train(dataset_m, lr, metric, logger, selective, orthogonal)

        if i % int((num_activae_clients * cfg["active_rate"]) + 1) == 0:
            _time = (time.time() - start_time) / (i + 1)
            epoch_finished_time = datetime.timedelta(seconds=_time * (num_active_clients - i - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg["global"]["num_epochs"] - epoch) * _time * num_active_clients)
            )
            exp_progress = 100.0 * i / num_active_clients
            info = {
                "info": [
                    "Model: {}".format(cfg["model_tag"]),
                    "Train Epoch (C): {}({:.0f}%)".format(epoch, exp_progress),
                    "Learning rate: {:.6f}".format(lr),
                    "ID: {}({}/{})".format(client_id[i], i + 1, num_active_clients),
                    "Epoch Finished Time: {}".format(epoch_finished_time),
                    "Experiment Finished Time: {}".format(exp_finished_time),
                ]
            }
            logger.append(info, "train", mean=False)
            print(logger.write("train", metric.metric_name["train"]))
    logger.safe(False)
    return


def Warmup(
    dataset: Dataset,
    server: Server,
    optimizer: torch.optim.Optimizer,
    metric: Metric,
    logger: Logger,
    rounds: int,
):
    """does warmup training on the server

    Args:
        dataset (Dataset): supeverised training dataset
        server (Server): Server object
        optimizer (torch.optim.Optimizer): optimizer
        metric (Metric): metric object
        logger (Logger): logger object
        rounds (int): number of warmup rounds

    """
    Logger.safe(True)
    for _ in range(rounds):
        server.train(dataset, optimizer, metric, logger)
        logger.reset()
    return server


def train_server(
    dataset: Dataset, server: Server, optimizer: optim.Optimizer, metric: Metric, logger: Logger, epoch: int
):
    logger.safe(True)
    start_time = time.time()
    lr = optimizer.param_groups[0]["lr"]
    server.train(dataset, lr, metric, logger)
    _time = time.time() - start_time
    epoch_finished_time = datetime.timedelta(seconds=round((cfg["global"]["num_epochs"] - epoch) * _time))
    info = {
        "info": [
            "Model: {}".format(cfg["model_tag"]),
            "Train Epoch (S): {}({:.0f}%)".format(epoch, 100.0),
            "Learning rate: {:.6f}".format(lr),
            "Epoch Finished Time: {}".format(epoch_finished_time),
        ]
    }
    logger.append(info, "train", mean=False)
    print(logger.write("train", metric.metric_name["train"]))
    logger.safe(False)
    return


def test(data_loader: DataLoader, model: nn.Module, metric: Metric, logger: Logger, epoch: int):
    logger.safe(True)
    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input["data"].size(0)
            input = to_device(input, cfg["device"])
            output = model(input)
            output["loss"] = output["loss"].mean() if cfg["world_size"] > 1 else output["loss"]
            evaluation = metric.evaluate(metric.metric_name["test"], input, output)
            logger.append(evaluation, "test", input_size)
        info = {"info": ["Model: {}".format(cfg["model_tag"]), "Test Epoch: {}({:.0f}%)".format(epoch, 100.0)]}
        logger.append(info, "test", mean=False)
        print(logger.write("test", metric.metric_name["test"]))
    logger.safe(False)
    return


def record_result(
    model: nn.Module,
    server: Server,
    client: Client,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    supervised_idx: List[int],
    data_loader: Dict[str, DataLoader],
    data_split: Dict[str, Dict[int, List[int]]],
    logger: Logger,
    metric: Metric,
    epoch: int,
):
    scheduler.step()
    model.load_state_dict(server.model_state_dict)
    test(data_loader["test"], model, metric, logger, epoch)
    result = {
        "cfg": cfg,
        "epoch": epoch + 1,
        "server": server,
        "client": client,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "supervised_idx": supervised_idx,
        "data_split": data_split,
        "logger": logger,
    }
    save(result, "./output/model/{}_checkpoint.pt".format(cfg["model_tag"]))
    if metric.compare(logger.mean["test/{}".format(metric.pivot_name)]):
        metric.update(logger.mean["test/{}".format(metric.pivot_name)])
        shutil.copy(
            "./output/model/{}_checkpoint.pt".format(cfg["model_tag"]),
            "./output/model/{}_best.pt".format(cfg["model_tag"]),
        )
    logger.reset()


if __name__ == "__main__":
    main()
