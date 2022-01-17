"""Gather utilitary functions for randomness control and parameters checking."""


from typing import List, Tuple

import argparse

import random
import numpy as np
from numpy import ndarray

import torch
from torch.nn import Module
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import Optimizer


from datasets import PartitionGenerator
from nn import LinRegNet, LogRegNet, MNISTNet, TorchNet, ResNet18, ResNet50
from optim import DistSGD, Signum


def set_seed(seed: int):
    """Fix seed for current run.

    Parameters
    ----------
    seed : int
        Global seeed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def str2bool(v: str) -> bool:
    """An easy way to handle boolean options.

    Parameters
    ----------
    v : str
        Argument value.

    Returns
    -------
    str2bool(v) : bool
        Corresponding boolean value, if it exists.

    Raises
    ------
    argparse.ArgumentTypeError
        If the entry cannot be converted to a boolean.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def check_nn(data_type: str, model_name: str, optim_name: str) -> Tuple[str, ...]:
    """Check whether chosen parameters for the nn are compatible.

    Parameters
    ----------
    data_type : {"logreg", "linreg", "mnist", "imagenet"}
        Name of the dataset.

    model_name : {"logregnet", "linregnet", "torchnet", "mnistnet", "resnet18", "resnet50"}
        Name of the neural network to use for training.

    optim_name : {"distsgd", "signsgd", "signum"}
        Name of the optimizer.

    Returns
    -------
    data_type : {"logreg", "linreg", "mnist", "imagenet"}
        Name of the dataset.

    model_name : {"logregnet", "linregnet", "torchnet", "mnistnet", "resnet18", "resnet50"}
        Name of the neural network to use for training.

    optim_name : {"distsgd", "signsgd", "signum"}
        Name of the optimizer.

    task : {"classification", "regression"}
        If "classification" task, will compute the accuracy, otherwise only the loss.

    Raises
    ------
    ValueError
        If some of the parameters are not supported.
    """
    # Unknown parameters
    available_nn = {"torchnet", "mnistnet", "resnet18", "resnet50", "linregnet", "logregnet"}
    if model_name not in available_nn:
        error_msg = f"Neural network {model_name} checking error." 
        error_msg += f" Please choose one from {available_nn}."
        raise ValueError(error_msg)

    available_data = {"mnist", "imagenet", "linreg", "logreg"}
    if data_type not in available_data:
        error_msg = f"Dataset {data_type} checking error. Please choose one from {available_data}."
        raise ValueError(error_msg)

    availabe_optim = {"distsgd", "signum", "signsgd"}
    if optim_name not in availabe_optim:
        error_msg = f"Optimizer {optim_name} is not supported."
        error_msg += f" Please shoose one from {availabe_optim}."
        raise ValueError(error_msg)

    # Incompatible parameters
    if data_type == "mnist" and model_name not in {"torchnet", "mnistnet"}:
        warning_msg = f"WARNING: Dataset {data_type} and model {model_name} are not compatible."
        warning_msg += " Setting model to torchnet."
        print(warning_msg)
        model_name = "torchnet"

    if data_type == "imagenet" and model_name not in {"resnet18", "resnet50"}:
        warning_msg = f"WARNING: Dataset {data_type} and model {model_name} are not compatible."
        warning_msg += " Setting model to resnet18."
        print(warning_msg)
        model_name = "resnet18"

    if data_type == "linreg" and model_name != "linregnet":
        warning_msg = f"WARNING: Dataset {data_type} and model {model_name} are not compatible."
        warning_msg += " Setting model to linregnet."
        print(warning_msg)
        model_name = "linregnet"

    if data_type == "logreg" and model_name != "logregnet":
        warning_msg = f"WARNING: Dataset {data_type} and model {model_name} are not compatible."
        warning_msg += " Setting model to logregnet."
        print(warning_msg)
        model_name = "logregnet"

    # Task
    if data_type == "linreg":
        task = "regression"
    else:
        task = "classification"

    return data_type, model_name, optim_name, task


def check_proc(world_size: int, blind_size: int, byz_size: int) -> Tuple[int, ...]:
    """Check whether chosen parameters for the processes are compatible.

    Parameters
    ----------
    world_size : int
        Total amount of processes.

    blind_size : int
       Number of blind adversaries that invert their gradients signs.

    byz_size : int
        Number of Byzantine adversaries.

    Returns
    -------
    world_size : int
        Total amount of processes.

    blind_size : int
       Number of blind adversaries that invert their gradients signs.

    byz_size : int
        Number of Byzantine adversaries.
    """
    # Values range
    if world_size <= 0:
        warning_msg = f"WARNING: Number of processors was set to {world_size}. Changed it to 1."
        print(warning_msg)
        world_size = 1

    if blind_size > world_size:
        warning_msg = "WARNING: Number of blind adversaries greater than world size"
        warning_msg += f" ({blind_size} > {world_size}). Changed it to {world_size}."
        print(warning_msg)
        blind_size = world_size

    left_size = world_size - blind_size
    if byz_size > left_size:
        warning_msg = "WARNING: Number of Byzantine adversaries greater than left ones"
        warning_msg += f" ({byz_size} > {left_size}). Changed it to {left_size}."
        print(warning_msg)
        byz_size = left_size

    return world_size, blind_size, byz_size


def check_plot(metrics: List[str], task: str) -> List[str]:
    """Check that the metrics correspond to the task that was run.

    Parameters
    ----------
    metrics : list of str
        Names of the metrics to plot.

    task : {"classification", "regression"}
        If "classification" task, will compute the accuracy, otherwise only the loss.

    Returns
    -------
    new_metrics : list of str
        Names of the metrics to plot.
    """
    new_metrics = []

    for metric in metrics:

        if task == "regression" and metric == "acc":
            print("WARNING: You wanted to plot accuracy for a regression task. Passed.")
        else:
            new_metrics.append(metric)

    return new_metrics


def init_model(model_name: str) -> Module:
    """Initialize a model based on its name.

    Parameters
    ----------
    model_name : {"logregnet", "linregnet", "torchnet", "mnistnet", "resnet18", "resnet50"}
        Name of the neural network to use for training.

    Returns
    -------
    model : torch.nn.Module
        A neural network to be trained and evaluated on the datasets.
    """
    model: Module

    if model_name == "linregnet":
        model = LinRegNet()
    elif model_name == "logregnet":
        model = LogRegNet()
    elif model_name == "torchnet":
        model = TorchNet(10)
    elif model_name == "mnistnet":
        model = MNISTNet(10)
    elif model_name == "resnet18":
        model = ResNet18(10)
    elif model_name == "resnet50":
        model = ResNet50(10)
    else:
        raise ValueError(f"Unknown model {model_name}.")

    return model


def init_loss(data_type: str) -> Module:
    """Initialize a loss function based on the considered dataset.

    Parameters
    ----------
    data_type : {"logreg", "linreg", "mnist", "imagenet"}
        Name of the dataset.

    Returns
    -------
    loss_fn : torch.nn.Module
        A loss function. Linear regression problem is associated to a mean squared error loss
        while the other datasets and models can be trained with the cross entropy loss.
    """
    loss_fn : Module

    if data_type == "linreg":
        loss_fn = MSELoss()
    else:
        loss_fn = CrossEntropyLoss()

    return loss_fn


def init_optim(optim_name: str, model: Module, blind_list: ndarray, byz_list: ndarray, 
               lr: float = 1e-3, momentum: float = 0.0, weight_decay: float = 0.0) -> Optimizer:
    """Initialize an optimizer based on its name.

    Parameters
    ----------
    optim_name : {"distsgd", "signsgd", "signum"}
        Name of the optimizer.

    blind_list : numpy.ndarray
        Ranks of blind adversaries inverting their gradients signs.

    byz_list : numpy.ndarray
        Ranks of Byzantine adversaries.

    lr : float, default=0.0001
        Learning rate for the optimizer.

    momentum : float, default=0.0
        Momentum for the optimizer.

    weight_decay : float, default=0.0
        Weight decay for the optimizer.

    Returns
    -------
    optimizer : torch.optim.Optimizer
        An optimizer to train the model with.
    """
    optimizer: Optimizer

    if optim_name == "distsgd":
        optimizer = DistSGD(model.parameters(), blind_list, byz_list, lr=lr, momentum=momentum)
    elif optim_name == "signsgd":
        optimizer = Signum(model.parameters(), blind_list, byz_list, lr=lr, momentum=0.,
                           weight_decay=weight_decay)
    elif optim_name == "signum":
        optimizer = Signum(model.parameters(), blind_list, byz_list, lr=lr, momentum=momentum,
                           weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer {optim_name}.")

    return optimizer


def init_data(data_type: str) -> PartitionGenerator:
    """Initialize a dataset based on its name.

    Parameters
    ----------
    data_type : {"logreg", "linreg", "mnist", "imagenet"}
        Name of the dataset.

    Returns
    -------
    data : `src.datasets.PartitionGenerator`
        An instance of `datasets.PartitionGenerator` to handle partitions for processes.
    """
    data = PartitionGenerator(data_type, shuffle=True)

    return data
