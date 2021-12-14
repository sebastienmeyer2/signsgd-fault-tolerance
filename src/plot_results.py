"""Utilitary functions that allow to plot results."""

# Import Python packages
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import List

# Import our own packages
import utils as ut

# Packages parameters
font = {"size": 14}
mpl.rc("font", **font)


def plot_save_metric(metric: str, world_size: int, blind_inv_size: int, byz_adv_size: int,
                     data_type: str, model_name: str, optim_name: str, training: bool = False):
    """
    Plot and save training and test loss for specified run.

    Parameters
    ----------
    metric : str
        Name of the metric as in the `argparse` arguments.

    world_size : int
        Total amount of processes

    blind_inv_size : int
        Number of blind adversaries in the run.

    byz_adv_size : int
        Number of Byzantine adversaries in the run.

    data_type : {"MNIST", "ImageNet", "LinReg", "LogReg"}
        Name of the dataset.

    model_name : {"TorchNet", "MNISTNet", "ResNet18", "ResNet50", "LinRegNet", "LogRegNet"}
        Name of the neural network to use for training. "TorchNet" and "MNISTNet" are compatible
        with "MNIST" dataset, while "ResNetX" are compatible with "ImageNet". "LinRegNet" is only
        compatible with "LinReg" dataset, as well as "LogRegNet" and "LogReg".

    optim_name : {"DistSGD", "Signum", "SignSGD"}
        Name of the optimizer.

    training : bool, default=False
        If True, will plot the training score, otherwise the test score.
    """
    metric_low = metric.lower()

    # Path to data
    scores_folder = "results/"

    if not os.path.exists(scores_folder):
        raise ValueError("Results folder is missing.")

    scores_folder += data_type + "_" + model_name + "_" + optim_name + "/"

    if not os.path.exists(scores_folder):
        error_msg = "No folder associated to dataset {}, model {} and/or optimizer {}."
        dmo_values = (data_type, model_name, optim_name)
        raise ValueError(error_msg.format(*dmo_values))

    scores_folder += "n" + str(world_size) + "_byz" + str(byz_adv_size) + "_inv" + \
        str(blind_inv_size) + "/"

    if not os.path.exists(scores_folder):
        error_msg = "No folder associated to nprocs {}, Byzantine {} and blind {}."
        nbi_values = (world_size, byz_adv_size, blind_inv_size)
        raise ValueError(error_msg.format(*nbi_values))

    path_to_score = scores_folder + metric_low + "_p0.csv"

    # Plot training and test score
    score_data = pd.read_csv(path_to_score)

    T = score_data["Epoch"]
    if training:
        score = score_data["Training " + metric_low]
    else:
        score = score_data["Test " + metric_low]

    pl_label = "Training " if training else "Test "
    pl_label += metric_low
    plt.plot(T, score, label=pl_label, color="blue")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel(metric)

    path_to_file = scores_folder + metric_low + "_p0.jpg"
    plt.tight_layout()
    plt.savefig(path_to_file)
    plt.close()


def plot_save_loss_acc(world_size: int, blind_inv_size: int, byz_adv_size: int, data_type: str,
                       model_name: str, optim_name: str) -> None:
    """
    Plot and save training and test accuracy & loss for specified run.

    Parameters
    ----------
    world_size : int
        Total amount of processes

    blind_inv_size : int
        Number of blind adversaries in the run.

    byz_adv_size : int
        Number of Byzantine adversaries in the run.

    data_type : {"MNIST", "ImageNet", "LinReg", "LogReg"}
        Name of the dataset.

    model_name : {"TorchNet", "MNISTNet", "ResNet18", "ResNet50", "LinRegNet", "LogRegNet"}
        Name of the neural network to use for training. "TorchNet" and "MNISTNet" are compatible
        with "MNIST" dataset, while "ResNetX" are compatible with "ImageNet". "LinRegNet" is only
        compatible with "LinReg" dataset, as well as "LogRegNet" and "LogReg".

    optim_name : {"DistSGD", "Signum", "SignSGD"}
        Name of the optimizer.
    """
    # Path to data
    scores_folder = "results/"

    if not os.path.exists(scores_folder):
        raise ValueError("Results folder is missing.")

    scores_folder += data_type + "_" + model_name + "_" + optim_name + "/"

    if not os.path.exists(scores_folder):
        error_msg = "No folder associated to dataset {}, model {} and/or optimizer {}."
        dmo_values = (data_type, model_name, optim_name)
        raise ValueError(error_msg.format(*dmo_values))

    scores_folder += "n" + str(world_size) + "_byz" + str(byz_adv_size) + "_inv" + \
        str(blind_inv_size) + "/"

    if not os.path.exists(scores_folder):
        error_msg = "No folder associated to nprocs {}, Byzantine {} and blind {}."
        nbi_values = (world_size, byz_adv_size, blind_inv_size)
        raise ValueError(error_msg.format(*nbi_values))

    path_to_acc = scores_folder + "acc_p0.csv"
    path_to_loss = scores_folder + "loss_p0.csv"

    # Retrieve data
    acc_data = pd.read_csv(path_to_acc)
    loss_data = pd.read_csv(path_to_loss)

    T = acc_data["Epoch"]
    training_acc = acc_data["Training accuracy"]
    test_acc = acc_data["Test accuracy"]
    training_loss = loss_data["Training loss"]
    test_loss = loss_data["Test loss"]

    # Plot training and test of both metrics
    fig, ax = plt.subplots(ncols=2, figsize=(12, 6))

    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].plot(T, training_loss, label="Training loss", color="blue")
    ax[0].plot(T, test_loss, label="Test loss", color="red")
    ax[0].tick_params(axis="y")
    ax[0].legend()

    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].plot(T, training_acc, label="Training Accuracy", color="blue")
    ax[1].plot(T, test_acc, label="Test Accuracy", color="red")
    ax[1].tick_params(axis="y")
    ax[1].legend()

    title = "Loss & Accuracy | {} blind, {} Byzantine adversaries out of {} processes"
    title_values = (blind_inv_size, byz_adv_size, world_size)
    fig.suptitle(title.format(*title_values))

    path_to_file = scores_folder + "acc_loss_p0.jpg"
    plt.tight_layout()
    plt.savefig(path_to_file)
    plt.close()


def compare_metric_blind_adv(metric: str, world_size: int, blind_inv_sizes: np.ndarray,
                             byz_adv_size: int, data_type: str, model_name: str, optim_name: str,
                             training: bool = False):
    """
    Plot and save training and test loss for different amounts of blind adversaries.

    Parameters
    ----------
    metric : str
        Name of the metric as in the `argparse` arguments.

    world_size : int
        Total amount of processes

    blind_inv_sizes : np.ndarray
        Number of blind adversaries in each run.

    byz_adv_size : int
        Number of Byzantine adversaries in all runs.

    data_type : {"MNIST", "ImageNet", "LinReg", "LogReg"}
        Name of the dataset.

    model_name : {"TorchNet", "MNISTNet", "ResNet18", "ResNet50", "LinRegNet", "LogRegNet"}
        Name of the neural network to use for training. "TorchNet" and "MNISTNet" are compatible
        with "MNIST" dataset, while "ResNetX" are compatible with "ImageNet". "LinRegNet" is only
        compatible with "LinReg" dataset, as well as "LogRegNet" and "LogReg".

    optim_name : {"DistSGD", "Signum", "SignSGD"}
        Name of the optimizer.

    training : bool, default=False
        If True, will plot the training score, otherwise the test score.
    """
    n_plots = len(blind_inv_sizes)

    color = iter(plt.cm.brg(np.linspace(0, 1, n_plots)))

    metric_low = metric.lower()

    # Path to data
    scores_folder = "results/"

    if not os.path.exists(scores_folder):
        raise ValueError("Results folder is missing.")

    scores_folder += data_type + "_" + model_name + "_" + optim_name + "/"

    if not os.path.exists(scores_folder):
        error_msg = "No folder associated to dataset {}, model {} and/or optimizer {}."
        dmo_values = (data_type, model_name, optim_name)
        raise ValueError(error_msg.format(*dmo_values))

    for i in range(n_plots):

        folder_i = scores_folder + "n" + str(world_size) + "_byz" + str(byz_adv_size) + "_inv" + \
            str(blind_inv_sizes[i]) + "/"

        if not os.path.exists(folder_i):
            error_msg = "No folder associated to nprocs {}, Byzantine {} and blind {}."
            nbi_values = (world_size, byz_adv_size, blind_inv_sizes[i])
            raise ValueError(error_msg.format(*nbi_values))

        path_to_score = folder_i + metric_low + "_p0.csv"

        # Retrieve data
        score_data = pd.read_csv(path_to_score)

        T = score_data["Epoch"]
        if training:
            score = score_data["Training " + metric_low]
        else:
            score = score_data["Test " + metric_low]

        if metric == "Accuracy":
            score = 100*score.to_numpy()

        # Plot training and test score
        c = next(color)

        score_label = r"{}% blind adv.".format(int(100*blind_inv_sizes[i]/world_size))
        plt.plot(T, score, label=score_label, color=c)

    # Legends
    plt.legend()
    plt.xlabel("Epoch")
    y_label = metric
    if metric == "Accuracy":
        y_label += " (%)"
    plt.ylabel(y_label)

    # Saving the graph
    path_to_file = scores_folder + "n" + str(world_size) + "_" + metric_low + "_comp_blind_inv.jpg"
    plt.tight_layout()
    plt.savefig(path_to_file)
    plt.close()


def compare_metric_byz(metric: str, world_size: int, blind_inv_size: int,
                       byz_adv_sizes: List[float], data_type: str, model_name: str,
                       optim_name: str, training: bool = False):
    """
    Plot and save training and test loss for different amounts of Byzantine adversaries.

    Parameters
    ----------
    metric : str
        Name of the metric as in the `argparse` arguments.

    world_size : int
        Total amount of processes

    blind_inv_size : int
        Number of blind adversaries in all runs.

    byz_adv_sizes : List[float]
        Number of Byzantine adversaries in each run.

    data_type : {"MNIST", "ImageNet", "LinReg", "LogReg"}
        Name of the dataset.

    model_name : {"TorchNet", "MNISTNet", "ResNet18", "ResNet50", "LinRegNet", "LogRegNet"}
        Name of the neural network to use for training. "TorchNet" and "MNISTNet" are compatible
        with "MNIST" dataset, while "ResNetX" are compatible with "ImageNet". "LinRegNet" is only
        compatible with "LinReg" dataset, as well as "LogRegNet" and "LogReg".

    optim_name : {"DistSGD", "Signum", "SignSGD"}
        Name of the optimizer.

    training : bool, default=False
        If True, will plot the training score, otherwise the test score.
    """
    n_plots = len(byz_adv_sizes)

    color = iter(plt.cm.brg(np.linspace(0, 1, n_plots)))

    metric_low = metric.lower()

    # Path to data
    scores_folder = "results/"

    if not os.path.exists(scores_folder):
        raise ValueError("Results folder is missing.")

    scores_folder += data_type + "_" + model_name + "_" + optim_name + "/"

    if not os.path.exists(scores_folder):
        error_msg = "No folder associated to dataset {}, model {} and/or optimizer {}."
        dmo_values = (data_type, model_name, optim_name)
        raise ValueError(error_msg.format(*dmo_values))

    for i in range(n_plots):

        folder_i = scores_folder + "n" + str(world_size) + "_byz" + str(byz_adv_sizes[i]) + \
            "_inv" + str(blind_inv_size) + "/"

        if not os.path.exists(folder_i):
            error_msg = "No folder associated to nprocs {}, Byzantine {} and blind {}."
            nbi_values = (world_size, byz_adv_sizes[i], blind_inv_size)
            raise ValueError(error_msg.format(*nbi_values))

        path_to_score = folder_i + metric_low + "_p0.csv"

        # Retrieve data
        score_data = pd.read_csv(path_to_score)

        T = score_data["Epoch"]
        if training:
            score = score_data["Training " + metric_low]
        else:
            score = score_data["Test " + metric_low]

        if metric == "Accuracy":
            score = 100*score.to_numpy()

        # Plot training and test score
        c = next(color)

        score_label = r"{}% Byzantine adv.".format(int(100*byz_adv_sizes[i]/world_size))
        plt.plot(T, score, label=score_label, color=c)

    # Legends
    plt.legend()
    plt.xlabel("Epoch")
    y_label = metric
    if metric == "Accuracy":
        y_label += " (%)"
    plt.ylabel(y_label)

    # Saving the graph
    path_to_file = scores_folder + "n" + str(world_size) + "_" + metric_low + "_comp_byzantine.jpg"
    plt.tight_layout()
    plt.savefig(path_to_file)
    plt.close()


if __name__ == "__main__":

    # Command lines
    parser = argparse.ArgumentParser(description="Evaluate your models tolerance to adversaries.")

    # Number of processes
    parser.add_argument("nprocs",
                        type=int,
                        help="""
                             Choose the total amount of processes to run in parallel. This number \
                             contains healthy workers aswell as possible adversaries to be set up \
                             with optional arguments.
                             """)

    # Data type and corresponding neural network & optimizer
    parser.add_argument("dataset",
                        type=str,
                        help="""Choose the name of the dataset.""",
                        choices=["MNIST", "ImageNet", "LinReg", "LogReg"])

    parser.add_argument("-n",
                        "--net",
                        type=str,
                        help="""
                             Choose the name of the neural network to use for training. \
                             "TorchNet" and "MNISTNet" are compatible with MNIST dataset, while \
                             "ResNet18" and "ResNet50" are compatible with ImageNet. "LinRegNet" \
                             is only compatible with LinReg dataset. Default: "MNISTNet".
                             """,
                        choices=["TorchNet", "MNISTNet", "ResNet18", "ResNet50", "LinRegNet",
                                 "LogRegNet"],
                        default="MNISTNet")

    parser.add_argument("-o",
                        "--optimizer",
                        type=str,
                        help="""Choose an optimizer. Default: "SignSGD".""",
                        choices=["DistSGD", "Signum", "SignSGD"],
                        default="Signum")

    # Sub command lines for mode
    subparsers = parser.add_subparsers(dest="subcommands",
                                       description="""
                                                   Please choose a mode of evaluation from "Plot" \
                                                   and "Comparison". "Plot" will create a graph of \
                                                   one run if it exists. "Comparison" will take \
                                                   several proportions of blind adversaries or \
                                                   several amounts of Byzantine adversaries and \
                                                   plot a comparison of a chosen metric.
                                                   """)

    parser_plot = subparsers.add_parser("Plot")
    parser_plot.add_argument("metric",
                             type=str,
                             help="""
                                  Choose to plot the "Loss", "Accuracy" or "Both" for a given \
                                  run.
                                  """,
                             choices=["Loss", "Accuracy", "Both"])

    parser_plot.add_argument("-i",
                             "--blindinv",
                             type=int,
                             help="""
                                  Choose a number of blind adversaries inverting their \
                                  gradients signs. Default: 0.""",
                             default=0)

    parser_plot.add_argument("-b",
                             "--byzantine",
                             type=int,
                             help="""Choose a NUMBER of Byzantine adversaries. Default: 0.""",
                             default=0)

    parser_comp_inv = subparsers.add_parser("Comparison")
    parser_comp_inv.add_argument("metric",
                                 type=str,
                                 help="""Choose to plot the "Loss" or "Accuracy" for comparison.""",
                                 choices=["Loss", "Accuracy"])

    parser_comp_inv.add_argument("-c",
                                 "--blindinvlist",
                                 type=lambda s: [int(x) for x in s.split(",")],
                                 help="""
                                      Type NUMBERS of blind adversaries to compare. You \
                                      should separate each proportion with a ",". Example: \
                                      "0,6". Default: "0".
                                      """,
                                 default="0")

    parser_comp_inv.add_argument("-d",
                                 "--byzantinelist",
                                 type=lambda s: [int(x) for x in s.split(",")],
                                 help="""
                                      Choose NUMBERS of Byzantine adversaries. You should separate \
                                      each value with a ",". Example: "0,1,2".
                                      """,
                                 default="0")

    # End of command lines
    args = parser.parse_args()

    # Number of processes
    world_size = args.nprocs
    print("In total, there were {} processes.".format(world_size))

    # Data and neural network
    data_type = args.dataset
    model_name = args.net
    optim_name = args.optimizer

    data_type, model_name, optim_name = ut.check_params(data_type, model_name, optim_name)

    # Mode selection
    mode = args.subcommands

    # Plotting loss
    if mode == "Plot":

        metric = args.metric

        # Establish number of processes that are blind adversaries
        blind_inv_size = args.blindinv
        print("{} blind adversaries were inverting their gradients signs.".format(blind_inv_size))

        # Establish number of processes that are Byzantine adversaries
        byz_adv_size = args.byzantine
        print("{} Byzantine adversaries were sending arbitrary values.".format(byz_adv_size))

        if metric == "Both":
            plot_save_loss_acc(world_size, blind_inv_size, byz_adv_size, data_type, model_name,
                               optim_name)
        else:
            plot_save_metric(metric, world_size, blind_inv_size, byz_adv_size, data_type,
                             model_name, optim_name)

    elif mode == "Comparison":

        metric = args.metric

        # Establish numbers of processes that are blind adversaries
        blind_inv_sizes = args.blindinvlist
        print("{} blind adversaries were inverting their gradients signs.".format(blind_inv_sizes))

        # Establish numbers of processes that are Byzantine adversaries
        byz_adv_sizes = args.byzantinelist
        print("{} Byzantine adversaries were sending arbitrary values.".format(byz_adv_sizes))

        if len(blind_inv_sizes) > 1 and len(byz_adv_sizes) > 1:
            error_msg = "Comparing with several values of blind and Byzantine adversaries is not \
                supported"
            raise ValueError(error_msg)

        if len(blind_inv_sizes) > 1:
            compare_metric_blind_adv(metric, world_size, blind_inv_sizes, byz_adv_sizes[0],
                                     data_type, model_name, optim_name)
        elif len(byz_adv_sizes) > 1:
            compare_metric_byz(metric, world_size, blind_inv_sizes[0], byz_adv_sizes, data_type,
                               model_name, optim_name)

    else:
        raise ValueError("Unknown mode {}. Please choose from available modes.".format(mode))
