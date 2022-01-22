"""Utilitary functions that allow to plot results."""


import os

from typing import List

import argparse

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler


import utils as ut


plt.style.use("seaborn-darkgrid")
mpl.rcParams["font.size"] = 18
mpl.rcParams["axes.prop_cycle"] = cycler("color", plt.get_cmap("Dark2")(np.linspace(0, 1, 5)))


def plot_save_metrics(metrics: List[str], world_size: int, blind_size: int, byz_size: int,
                      folder_partname: str, optim_names: List[str], include_title: bool = False):
    """Plot and save training and test accuracy and loss for specified run.

    Parameters
    ----------
    metrics : list of str
        Names of the metrics to plot.

    world_size : int
        Total amount of processes.

    blind_size : int
        Number of blind adversaries inverting their gradient signs.

    byz_size : int
        Number of Byzantines adversaries.

    folder_partname : str
        Specific partname for the folder to save scores results, does not include the optimizer
        name.

    optim_names : list of str
        Names of the optimizers to compare.

    include_title : bool, default=False
        If True, will add a suptitle for the graph, otherwise no title will be added.

    Raises
    ------
    ValueError
        If the path to the results folder does not exist.
    """
    n_optims = len(optim_names)
    n_metrics = len(metrics)

    # Path to data
    results_folder = "results/"

    if not os.path.exists(results_folder):
        raise ValueError("Results folder is missing.")

    # Plot training and test of metrics
    nrows = n_metrics
    ncols = n_optims

    width = 6*n_optims
    heigth = 6*n_metrics

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, heigth), squeeze=False)

    for i in range(n_metrics):

        for j in range(n_optims):

            metric = metrics[i]
            optim_name = optim_names[j]

            # Complete the path
            scores_folder = results_folder + folder_partname + optim_name + "/"

            if not os.path.exists(scores_folder):
                error_msg = f"No folder associated to dataset {data_type},"
                error_msg += f" model {model_name} and/or optimizer {optim_name}."
                raise ValueError(error_msg)

            scores_folder += "n" + str(world_size) + "_byz" + str(byz_size) + "_inv" + \
                str(blind_size) + "/"

            if not os.path.exists(scores_folder):
                error_msg = f"No folder associated to nprocs {world_size},"
                error_msg += f" Byzantine {byz_size} and blind {blind_size}."
                raise ValueError(error_msg)

            path_to_metric = scores_folder + metric + "_p0.csv"

            # Retrieve data
            metric_data = pd.read_csv(path_to_metric)

            T = metric_data["Epoch"]
            training_metric = metric_data[f"Training {metric}"]
            test_metric = metric_data[f"Test {metric}"]

            # Plot this metric
            y_label_name = metric
            if metric == "acc":
                y_label_name += "uracy"

            if i == nrows-1:
                ax[i][j].set_xlabel("Epoch")
            if j == 0:
                ax[i][j].set_ylabel(y_label_name.capitalize())
            if i == 0:
                title_name = optim_name.capitalize()
                if "sgd" in optim_name:
                    title_name = title_name[:-3] + title_name[-3:].upper()
                ax[i][j].set_title(title_name)
            ax[i][j].plot(T, training_metric, label=f"Training {y_label_name}")
            ax[i][j].plot(T, test_metric, label=f"Test {y_label_name}")
            ax[i][j].tick_params(axis="y")
            ax[i][j].legend()

    # Figure suptitle
    if include_title:
        fig.suptitle(f"{blind_size} blind, {byz_size} Byz. / {world_size}")

    # Saving the graph
    path_to_file = results_folder + folder_partname + "n" + str(world_size)
    path_to_file += "_i" + str(blind_size)
    path_to_file += "_b" + str(byz_size)

    for j in range(n_optims):
        path_to_file += "_" + optim_names[j]
    for i in range(n_metrics):
        path_to_file += "_" + metrics[i]
    path_to_file += ".jpg"

    plt.tight_layout()
    plt.savefig(path_to_file)
    plt.close()


def compare_save_metrics(metrics: List[str], world_size: int, blind_sizes: List[int],
                         byz_sizes: List[int], folder_partname: str, optim_names: List[str],
                         training: bool = False, include_title: bool = False):
    """Plot and save training and test **metrics** for different amounts of blind adversaries.

    Parameters
    ----------
    metrics : list of str
        Names of the metrics to plot.

    world_size : int
        Total amount of processes.

    blind_sizes : list of int
        Numbers of blind adversaries inverting their gradient signs.

    byz_sizes : list of int
        Numbers of Byzantines adversaries.

    folder_partname : str
        Specific partname for the folder to save scores results, does not include the optimizer
        name.

    optim_names : list of str
        Names of the optimizers to compare.

    training : bool, default=False
        If True, will plot the training score, otherwise the test score.

    include_title : bool, default=False
        If True, will add a suptitle for the graph, otherwise no title will be added.

    Raises
    ------
    ValueError
        If the numbers of adversaries are not compatible or path to the optimizer results folder
        does not exist.
    """
    # Global parameters
    if len(blind_sizes) > 1 and len(byz_sizes) > 1:

        error_msg = "Comparing with several values of blind and Byzantine adversaries is not \
            supported"
        raise ValueError(error_msg)

    elif len(blind_sizes) > 1 and len(byz_sizes) == 1:

        single_adv = byz_sizes[0]
        single_str = "_b"
        single_name = " Byz."
        several_adv = blind_sizes
        several_str = "_i"
        several_name = " blind"

    elif len(blind_sizes) == 1 and len(byz_sizes) > 1:

        single_adv = blind_sizes[0]
        single_str = "_i"
        single_name = " blind"
        several_adv = byz_sizes
        several_str = "_b"
        several_name = " Byz."

    else:

        error_msg = "Unsupported numbers of adversaries."
        raise ValueError(error_msg)

    n_metrics = len(metrics)
    n_optims = len(optim_names)
    n_plots = len(several_adv)

    # Path to data
    results_folder = "results/"

    if not os.path.exists(results_folder):
        raise ValueError("Results folder is missing.")

    # Plot training or test of metrics
    nrows = n_metrics
    ncols = n_optims

    width = 6*n_optims
    heigth = 6*n_metrics

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, heigth), squeeze=False)

    for i in range(n_metrics):

        for j in range(n_optims):

            metric = metrics[i]
            optim_name = optim_names[j]

            color = iter(plt.get_cmap("Dark2")(np.linspace(0, 1, n_plots)))

            # Complete the path
            scores_folder = results_folder + folder_partname + optim_name + "/"

            if not os.path.exists(scores_folder):
                error_msg = f"No folder associated to dataset {data_type},"
                error_msg += f" model {model_name} and/or optimizer {optim_name}."
                raise ValueError(error_msg)

            # Plot for each blind adversaries proportion
            for k in range(n_plots):

                if single_str == "_b":

                    folder_k = scores_folder + "n" + str(world_size) + "_byz" + \
                        str(single_adv) + "_inv" + str(several_adv[k]) + "/"

                else:

                    folder_k = scores_folder + "n" + str(world_size) + "_byz" + \
                        str(several_adv[k]) + "_inv" + str(single_adv) + "/"

                if not os.path.exists(folder_k):
                    warning_msg = f"WARNING: No folder associated to nprocs {world_size},"
                    warning_msg += f" {single_adv} {single_name} and"
                    warning_msg += f" {several_adv[k]} {several_name} adversaries. Passed!"
                    print(warning_msg)
                    c = next(color)
                    continue

                path_to_metric = folder_k + metric + "_p0.csv"

                # Retrieve data
                metric_data = pd.read_csv(path_to_metric)

                T = metric_data["Epoch"]
                if training:
                    plot_metric = metric_data[f"Training {metric}"]
                else:
                    plot_metric = metric_data[f"Test {metric}"]

                if metric == "acc":
                    plot_metric = 100*plot_metric.to_numpy()

                # Plot training and test score
                c = next(color)

                adv_label = fr"{int(100*several_adv[k]/world_size)}% {several_name}"
                ax[i][j].plot(T, plot_metric, label=adv_label, color=c)

            # Legends
            if i == nrows-1:
                ax[i][j].set_xlabel("Epoch")
            if j == 0:
                y_label_name = metric.capitalize()
                if metric == "acc":
                    y_label_name += "uracy (%)"
                ax[i][j].set_ylabel(y_label_name)
            if i == 0:
                title_name = optim_name.capitalize()
                if "sgd" in optim_name:
                    title_name = title_name[:-3] + title_name[-3:].upper()
                ax[i][j].set_title(title_name)
            ax[i][j].legend()

    # Figure suptitle
    if include_title:
        fig.suptitle(f"{several_adv} blind, {single_adv} Byz. / {world_size}")

    # Saving the graph
    path_to_file = results_folder + folder_partname + "n" + str(world_size)
    path_to_file += several_str + str(several_adv[0])
    for k in range(1, n_plots):
        path_to_file += "_" + str(several_adv[k])
    path_to_file += single_str + str(single_adv)

    for j in range(n_optims):
        path_to_file += "_" + optim_names[j]
    for i in range(n_metrics):
        path_to_file += "_" + metrics[i]
    path_to_file += ".jpg"

    plt.tight_layout()
    plt.savefig(path_to_file)
    plt.close()


if __name__ == "__main__":

    # Command lines
    parser_desc = """Main file to plot comparisons of distributed optimizers."""
    parser = argparse.ArgumentParser(description=parser_desc, add_help=False)

    parser.add_argument("-h",
                        "--help",
                        action="help",
                        default=argparse.SUPPRESS,
                        help="If selected, will show the help message and exit.")

    # Number of processes
    parser.add_argument("nprocs",
                        type=int,
                        help="""
                             Number of processes. This number contains healthy workers as well \
                             as eventual adversaries to be set up with optional arguments.
                             """)

    # Data type and corresponding neural network & optimizer
    parser.add_argument("dataset",
                        type=str,
                        help="""
                             Name of the dataset from "linreg", "logreg", "mnist" and "imagenet". \
                             The "linreg" and "logreg" datasets are generated internally during \
                             the runs and can be used to demonstrate the algorithm. The "mnist" \
                             dataset contains 50,000 training and 10,000 test images of size \
                             28x28x1 representing handwritten digits. The "imagenet" dataset \
                             contains more than 1,000,000 training and 100,000 test images of \
                             1,000 different classes.
                             """)

    parser.add_argument("-n",
                        "--net",
                        type=str,
                        help="""
                             Name of the neural network from "linregnet", "logregnet", "torchnet, \
                             "mnistnet", "resnet18" and "resnet50". "linregnet" is only \
                             compatible with linreg dataset, as well as "logregnet" with logreg. \
                             "torchnet" and "mnistnet" are compatible with mnist dataset, while \
                             "resnet18" and "resnet50" are compatible with imagenet. Default: \
                             "mnistnet".
                             """,
                        default="mnistnet")

    parser.add_argument("optimizers",
                        type=lambda s: [x.lower() for x in s.split(",")],
                        help="""
                             Names of optimizers from "distsgd", "signsgd" and "signum" and \
                             separated with commas. "distsgd" is the default stochastic gradient \
                             descent with a distributed support, "signsgd" is the distributed \
                             optimizer detailed in the paper and "signum" is equivalent to \
                             "signsgd" with a momentum parameter. Example: "distsgd,signsgd".
                             """)

    # Graph parameters
    parser.add_argument("--title",
                        type=str,
                        help="""
                             If True, will add a title to the graph with the numbers of \
                             adversaries in the different runs. Default: False.
                             """,
                        default="False")

    # Wanted metrics
    parser.add_argument("metrics",
                        type=lambda s: [x.lower() for x in s.split(",")],
                        help="""
                             Names of metrics to plot, from "acc" and "loss" and separated with \
                             commas. Example: "acc,loss".
                             """)

    # Sub command lines for mode
    subparsers = parser.add_subparsers(dest="subcommands",
                                       description="""
                                                   Mode of evaluation from "plot" and \
                                                   "comparison". "plot" will create a graph of \
                                                   one distributed setting, if it exists. \
                                                   "comparison" will take several numbers of \
                                                   blind adversaries or several numbers of \
                                                   Byzantine adversaries and plot a comparison of \
                                                   chosen metrics.
                                                   """)

    # Plot option: an unique distributed setting and training/test/optimizer comparison
    parser_plot = subparsers.add_parser("plot",
                                        help="""Plot metrics for a given run.""")

    parser_plot.add_argument("-i",
                             "--blind-inv",
                             type=int,
                             help="""
                                  Number of blind adversaries inverting their gradients signs. \
                                  Default: 0.
                                  """,
                             default=0)

    parser_plot.add_argument("-b",
                             "--byzantine",
                             type=int,
                             help="""
                                  Number of Byzantine adversaries. This number must not be \
                                  greater than the number of processes minus the number of blind \
                                  adversaries. Default: 0.
                                  """,
                             default=0)

    # Comparison option: several distributed settings and optimizer comparison
    parser_comp_inv = subparsers.add_parser("comparison",
                                            help="""Compare metrics for several runs.""")

    parser_comp_inv.add_argument("--blind-inv-list",
                                 type=lambda s: [int(x) for x in s.split(",")],
                                 help="""
                                      Numbers of blind adversaries separated with commas. \
                                      Example: 0,1,2,3,4. Default: 0.
                                      """,
                                 default="0")

    parser_comp_inv.add_argument("--byzantine-list",
                                 type=lambda s: [int(x) for x in s.split(",")],
                                 help="""
                                      Numbers of Byzantine adversaries separated with commas. \
                                      Example: 0,1,2,3,4. Default: 0.
                                      """,
                                 default="0")

    # Parse parameters
    args = parser.parse_args()

    world_size = args.nprocs

    data_type = args.dataset.lower()
    model_name = args.net.lower()
    optim_names = args.optimizers

    include_title = ut.str2bool(args.title)

    # Control parameters format and compatibility
    task = "classification"
    for i, optim_name in enumerate(optim_names):
        data_type, model_name, optim_names[i], task = \
            ut.check_nn(data_type, model_name, optim_name)

    metrics = ut.check_plot(args.metrics, task)

    # Path parameters
    folder_partname = data_type + "_" + model_name + "_"

    # Mode selection
    mode = args.subcommands

    # Plot or compare metrics
    if mode == "plot":

        # Establish number of processes that are blind adversaries
        blind_size = args.blind_inv

        # Establish number of processes that are Byzantine adversaries
        byz_size = args.byzantine

        # Control parameters format and compatibility
        world_size, blind_size, byz_size = ut.check_proc(world_size, blind_size, byz_size)

        # Summary of the experiment
        summary = f"In total, there were {world_size} processes,"
        summary += f" of which {blind_size} were blind adversaries"
        summary += f" and {byz_size} were Byzantine adversaries."
        summary += f""" Optimizers were {",".join([str(o) for o in optim_names])}."""
        print(summary)

        # Plot and save metrics evolution
        plot_save_metrics(metrics, world_size, blind_size, byz_size, folder_partname,
                          optim_names, include_title=include_title)

    elif mode == "comparison":

        # Establish numbers of processes that are blind adversaries
        blind_sizes = args.blind_inv_list

        # Establish numbers of processes that are Byzantine adversaries
        byz_sizes = args.byzantine_list

        # Control parameters format and compatibility
        for i, _ in enumerate(blind_sizes):
            for j, _ in enumerate(byz_sizes):
                world_size, blind_sizes[i], byz_sizes[j] = \
                    ut.check_proc(world_size, blind_sizes[i], byz_sizes[j])

        # Summary of the experiments
        summary = f"In total, there were {world_size} processes,"
        summary += f""" of which {",".join([str(i) for i in blind_sizes])} were blind adversaries"""
        summary += f""" and {",".join([str(b) for b in byz_sizes])} were Byzantine adversaries."""
        summary += f""" Optimizers were {",".join([str(o) for o in optim_names])}."""
        print(summary)

        # Compare and save metrics evolution
        compare_save_metrics(metrics, world_size, blind_sizes, byz_sizes, folder_partname,
                             optim_names, training=False, include_title=include_title)

    else:

        raise ValueError(f"Unknown mode {mode}. Please choose from available modes.")
