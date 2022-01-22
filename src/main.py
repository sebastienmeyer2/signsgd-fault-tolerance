"""Main file to train and evaluate models with distributed optimizers."""


import os

from typing import List

import argparse

import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


import utils as ut
from datasets import PartitionGenerator
from dist_training import train_eval_dist


def run(rank: int, world_size: int, blind_list: List[int], byz_list: List[int], seed: int,
        data_type: str, model_name: str, optim_name: str, lr: float = 1e-3,
        lr_decay_step: int = 30, lr_decay_rate: float = 0.1, momentum: float = 0.0,
        weight_decay: float = 0.0, n_epochs: int = 10, task: str = "classification",
        save_loss: bool = True, save_acc: bool = True, folder_name: str = "tmp",
        save_step: int = 10, verbose: int = 1):
    """Run experiment.

    Parameters
    ----------
    rank : int
        Rank of current process.

    world_size : int
        Total amount of processes.

    blind_list : list of int
        Ranks of blind adversaries inverting their gradients signs.

    byz_list : list of int
        Ranks of Byzantine adversaries.

    seed : int
        Seed to use everywhere for reproducibility.

    data_type : {"logreg", "linreg", "mnist", "imagenet"}
        Name of the dataset.

    model_name : {"logregnet", "linregnet", "torchnet", "mnistnet", "resnet18", "resnet50"}
        Name of the neural network to use for training.

    optim_name : {"distsgd", "signsgd", "signum"}
        Name of the optimizer.

    lr : float, default=0.0001
        Learning rate for the optimizer.

    lr_decay_step : int, default=30
        Number of steps between each modification of the learning rate.

    lr_decay_rate : float, default=0.1
        Value by which the learning rate is multiplied every *lr_decay_step*.

    momentum : float, default=0.0
        Momentum for the optimizer.

    weight_decay : float, default=0.0
        Weight decay for the optimizer.

    n_epochs : int, default=10
        Number of training epochs.

    task : {"classification", "regression"}, default="classification"
        If "classification" task, will compute the accuracy, otherwise only the loss.

    save_loss : bool, default=True
        If True, training and test loss for each epoch and rank will be saved in csv files.

    save_acc : bool, default=True
        If True, training and test accuracy for each epoch and rank will be saved in csv files.

    save_step : int, default=10
        Number of steps between each save of the metrics. A smaller step is convenient for longer
        training in order not to lose all the work in case of early stopping.

    folder_name : str, default="tmp"
        Specific name for the folder to save scores results.

    verbose : {0, 1, 2}, default=1
        If 2, will print all metrics of all processes along epochs. If 1, will only show a
        progress bar for the rank 0 process with server's scores. If 0, will only show a simple
        progress bar.
    """
    # Initialize the process
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12350"

    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Fix seed
    ut.set_seed(seed)

    # Get training and test sets
    batch_size = int(128 / float(world_size))  # prevent too big batches

    data = PartitionGenerator(data_type, shuffle=True)
    trainset, testset = data.get_partition(rank, world_size)

    # Model & Optimizer initialization
    model = ut.init_model(model_name)
    loss_fn = ut.init_loss(data_type)
    optimizer = ut.init_optim(optim_name, model, blind_list, byz_list, lr=lr, momentum=momentum,
                              weight_decay=weight_decay)

    # Train the process
    train_eval_dist(rank, world_size, trainset, testset, model, loss_fn, optimizer,
                    lr_decay_step=lr_decay_step, lr_decay_rate=lr_decay_rate,
                    batch_size=batch_size, n_epochs=n_epochs, task=task, save_loss=save_loss,
                    save_acc=save_acc, folder_name=folder_name, save_step=save_step,
                    verbose=verbose)

    # Clean the process
    dist.destroy_process_group()


if __name__ == "__main__":

    if not torch.distributed.is_available():
        error_msg = "PyTorch support for distributed applications is not enabled on your machine."
        raise ValueError(error_msg)

    # Command lines
    parser_desc = """Main file to train and evaluate models with distributed optimizers."""
    parser = argparse.ArgumentParser(description=parser_desc, add_help=False)

    parser.add_argument("-h",
                        "--help",
                        action="help",
                        default=argparse.SUPPRESS,
                        help="If selected, will show the help message and exit.")

    # Number of processes and adversaries
    parser.add_argument("nprocs",
                        type=int,
                        help="""
                             Number of processes. This number contains healthy workers as well \
                             as eventual adversaries to be set up with optional arguments.
                             """)

    parser.add_argument("-i",
                        "--blind-inv",
                        type=int,
                        help="""
                             Number of blind adversaries inverting their gradients signs. \
                             Default: 0.
                             """,
                        default=0)

    parser.add_argument("-b",
                        "--byzantine",
                        type=int,
                        help="""
                             Number of Byzantine adversaries. This number must not be greater \
                             than the number of processes minus the number of blind adversaries. \
                             Default: 0.
                             """,
                        default=0)

    # Seed selection
    parser.add_argument("-s",
                        "--seed",
                        type=int,
                        help="""The seed to use everywhere for reproducibility. Default: 42.""",
                        default=42)

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
                             Name of the neural network from "linregnet", "logregnet", "torchnet", \
                             "mnistnet", "resnet18" and "resnet50". "linregnet" is only \
                             compatible with linreg dataset, as well as "logregnet" with logreg. \
                             "torchnet" and "mnistnet" are compatible with mnist dataset, while \
                             "resnet18" and "resnet50" are compatible with imagenet. Default: \
                             "mnistnet".
                             """,
                        default="mnistnet")

    parser.add_argument("-o",
                        "--optimizer",
                        type=str,
                        help="""
                             Name of the optimizer from "distsgd", "signsgd" and "signum". \
                             "distsgd" is the default stochastic gradient descent with a \
                             distributed support, "signsgd" is the distributed optimizer detailed \
                             in the paper and "signum" is equivalent to "signsgd" with a momentum \
                             parameter. Default: "signum".
                             """,
                        default="signum")

    # Training parameters
    parser.add_argument("-e",
                        "--epochs",
                        type=int,
                        help="""Number of training epochs. Default: 10.""",
                        default=10)

    parser.add_argument("--lr",
                        type=float,
                        help="""Learning rate. Default: 0.001.""",
                        default=1e-3)

    parser.add_argument("--lr-decay-step",
                        type=int,
                        help="""
                             Number of steps between each modification of the learning rate. \
                             Default: 30.
                             """,
                        default=30)

    parser.add_argument("--lr-decay-rate",
                        type=float,
                        help="""
                             Value by which the learning rate is multiplied every lr_decay_step. \
                             Default: 0.1.
                             """,
                        default=0.1)

    parser.add_argument("--momentum",
                        type=float,
                        help="""
                             Momentum parameter. Only available for "distsgd" and "signum" \
                             optimizers. Default: 0.
                             """,
                        default=0.0)

    parser.add_argument("--weight-decay",
                        type=float,
                        help="""
                             Weight decay. Only available for "signsgd" and "signum" optimizers. \
                             Default: 0.
                             """,
                        default=0.0)

    # Saving results
    parser.add_argument("--loss",
                        type=str,
                        help="""
                             If True, training and test loss for each epoch will be saved in csv \
                             files. Verbose argument allows to select if files have to be written \
                             only for the mean scores or for each process. Default: True.
                             """,
                        default="True")

    parser.add_argument("--acc",
                        type=str,
                        help="""
                             If True, training and test accuracy for each epoch will be saved in \
                             csv files. Verbose argument allows to select if files have to be \
                             written only for the mean scores or for each process. Default: True.
                             """,
                        default="True")

    parser.add_argument("--save-step",
                        type=int,
                        help="""
                             Number of steps between each save of the metrics. A smaller step is \
                             convenient for longer training in order not to lose all the work in \
                             case of early stopping. Default: 10.
                             """,
                        default=10)

    parser.add_argument("-v",
                        "--verbose",
                        type=int,
                        help="""
                             If 2, will print all metrics of all processes along epochs. If 1, \
                             will show a progress bar with the mean scores. If 0, will show a raw \
                             progress bar. If loss or acc saving options are True, a verbose of 1 \
                             will allow writing files only for the mean scores while a verbose of \
                             2 will allow writing files for each process. Default: 1.
                             """,
                        default=1)

    # Parse parameters
    args = parser.parse_args()

    seed = args.seed

    world_size = args.nprocs
    blind_size = args.blind_inv
    byz_size = args.byzantine

    n_epochs = args.epochs
    lr = args.lr
    lr_decay_step = args.lr_decay_step
    lr_decay_rate = args.lr_decay_rate
    momentum = args.momentum
    weight_decay = args.weight_decay

    data_type = args.dataset.lower()
    model_name = args.net.lower()
    optim_name = args.optimizer.lower()

    save_loss = ut.str2bool(args.loss)
    save_acc = ut.str2bool(args.acc)
    save_step = args.save_step
    verbose = args.verbose

    # Control parameters format and compatibility
    world_size, blind_size, byz_size = ut.check_proc(world_size, blind_size, byz_size)
    data_type, model_name, optim_name, task = ut.check_nn(data_type, model_name, optim_name)

    # Set global seed
    ut.set_seed(seed)

    # Establish which processes are blind adversaries that invert their gradients signs
    blind_list = list(np.random.choice(world_size, size=blind_size, replace=False))

    # Establish which processes are Byzantine adversaries
    byz_list = list(np.random.choice([rank for rank in range(world_size) if rank not in blind_list],
                                     size=byz_size, replace=False))

    # Summary of the experiment
    if verbose in {1, 2}:

        summary = f"In total, there are {world_size} processes,"
        summary += f" of which {blind_size} are blind adversaries"
        summary += f" and {byz_size} are Byzantine adversaries."
        summary += f" Optimizer is {optim_name}."
        print(summary)

    # Path parameters
    folder_name = data_type + "_" + model_name + "_" + optim_name

    # Run experiment
    processes = []
    mp.set_start_method("spawn")

    for rank in range(world_size):

        run_args = (rank, world_size, blind_list, byz_list, seed, data_type, model_name,
                    optim_name, lr, lr_decay_step, lr_decay_rate, momentum, weight_decay, n_epochs,
                    task, save_loss, save_acc, folder_name, save_step, verbose)
        p = mp.Process(target=run, args=run_args)
        p.start()

        processes.append(p)

    for p in processes:

        p.join()
