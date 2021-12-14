"""Main file to train and evaluate models with distributed optimizers such as Signum."""


# Import Python packages
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp
from typing import Callable

# Import our own packages
import utils as ut
from model_training import train_process


def run(fn: Callable, world_size: int, blind_inv_adv: np.ndarray, byz_adv: np.ndarray, seed: int,
        data_type: str, model_name: str, optim_name: str, n_epochs: int = 10,
        save_score: bool = True, verbose: bool = True):
    """
    Create processes and run models.

    Parameters
    ----------
    fn : Callable
        A function containing all steps to proceed with.

    world_size : int
        Total amount of processes

    blind_inv_adv : np.ndarray
        Array containing the ranks of blind adversaries that invert their gradients signs.

    byz_adv : np.ndarray
        Ranks of Byzantine adversaries.

    seed : int
        Global seed.

    data_type : {"MNIST", "ImageNet", "LinReg", "LogReg"}
        Name of the dataset.

    model_name : {"TorchNet", "MNISTNet", "ResNet18", "ResNet50", "LinRegNet", "LogRegNet"}
        Name of the neural network to use for training. "TorchNet" and "MNISTNet" are compatible
        with "MNIST" dataset, while "ResNetX" are compatible with "ImageNet". "LinRegNet" is only
        compatible with "LinReg" dataset, as well as "LogRegNet" and "LogReg".

    optim_name : {"DistSGD", "Signum", "SignSGD"}
        Name of the optimizer.

    n_epochs : int, default=10
        Number of training epochs.

    save_score : bool, default=True
        If True, scores (loss and accuracy) for each epoch will be saved in csv files.

    verbose : boool, default=True
        If True, will print all metrics of all processes along epochs. If False, will only show a
        progress bar for the rank 0 process.
    """
    fn_args = (world_size, blind_inv_adv, byz_adv, seed, data_type, model_name, optim_name,
               n_epochs, save_score, verbose)

    mp.spawn(train_process, args=fn_args, nprocs=world_size, join=True, start_method="spawn")


if __name__ == "__main__":

    if not torch.distributed.is_available():
        error_msg = "PyTorch support for distributed applications is not enabled on your machine."
        raise ValueError(error_msg)

    # Command lines
    parser_desc = "Main file to train and evaluate the SignSGD optimizer."
    parser = argparse.ArgumentParser(description=parser_desc)

    # Number of processes
    parser.add_argument("nprocs",
                        type=int,
                        help="""
                             Choose the total amount of processes to run in parallel. This number \
                             contains healthy workers aswell as possible adversaries to be set up \
                             with optional arguments.
                             """)

    parser.add_argument("-i",
                        "--blindinv",
                        type=int,
                        help="""
                             Choose a NUMBER of blind adversaries inverting their gradients \
                             signs. Default: 0.
                             """,
                        default=0)

    parser.add_argument("-b",
                        "--byzantine",
                        type=int,
                        help="""
                             Choose a NUMBER of Byzantine adversaries. This number must not be \
                             greater than nprocs minus the number of blind adversaries. Default: 0.
                             """,
                        default=0)

    # Seed selection
    parser.add_argument("-s",
                        "--seed",
                        type=int,
                        help="""Choose a global seed shared by all processes. Default: 42.""",
                        default=42)

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
                             is only compatible with LinReg dataset, as well as "LogRegNet" and \
                             LogReg. Default: "MNISTNet".
                             """,
                        choices=["TorchNet", "MNISTNet", "ResNet18", "ResNet50", "LinRegNet",
                                 "LogRegNet"],
                        default="MNISTNet")

    parser.add_argument("-o",
                        "--optimizer",
                        type=str,
                        help="""Choose an optimizer. Default: "Signum".""",
                        choices=["DistSGD", "Signum", "SignSGD"],
                        default="Signum")

    # Number of training epochs
    parser.add_argument("-e",
                        "--epochs",
                        type=int,
                        help="""Number of training epochs. Default: 10.""",
                        default=10)

    # Saving results
    parser.add_argument("-m",
                        "--metrics",
                        type=str,
                        help="""Choose whether to save scores after each epoch. Default: True.""",
                        default="True")

    parser.add_argument("-v",
                        "--verbose",
                        type=str,
                        help="""
                             If True, will print all metrics of all processes along epochs. \
                             If False, will only show a progress bar for the rank 0 process.
                             Default: False.
                             """,
                        default="False")

    # End of command lines
    args = parser.parse_args()

    # Set global seed
    seed = args.seed
    ut.set_seed(seed)

    # Data and neural network
    data_type = args.dataset
    model_name = args.net
    optim_name = args.optimizer

    data_type, model_name, optim_name = ut.check_params(data_type, model_name, optim_name)

    # Number of processes
    world_size = args.nprocs
    print("In total, there are {} processes.".format(world_size))

    # Establish which processes are blind adversaries that invert their gradients signs
    blind_inv_size = args.blindinv
    print("{} blind adversaries are inverting their gradients signs.".format(blind_inv_size))
    blind_inv_adv = np.random.choice(world_size, size=blind_inv_size)

    # Establish which processes are Byzantine adversaries
    byz_adv_size = args.byzantine
    print("{} Byzantine adversaries can send arbitrary values.".format(byz_adv_size))
    byz_adv = np.random.choice([rank for rank in range(world_size) if rank not in blind_inv_adv],
                               size=byz_adv_size)

    # Training parameters
    n_epochs = args.epochs
    save_score = ut.str2bool(args.metrics)
    verbose = ut.str2bool(args.verbose)

    run(train_process, world_size, blind_inv_adv, byz_adv, seed, data_type, model_name, optim_name,
        n_epochs=n_epochs, save_score=save_score, verbose=verbose)
