"""Training and evaluation functions are gathered in this file."""


import os

from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.distributed as dist
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset


def train_eval_dist(rank: int, world_size: int, trainset: Dataset, testset: Optional[Dataset],
                    model: Module, loss_fn: Module, optimizer: Optimizer, lr_decay_step: int = 30,
                    lr_decay_rate: float = 0.1, batch_size: int = 64, n_epochs: int = 10,
                    task: str = "classification", save_loss: bool = True, save_acc: bool = True,
                    folder_name: str = "tmp", verbose: int = 1):
    """Train the **model** with given **rank** process.

    Parameters
    ----------
    rank : int
        Rank of current process.

    world_size : int
        Total amount of processes.

    trainset : torch.utils.data.Dataset
        A dataloader instance for the training samples.

    testset : torch.utils.data.Dataset, optional
        A dataloader instance for the test samples.

    model : torch.nn.Module
        A neural network to be trained and evaluated on the datasets.

    loss_fn : torch.nn.Module
        A loss function.

    optimizer : torch.optim.Optimizer
        An optimizer to train the model with.

    lr_decay_step : int, default=30
        Number of steps between each modification of the learning rate.

    lr_decay_rate : float, default=0.1
        Value by which the learning rate is multiplied every *lr_decay_step*.

    batch_size : int, default=64
        Training and test batch size.

    n_epochs : int, default=10
        Number of training epochs.

    task : {"classification", "regression"}, default="classification"
        If "classification" task, will compute the accuracy, otherwise only the loss.

    save_loss : bool, default=True
        If True, training and test loss for each epoch and rank will be saved in csv files.

    save_acc : bool, default=True
        If True, training and test accuracy for each epoch and rank will be saved in csv files.

    folder_name : str, default="tmp"
        Specific name for the folder to save scores results.

    verbose : {0, 1, 2}, default=1
        If 2, will print all metrics of all processes along epochs. If 1, will only show a
        progress bar for the rank 0 process with server's scores. If 0, will only show a simple
        progress bar.
    """
    # Build up a dataloader
    trainloader = DataLoader(trainset, batch_size=batch_size)

    # Learning rate scheduler (improve learning)
    scheduler = StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_rate)
    current_lr = scheduler.get_last_lr()[0]

    # Metrics lists
    train_loss_value = -1
    train_loss_list = []

    train_loss_epoch = -1
    test_loss_epoch = -1
    loss_list = []

    train_acc_value = -1
    train_acc_list = []

    train_acc_epoch = -1
    test_acc_epoch = -1
    acc_list = []

    # Progress bar
    if verbose in {1, 2} and rank == 0:  # show a progress bar for the server

        pbar = tqdm(range(n_epochs), desc=f"Rank 0/{world_size-1}; LR {current_lr}; First epoch...")

    elif verbose == 0 and rank == 0:  # show a simple progress bar

        pbar = tqdm(range(n_epochs))

    else:

        pbar = range(n_epochs)

    # Start training
    for epoch in pbar:

        # Training epoch
        model.train()

        for x_train, y_train in trainloader:

            # Setting grad to zero
            optimizer.zero_grad()

            # Model output & loss
            y_lsm_pred = model(x_train)
            if task == "classification":
                train_loss = loss_fn(y_lsm_pred, y_train.flatten())
            else:
                train_loss = loss_fn(y_lsm_pred.flatten(), y_train.flatten())

            # Optimization step
            train_loss.backward()

            optimizer.step()

            # Compute scores on training set
            train_loss_value = train_loss.item()
            train_loss_list.append(train_loss_value)

            if task == "classification":
                y_pred = torch.argmax(y_lsm_pred, dim=1)
                train_acc_value = np.sum((y_pred == y_train.flatten()).detach().numpy())
                train_acc_list.append(train_acc_value)

        # Update the learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # After each training epoch, we evaluate the model, which is common to all processes
        if testset is not None and (rank == 0 or verbose == 2):

            test_loss_epoch, test_acc_epoch = eval_dist(model, testset, loss_fn,
                                                        batch_size=batch_size, task=task)

        dist.barrier()  # make sure that all the processes are here after evaluation

        # Append the computed scores
        train_loss_epoch = np.mean(train_loss_list)
        train_loss_list = []
        loss_list.append([epoch+1, train_loss_epoch, test_loss_epoch])

        if task == "classification":
            train_acc_epoch = np.sum(train_acc_list)/len(trainset)
            train_acc_list = []
            acc_list.append([epoch+1, train_acc_epoch, test_acc_epoch])

        # If we want to print training information, we have to get the scores to the server
        if verbose in {1, 2}:

            all_scores = torch.zeros([world_size, 4])

            if rank != 0:

                all_scores[rank][0] = train_loss_epoch
                all_scores[rank][1] = test_loss_epoch
                all_scores[rank][2] = train_acc_epoch
                all_scores[rank][3] = test_acc_epoch

                dist.send(all_scores[rank], 0)

            else:

                all_scores[rank][0] = train_loss_epoch
                all_scores[rank][1] = test_loss_epoch
                all_scores[rank][2] = train_acc_epoch
                all_scores[rank][3] = test_acc_epoch

                for worker in range(1, world_size):

                    dist.recv(all_scores[worker], src=worker)

            dist.barrier()  # wait for the server to know about others' scores

            # Update the progress bar to show current mean scores
            if rank == 0:

                # Compute mean training scores
                mean_train_loss_epoch = 0.
                mean_train_acc_epoch = 0.

                for worker in range(world_size):

                    mean_train_loss_epoch += all_scores[worker][0].item()
                    mean_train_acc_epoch += all_scores[worker][2].item()

                mean_train_loss_epoch /= world_size
                mean_train_acc_epoch /= world_size

                # Update the progress bar
                msg_server = f"Epoch {epoch}; LR {current_lr};"
                msg_server += f" Mean training loss {np.around(mean_train_loss_epoch, 4)};"
                msg_server += f" Test loss {np.around(test_loss_epoch, 4)};"
                if task == "classification":
                    msg_server += f" Mean training acc. {np.around(mean_train_acc_epoch, 4)};"
                    msg_server += f" Test acc. {np.around(test_acc_epoch, 4)}"
                pbar.set_description(msg_server)

            # If verbose is 2, print a summary of all processes current scores
            if verbose == 2:

                if rank == 0:

                    msg_server = "\n\n"

                    for worker in range(world_size):

                        msg_worker = f"Rank {worker}/{world_size-1}; Epoch {epoch};"
                        msg_worker += \
                            f" Training loss {np.around(all_scores[worker][0].item(), 4)};"
                        msg_worker += f" Test loss {np.around(all_scores[worker][1].item(), 4)};"
                        if task == "classification":
                            msg_worker += \
                                f" Training accuracy {np.around(all_scores[worker][2].item(), 4)};"
                            msg_worker += \
                                f" Test accuracy {np.around(all_scores[worker][3].item(), 4)}"
                        msg_server += msg_worker + "\n"

                    print(msg_server)

                dist.barrier()  # wait for the server to print the scores

    # Saving scores
    byz_size = optimizer.get_byz_size()
    blind_size = optimizer.get_blind_size()
    metrics_scores = {}

    if save_loss:
        metrics_scores["loss"] = loss_list

    if task == "classification" and save_acc:
        metrics_scores["acc"] = acc_list

    save_metrics(rank, world_size, blind_size, byz_size, metrics_scores, folder_name,
                 verbose=verbose)

    dist.barrier()  # wait for all processes to finish


def eval_dist(model: Module, testset: Dataset, loss_fn: Module, batch_size: int = 64,
              task: str = "classification") -> Tuple[float]:
    """Evaluate a model in a distributed setting.

    Parameters
    ----------
    model : torch.nn.Module
        A neural network to be evaluated.

    testset : torch.utils.data.Dataset
        A dataloader instance for the test samples.

    loss_fn : torch.nn.Module
        A loss function.

    batch_size : int, default=64
        Training and test batch size.

    task : {"classification", "regression"}, default="classification"
        If "classification" task, will compute the accuracy, otherwise only the loss.

    Returns
    -------
    test_loss : float
        Value of the loss of the **model** on test samples.

    test_acc : float
        Value of the accuracy of the **model** on test samples.
    """
    # Build a dataloader
    testloader = DataLoader(testset, batch_size=batch_size)

    model.eval()

    test_loss_list = []
    test_acc_list = []

    for x_test, y_test in testloader:

        # Model output & loss
        y_lsm_pred = model(x_test)

        if task == "classification":
            test_loss = loss_fn(y_lsm_pred, y_test.flatten())
        else:
            test_loss = loss_fn(y_lsm_pred.flatten(), y_test.flatten())

        # Compute loss on test
        test_loss_value = test_loss.item()
        test_loss_list.append(test_loss_value)

        # Compute accuracy on test
        if task == "classification":
            y_pred = torch.argmax(y_lsm_pred, dim=1)
            test_acc_value = np.sum((y_pred == y_test.flatten()).detach().numpy())
            test_acc_list.append(test_acc_value)

    test_loss = np.mean(test_loss_list) if len(test_loss_list) > 0 else -1
    test_acc = np.sum(test_acc_list)/len(testset) if len(test_acc_list) > 0 else -1

    model.train()

    return test_loss, test_acc


def save_metrics(rank: int, world_size: int, blind_size: int, byz_size: int,
                 metrics_scores: Dict[str, List[float]], folder_name: str, verbose: int = 1):
    """Write a csv file for each metric computed.

    Parameters
    ----------
    rank : int
        Rank of current process.

    world_size : int
        Total amount of processes.

    blind_size : int
        Number of blind adversaries inverting their gradient signs.

    byz_size : int
        Number of Byzantines adversaries.

    metrics_scores : dict
        A dictionary whose keys are the saved metrics and values are the list of training and test
        scores. Therefore, the values of the dictionary must be arrays of dimension 3.

    folder_name : str
        Specific name for the folder to save scores results.

    verbose : {0, 1, 2}, default=1
        If 2, will save all scores of all processes along epochs. If 0 or 1, will only save
        the scores of the server.
    """
    for metric, metric_scores in metrics_scores.items():

        metric_col = ["Epoch", f"Training {metric}", f"Test {metric}"]
        metric_df = pd.DataFrame(metric_scores, columns=metric_col)

        # Create necessary folders
        scores_folder = "results/"

        if rank == 0:
            if not os.path.exists(scores_folder):
                os.makedirs(scores_folder)
        dist.barrier()  # problems can occur if multiple processes run makedirs in parallel

        scores_folder += folder_name + "/"

        if rank == 0:
            if not os.path.exists(scores_folder):
                os.makedirs(scores_folder)
        dist.barrier()  # problems can occur if multiple processes run makedirs in parallel

        scores_folder += "n" + str(world_size) + "_byz" + str(byz_size) + "_inv" + \
            str(blind_size) + "/"

        if rank == 0:
            if not os.path.exists(scores_folder):
                os.makedirs(scores_folder)
        dist.barrier()  # problems can occur if multiple processes run makedirs in parallel

        # Save in csv format
        path_to_metric = scores_folder + metric + "_p" + str(rank) + ".csv"

        if verbose == 2 or rank == 0:

            metric_df.to_csv(path_to_metric, columns=metric_df.columns, header=True, index=False)

        dist.barrier()  # make sure that all processes have saved their scores
