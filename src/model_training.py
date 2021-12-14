"""Training and evaluation functions are gathered in this file."""


# Import Python packages
import os
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from typing import List, Union
from tqdm import tqdm

# Import our own packages
import utils as ut
from nn import TorchNet, MNISTNet, ResNet18, ResNet50, LinRegNet, LogRegNet
from optim import DistSGD, Signum


def train_process(rank: int, world_size: int, blind_inv_adv: np.ndarray, byz_adv: np.ndarray,
                  seed: int, data_type: str, model_name: str, optim_name: str, n_epochs: int = 10,
                  save_score: bool = True, verbose: bool = True):
    """
    Setup a process of given **rank** and train the local model.

    Parameters
    ----------
    rank : int
        Rank of current process.

    world_size : int
        Total amount of processes.

    blind_inv_adv : numpy.ndarray
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

    optim_name : {"DistSGD", "Signum"}
        Name of the optimizer.

    n_epochs : int, default=1
        Number of training epochs.

    save_score : bool, default=True
        If True, scores (loss and accuracy) for each epoch will be saved in csv files.

    verbose : boool, default=True
        If True, will print all metrics of all processes along epochs. If False, will only show a
        progress bar for the rank 0 process.

    Raises
    ------
    ValueError
        If the name of the model is not contained in the specified choices.
    """
    # Initialize the process
    setup(rank, world_size)

    # Fix seed
    ut.set_seed(seed)

    # Start training
    bs = int(128 / float(world_size))  # prevent too big batches

    train_set, test_set = ut.build_train_test_set(rank, world_size, data_type, bs, shuffle=True)

    model: Union[TorchNet, MNISTNet, ResNet18, ResNet50, LinRegNet]
    if model_name == "TorchNet":
        model = TorchNet(10)
    elif model_name == "MNISTNet":
        model = MNISTNet(10)
    elif model_name == "ResNet18":
        model = ResNet18(1000)
    elif model_name == "ResNet50":
        model = ResNet50(1000)
    elif model_name == "LinRegNet":
        model = LinRegNet()
    elif model_name == "LogRegNet":
        model = LogRegNet()
    else:
        raise ValueError("Unsupported nn {} in training function.".format(model_name))

    optimizer: Union[DistSGD, Signum]
    if optim_name == "DistSGD":
        optimizer = DistSGD(model.parameters(), blind_inv_adv, byz_adv, lr=1e-4)
    elif optim_name == "Signum":
        optimizer = Signum(model.parameters(), blind_inv_adv, byz_adv, lr=1e-4, momentum=0.9,
                           weight_decay=0.)
    elif optim_name == "SignSGD":
        optimizer = Signum(model.parameters(), blind_inv_adv, byz_adv, lr=1e-4, momentum=0.,
                           weight_decay=0.)
    else:
        raise ValueError("Unsupported optimizer {} in training function.".format(optim_name))

    loss_fn: Union[torch.nn.MSELoss, torch.nn.NLLLoss]
    if data_type == "LinReg":
        loss_fn = torch.nn.MSELoss()
    elif data_type in {"MNIST", "ImageNet", "LogReg"}:
        loss_fn = torch.nn.NLLLoss()
    else:
        raise ValueError("No loss function for dataset {} in training function.".format(data_type))

    train_loss_value = 0.
    train_loss_list: List[float] = []
    test_loss_value = 0.
    test_loss_list = []

    train_loss_epoch = 0.
    test_loss_epoch = 0.
    loss_list = []

    train_acc_value = 0.
    train_acc_list = []
    test_acc_value = 0.
    test_acc_list = []

    train_acc_epoch = 0.
    test_acc_epoch = 0.
    acc_list = []

    if verbose or rank == 0:
        values = (rank, world_size-1, 0, 0.)
        pbar = tqdm(range(n_epochs), desc="Rank {}/{}; Epoch {}; Test loss {}".format(*values))
    else:
        pbar = range(n_epochs)
    for epoch in pbar:

        # Training step
        for x_train, y_train in train_set:

            # Setting grad to zero
            optimizer.zero_grad()

            # Model output & loss
            y_lsm_pred = model(x_train)
            # print(y_lsm_pred, y_train.flatten())
            if data_type == "LinReg":
                train_loss = loss_fn(y_lsm_pred.flatten(), y_train.flatten())
            else:
                train_loss = loss_fn(y_lsm_pred, y_train.flatten())

            # Compute scores on training
            train_loss_value = train_loss.item()

            train_loss_list.append(train_loss_value)

            if data_type in {"MNIST", "ImageNet", "LogReg"}:
                y_pred = torch.argmax(y_lsm_pred, dim=1)
                train_acc_value = (y_pred == y_train).float().mean()
                train_acc_list.append(train_acc_value)

            # Optimization step
            train_loss.backward()

            optimizer.step()

        # After each epoch of training, we evaluate the model, which is common to all processes
        if rank == 0:
            for x_test, y_test in test_set:

                # Model output & loss
                y_lsm_pred = model(x_test)

                if data_type == "LinReg":
                    test_loss = loss_fn(y_lsm_pred.flatten(), y_test.flatten())
                else:
                    test_loss = loss_fn(y_lsm_pred, y_test.flatten())

                # Compute scores on test
                test_loss_value = test_loss.item()
                test_loss_list.append(test_loss_value)

                if data_type in {"MNIST", "ImageNet", "LogReg"}:
                    y_pred = torch.argmax(y_lsm_pred, dim=1)
                    test_acc_value = (y_pred == y_test).float().mean()
                    test_acc_list.append(test_acc_value)

            test_loss_epoch = np.mean(test_loss_list)
            test_loss_list = []

            if data_type in {"MNIST", "ImageNet", "LogReg"}:
                test_acc_epoch = np.mean(test_acc_list)
                test_acc_list = []

        dist.barrier()  # ensure every process are here after evaluation

        train_loss_epoch = np.mean(train_loss_list)
        train_loss_list = []
        loss_list.append([epoch+1, train_loss_epoch, test_loss_epoch])

        if data_type in {"MNIST", "ImageNet", "LogReg"}:
            train_acc_epoch = np.mean(train_acc_list)
            train_acc_list = []
            acc_list.append([epoch+1, train_acc_epoch, test_acc_epoch])

        if verbose:
            msg_worker = "Rank {}/{}; Epoch {}; Training loss {}; Test loss {}; Test accuracy {}"
            v_worker = (rank, world_size, epoch, train_loss_epoch, test_loss_epoch, test_acc_epoch)
            print(msg_worker.format(*v_worker))

        if verbose or rank == 0:
            msg_server = "Rank {}/{}; Epoch {}; Test loss {}; Test accuracy {}"
            v_server = (rank, world_size-1, epoch, test_loss_epoch, test_acc_epoch)
            pbar.set_description(msg_server.format(*v_server))

    # Saving scores
    if save_score:

        loss_df = pd.DataFrame(loss_list, columns=["Epoch", "Training loss", "Test loss"])
        acc_df = pd.DataFrame(acc_list, columns=["Epoch", "Training accuracy", "Test accuracy"])

        # Create necessary folders
        byz_adv_size = len(byz_adv)
        blind_inv_size = len(blind_inv_adv)

        scores_folder = "results/"

        if rank == 0:
            if not os.path.exists(scores_folder):
                os.makedirs(scores_folder)
        dist.barrier()

        scores_folder += data_type + "_" + model_name + "_" + optim_name + "/"

        if rank == 0:
            if not os.path.exists(scores_folder):
                os.makedirs(scores_folder)
        dist.barrier()

        scores_folder += "n" + str(world_size) + "_byz" + str(byz_adv_size) + "_inv" + \
            str(blind_inv_size) + "/"

        if rank == 0:
            if not os.path.exists(scores_folder):
                os.makedirs(scores_folder)
        dist.barrier()

        path_to_loss = scores_folder + "loss_p" + str(rank) + ".csv"
        loss_df.to_csv(path_to_loss, columns=loss_df.columns, header=True, index=False)

        # Save data
        if data_type in {"MNIST", "ImageNet", "LogReg"}:
            path_to_acc = scores_folder + "accuracy_p" + str(rank) + ".csv"
            acc_df.to_csv(path_to_acc, columns=acc_df.columns, header=True, index=False)

    # Clean the process
    cleanup()


def setup(rank: int, world_size: int, backend: str = "gloo"):
    """
    Initialize the distributed environment.

    Parameters
    ----------
    rank : int
        Rank of current process.

    world_size : int
        Total amount of processes.

    backend : str, default="gloo"
        Backend to use in the distributed environment.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    """Clean the current process group."""
    dist.destroy_process_group()
