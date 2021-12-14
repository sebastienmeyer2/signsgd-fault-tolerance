"""Gather utilitary functions for randomness control and data handling."""


# Import Python packages
from typing import List, Tuple
import argparse
import numpy as np
import torch
from numpy.random import multivariate_normal, normal, randn
from scipy.linalg.special_matrices import toeplitz
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class LinRegDataset(Dataset):
    """Dataset of generated linear regression data."""

    def __init__(self, data: Tuple[np.ndarray, ...]):
        """
        Initialize a dataset of linear regression data.

        Parameters
        ----------
        data : Tuple[np.ndarray, ...]
            Tuple containing the samples and corresponding labels.
        """
        X, y = data
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

        self.transform = None

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns
        -------
        length : int
            Length of the dataset.
        """
        length = len(self.y)

        return length

    def __getitem__(self, index) -> Tuple[torch.Tensor, ...]:
        """
        Return data point(s) at position **idx**.

        Parameters
        ----------
        index : Union[int, List[int]]
            Index or list of indices of selected data points.
        """
        samples = self.X[index]
        labels = self.y[index].flatten()

        if self.transform:
            samples = self.transform(samples)
            labels = self.transform(labels)

        return samples, labels


class LogRegDataset(Dataset):
    """Dataset of generated logistic regression data."""

    def __init__(self, data: Tuple[np.ndarray, ...]):
        """
        Initialize a dataset of logistic regression data.

        Parameters
        ----------
        data : Tuple[np.ndarray, ...]
            Tuple containing the samples and corresponding labels.
        """
        X, y = data
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y)

        self.transform = None

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns
        -------
        length : int
            Length of the dataset.
        """
        length = len(self.y)

        return length

    def __getitem__(self, index) -> Tuple[torch.Tensor, ...]:
        """
        Return data point(s) at position **idx**.

        Parameters
        ----------
        index : Union[int, List[int]]
            Index or list of indices of selected data points.
        """
        samples = self.X[index]
        labels = self.y[index].flatten()

        if self.transform:
            samples = self.transform(samples)
            labels = self.transform(labels)

        return samples, labels


class Partition(Dataset):
    """Wrapper around a dataset and indices for a partition."""

    def __init__(self, data: Dataset, indices: List[int]):
        """
        Initialize a partition of a dataset.

        Parameters
        ----------
        data : `torch.utils.data.Dataset`
            Complete dataset.

        indices : List[int]
            List of indices of the dataset that can be used in this partition.
        """
        self.data = data
        self.indices = indices

    def __len__(self) -> int:
        """
        Return the length of the partition.

        Returns
        -------
        length : int
            Length of the partition.
        """
        length = len(self.indices)
        return length

    def __getitem__(self, index) -> torch.Tensor:
        """
        Return data point(s) at position **index**.

        Parameters
        ----------
        index : Union[int, List[int]]
            Index or list of indices of selected data points.
        """
        data_idx = self.indices[index]
        return self.data[data_idx]


def set_seed(seed):
    """
    Fix seed for current run.

    Parameters
    ----------
    seed : int
        Global seeed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_train_test_set(rank: int, world_size: int, data_type: str, batch_size: int,
                         shuffle: bool = True) -> Tuple[DataLoader, ...]:
    """
    Create the training and test sets for specified **data_type** name.

    Parameters
    ----------
    rank : int
        Rank of current process.

    world_size : int
        Total amount of processes.

    data_type : {"MNIST", "ImageNet", "LinReg", "LogReg"}
        Name of the dataset.

    batch_size : int
        Batch size.

    shuffle : bool, default=True
        If True, data will be shuffled in both training and test set.

    Returns
    -------
    train_set : `torch.utils.data.DataLoader`
        Training set built from a partition of the complete dataset.

    test_set : `torch.utils.data.DataLoader`
        Test set.
    """
    # Retrieve training and test sets
    if data_type == "MNIST":

        train = datasets.MNIST("data/", train=True, download=True, transform=transforms.ToTensor())
        test = datasets.MNIST("data/", train=False, download=True, transform=transforms.ToTensor())

    elif data_type == "ImageNet":

        train = datasets.ImageNet("data/", split="train", download=True,
                                  transform=transforms.ToTensor())
        test = datasets.ImageNet("data/", split="test", download=True,
                                 transform=transforms.ToTensor())

    elif data_type == "LinReg":

        X_y_train, X_y_test = simu_linreg()
        train = LinRegDataset(X_y_train)
        test = LinRegDataset(X_y_test)

    elif data_type == "LogReg":

        X_y_train, X_y_test = simu_logreg()
        train = LogRegDataset(X_y_train)
        test = LogRegDataset(X_y_test)

    else:
        raise ValueError("Dataset {} not supported in dataset creation.".format(data_type))

    # Shuffle all the indices, here it is vital that the seed is shared among all processes
    n = len(train)
    all_indices = [i for i in range(n)]
    np.random.shuffle(all_indices)

    # Select the data
    partition_size = n // world_size
    first_idx = (rank*n) // world_size
    last_idx = first_idx + partition_size
    partition_indices = all_indices[first_idx:last_idx]

    partition = Partition(train, partition_indices)

    # DataLoader wrapper
    train_set = DataLoader(partition, batch_size=batch_size, shuffle=True)
    test_set = DataLoader(test, batch_size=batch_size, shuffle=True)

    return (train_set, test_set)


def simu_linreg(n_features: int = 20, n_samples: int = 10000, corr: float = 0.5,
                std: float = 0.5) -> Tuple[Tuple[np.ndarray, ...], ...]:
    """
    Simulation of a linear regression model with Gaussian features and a Toeplitz covariance, with
    Gaussian noise.

    Parameters
    ----------
    n_features : int
        Number of features to simulate.

    n_samples : int, default=1000
        Number of samples to simulate.

    corr : float, default=0.5
        Correlation of the features.

    std : float, default=0.5
        Standard deviation of the noise.

    Returns
    -------
    train_set : Tuple[np.ndarray]
        Training data.

    test_set : Tuple[np.ndarray]
        Test data.
    """
    # Weight creation
    w0 = normal(loc=2, scale=1, size=n_features)

    # Construction of a covariance matrix
    cov = toeplitz(corr ** np.arange(0, n_features))

    # Simulation of features
    X = multivariate_normal(np.zeros(n_features), cov, size=n_samples)

    # Simulation of the labels
    y = X.dot(w0) + std * randn(n_samples)

    # Create training and test sets
    train_size = int(0.8*n_samples)

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    train_set = (X_train, y_train)
    test_set = (X_test, y_test)

    return train_set, test_set


def sigmoid(t: np.ndarray) -> np.ndarray:
    """
    Sigmoid function.

    Parameters
    ----------
    t : np.ndarray
        Inputs.

    Returns
    -------
    sig_t : np.ndarray
        Sigmoid of inputs.
    """
    sig_t = np.zeros(t.shape)

    # Separate where t is nonnegative
    pos_indices = t > 0

    sig_t[pos_indices] = 1/(1 + np.exp(-t[pos_indices]))
    exp_t = np.exp(t[~pos_indices])
    sig_t[~pos_indices] = exp_t/(1 + exp_t)

    return sig_t


def simu_logreg(n_features: int = 20, n_samples: int = 10000,
                corr: float = 0.5) -> Tuple[Tuple[np.ndarray, ...], ...]:
    """
    Simulation of a logistic regression model with Gaussian features and a Toeplitz covariance.

    Parameters
    ----------
    n_features : int
        Number of features to simulate.

    n_samples : int, default=1000
        Number of samples to simulate.

    corr : float, default=0.5
        Correlation of the features.

    Returns
    -------
    train_set : Tuple[np.ndarray]
        Training data.

    test_set : Tuple[np.ndarray]
        Test data.
    """
    # Weight creation
    w0 = normal(loc=2, scale=1, size=n_features)

    # Construction of a covariance matrix
    cov = toeplitz(corr ** np.arange(0, n_features))

    # Simulation of features
    X = multivariate_normal(np.zeros(n_features), cov, size=n_samples)

    # Simulation of the labels
    p = sigmoid(X.dot(w0))
    y = np.random.binomial(1, p, size=n_samples)

    # Create training and test sets
    train_size = int(0.8*n_samples)

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    train_set = (X_train, y_train)
    test_set = (X_test, y_test)

    return train_set, test_set


def str2bool(v: str) -> bool:
    """
    An easy way to handle boolean options.

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
    `argparse.ArgumentTypeError`
        If the entry cannot be converted to a boolean.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def restricted_float(x: str) -> float:
    """
    An easy war to handle range for floats.

    Parameters
    ----------
    x : float
        Argument value.

    Returns
    -------
    y : float
        Terminal value if **x** is between zero and one.

    Raises
    ------
    `argparse.ArgumentTypeError`
        If **x** cannot be converted to a float or is not in specified range.
    """
    try:
        y = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("{} not a floating-point literal.".format(x))

    if y < 0. or y > 1.:
        raise argparse.ArgumentTypeError("{} not in range [0.0, 1.0].".format(y))

    return y


def check_params(data_type: str, model_name: str, optim_name: str) -> Tuple[str, ...]:
    """
    Check whether chosen parameters are compatible.

    Parameters
    ----------
    data_type : {"MNIST", "ImageNet", "LinReg", "LogReg"}
        Name of the dataset.

    model_name : {"TorchNet", "MNISTNet", "ResNet18", "ResNet50", "LinRegNet", "LogRegNet"}
        Name of the neural network to use for training. "TorchNet" and "MNISTNet" are compatible
        with "MNIST" dataset, while "ResNetX" are compatible with "ImageNet". "LinRegNet" is only
        compatible with "LinReg" dataset, as well as "LogRegNet" and "LogReg".

    optim_name : {"DistSGD", "Signum", "Signum"}
        Name of the optimizer.

    Returns
    -------
    data_type : {"MNIST", "ImageNet", "LinReg", "LogReg"}
        Name of the dataset.

    model_name : {"TorchNet", "MNISTNet", "ResNet18", "ResNet50", "LinRegNet", "LogRegNet"}
        Name of the neural network to use for training. "TorchNet" and "MNISTNet" are compatible
        with "MNIST" dataset, while "ResNetX" are compatible with "ImageNet". "LinRegNet" is only
        compatible with "LinReg" dataset, as well as "LogRegNet" and "LogReg".

    optim_name : {"DistSGD", "Signum", "SignSGD"}
        Name of the optimizer.

    Raises
    ------
    ValueError
        If the dataset or model are not supported or compatible.
    """
    # Unknown parameters
    available_nn = {"TorchNet", "MNISTNet", "ResNet18", "ResNet50", "LinRegNet", "LogRegNet"}
    if model_name not in available_nn:
        error_msg = "Neural network {} checking error. Please choose one from {}."
        raise ValueError(error_msg.format(model_name, available_nn))

    available_data = {"MNIST", "ImageNet", "LinReg", "LogReg"}
    if data_type not in available_data:
        error_msg = "Dataset {} checking error. Please choose one from {}."
        raise ValueError(error_msg.format(data_type, available_data))

    availabe_optim = {"DistSGD", "Signum", "SignSGD"}
    if optim_name not in availabe_optim:
        error_msg = "Optimizer {} is not supported. Please shoose one from {}."
        raise ValueError(error_msg.format(optim_name, availabe_optim))

    # Incompatible parameters
    if data_type == "MNIST" and model_name not in {"TorchNet", "MNISTNet"}:
        warning_msg = "WARNING: Dataset {} and model {} are not compatible. Setting model to {}."
        print(warning_msg.format(data_type, model_name, "TorchNet"))
        model_name = "TorchNet"

    if data_type == "ImageNet" and model_name not in {"ResNet18", "ResNet50"}:
        warning_msg = "WARNING: Dataset {} and model {} are not compatible. Setting model to {}."
        print(warning_msg.format(data_type, model_name, "ResNet18"))
        model_name = "ResNet18"

    if data_type == "LinReg" and model_name != "LinRegNet":
        warning_msg = "WARNING: Dataset {} and model {} are not compatible. Setting model to {}."
        print(warning_msg.format(data_type, model_name, "LinRegNet"))
        model_name = "LinRegNet"

    if data_type == "LogReg" and model_name != "LogRegNet":
        warning_msg = "WARNING: Dataset {} and model {} are not compatible. Setting model to {}."
        print(warning_msg.format(data_type, model_name, "LogRegNet"))
        model_name = "LogRegNet"

    return data_type, model_name, optim_name
