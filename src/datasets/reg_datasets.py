"""Contain a simple linear and logistic regression datasets parameters estimation.."""


from typing import Callable, Optional, Tuple

import numpy as np
from numpy import ndarray
from scipy.linalg import toeplitz
from sklearn.preprocessing import StandardScaler

import torch
from torch import Tensor
from torch.utils.data import Dataset


class LinRegDataset(Dataset):
    """Dataset of generated linear regression data."""

    def __init__(self, data: Tuple[ndarray, ...], transform: Optional[Callable] = None):
        """Initialize a dataset of linear regression data.

        Parameters
        ----------
        data : tuple of numpy.ndarray
            Tuple containing the samples and corresponding labels.

        transform : callable, optional
            A transform to apply to the data.
        """
        X, y = data
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

        self.transform = transform

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns
        -------
        length : int
            Length of the dataset.
        """
        length = len(self.y)

        return length

    def __getitem__(self, index) -> Tuple[Tensor, ...]:
        """Return data point(s) at position **idx**.

        Parameters
        ----------
        index : int or list of int
            Index or list of indices of selected data points.
        """
        samples = self.X[index]
        labels = self.y[index].flatten()

        return samples, labels


class LogRegDataset(Dataset):
    """Dataset of generated logistic regression data."""

    def __init__(self, data: Tuple[ndarray, ...], transform: Optional[Callable] = None):
        """Initialize a dataset of logistic regression data.

        Parameters
        ----------
        data : tuple of numpy.ndarray
            Tuple containing the samples and corresponding labels.

        transform : callable, optional
            A transform to apply to the data.
        """
        X, y = data
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

        self.transform = transform

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns
        -------
        length : int
            Length of the dataset.
        """
        length = len(self.y)

        return length

    def __getitem__(self, index) -> Tuple[Tensor, ...]:
        """Return data point(s) at position **idx**.

        Parameters
        ----------
        index : int or list of int
            Index or list of indices of selected data points.
        """
        samples = self.X[index]
        labels = self.y[index].flatten()

        return samples, labels


def simu_linreg(n_features: int = 20, n_samples: int = 10000, corr: float = 0.3,
                std: float = 0.5) -> Tuple[Tuple[ndarray, ...], ...]:
    """Simulation of a linear regression model with Gaussian features.

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
    train_set : tuple of numpy.ndarray
        Training data.

    test_set : tuple of numpy.ndarray
        Test data.
    """
    # Weight creation
    w0 = np.random.normal(loc=2, scale=1, size=n_features)

    # Construction of a covariance matrix
    cov = toeplitz(corr ** np.arange(0, n_features))

    # Simulation of features
    X = np.random.multivariate_normal(np.zeros(n_features), cov, size=n_samples)

    # Simulation of the labels
    y = X.dot(w0) + std * np.random.randn(n_samples)

    # Create training and test sets
    train_size = int(0.8*n_samples)

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Rescaling the data
    sc = StandardScaler()

    X_y_train = np.concatenate((X_train, np.reshape(y_train, (len(y_train), 1))), axis=1)
    X_y_test = np.concatenate((X_test, np.reshape(y_test, (len(y_test), 1))), axis=1)

    X_y_train = sc.fit_transform(X_y_train)
    X_y_test = sc.transform(X_y_test)

    X_train, X_test = X_y_train[:, :-1], X_y_test[:, :-1]
    y_train, y_test = X_y_train[:, -1], X_y_test[:, -1]

    # Dataset format
    train_set = (X_train, y_train)
    test_set = (X_test, y_test)

    return train_set, test_set


def simu_logreg(n_features: int = 20, n_samples: int = 10000,
                corr: float = 0.3) -> Tuple[Tuple[ndarray, ...], ...]:
    """Simulation of a logistic regression model with Gaussian features and a Toeplitz covariance.

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
    train_set : tuple of numpy.ndarray
        Training data.

    test_set : tuple of numpy.ndarray
        Test data.
    """
    # Weight creation
    w0 = np.random.normal(loc=2, scale=1, size=n_features)

    # Construction of a covariance matrix
    cov = toeplitz(corr ** np.arange(0, n_features))

    # Simulation of features
    X = np.random.multivariate_normal(np.zeros(n_features), cov, size=n_samples)

    # Simulation of the labels
    p = sigmoid(X.dot(w0))
    y = np.random.binomial(1, p, size=n_samples)

    # Create training and test sets
    train_size = int(0.8*n_samples)

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Rescaling the data
    sc = StandardScaler()

    X_y_train = np.concatenate((X_train, np.reshape(y_train, (len(y_train), 1))), axis=1)
    X_y_test = np.concatenate((X_test, np.reshape(y_test, (len(y_test), 1))), axis=1)

    X_y_train = sc.fit_transform(X_y_train)
    X_y_test = sc.transform(X_y_test)

    X_train, X_test = X_y_train[:, :-1], X_y_test[:, :-1]
    y_train, y_test = X_y_train[:, -1], X_y_test[:, -1]

    # Dataset format
    train_set = (X_train, y_train)
    test_set = (X_test, y_test)

    return train_set, test_set


def sigmoid(t: ndarray) -> ndarray:
    """Sigmoid function.

    Parameters
    ----------
    t : numpy.ndarray
        Inputs.

    Returns
    -------
    sig_t : numpy.ndarray
        Sigmoid of inputs.
    """
    sig_t = np.zeros(t.shape)

    # Separate where t is nonnegative
    pos_indices = t > 0

    sig_t[pos_indices] = 1/(1 + np.exp(-t[pos_indices]))
    exp_t = np.exp(t[~pos_indices])
    sig_t[~pos_indices] = exp_t/(1 + exp_t)

    return sig_t
