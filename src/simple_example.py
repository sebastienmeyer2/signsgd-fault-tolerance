"""A simple example to show that Signum is converging."""


# Import Python packages
import numpy as np
from typing import Callable, List, Tuple
from numpy.random import multivariate_normal, randn
from scipy.linalg.special_matrices import toeplitz
from sklearn.linear_model import LinearRegression


class Worker:
    """Simulate a worker."""

    def __init__(self, x: np.ndarray, batch_size: int = 8, lr: float = 1e-4, beta: float = 0.9,
                 weight_decay: float = 0.):
        """
        Initialize a worker instance.

        Parameters
        ----------
        x : np.ndarray
            Initial value.

        batch_size : int, default=8
            Batch size.

        lr : float, default=1e-4
            Learning rate.

        beta : float, default=0.9
            Momentum.

        weight_decay : float, default=0.
            Weight decay for the optimizer.
        """
        self.x = x

        self.batch_size = batch_size

        self.beta = beta
        self.lr = lr
        self.weight_decay = weight_decay

        self.v_m = 0.

    def compute_sign(self, grad_fn: Callable, inputs: Tuple[np.ndarray, ...]) -> int:
        """Compute sign of gradients wrt a batch of data.

        Parameters
        ----------
        grad_fn : Callable
            Function that computed the gradient.

        inputs : Tuple[np.ndarray, ...]
            Input data.

        Returns
        -------
        sg_v_m : int
            Sign of gradients, as a result of Signum algorithm.
        """
        g_m = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            g_m += grad_fn(inputs, self.x)
        g_m = np.mean(g_m)

        self.v_m = (1 - self.beta)*g_m + self.beta*self.v_m
        sg_v_m = 2*(self.v_m >= 0) - 1

        return sg_v_m

    def update(self, sg_V: int):
        """
        Update **x** with computed sign.

        Parameters
        ----------
        sg_V : int
            Sign as gathered in the server.
        """
        self.x -= self.lr * (sg_V + self.weight_decay*self.x)


class Server:
    """Simulate a server instance that gathers a list of workers."""

    def __init__(self, workers: List[Worker], batch_size: int = 8):
        """
        Initialize the server.

        Parameters
        ----------
        workers : List[`Worker`]
            List of workers instances.

        batch_size : int, default=8
            Batch size.
        """
        self.workers = workers
        self.M = len(workers)

        self.batch_size = batch_size

    def aggregate_sign(self, workers_sg: np.ndarray) -> int:
        """
        Aggregate gradients signs of workers and compute final sign.

        Parameters
        ----------
        workers_sg : np.ndarray
            Gradients signs of workers.

        Returns
        -------
        sg_V : int
            Sign of aggregated workers signs.
        """
        V = 0.

        assert len(workers_sg) == self.M

        for m in range(self.M):

            V += workers_sg[m]

        sg_V = 2*(V >= 0) - 1

        return sg_V

    def step(self, grad_fn: Callable, workers_inputs: List[Tuple[np.ndarray, ...]]):
        """
        Run a step of optimization.

        Parameters
        ----------
        grad_fn : Callable
            Function that computes the gradient.

        workers_inputs : List[np.ndarray]
            Input data for each worker.
        """
        assert len(workers_inputs) == self.M

        # Compute batch gradient on each worker
        workers_sg = np.zeros(self.M)

        for m in range(self.M):

            workers_sg[m] = self.workers[m].compute_sign(grad_fn, workers_inputs[m])

        # Aggregate signs
        sg_V = self.aggregate_sign(workers_sg)

        # Update x on each worker
        for m in range(self.M):

            self.workers[m].update(sg_V)


def grad_linreg(inputs: Tuple[np.ndarray, ...], w: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of linear regression.

    Parameters
    ----------
    inputs : Tuple[np.ndarray, ...]
        Input data.

    w : np.ndarray
        Weights.

    Returns
    -------
    grad[0] : np.ndarray
        Value of the gradient at **inputs**.
    """
    X, y = inputs
    grad = 2 * X.T.dot(X.dot(w) - y)

    return grad[0]


def simu_linreg(w0: np.ndarray, n_samples: int = 1000, corr: float = 0.5,
                std: float = 0.5) -> Tuple[np.ndarray, ...]:
    """
    Simulation of a linear regression model with Gaussian features and a Toeplitz covariance, with
    Gaussian noise.

    Parameters
    ----------
    w0 : numpy.ndarray, shape=(n_features,)
        Model weights.

    n_samples : int, default=1000
        Number of samples to simulate.

    corr : float, default=0.5
        Correlation of the features.

    std : float, default=0.5
        Standard deviation of the noise.

    Returns
    -------
    X : numpy.ndarray, shape=(n_samples, n_features)
        Simulated features matrix. It contains samples of a centered Gaussian  vector with Toeplitz
        covariance.

    y : numpy.ndarray, shape=(n_samples,)
        Simulated labels.
    """
    n_features = w0.shape[0]

    # Construction of a covariance matrix
    cov = toeplitz(corr ** np.arange(0, n_features))

    # Simulation of features
    X = multivariate_normal(np.zeros(n_features), cov, size=n_samples)

    # Simulation of the labels
    y = X.dot(w0) + std * randn(n_samples)

    return X, y


if __name__ == "__main__":

    # Linear regression simulation
    n_samples = 50000
    w0 = np.array([0.5])

    X, y = simu_linreg(w0, n_samples=n_samples, corr=0.3, std=0.5)
    data = (X, y)

    # Initialization of processes
    batch_size = 8
    M = 10
    x0 = np.array([0.])
    lr = 0.0001

    workers = [Worker(x0, lr=lr) for m in range(M)]
    server = Server(workers)

    # Optimization with Signum
    nb_iter = n_samples//(M*batch_size)
    idx = 0
    while idx < nb_iter:

        idx_st = idx*(M*batch_size)
        idx_end = idx_st + batch_size

        workers_inputs: List[Tuple[np.ndarray, ...]] = []
        for m in range(M):

            workers_inputs.append((X[idx_st:idx_end], y[idx_st:idx_end]))

            idx_st = idx_end
            idx_end += batch_size

        server.step(grad_linreg, workers_inputs)

        print(server.workers[0].x)

        idx += 1

    # Comparison with the true value
    print("Approx. solution: ", LinearRegression().fit(X, y).coef_)
