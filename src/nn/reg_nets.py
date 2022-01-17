"""Contain simple neural networks for linear and logistic regression parameters estimation."""


from torch import Tensor
from torch.nn import Module, Sequential
from torch.nn import Linear, ReLU


class LinRegNet(Module):
    """Basic neural network to evaluate linear regression parameters."""

    def __init__(self):
        """Initialize the neural network.

        Parameters
        ----------
        n_classes : int
            Number of classes.
        """
        super().__init__()

        self.classifier = Sequential(
            Linear(20, 40),
            ReLU(inplace=True),
            Linear(40, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Perform a forward pass on input data.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        x : torch.Tensor
            Output data.
        """
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


class LogRegNet(Module):
    """Basic neural network to evaluate logistic regression parameters."""

    def __init__(self):
        """Initialize the neural network.

        Parameters
        ----------
        n_classes : int
            Number of classes.
        """
        super().__init__()

        self.classifier = Sequential(
            Linear(20, 40),
            ReLU(inplace=True),
            Linear(40, 2)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Perform a forward pass on input data.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        x : torch.Tensor
            Output data.
        """
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
