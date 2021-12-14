"""Contain a simple neural network for logistic regression parameters estimation."""


# Import Python packages
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn import Linear


class LogRegNet(Module):
    """Basic neural network to evaluate logistic regression parameters."""

    def __init__(self):
        """
        Initialize the neural network.

        Parameters
        ----------
        n_classes : int
            Number of classes.
        """
        super(LogRegNet, self).__init__()

        self.fc1 = Linear(20, 40)
        self.fc2 = Linear(40, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass on input data.

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
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        x = F.log_softmax(x, dim=1)

        return x
