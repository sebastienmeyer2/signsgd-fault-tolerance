"""Contain neural networks that yield good results on MNIST."""


# Import Python packages
import torch
import torch.nn.functional as F
from torch.nn import Module, Sequential
from torch.nn import Conv2d, Dropout, Linear, MaxPool2d, ReLU


class TorchNet(Module):
    """Neural network given in the tutorials of PyTorch."""

    def __init__(self, n_classes, in_channels: int = 1):
        """
        Initialize the neural network.

        Parameters
        ----------
        n_classes : int
            Number of classes.

        in_channels : int, default=1
            Number of channels in the input data. For instance, it can be 1 if the data are
            images in gray scale or 3 if they are RGB.
        """
        super(TorchNet, self).__init__()

        self.conv1 = Conv2d(in_channels, 32, 3, stride=1)
        self.conv2 = Conv2d(32, 64, 3, stride=1)
        self.dropout1 = Dropout(p=0.25)
        self.dropout2 = Dropout(p=0.5)
        self.fc1 = Linear(9216, 128)
        self.fc2 = Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass on input data.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        output : torch.Tensor
            Output data.
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        return output


class MNISTNet(Module):
    """Neural network which usually gives good results on MNIST."""

    def __init__(self, n_classes: int, in_channels: int = 1):
        """
        Initialize the neural network.

        Parameters
        ----------
        n_classes : int
            Number of classes.

        in_channels : int, default=1
            Number of channels in the input data. For instance, it can be 1 if the data are
            images in gray scale or 3 if they are RGB.
        """
        super(MNISTNet, self).__init__()

        self.cnn = Sequential(
            Conv2d(in_channels, 64, 5, stride=1, padding=2),
            ReLU(inplace=True),
            Conv2d(64, 64, 5, stride=1, padding=2),
            ReLU(inplace=True),
            MaxPool2d(2),
            Dropout(p=0.25),

            Conv2d(64, 64, 3, stride=1, padding=1),
            ReLU(inplace=True),
            Conv2d(64, 64, 3, stride=1, padding=1),
            ReLU(inplace=True),
            MaxPool2d(2, stride=2),
            Dropout(p=0.25)
        )

        self.fc = Sequential(
            Linear(3136, 128),
            ReLU(inplace=True),
            Dropout(p=0.5),
            Linear(128, n_classes)
        )

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
        # Apply convolutional layers
        x = self.cnn(x)

        # Apply linear layers
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        x = F.log_softmax(x, dim=1)

        return x
