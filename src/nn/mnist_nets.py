"""Contain neural networks that yield good results on MNIST."""


from torch import Tensor
from torch.nn import Module, Sequential
from torch.nn import Conv2d, Dropout, Linear, MaxPool2d, ReLU


class TorchNet(Module):
    """Neural network given in the tutorials of PyTorch."""

    def __init__(self, n_classes, in_channels: int = 1):
        """Initialize the neural network.

        Parameters
        ----------
        n_classes : int
            Number of classes.

        in_channels : int, default=1
            Number of channels in the input data. For instance, it can be 1 if the data are
            images in gray scale or 3 if they are RGB.
        """
        super().__init__()

        self.features = Sequential(
            Conv2d(in_channels, 32, 3, stride=1),
            ReLU(inplace=True),
            Conv2d(32, 64, 3, stride=1),
            ReLU(inplace=True),

            MaxPool2d(2, stride=2),
            Dropout(p=0.25)
        )

        self.classifier = Sequential(
            Linear(9216, 128),
            ReLU(inplace=True),
            Dropout(p=0.5),
            Linear(128, n_classes)
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
        x = self.features(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


class MNISTNet(Module):
    """Neural network which usually gives good results on MNIST."""

    def __init__(self, n_classes: int, in_channels: int = 1):
        """Initialize the neural network.

        Parameters
        ----------
        n_classes : int
            Number of classes.

        in_channels : int, default=1
            Number of channels in the input data. For instance, it can be 1 if the data are
            images in gray scale or 3 if they are RGB.
        """
        super().__init__()

        self.features = Sequential(
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

        self.classifier = Sequential(
            Linear(3136, 128),
            ReLU(inplace=True),
            Dropout(p=0.5),
            Linear(128, n_classes)
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
        x = self.features(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
