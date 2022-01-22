"""Implement various ResNet neural networks.

References
----------
.. [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun. *Deep Residual Learning for Image
   Recognition.* December 2015. (Available at: https://arxiv.org/abs/1512.03385)
"""


from torch import Tensor
from torch.nn import Module, Sequential
from torch.nn import AdaptiveAvgPool2d, BatchNorm2d, Conv2d, Linear, MaxPool2d, ReLU


class ResNetEntry(Module):
    """Entry module of any ResNet neural network, which corresponds to a first convolution."""

    def __init__(self, in_channels: int = 3):
        """Initialize the entry module of ResNet.

        Parameters
        ----------
        in_channels : int, default=3
            Number of channels in the input data. For instance, it can be 1 if the data are
            images in gray scale or 3 if they are RGB.
        """
        super().__init__()

        self.entry = Sequential(
            Conv2d(in_channels, 64, 7, stride=2, padding=3, padding_mode="replicate"),
            BatchNorm2d(64),
            ReLU(inplace=True),

            MaxPool2d(3, stride=2, padding=1)
        )

    def forward(self, x: Tensor) -> Tensor:
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
        x = self.entry(x)

        return x


class ResidualBlock50(Module):
    """Residual block of ResNet50 which has to be repeated."""

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, downsampling: bool):
        """Initialize a residual block of ResNet50.

        Parameters
        ----------
        in_channels : int
            Number of channels at the entry of this residual block.

        mid_channels : int
            Since residual blocks of ResNet50 are made of three layers, this parameters controls
            the number of channels in the middle layer.

        out_channels : int
            Number of channels at the exit of this residual block.

        downsampling : bool
            If True, width and heigth of the data are divided by two at the beginning of this
            residual block.
        """
        super().__init__()

        self.shortcut_conv = (in_channels != out_channels)

        in_stride = 2 if downsampling else 1
        self.out_channels = out_channels

        self.plain = Sequential(
            Conv2d(in_channels, mid_channels, kernel_size=1, stride=in_stride),
            BatchNorm2d(mid_channels),
            ReLU(inplace=True),
            Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1,
                   padding_mode="replicate"),
            BatchNorm2d(mid_channels),
            ReLU(inplace=True),
            Conv2d(mid_channels, out_channels, kernel_size=1, stride=1),
            BatchNorm2d(out_channels),
            ReLU(inplace=True)
        )

        self.shortcut = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=1, stride=in_stride),
            BatchNorm2d(out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Perform a forward pass on input data.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        h_x : torch.Tensor
            Output data.
        """
        # Apply the plain layers
        f_x = self.plain(x)

        # Apply the rescaling - if needed
        x = self.shortcut(x) if self.shortcut_conv else x

        # Compute the residual
        h_x = f_x + x

        return h_x


class ResidualBlock18(Module):
    """Residual block of ResNet18 which has to be repeated."""

    def __init__(self, in_channels: int, out_channels: int, downsampling: bool):
        """Initialize a residual block of ResNet18.

        Parameters
        ----------
        in_channels : int
            Number of channels at the entry of this residual block.

        out_channels : int
            Number of channels at the exit of this residual block.

        downsampling : bool
            If True, width and heigth of the data are divided by two at the beginning of this
            residual block.
        """
        super().__init__()

        self.shortcut_conv = (in_channels != out_channels)

        in_stride = 2 if downsampling else 1

        self.plain = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, stride=in_stride, padding=1,
                   padding_mode="replicate"),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
            Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,
                   padding_mode="replicate"),
            BatchNorm2d(out_channels),
            ReLU(inplace=True)
        )

        self.shortcut = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=1, stride=in_stride),
            BatchNorm2d(out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Perform a forward pass on input data.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        h_x : torch.Tensor
            Output data.
        """
        # Apply the plain layers
        f_x = self.plain(x)

        # Apply the rescaling - if needed
        x = self.shortcut(x) if self.shortcut_conv else x

        # Compute the residual
        h_x = f_x + x

        return h_x


class ResNetExit(Module):
    """Exit module of any ResNet neural network, which corresponds to a fully connected layer."""

    def __init__(self, num_features: int, n_classes: int):
        """Initialize the exit module of ResNet.

        Parameters
        ----------
        num_features : int
            Number of features at the beginning of the fully connected layer.

        n_classes : int
            Number of classes.
        """
        super().__init__()

        self.pool = AdaptiveAvgPool2d(1)
        self.classifier = Linear(num_features, n_classes, bias=True)

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
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


class ResNet18(Module):
    """ResNet18 neural network."""

    def __init__(self, n_classes: int, in_channels: int = 3):
        """Initialize ResNet18.

        Parameters
        ----------
        n_classes : int
            Number of classes.

        in_channels : int, default=3
            Number of channels in the input data. For instance, it can be 1 if the data are
            images in gray scale or 3 if they are RGB.
        """
        super().__init__()

        # Entry module
        self.entry = ResNetEntry(in_channels=in_channels)

        # Residual blocks
        self.blocks = Sequential()
        # conv2_x
        self.blocks.add_module("conv2_1", ResidualBlock18(64, 64, False))
        self.blocks.add_module("conv2_2", ResidualBlock18(64, 64, False))
        # conv3_x
        self.blocks.add_module("conv3_1", ResidualBlock18(64, 128, True))
        self.blocks.add_module("conv3_2", ResidualBlock18(128, 128, False))
        # conv4_x
        self.blocks.add_module("conv4_1", ResidualBlock18(128, 256, True))
        self.blocks.add_module("conv4_2", ResidualBlock18(256, 256, False))
        # conv5_x
        self.blocks.add_module("conv5_1", ResidualBlock18(256, 512, True))
        self.blocks.add_module("conv5_2", ResidualBlock18(512, 512, False))

        # Exit module
        self.exit = ResNetExit(512, n_classes)

        # Final softmax layer
        # self.softmax = Softmax(dim=0)

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
        # Apply entry layers
        x = self.entry(x)

        # Apply residual blocks
        x = self.blocks(x)

        # Apply final layers
        x = self.exit(x)
        # x = self.softmax(x)

        return x


class ResNet50(Module):
    """ResNet50 neural network."""

    def __init__(self, n_classes, in_channels=3):
        """Initialize ResNet50.

        Parameters
        ----------
        n_classes : int
            Number of classes.

        in_channels : int, default=3
            Number of channels in the input data. For instance, it can be 1 if the data are
            images in gray scale or 3 if they are RGB.
        """
        super().__init__()

        # Entry module
        self.entry = ResNetEntry(in_channels=in_channels)

        # Residual blocks
        self.blocks = Sequential()
        # conv2_x
        self.blocks.add_module("conv2_1", ResidualBlock50(64, 64, 256, False))
        for i in range(2, 4):
            self.blocks.add_module("conv2_{}".format(i), ResidualBlock50(256, 64, 256, False))
        # conv3_x
        self.blocks.add_module("conv3_1", ResidualBlock50(256, 128, 512, True))
        for i in range(2, 5):
            self.blocks.add_module("conv3_{}".format(i), ResidualBlock50(512, 128, 512, False))
        # conv4_x
        self.blocks.add_module("conv4_1", ResidualBlock50(512, 256, 1024, True))
        for i in range(2, 7):
            self.blocks.add_module("conv4_{}".format(i), ResidualBlock50(1024, 256, 1024, False))
        # conv5_x
        self.blocks.add_module("conv5_1", ResidualBlock50(1024, 512, 2048, True))
        for i in range(2, 4):
            self.blocks.add_module("conv5_{}".format(i), ResidualBlock50(2048, 512, 2048, False))

        # Exit module
        self.exit = ResNetExit(2048, n_classes)

        # Final softmax layer
        # self.softmax = Softmax(dim=0)

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
        # Apply entry layers
        x = self.entry(x)

        # Apply residual blocks
        x = self.blocks(x)

        # Apply final layers
        x = self.exit(x)
        # x = self.softmax(x)

        return x
