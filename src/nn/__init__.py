"""Contains various neural networks adapted to specific datasets."""


from nn.linreg_net import LinRegNet
from nn.logreg_net import LogRegNet
from nn.mnist_nets import MNISTNet, TorchNet
from nn.resnets import ResNet18, ResNet50


__all__ = ["LinRegNet", "LogRegNet", "MNISTNet", "TorchNet", "ResNet18", "ResNet50"]
