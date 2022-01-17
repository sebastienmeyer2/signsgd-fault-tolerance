"""Contains various neural networks adapted to specific datasets."""


from nn.mnist_nets import TorchNet, MNISTNet
from nn.reg_nets import LogRegNet, LinRegNet
from nn.resnets import ResNet18, ResNet50


__all__ = ["LogRegNet", "LinRegNet", "TorchNet", "MNISTNet", "ResNet18", "ResNet50"]
