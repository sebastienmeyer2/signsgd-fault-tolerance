"""Implement paritions classes to split data amongst processes."""


from typing import List, Tuple

import numpy as np

from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets, transforms


from datasets.reg_datasets import simu_linreg, simu_logreg
from datasets.reg_datasets import LinRegDataset, LogRegDataset


class Partition(Dataset):
    """Wrapper around a dataset and indices for a partition."""

    def __init__(self, data: Dataset, indices: List[int]):
        """Initialize a partition of a dataset.

        Parameters
        ----------
        data : torch.utils.data.Dataset
            Complete dataset.

        indices : list of int
            List of indices of the dataset that can be used in this partition.
        """
        self.data = data
        self.indices = indices

    def __len__(self) -> int:
        """Return the length of the partition.

        Returns
        -------
        length : int
            Length of the partition.
        """
        length = len(self.indices)
        return length

    def __getitem__(self, index) -> Tensor:
        """Return data point(s) at position **index**.

        Parameters
        ----------
        index : int or list of int
            Index or list of indices of selected data points.
        """
        data_idx = self.indices[index]
        return self.data[data_idx]


class PartitionGenerator():
    """Wrapper around torch.utils.data.Dataset objects for partitioning."""

    def __init__(self, data_type: str, shuffle: bool = True):
        """Get the datasets.

        Parameters
        ----------
        data_type : {"logreg", "linreg", "mnist", "imagenet"}
            Name of the dataset.

        shuffle : bool, default=True
            If True, will shuffle all the indices of the training set before splitting it into
            partitions.
        """
        # Retrieve training and test sets
        if data_type == "linreg":

            X_y_train, X_y_test = simu_linreg()
            self.train = LinRegDataset(X_y_train)
            self.test = LinRegDataset(X_y_test)

        elif data_type == "logreg":

            X_y_train, X_y_test = simu_logreg()
            self.train = LogRegDataset(X_y_train)
            self.test = LogRegDataset(X_y_test)

        elif data_type == "mnist":

            self.train = datasets.MNIST("data/", train=True, download=True,
                                        transform=transforms.ToTensor())
            self.test = datasets.MNIST("data/", train=False, download=True,
                                       transform=transforms.ToTensor())

        elif data_type == "imagenet":

            imagenet_transforms = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            self.train = datasets.ImageNet("data/", split="train", download=True,
                                           transform=imagenet_transforms)
            self.test = datasets.ImageNet("data/", split="test", download=True,
                                          transform=imagenet_transforms)

        else:
            raise ValueError("Dataset {} not supported in dataset creation.".format(data_type))

        # Shuffle all the indices, here it is vital that the seed is shared among all processes
        n = len(self.train)
        self.all_indices = [i for i in range(n)]  # pylint: disable=unnecessary-comprehension
        if shuffle:
            np.random.shuffle(self.all_indices)

    def get_partition(self, rank: int, world_size: int) -> Tuple[Partition, Dataset]:
        """Return the partition assigned to **rank** process.

        Parameters
        ----------
        rank : int
            Rank of current process.

        world_size : int
            Total amount of processes.

        Returns
        -------
        partition : `src.datasets.Partition`
            A partition of the training set.

        test_set : torch.utils.data.Dataset
            The test set.
        """
        n = len(self.all_indices)
        partition_size = n // world_size

        first_idx = (rank*n) // world_size
        last_idx = first_idx + partition_size
        partition_indices = self.all_indices[first_idx:last_idx]

        partition = Partition(self.train, partition_indices)
        test_set = self.test

        return partition, test_set
