"""Contain our own optimizer class for a distributed SGD."""


# Import Python packages
import numpy as np
import torch
import torch.distributed as dist
from typing import Callable, Iterable, Optional
from torch.optim import Optimizer


class DistSGD(Optimizer):
    """Extend the base `torch.optim.Optimizer` class from PyTorch for distributed SGD."""

    def __init__(self, params: Iterable, blind_inv_adv: np.ndarray, byz_adv: np.ndarray,
                 lr: float = 1e-4):
        """
        Initialize a distributed SGD optimizer.

        Parameters
        ----------
        params : Iterator
            Model parameters.

        blind_inv_adv : numpy.ndarray
            Array containing the ranks of blind adversaries that invert their gradients signs.

        byz_adv : np.ndarray
            Ranks of Byzantine adversaries.

        lr : float, default=0.0001
            Learning rate.
        """
        if lr < 0.:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)

        super(DistSGD, self).__init__(params, defaults)

        # Handling of adversaries
        self.blind_inv_adv = blind_inv_adv
        self.byz_adv = byz_adv

    def __setstate__(self, state):
        """
        Set the state of this instance.

        Parameters
        ----------
        state :
            New state.
        """
        super(DistSGD, self).__setstate__(state)

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Perform one step of optimization.

        Parameters
        ----------
        closure : Optional[Callable], default=None
            Callable function to evaluate the loss.

        Returns
        -------
        loss : Optional[float]
            Computed loss if a closure is applied.
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            lr = group["lr"]

            for p in group["params"]:

                if p.grad is None:
                    continue

                # d_p is the batch gradient on current worker and parameter
                d_p = p.grad.data

                # Handling of blind adversaries
                if rank in self.blind_inv_adv:
                    d_p = -d_p

                # Gradients are picked up by Byzantine workers, eventually modifying their own...
                if len(self.byz_adv) > 0:  # one Byzantine is enough to crush SGD

                    all_d_p = torch.zeros([world_size] + list(d_p.size()))
                    adv = self.byz_adv[0]

                    if rank != adv:

                        all_d_p[rank] = d_p
                        dist.send(all_d_p[rank], adv)

                    else:

                        non_adv = [i for i in range(world_size) if i != adv]

                        for worker in non_adv:

                            dist.recv(all_d_p[worker], src=worker)

                    dist.barrier()  # wait for the Byzantine to know about others gradients signs

                    if rank == adv:

                        agg_d_p = torch.sum(all_d_p, dim=0)

                        for indices, _ in np.ndenumerate(agg_d_p):

                            d_p[indices] = -agg_d_p[indices]

                    dist.barrier()  # wait for the Byzantine to compute its plot

                # ...and averaged (we omit the idea of server here)
                dist.all_reduce(d_p, op=dist.ReduceOp.SUM)

                dist.barrier()  # wait for everyone to get the server's information

                # Weights are modified according to the sign of summed gradients signs
                p.data.add_(d_p/world_size, alpha=-lr)

        return loss
