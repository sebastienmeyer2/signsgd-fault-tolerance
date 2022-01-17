"""Contain our own optimizer class for a distributed SGD."""


from typing import Callable, Iterable, List, Optional

import torch
import torch.distributed as dist
from torch.optim import Optimizer
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


class DistSGD(Optimizer):
    """Extend the base `torch.optim.Optimizer` class from PyTorch for distributed SGD."""

    def __init__(self, params: Iterable, blind_list: Optional[List[int]] = None,
                 byz_list: Optional[List[int]] = None, lr: float = 1e-3, momentum: float = 0.0):
        """Initialize a distributed SGD optimizer.

        Parameters
        ----------
        params : iterable
            Model parameters.

        blind_list : list of int, optional
            Ranks of blind adversaries inverting their gradients signs.

        byz_list : list of int, optional
            Ranks of Byzantine adversaries.

        lr : float, default=0.001
            Learning rate.

        momentum : float, default=0.0
            Value of momentum.
        """
        if lr < 0.:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0. or momentum > 1.:
            raise ValueError(f"Invalid momentum parameter: {momentum}")

        defaults = dict(lr=lr, momentum=momentum)

        super().__init__(params, defaults)

        # Handling of blind adversaries
        self.blind_list = blind_list
        self.blind_size = len(blind_list) if self.blind_list is not None else 0

        # Handling of Byzantine adversaries
        self.byz_list = byz_list
        self.byz_size = len(byz_list) if self.byz_list is not None else 0

        self.groups = []  # sharing of data between the Byzantine server and other processes
        self.byz_server = None

        if self.byz_size > 0:

            self.byz_server = self.byz_list[0]

            for i in range(dist.get_world_size()):

                if i != self.byz_server:
                    self.groups.append(dist.new_group([i, self.byz_server]))
                else:
                    self.groups.append(None)  # make sure we have the correct length

    def get_blind_size(self) -> int:
        """Return the number of blind adversaries inverting their gradient signs.

        Returns
        -------
        blind_size : int
            Number of blind adversaries inverting their gradient signs.
        """
        blind_size = self.blind_size

        return blind_size

    def get_byz_size(self) -> int:
        """Return the number of Byzantine adversaries.

        Returns
        -------
        byz_size : int
            Number of Byzantine adversaries.
        """
        byz_size = self.byz_size

        return byz_size

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform one step of optimization.

        Parameters
        ----------
        closure : callable, optional
            Callable function to evaluate the loss.

        Returns
        -------
        loss : float, optional
            Computed loss if a closure is applied.
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        f = self.byz_size

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            lr = group["lr"]
            momentum = group["momentum"]

            all_d_p = []

            # First step: Workers compute the signs of their gradients and send them
            for p in group["params"]:

                if p.grad is None:
                    continue

                # Batch gradient of current worker
                d_p = p.grad.data

                # In case of momentum, we are keeping a proportion of past gradient in memory
                if momentum >= 0.:

                    param_state = self.state[p]

                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                    else:
                        buf = param_state["momentum_buffer"]

                    buf.mul_(momentum).add_(d_p, alpha=1-momentum)
                    d_p.copy_(buf)

                # Blind adversaries inverse their gradients
                if self.blind_list is not None and rank in self.blind_list:
                    d_p.mul_(-1)

                # All gradients are gathered to make computation easier
                all_d_p.append(d_p)

            # Second step: Gradients are intercepted by Byzantine workers
            if f > 0:

                other_workers = [i for i in range(world_size) if i != self.byz_server]

                # Tensors are flattened in order to make computation easier
                all_d_p_f = _flatten_dense_tensors(all_d_p)

                # The Byzantine server will gather all the gradients signs
                if rank == self.byz_server:

                    all_d_p_f_list = torch.zeros([world_size] + list(all_d_p_f.size()))
                    all_d_p_f_list[self.byz_server] = all_d_p_f.clone()  # his own tensor

                    for worker in other_workers:

                        all_d_p_f_temp = torch.zeros(all_d_p_f.size())
                        dist.broadcast(all_d_p_f_temp, worker, group=self.groups[worker])
                        all_d_p_f_list[worker] = all_d_p_f_temp  # others' tensors

                else:

                    dist.broadcast(all_d_p_f, rank, group=self.groups[rank])

                # Then, the Byzantine server will deploy its strategy
                if rank == self.byz_server:

                    # The Byzantine server opposes to all the other workers
                    agg_d_p_f = all_d_p_f_list[other_workers].sum(0)
                    all_d_p_f.copy_(-agg_d_p_f)

                # Recover initial shapes
                new_all_d_p = _unflatten_dense_tensors(all_d_p_f, all_d_p)

                for d_p, new_d_p in zip(all_d_p, new_all_d_p):

                    d_p.copy_(new_d_p)

            # Third step: Server sends back the aggregation and workers update their parameters
            for p in group["params"]:

                if p.grad is None:
                    continue

                d_p = p.grad.data

                dist.barrier()  # everyone must have the gradient of correct index

                # Final values are summed (we omit the idea of server here)
                dist.all_reduce(d_p, op=dist.ReduceOp.SUM)

                # Weights are modified according to the sign of summed gradients signs
                p.data.add_(d_p, alpha=-lr)

        return loss
