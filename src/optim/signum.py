"""Contain our own optimizer class for Signum algorithm.

References
----------
.. [1] Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli and Anima Anandkumar.
    *SignSGD: Compressed Optimisation for Non-Convex Problems*. (Available at:
    https://arxiv.org/abs/1802.04434)
"""


from typing import Callable, Iterable, List, Optional

import torch
import torch.distributed as dist
from torch.optim import Optimizer
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


class Signum(Optimizer):
    """Extend the base `torch.optim.Optimizer` class from PyTorch to the Signum algorithm."""

    def __init__(self, params: Iterable, blind_list: Optional[List[int]] = None,
                 byz_list: Optional[List[int]] = None, lr: float = 1e-4, momentum: float = 0.9,
                 weight_decay: float = 0.0):
        r"""Initialize an instance of Signum optimizer.

        Parameters
        ----------
        params : iterable
            Model parameters.

        blind_list : list of int, optional
            Ranks of blind adversaries inverting their gradients signs.

        byz_list : list of int, optional
            Ranks of Byzantine adversaries.

        lr : float, default=0.0001
            Learning rate, denoted \(\eta\) in the algorithm description.

        momentum : float, default=0.9
            Value of momentum, denoted \(\beta\) in the algorithm description. If **momentum** is
            equal to zero, yields *SignSGD*.

        weight_decay : float, default=0.0
            Value of the weight decay, denoted \(\lambda\) in the algorithm description.
        """
        if lr < 0.:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0. or momentum > 1.:
            raise ValueError(f"Invalid momentum parameter: {momentum}")
        if weight_decay < 0.:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)

        super().__init__(params, defaults)

        # Handling of blind adversaries
        self.blind_list = blind_list
        self.blind_size = len(blind_list) if blind_list is not None else 0

        # Handling of Byzantine adversaries
        self.byz_list = byz_list
        self.byz_size = len(byz_list) if byz_list is not None else 0

        self.groups = []  # sharing of data between the Byzantine server and other processes
        self.byz_server = None

        if self.byz_list is not None and self.byz_size > 0:

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
            loss = closure()

        for group in self.param_groups:

            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

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

                # Compute the sign of current gradient
                d_p.sign_()

                # Blind adversaries inverse their gradients
                if self.blind_list is not None and rank in self.blind_list:
                    d_p.mul_(-1)

                # All gradients are gathered to make computation easier
                all_d_p.append(d_p)

            # Second step: Gradients are intercepted by Byzantine workers
            if self.byz_list is not None and f > 0:

                # One of the Byzantines will be the Byzantine server
                other_byz = [i for i in self.byz_list if i != self.byz_server]

                # Tensors are flattened in order to make computation easier
                all_d_p_f = _flatten_dense_tensors(all_d_p)

                # The Byzantine server will gather all the gradients signs
                if rank == self.byz_server:

                    not_byz = [i for i in range(world_size) if i not in self.byz_list]

                    all_d_p_f_list = torch.zeros([world_size] + list(all_d_p_f.size()))

                    for worker in not_byz:

                        all_d_p_f_temp = torch.zeros(all_d_p_f.size())
                        dist.broadcast(all_d_p_f_temp, worker, group=self.groups[worker])
                        all_d_p_f_list[worker] = all_d_p_f_temp  # tensors from not Byzantines

                    # Then, the Byzantine server will deploy its strategy
                    agg_d_p_f = all_d_p_f_list.sum(0)

                    # If Byzantines are not able to reverse the sign, they go against it
                    all_d_p_f[agg_d_p_f > f] = -f

                    all_d_p_f[agg_d_p_f < -f] = +f

                    # If Byzantines have a chance to kill the learning process, they will first
                    # kill the aggregate of all honest workers and then oscillate around zero
                    mod = torch.Tensor([2])

                    idx_pos = ((agg_d_p_f - f/2).abs() <= f/2)
                    agg_pos = agg_d_p_f[idx_pos]
                    all_d_p_f[idx_pos] = (-agg_pos - (f-agg_pos).remainder(mod))

                    idx_neg = ((agg_d_p_f + f/2).abs() <= f/2)
                    agg_neg = agg_d_p_f[idx_neg]
                    all_d_p_f[idx_neg] = (-agg_neg + (f+agg_neg).remainder(mod))

                # Other Byzantines become silent
                elif rank in other_byz:

                    all_d_p_f.fill_(0)

                # Healthy workers and blind adversaries send their information
                else:

                    dist.broadcast(all_d_p_f, rank, group=self.groups[rank])

                # Recover initial shapes (for all processes)
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

                # In case of weight decay, current weights are slightly diminished
                if weight_decay >= 0.0:
                    p.data.mul_(1 - lr*weight_decay)

                # Weights are modified according to the sign of summed gradients signs
                p.data.add_(d_p.sign(), alpha=-lr)

        return loss
