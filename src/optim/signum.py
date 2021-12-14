"""Contain our own optimizer class for Signum algorithm."""


# Import Python packages
import numpy as np
import torch
import torch.distributed as dist
from typing import Callable, Iterable, Optional
from torch.optim import Optimizer


class Signum(Optimizer):
    """
    Extend the base `torch.optim.Optimizer` class from PyTorch to the Signum algorithm.

    References
    ----------
    .. [1] Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli and Anima Anandkumar.
       *SignSGD: Compressed Optimisation for Non-Convex Problems*. (Available at:
       https://arxiv.org/abs/1802.04434)
    """

    def __init__(self, params: Iterable, blind_inv_adv: np.ndarray, byz_adv: np.ndarray,
                 lr: float = 1e-4, momentum: float = 0.9, weight_decay: float = 0.):
        r"""
        Initialize an instance of Signum optimizer.

        Parameters
        ----------
        params : Iterator
            Model parameters.

        blind_inv_adv : numpy.ndarray
            Array containing the ranks of blind adversaries that invert their gradients signs.

        byz_adv : np.ndarray
            Ranks of Byzantine adversaries.

        lr : float, default=0.0001
            Learning rate, denoted \(\eta\) in the algorithm description.

        momentum : float, default=0.9
            Value of momentum, denoted \(\beta\) in the algorithm description.

        weight_decay : float, default=0.0
            Value of the weight decay, denoted \(\lambda\) in the algorithm description.
        """
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum parameter: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)

        super(Signum, self).__init__(params, defaults)

        # Handling of blind adversaries
        self.blind_inv_adv = blind_inv_adv

        # Handling of Byzantine adversaries
        self.byz_adv = byz_adv
        self.f = len(byz_adv)
        self.groups = []
        self.byz_server = None
        if self.f > 0:

            self.byz_server = self.byz_adv[0]

            for i in range(dist.get_world_size()):

                if i != self.byz_server:
                    self.groups.append(dist.new_group([i, self.byz_server]))
                else:
                    self.groups.append(None)  # make sure we have the correct length

    def __setstate__(self, state):
        """
        Set the state of this instance.

        Parameters
        ----------
        state :
            New state.
        """
        super(Signum, self).__setstate__(state)

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
            loss = closure()

        for group in self.param_groups:

            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:

                if p.grad is None:
                    continue

                # d_p is the batch gradient on current worker and parameter
                d_p = p.grad.data

                # Handling of blind adversaries
                if rank in self.blind_inv_adv:
                    d_p = -d_p

                # In case of momentum, we are keeping a proportion of past gradient in memory
                if momentum >= 0.0:

                    param_state = self.state[p]

                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                    else:
                        buf = param_state["momentum_buffer"]

                    buf.mul_(momentum).add_(d_p, alpha=1-momentum)
                    d_p.copy_(buf)

                # Sign of workers are gathered...
                d_p_sg = torch.sign(d_p)

                # ...then picked up by Byzantine workers, eventually modifying their own...
                if self.f > 0:

                    # The Byzantine server will gather all the gradients signs
                    if rank == self.byz_server:

                        other_workers = [i for i in range(world_size) if i != self.byz_server]

                        d_p_sg_list = torch.zeros([world_size] + list(d_p_sg.size()))
                        d_p_sg_list[self.byz_server] = d_p_sg.clone()  # his own tensor

                        for worker in other_workers:

                            d_p_temp = torch.zeros(d_p_sg.size())
                            dist.broadcast(d_p_temp, worker, group=self.groups[worker])
                            d_p_sg_list[worker] = d_p_temp  # others tensors

                    else:

                        dist.broadcast(d_p_sg, rank, group=self.groups[rank])

                    dist.barrier()  # wait for the Byzantines to know all the gradients signs

                    # Then, the Byzantine server will decide what strategy is the best one
                    if rank == self.byz_server:

                        not_byz = [i for i in range(world_size) if i not in self.byz_adv]

                        agg_d_p_sg = torch.sum(d_p_sg_list[not_byz], dim=0).detach().numpy()

                        for indices, sg_sum in np.ndenumerate(agg_d_p_sg):

                            # If Byzantines are not able to reverse the sign, they just go against
                            # it
                            if sg_sum > self.f:

                                for byz in self.byz_adv:

                                    d_p_sg_list[byz][indices] = -1

                            elif sg_sum < -self.f:

                                for byz in self.byz_adv:

                                    d_p_sg_list[byz][indices] = +1

                            # If Byzantines have a chance to kill the learning process, they will
                            # first kill the aggregated of all honest workers and then oscillate
                            # around zero
                            if sg_sum <= self.f and sg_sum >= 0:

                                k = sg_sum
                                start_pos = np.random.choice([True, False])
                                for byz in self.byz_adv:

                                    if k > 0:

                                        d_p_sg_list[byz][indices] = -1
                                        k -= 1
                                    else:

                                        if start_pos:
                                            d_p_sg_list[byz][indices] = +1
                                        else:
                                            d_p_sg_list[byz][indices] = -1
                                        start_pos = not start_pos

                            elif sg_sum >= -self.f and sg_sum <= 0:

                                k = sg_sum
                                start_pos = np.random.choice([True, False])
                                for byz in self.byz_adv:

                                    if k < 0:

                                        d_p_sg_list[byz][indices] = +1
                                        k += 1

                                    else:

                                        if start_pos:
                                            d_p_sg_list[byz][indices] = +1
                                        else:
                                            d_p_sg_list[byz][indices] = -1
                                        start_pos = not start_pos

                    dist.barrier()  # wait for the Byzantine server to have computed its strategy

                    # Modified tensors will now be sent back to the workers
                    if rank == self.byz_server:

                        other_byz = [i for i in self.byz_adv if i != self.byz_server]

                        for byz in other_byz:

                            dist.broadcast(d_p_sg_list[byz], self.byz_server,
                                           group=self.groups[byz])

                        d_p_sg = d_p_sg_list[self.byz_server]  # changed its own tensor

                    elif rank in self.byz_adv and rank != self.byz_server:

                        dist.broadcast(d_p_sg, self.byz_server, group=self.groups[rank])

                    dist.barrier()

                # ...and summed (we omit the idea of server here)
                dist.all_reduce(d_p_sg, op=dist.ReduceOp.SUM)

                dist.barrier()  # wait for everyone to get the server's information

                # In case of weight decay, current weights are slightly diminished
                if weight_decay >= 0.0:
                    p.data.mul_(1 - lr*weight_decay)

                # Weights are modified according to the sign of summed gradients signs
                p.data.add_(torch.sign(d_p_sg), alpha=-lr)

        return loss
