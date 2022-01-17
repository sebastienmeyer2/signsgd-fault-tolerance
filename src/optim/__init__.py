"""Contains custom distributed optimizers."""


from optim.distsgd import DistSGD
from optim.signum import Signum


__all__ = ["DistSGD", "Signum"]
