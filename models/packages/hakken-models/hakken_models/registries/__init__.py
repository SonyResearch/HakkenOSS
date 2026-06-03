from torch import nn, optim
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import LRScheduler

from .base import Registry


# Specialized registries
class LossRegistry(Registry[nn.Module]):
    """Registry for loss functions (nn.Module subclasses)."""

    pass


class OptimizerRegistry(Registry[Optimizer]):
    """Registry for optimizers (torch.optim.Optimizer subclasses)."""

    pass


class SchedulerRegistry(Registry[LRScheduler]):
    """Registry for learning rate schedulers (LRScheduler subclasses)."""

    pass


# Create instances
loss_registry = LossRegistry("Loss")
optimizer_registry = OptimizerRegistry("Optimizer")
scheduler_registry = SchedulerRegistry("Scheduler")


# Register built-in PyTorch losses
loss_registry.register_class(nn.MarginRankingLoss)


# Register built-in optimizers

optimizer_registry.register_class(optim.Adam)
optimizer_registry.register_class(optim.SGD)
optimizer_registry.register_class(optim.RMSprop)
optimizer_registry.register_class(optim.AdamW)
optimizer_registry.register_class(optim.Adagrad)
optimizer_registry.register_class(optim.Adamax)
optimizer_registry.register_class(optim.SparseAdam)

# Register built-in schedulers

scheduler_registry.register_class(lr_scheduler.StepLR)
scheduler_registry.register_class(lr_scheduler.ExponentialLR)
scheduler_registry.register_class(lr_scheduler.CosineAnnealingLR)
scheduler_registry.register_class(lr_scheduler.ReduceLROnPlateau)
scheduler_registry.register_class(lr_scheduler.OneCycleLR)
scheduler_registry.register_class(lr_scheduler.CosineAnnealingWarmRestarts)
scheduler_registry.register_class(lr_scheduler.LinearLR)
scheduler_registry.register_class(lr_scheduler.PolynomialLR)


__all__ = ["Registry", "loss_registry", "optimizer_registry", "scheduler_registry"]
