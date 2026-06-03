"""Loss function registry. Isolated to avoid circular imports with loss modules."""

from torch.nn import Module

from hakken_models.registries.base import Registry


class LossFnRegistry(Registry[Module]):
    pass


loss_fn_registry = LossFnRegistry("LossFn")
