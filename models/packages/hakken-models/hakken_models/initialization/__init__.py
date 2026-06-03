from hakken_models.registries.base import Registry

from .base import BaseInitStrategy
from .xavier import XavierConfig, XavierNormal, XavierUniform


class InitStrategyRegistry(Registry[BaseInitStrategy]):
    pass


init_strategy_registry = InitStrategyRegistry("InitStrategy")

init_strategy_registry.register_class(XavierNormal)
init_strategy_registry.register_class(XavierUniform)

__all__ = [
    "BaseInitStrategy",
    "XavierConfig",
    "XavierNormal",
    "XavierUniform",
    "init_strategy_registry",
]
