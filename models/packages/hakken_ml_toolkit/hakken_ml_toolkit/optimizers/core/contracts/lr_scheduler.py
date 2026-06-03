from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from hakken_ml_toolkit.optimizers.core.contracts.optimizer import OptimizerProtocol


class LRSchedulerProtocol(Protocol):
    def __init__(self, optimizer: OptimizerProtocol, **kwargs) -> None:
        pass

    def step(self, metrics: float | None = None) -> None:
        pass
