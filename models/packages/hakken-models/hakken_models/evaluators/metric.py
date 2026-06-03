from typing import Any, Protocol, Self, runtime_checkable

import torch


@runtime_checkable
class MetricLike(Protocol):
    def update(self, *args: Any, **kwargs: Any) -> None: ...

    def compute(self) -> Any: ...

    def reset(self) -> None: ...

    def to(self, device: torch.device | str) -> Self: ...
