from collections.abc import Iterable
from typing import Protocol


class OptimizerProtocol(Protocol):
    def __init__(self, param: Iterable, **kwargs) -> None:
        pass

    def step(self) -> None:
        pass
