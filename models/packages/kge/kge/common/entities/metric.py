from abc import ABC, abstractmethod

from torch.nn import Module


class MetricI(Module, ABC):
    @abstractmethod
    def reset(self) -> None:
        """
        Resets the metric to its initial state.
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def compute(self) -> float:
        pass
