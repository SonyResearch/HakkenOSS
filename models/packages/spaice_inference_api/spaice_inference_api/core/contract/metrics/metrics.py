from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, cast

from typing_extensions import Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

MetricLabels = dict[str, str | int | float | None]


class ICounter(Protocol):
    def inc(self, amount: float = 1, exemplar: dict[str, str] | None = None) -> None:
        raise NotImplementedError


class IGauge(Protocol):
    def inc(self, amount: float = 1) -> None:
        raise NotImplementedError

    def dec(self, amount: float = 1) -> None:
        raise NotImplementedError

    def set(self, value: float) -> None:
        raise NotImplementedError


class IHistogram(Protocol):
    def observe(self, amount: float, exemplar: dict[str, str] | None = None) -> None:
        raise NotImplementedError


class Metric(Protocol):
    @property
    def value(self) -> str:
        pass

    @property
    def description(self) -> str:
        pass


class IMetrics(ABC):
    @abstractmethod
    def __init__(
        self, common_key_prefix: str = "", common_labels: MetricLabels | None = None
    ) -> None:
        pass

    @abstractmethod
    def counter(self, key: Metric, labels: MetricLabels | None = None, description="") -> ICounter:
        pass

    @abstractmethod
    def gauge(self, key: Metric, labels: MetricLabels | None = None, description="") -> IGauge:
        pass

    @abstractmethod
    def histogram(
        self,
        key: Metric,
        labels: MetricLabels | None = None,
        buckets: Sequence[float | str] = (),
        description="",
    ) -> IHistogram:
        pass

    @abstractmethod
    def report(self) -> bytes:
        pass


class IMetricsType(str, Enum):
    def __new__(cls, *args, **_kwds):
        obj = str.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, _: str, description: str | None = None):
        self._description_ = description

    def __str__(self):
        return self.value

    @property
    def description(self) -> str:
        return self._description_ or ""

    @property
    def value(self) -> str:
        return self._value_ or ""

    @classmethod
    def contains(cls, o: str) -> bool:
        return o in cls._value2member_map_

    @classmethod
    def get(cls, o: str) -> IMetricsType:
        return cast("IMetricsType", cls._value2member_map_[o])
