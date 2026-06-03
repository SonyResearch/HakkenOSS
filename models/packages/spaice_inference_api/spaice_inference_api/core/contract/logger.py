import logging
from abc import ABC, abstractmethod
from typing import Any

from typing_extensions import TypedDict

LoggerToken = "logger"


class PossibleLabelValues(TypedDict, total=False):
    values: list[str]
    max_length: int


class ILogger(ABC, logging.Logger):
    @abstractmethod
    def initialize(
        self,
        labels: dict[str, Any] | None = None,
        possible_label_values: dict[str, PossibleLabelValues] | None = None,
    ):
        pass

    @abstractmethod
    def add_labels(self, labels: dict[str, Any] | None = None):
        pass

    @abstractmethod
    def remove_labels(self, labels: list[str]):
        pass

    @abstractmethod
    def set_labels(self, labels: dict[str, Any] | None = None):
        pass

    @abstractmethod
    def extend(self, name: str, labels: dict[str, Any] | None = None) -> "ILogger":
        pass
