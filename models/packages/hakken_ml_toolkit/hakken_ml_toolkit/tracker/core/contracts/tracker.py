from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path  # noqa: TC003
from typing import Any, Generic, TypeVar

from loguru import logger
from pydantic_settings import BaseSettings

from hakken_ml_toolkit.tracker.core.values.exceptions import TrackerFolderError


class TrackerConfig(BaseSettings):
    folder: Path | None = None
    persist: bool = True
    log: bool = False


T = TypeVar("T", bound=TrackerConfig)


class TrackerI(ABC, Generic[T]):
    def __init__(self, config: T):
        self.config = config
        self.persistance_is_enabled = config.persist
        self.logging_is_enabled = config.log
        self.step = 0

        if self.config.folder is not None:
            self.config.folder.mkdir(parents=True, exist_ok=True)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        return

    def folder(self) -> Path:
        if self.config.folder is None:
            raise TrackerFolderError()
        return self.config.folder

    def finish(self):
        pass

    def enable(self):
        self.persistance_is_enabled = True
        self.logging_is_enabled = True

    def disable(self):
        self.persistance_is_enabled = False
        self.logging_is_enabled = False

    def set_logging(self, value: bool):
        self.logging_is_enabled = value

    def set_persistance(self, value: bool):
        self.persistance_is_enabled = value

    @abstractmethod
    def _track_value(self, key: str, value: Any, step: int | None = None) -> None:
        pass

    def track_value(self, key: str, value: Any, step: int | None = None) -> None:
        if self.logging_is_enabled:
            logger.info(f"[{step}] {key}: {value}")
        self._track_value(key, value, step)

    @abstractmethod
    def _track_config(self, config: dict[str, Any]) -> None:
        pass

    def track_config(self, config: dict[str, Any]) -> None:
        if self.logging_is_enabled:
            for key, value in config.items():
                logger.info(f"[CONFIG] {key}: {value}")

        self._track_config(config)

    @abstractmethod
    def _track_data(self, data: dict[str, Any], step: int | None = None) -> None:
        pass

    def track_data(self, data: dict[str, Any], step: int | None = None) -> None:
        self._track_data(data, step)
        if self.logging_is_enabled:
            for key, value in data.items():
                logger.info(f"[{step}] {key}: {value}")

    @abstractmethod
    def track_table(
        self,
        key: str,
        columns: list[str],
        data: list[list[Any]],
        step: int | None = None,
    ) -> None:
        pass

    def increment_step(self) -> None:
        self.step += 1
        if self.logging_is_enabled:
            logger.info(f"[INCREMENT_STEP] {self.step}")
