# ruff: noqa: ARG002

from __future__ import annotations

from typing import Any

from hakken_ml_toolkit.tracker.core.contracts.tracker import TrackerConfig, TrackerI


class DummyTrackerConfig(TrackerConfig):
    pass


class DummyTracker(TrackerI[DummyTrackerConfig]):
    def __init__(self, config: DummyTrackerConfig | None = None):
        if config is None:
            config = DummyTrackerConfig()
        super().__init__(config=config)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

    def _track_value(self, key: str, value: Any, step: int | None = None) -> None:
        return

    def _track_data(self, data: dict[str, Any], step: int | None = None) -> None:
        return

    def _track_config(self, config: dict[str, Any]) -> None:
        return

    def track_table(
        self,
        key: str,
        columns: list[str],
        data: list[list[Any]],
        step: int | None = None,
    ) -> None:
        return

    def finish(self):
        return

    def __del__(self):
        self.finish()
