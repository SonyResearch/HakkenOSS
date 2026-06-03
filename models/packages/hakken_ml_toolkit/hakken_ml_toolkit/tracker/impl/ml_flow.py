from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mlflow
import pandas as pd

from hakken_ml_toolkit.ml_utils import RandomUtils
from hakken_ml_toolkit.tracker.core.contracts.tracker import TrackerConfig, TrackerI

if TYPE_CHECKING:
    from mlflow.tracking.fluent import ActiveRun


class MLFlowTrackerConfig(TrackerConfig):
    experiment_name: str | None = None
    run_name: str | None = None
    tracking_uri: str | None = None


class MLFlowTracker(TrackerI[MLFlowTrackerConfig]):
    def __init__(self, config: MLFlowTrackerConfig):
        super().__init__(config=config)
        if self.config.persist and self.config.tracking_uri:
            mlflow.set_tracking_uri(self.config.tracking_uri)
        if self.config.persist:
            mlflow.set_experiment(self.config.experiment_name)
        self.run_name = RandomUtils.generate_run_name(
            basename=self.config.run_name, use_local_random=True
        )

        self.run: ActiveRun | None = None

    def __enter__(self):
        if self.config.persist:
            self.run = mlflow.start_run(run_name=self.run_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

    def _track_value(self, key: str, value: Any, step: int | None = None) -> None:
        if not self.config.persist or self.run is None:
            return
        step = step if step is not None else self.step
        mlflow.log_metric(key, value, step=step)

    def _track_data(self, data: dict[str, Any], step: int | None = None) -> None:
        if not self.config.persist or self.run is None:
            return

        step = step if step is not None else self.step
        for key, value in data.items():
            mlflow.log_metric(key, value, step=step)

    def _track_config(self, config: dict[str, Any]) -> None:
        if not self.config.persist or self.run is None:
            return

        mlflow.log_params(config)

    def track_table(
        self,
        key: str,
        columns: list[str],
        data: list[list[Any]],
        step: int | None = None,
    ) -> None:
        if not self.config.persist or self.run is None:
            return

        artifact_file = f"{key}__{step}.json" if step is not None else f"{key}.json"

        df = pd.DataFrame(columns=columns, data=data)
        mlflow.log_table(data=df, artifact_file=artifact_file)

    def finish(self):
        if self.run:
            mlflow.end_run()
            self.run = None

    def __del__(self):
        self.finish()
