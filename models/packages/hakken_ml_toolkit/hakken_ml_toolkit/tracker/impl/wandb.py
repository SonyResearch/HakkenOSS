# ================================================================
# DO NOT USE IT
# ================================================================

from __future__ import annotations

from typing import Any

import pandas as pd
import wandb

from hakken_ml_toolkit.tracker.core.contracts.tracker import TrackerConfig, TrackerI


class WandBTrackerConfig(TrackerConfig):
    project: str = "my-project"
    run_name: str = "experiment-1"


class WandBTracker(TrackerI[WandBTrackerConfig]):
    def __init__(self, config: WandBTrackerConfig):
        super().__init__(config=config)

    def __enter__(self):
        wandb.init(project=self.config.project, name=self.config.run_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

    def _track_value(self, key: str, value: Any, step: int | None = None) -> None:
        if not self.config.persist:
            return
        step = step if step is not None else self.step
        wandb.log({"step": step, key: value})

    def _track_data(self, data: dict[str, Any], step: int | None = None) -> None:
        if not self.config.persist:
            return

        step = step if step is not None else self.step

        data["step"] = step
        wandb.log(data)

    def _track_config(self, config: dict[str, Any]) -> None:
        if not self.config.persist:
            return

        wandb.config.update(config)

    def track_table(
        self,
        key: str,
        columns: list[str],
        data: list[list[Any]],
        step: int | None = None,
    ) -> None:
        if not self.config.persist:
            return

        df = pd.DataFrame(columns=columns, data=data)

        wandb.log({key: wandb.Table(dataframe=df), "step": step})

    def finish(self):
        wandb.finish()

    def __del__(self):
        self.finish()
