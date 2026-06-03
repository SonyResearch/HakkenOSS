from __future__ import annotations

import json
from typing import Any

import pandas as pd
import torch
import yaml

from hakken_ml_toolkit.tracker.core.contracts.tracker import TrackerConfig, TrackerI


class FSTrackerConfig(TrackerConfig):
    pass


class FSTracker(TrackerI[FSTrackerConfig]):
    def __init__(self, config: FSTrackerConfig):
        super().__init__(config=config)

    def _track_value(self, key: str, value: Any, step: int | None = None) -> None:
        data = {key: value}

        self.track_data(data=data, step=step)

    def _track_config(self, config: dict[str, Any]) -> None:
        if not self.persistance_is_enabled:
            return

        file_path = self.folder() / "config.yaml"

        with file_path.open("w") as f:
            yaml.dump(config, f, default_flow_style=False)

    def _track_data(self, data: dict[str, Any], step: int | None = None) -> None:
        if not self.persistance_is_enabled:
            return

        step = step if step else self.step
        # Add dict to file in a new line
        if "step" in data:
            msg = "Step should not be in data"
            raise KeyError(msg)

        for key, value in data.items():
            file_path = self.folder() / f"{key}.txt"

            value_item = value
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    value_item = value.item()
                else:
                    raise NotImplementedError()

            row = {key: value_item, "step": step}

            with open(file_path, "a") as f:
                f.write(json.dumps(row))
                f.write("\n")

    def track_table(
        self,
        key: str,
        columns: list[str],
        data: list[list[Any]],
        step: int | None = None,
    ) -> None:
        if not self.persistance_is_enabled:
            return

        step = step if step else self.step

        df = pd.DataFrame(columns=columns, data=data)

        file_path = self.folder() / f"{key}__{step}.tsv"

        df.to_csv(file_path, sep="\t", index=False)
