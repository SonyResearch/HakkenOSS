from __future__ import annotations

import json
from typing import TYPE_CHECKING

import torch

from hakken_ml_toolkit.ml_utils.extras.scalers.core.contracts.scaler import (
    ScalerConfig,
    ScalerI,
)
from hakken_ml_toolkit.ml_utils.extras.scalers.core.values.exceptions import (
    ScalerNotFittedError,
    ZeroStandardDeviationError,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from hakken_ml_toolkit.ml_utils.extras.domain import TensorND


class StandardScalerConfig(ScalerConfig):
    mean: float = 0.0
    std: float = 1.0


class StandardScaler(ScalerI[StandardScalerConfig]):
    def __init__(self, config: StandardScalerConfig | None = None):
        if config is None:
            config = StandardScalerConfig()
        super().__init__(config)

        self.mean = self.config.mean
        self.std = self.config.std
        self.data_mean_: torch.Tensor | None = None
        self.data_std_: torch.Tensor | None = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

    def to(self, device: str | torch.device) -> None:
        if self.data_mean_ is not None:
            self.data_mean_ = self.data_mean_.to(device)
        if self.data_std_ is not None:
            self.data_std_ = self.data_std_.to(device)

    def fit(self, data: TensorND) -> None:
        self.data_mean_ = torch.mean(data, dim=0)
        self.data_std_ = torch.std(data, dim=0)
        if torch.any(self.data_std_ == 0):
            raise ZeroStandardDeviationError()

    def fit_from_iterator(
        self, iterator: Iterator[TensorND], num_batches: int | None = None
    ) -> None:
        sum_data: torch.Tensor
        sum_squared_data: torch.Tensor
        total_samples = 0
        first_batch = True

        for i, batch_tensor in enumerate(iterator):
            if num_batches is not None and i > num_batches:
                break
            batch = batch_tensor
            batch_size = batch.shape[0]
            total_samples += batch_size

            batch_sum = torch.sum(batch, dim=0)
            batch_sum_squared = torch.sum(batch**2, dim=0)

            if first_batch:
                sum_data = batch_sum
                sum_squared_data = batch_sum_squared
                first_batch = False
            else:
                sum_data += batch_sum
                sum_squared_data += batch_sum_squared

        self.data_mean_ = sum_data / total_samples

        data_var = sum_squared_data / total_samples - self.data_mean_**2

        self.data_std_ = torch.sqrt((total_samples / (total_samples - 1)) * data_var)

        if torch.any(self.data_std_ == 0):
            raise ZeroStandardDeviationError()

    def transform(self, data: TensorND) -> TensorND:
        if self.data_mean_ is None or self.data_std_ is None:
            raise ScalerNotFittedError()

        x_scaled = (data - self.data_mean_) / self.data_std_
        return self.mean + x_scaled * self.std

    def inverse_transform(self, data_norm: TensorND) -> TensorND:
        if self.data_mean_ is None or self.data_std_ is None:
            raise ScalerNotFittedError()

        x_original = (data_norm - self.mean) / self.std
        return x_original * self.data_std_ + self.data_mean_

    def save(self, json_path: str | Path):
        state_dict = {
            "mean": self.mean,
            "std": self.std,
            "data_mean_": (self.data_mean_.tolist() if self.data_mean_ is not None else None),
            "data_std_": (self.data_std_.tolist() if self.data_std_ is not None else None),
        }
        with open(str(json_path), "w") as f:
            json.dump(state_dict, f)

    @classmethod
    def load(cls, json_path: str) -> StandardScaler:
        with open(json_path) as f:
            state_dict = json.load(f)

        config = StandardScalerConfig(mean=state_dict["mean"], std=state_dict["std"])
        scaler = cls(config)
        scaler.data_mean_ = (
            torch.tensor(state_dict["data_mean_"]) if state_dict["data_mean_"] is not None else None
        )
        scaler.data_std_ = (
            torch.tensor(state_dict["data_std_"]) if state_dict["data_std_"] is not None else None
        )

        return scaler
