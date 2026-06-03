from __future__ import annotations

import json
from typing import TYPE_CHECKING

import torch
from loguru import logger

from hakken_ml_toolkit.ml_utils.extras.scalers.core.contracts.scaler import (
    ScalerConfig,
    ScalerI,
)
from hakken_ml_toolkit.ml_utils.extras.scalers.core.values.exceptions import (
    ScalerNotFittedError,
    ZeroRangeError,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from hakken_ml_toolkit.ml_utils.extras.domain import TensorND


class MinMaxScalerConfig(ScalerConfig):
    feature_range: tuple[float, float] = (0, 1)


class MinMaxScaler(ScalerI[MinMaxScalerConfig]):
    def __init__(self, config: MinMaxScalerConfig):
        super().__init__(config)
        self.feature_range = config.feature_range
        self.data_min_: torch.Tensor | None = None
        self.data_max_: torch.Tensor | None = None
        self.data_range_: torch.Tensor | None = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(feature_range={self.feature_range})"

    def to(self, device: str | torch.device) -> None:
        if self.data_min_ is not None:
            self.data_min_ = self.data_min_.to(device)
        if self.data_max_ is not None:
            self.data_max_ = self.data_max_.to(device)
        if self.data_range_ is not None:
            self.data_range_ = self.data_range_.to(device)

    def fit(self, data: TensorND) -> None:
        self.data_min_ = torch.min(data, dim=0)[0]
        self.data_max_ = torch.max(data, dim=0)[0]
        self.data_range_ = self.data_max_ - self.data_min_  # type: ignore
        if torch.any(self.data_range_ == 0):
            raise ZeroRangeError()

    def fit_from_iterator(
        self, iterator: Iterator[TensorND], num_batches: int | None = None
    ) -> None:
        data_min: torch.Tensor
        data_max: torch.Tensor
        first_batch = True
        for i, batch_tensor in enumerate(iterator):
            if num_batches is not None and i > num_batches:
                break
            batch = batch_tensor
            if first_batch:
                data_min = torch.min(batch, dim=0)[0]
                data_max = torch.max(batch, dim=0)[0]
                first_batch = False
            else:
                data_min = torch.min(data_min, torch.min(batch, dim=0)[0])
                data_max = torch.max(data_max, torch.max(batch, dim=0)[0])

        self.data_min_ = data_min
        self.data_max_ = data_max

        self.data_range_ = self.data_max_ - self.data_min_  # type: ignore

        # If any of the data_range_ is 0, we throw an error
        if torch.any(self.data_range_ == 0):
            raise ZeroRangeError()

    def transform(self, data: TensorND) -> TensorND:
        if self.data_range_ is None:
            raise ScalerNotFittedError()

        x_std = (data - self.data_min_) / self.data_range_  # type: ignore
        x_scaled = self.feature_range[0] + x_std * (self.feature_range[1] - self.feature_range[0])

        if torch.any(x_scaled < self.feature_range[0]) or torch.any(
            x_scaled > self.feature_range[1]
        ):
            logger.warning("Some values were outside of feature_range and have been clipped.")

        return torch.clamp(x_scaled, self.feature_range[0], self.feature_range[1])

    def inverse_transform(self, data_norm: TensorND) -> TensorND:
        range_min = self.feature_range[0]
        range_max = self.feature_range[1]

        if self.data_range_ is None or self.data_min_ is None:
            raise ScalerNotFittedError()

        if torch.any(data_norm < self.feature_range[0]) or torch.any(
            data_norm > self.feature_range[1]
        ):
            logger.warning("Some input values were outside of feature_range and have been clipped.")

        # Clip values between feature_range
        data_norm_clipped = torch.clamp(data_norm, range_min, range_max)

        x_original = (data_norm_clipped - range_min) / (range_max - range_min)
        return self.data_min_ + x_original * self.data_range_

    def save(self, json_path: str | Path):
        state_dict = {
            "feature_range": self.feature_range,
            "data_min_": (self.data_min_.tolist() if self.data_min_ is not None else None),
            "data_max_": (self.data_max_.tolist() if self.data_max_ is not None else None),
        }
        with open(str(json_path), "w") as f:
            json.dump(state_dict, f)

    @classmethod
    def load(cls, json_path: str) -> MinMaxScaler:
        with open(json_path) as f:
            state_dict = json.load(f)

        config = MinMaxScalerConfig(feature_range=tuple(state_dict["feature_range"]))
        scaler = cls(config)
        scaler.data_min_ = torch.tensor(state_dict["data_min_"])
        scaler.data_max_ = torch.tensor(state_dict["data_max_"])

        scaler.data_range_ = scaler.data_max_ - scaler.data_min_

        return scaler
