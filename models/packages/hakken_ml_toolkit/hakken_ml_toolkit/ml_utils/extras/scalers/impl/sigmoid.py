from __future__ import annotations

import json
from typing import TYPE_CHECKING

import torch
from loguru import logger
from pydantic import Field

from hakken_ml_toolkit.ml_utils.extras.scalers.core.contracts.scaler import (
    ScalerConfig,
    ScalerI,
)
from hakken_ml_toolkit.ml_utils.extras.scalers.core.values.exceptions import (
    ScalerNotFittedError,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from hakken_ml_toolkit.ml_utils.extras.domain import TensorND


class SigmoidScalerConfig(ScalerConfig):
    """Configuration for SigmoidScaler."""

    temperature: float | None = Field(
        default=1.0,
        ge=0.5,
        description=(
            "Scaling temperature (> 0.5). Controls smoothness of the sigmoid. "
            "If None it will be learnt."
        ),
    )
    fixed_data_min: list[float] | None = Field(
        default=None,
        description=(
            "Optional fixed data_min values. If None, data_min will be learned during fitting."
        ),
    )

    target_eps: float = Field(
        default=1e-2,
        gt=0.0,
        lt=0.5,
        description="Target epsilon so that median maps to 1 - eps.",
    )

    temperature_min: float = Field(
        default=1e-6,
        gt=0.0,
        description="Lower bound clamp for learned temperature for numerical stability.",
    )


class SigmoidScaler(ScalerI[SigmoidScalerConfig]):
    def __init__(self, config: SigmoidScalerConfig):
        super().__init__(config)
        self.learn_data_min = config.fixed_data_min is None
        self.fixed_data_min = config.fixed_data_min
        self.data_min_: torch.Tensor | None = None
        self.temperature_: torch.Tensor | None = None

    @property
    def data_min(self) -> torch.Tensor:
        if self.data_min_ is None:
            raise ScalerNotFittedError()
        return self.data_min_

    @property
    def temperature(self) -> torch.Tensor:
        if self.temperature_ is None:
            raise ScalerNotFittedError()
        return self.temperature_

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(temperature={self.temperature})"

    def to(self, device: str | torch.device) -> None:
        if self.data_min_ is not None:
            self.data_min_ = self.data_min_.to(device)
        if self.temperature_ is not None:
            self.temperature_ = self.temperature_.to(device)

    def _get_fixed_data_min(self, data: TensorND | None = None) -> torch.Tensor:
        data_min = torch.tensor(self.fixed_data_min, dtype=torch.float32)
        if data is not None and data_min.ndim == 1 and data.shape[1] != data_min.shape[0]:
            # Expand or validate length
            if data_min.numel() == 1:
                data_min = data_min.expand(data.shape[1])
            else:
                msg = "fixed_data_min length must match data features."
                raise ValueError(msg)
        return data_min

    def _learn_temperature_from_data(self, data: TensorND) -> torch.Tensor:
        """
        Learn per-feature temperature so that sigmoid(median/T) = 1 - eps.
        """
        eps = self.config.target_eps
        logit_scale = torch.log(torch.tensor((1 - eps) / eps, dtype=torch.float32))  # > 0
        med = torch.median(data, dim=0).values  # [D]

        temperature = med / logit_scale

        # Guard against degenerate/negative or too small values
        # If median <= min, set a small positive temperature (will make sigmoid very steep)

        return torch.clamp(temperature, min=self.config.temperature_min)

    def fit(self, data: TensorND) -> None:
        if self.learn_data_min:
            self.data_min_ = torch.min(data, dim=0)[0]
        else:
            self.data_min_ = self._get_fixed_data_min(data)

        if self.config.temperature is None:
            self.temperature_ = self._learn_temperature_from_data(data)
        else:
            self.temperature_ = torch.full_like(
                self.data_min_, fill_value=float(self.config.temperature)
            )

        logger.debug(
            "Fitted SigmoidScaler: data_min_.shape={}, temperature_.shape={}",
            tuple(self.data_min_.shape),
            tuple(self.temperature_.shape),
        )

    def fit_from_iterator(
        self, iterator: Iterator[TensorND], num_batches: int | None = None
    ) -> None:
        data_min: torch.Tensor | None = None
        sampled: list[torch.Tensor] = []

        for i, batch in enumerate(iterator):
            if num_batches is not None and i >= num_batches:
                break

            batch_min = torch.min(batch, dim=0).values
            data_min = batch_min if data_min is None else torch.minimum(data_min, batch_min)

            if self.config.temperature is None:
                sampled.append(batch)

        if data_min is None:
            msg = "fit_from_iterator received no batches."
            raise ValueError(msg)

        self.data_min_ = (
            data_min
            if self.learn_data_min
            else self._get_fixed_data_min(sampled[0] if sampled else None)
        )

        if self.config.temperature is None:
            if not sampled:
                msg = "Cannot learn temperature from iterator: no data sampled."
                raise ValueError(msg)
            data_for_median = torch.cat(sampled, dim=0)
            self.temperature_ = self._learn_temperature_from_data(data_for_median)
        else:
            self.temperature_ = torch.full_like(
                self.data_min_, fill_value=float(self.config.temperature)
            )

        logger.debug(
            "Fitted SigmoidScaler: data_min_.shape={}, temperature_.shape={}",
            tuple(self.data_min_.shape),
            tuple(self.temperature_.shape),
        )

    # -----------------------
    # Transformations
    # -----------------------
    def transform(self, data: TensorND) -> TensorND:
        z = (data - self.data_min) / self.temperature

        return torch.sigmoid(z)

    def inverse_transform(self, data_norm: TensorND) -> TensorND:
        return self.data_min + self.temperature * torch.logit(data_norm, eps=1e-7)

    def save(self, json_path: str | Path):
        state_dict = {
            "temperature": self.config.temperature,
            "fixed_data_min": self.fixed_data_min,
            "target_eps": self.config.target_eps,
            "temperature_min": self.config.temperature_min,
            "data_min_": (self.data_min_.tolist() if self.data_min_ is not None else None),
            "temperature_": (self.temperature_.tolist() if self.temperature_ is not None else None),
        }
        with open(str(json_path), "w") as f:
            json.dump(state_dict, f)

    @classmethod
    def load(cls, json_path: str) -> SigmoidScaler:
        with open(json_path) as f:
            state_dict = json.load(f)

        config = SigmoidScalerConfig(
            temperature=state_dict["temperature"],
            fixed_data_min=state_dict["fixed_data_min"],
            target_eps=state_dict["target_eps"],
            temperature_min=state_dict["temperature_min"],
        )
        scaler = cls(config)
        scaler.data_min_ = torch.tensor(state_dict["data_min_"], dtype=torch.float32)
        scaler.temperature_ = torch.tensor(state_dict["temperature_"], dtype=torch.float32)
        return scaler
