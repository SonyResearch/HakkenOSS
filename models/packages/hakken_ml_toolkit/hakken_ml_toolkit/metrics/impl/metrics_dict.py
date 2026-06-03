from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torchmetrics as tm

from hakken_ml_toolkit.metrics.core.contracts.metric import MetricConfig
from hakken_ml_toolkit.metrics.core.exceptions import UnknownReductionError

if TYPE_CHECKING:
    import torch
    from torchmetrics.aggregation import BaseAggregator

    from hakken_ml_toolkit.metrics.core.entities import FloatTensor1D


class MetricsDictConfig(MetricConfig):
    pass


class MetricsDict:
    def __init__(
        self,
        config: MetricsDictConfig,
        metrics_dict: dict[str, BaseAggregator] | None = None,
    ):
        self.config = config
        self.metrics_dict = {} if metrics_dict is None else metrics_dict

    def to(self, device: str | torch.device) -> None:
        for metric in self.metrics_dict.values():
            metric.to(device)

    def get_keys(self, filter: str | None = None):
        if filter is None:
            return list(self.metrics_dict.keys())
        return [k for k in self.metrics_dict if re.search(filter, k) is not None]

    def add(self, name: str) -> None:
        metric: BaseAggregator
        if self.config.reduce == "mean":
            metric = tm.MeanMetric()
        elif self.config.reduce == "sum":
            metric = tm.SumMetric()
        else:
            raise UnknownReductionError()
        self.metrics_dict[name] = metric

    def reset(self, regex: str | None = None) -> None:
        keys_list = self.get_keys(regex)
        for k, v in self.metrics_dict.items():
            if k in keys_list:
                v.reset()

    def update(self, name: str, value: FloatTensor1D | torch.Tensor | float) -> None:
        if name not in self.metrics_dict:
            self.add(name)

        self.metrics_dict[name].update(value)

    def compute(self, regex: str | None = None) -> dict[str, float]:
        keys_list = self.get_keys(regex)

        metric_dict = {}
        for k in keys_list:
            v = self.metrics_dict[k]
            value = v.compute().item()
            metric_dict[k] = value
        return metric_dict
