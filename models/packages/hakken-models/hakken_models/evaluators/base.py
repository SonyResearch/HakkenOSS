from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Generic, TypeVar, cast

import torch
from loguru import logger
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from hakken_models.core.configs.evaluator import MetricConfig
from hakken_models.core.utils.runtime import instantiate_from_string

from .metric import MetricLike

ModuleType = TypeVar("ModuleType", bound=Module)
BatchType = TypeVar("BatchType")


@dataclass
class MetricBundle:
    instance: MetricLike
    kwargs: dict[str, Any]
    config: MetricConfig
    relation_id: int | None = None

    def to(self, device: torch.device | str) -> "MetricBundle":
        self.instance = self.instance.to(device)
        return self


class HakkenModelEvaluator(ABC, Generic[ModuleType, BatchType]):
    def __init__(
        self,
        metrics_config: list[MetricConfig],
        max_num_batches: int = 1_000_000,
        num_relations: int | None = None,
    ) -> None:
        self.metrics_config = metrics_config
        self.max_num_batches = max_num_batches
        self.metrics: dict[str, MetricBundle] = {}
        self.num_relations = num_relations

        self._initialize_metrics()

    @cached_property
    def unique_prediction_modes(self) -> set[str]:
        """Get the unique prediction modes required by the metrics.

        Returns:
            A set of unique prediction modes.
        """
        prediction_modes = set()
        for metric_config in self.metrics_config:
            if metric_config.prediction_mode is not None:
                prediction_modes.add(metric_config.prediction_mode)

        return prediction_modes

    def _build_parameter_context(self) -> dict[str, Any]:
        return {
            "num_relations": self.num_relations,
            "max_num_batches": self.max_num_batches,
        }

    def _initialize_metrics(self) -> None:
        """Initialize all metrics from configuration."""

        context = self._build_parameter_context()
        for metric_config in self.metrics_config:
            metric_name = metric_config.name
            # Instantiate metric from class path
            metric_kwargs = metric_config.resolve_parameters(context)

            if metric_config.divide_by_relation:
                for relation_id in range(self.num_relations):
                    metric = instantiate_from_string(
                        metric_config.target_class, expected_type=MetricLike, **metric_kwargs
                    )
                    matric_name_i = f"{metric_name}_relation_{relation_id}"
                    self.metrics[matric_name_i] = MetricBundle(
                        instance=metric,
                        kwargs=metric_kwargs,
                        config=metric_config,
                        relation_id=relation_id,
                    )

            else:
                metric = instantiate_from_string(
                    metric_config.target_class, expected_type=MetricLike, **metric_kwargs
                )

                if metric_name in self.metrics:
                    raise ValueError(
                        f"Duplicate metric name '{metric_name}'. "
                        "Provide unique names in MetricConfig or use different class paths."
                    )

                self.metrics[metric_name] = MetricBundle(
                    instance=metric, kwargs=metric_kwargs, config=metric_config
                )

    def reset(self) -> None:
        """Reset all metrics to their initial state."""
        for metric in self.metrics.values():
            metric.instance.reset()

    @abstractmethod
    def update_from_batch(self, model: ModuleType, batch: BatchType) -> None:
        """Update metrics based on a single batch.

        Args:
            model: The model to evaluate
            batch: A single batch of data
        """
        pass

    @torch.no_grad()
    def update_from_dataloader(self, model: ModuleType, data_loader: DataLoader) -> None:
        model.eval()

        for i, batch in enumerate(
            tqdm(data_loader, desc="Processing batches", total=len(data_loader))
        ):
            if i >= self.max_num_batches:
                break

            self.update_from_batch(model, batch)

    def compute(self) -> list[dict]:
        """Compute and return the current values of all metrics.

        Returns:
            List of dictionaries, each containing:
                - "name": Metric name
                - "value": Computed metric value (as float)
                - Additional metadata from metric config kwargs

        Raises:
            RuntimeError: If metrics have not been updated or computation fails
        """
        if not self.metrics:
            raise RuntimeError("No metrics have been initialized. Cannot compute results.")

        results = []
        for name, metric_bundle in self.metrics.items():
            metric_kwargs = metric_bundle.kwargs
            metric = metric_bundle.instance
            metric_value = cast(Tensor, metric.compute())
            if metric_value.numel() != 1:
                metric_value = metric_value.mean()
                logger.warning(
                    f"Metric '{name}' returned a non-scalar value. "
                    "Averaging to get a single value for logging."
                )
            results.append({"name": name, "value": metric_value.item(), **metric_kwargs})
        return results

    def __repr__(self) -> str:
        """Return string representation of the evaluator."""
        metric_names = ", ".join(self.metrics.keys())
        return f"{self.__class__.__name__}(metrics=[{metric_names}])"
