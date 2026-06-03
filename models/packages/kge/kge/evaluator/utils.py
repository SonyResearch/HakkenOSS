import itertools
from typing import Any, Protocol

import torch
from hydra.utils import get_class

from kge.common.constants import TargetType
from kge.common.entities.metric import MetricI
from kge.common.types import FloatTensor1D
from kge.evaluator.entities import EvaluationMetric, MetricModel


class KGEEvalUtils(Protocol):
    @staticmethod
    def get_targets_from_sro_batch(sro_batch: torch.Tensor, target: TargetType) -> torch.Tensor:
        """Extract target column from SRO batch based on target type."""
        if target == TargetType.SUBJECT:
            return sro_batch[:, 0]
        if target == TargetType.RELATION:
            return sro_batch[:, 1]
        if target == TargetType.OBJECT:
            return sro_batch[:, 2]
        msg = f"Unsupported target type: {target}"
        raise ValueError(msg)

    @staticmethod
    def create_metrics_from_config(
        metrics_config: dict[str, MetricModel], **kwargs: Any
    ) -> tuple[list[EvaluationMetric], set[TargetType]]:
        """Create evaluation metrics from configuration."""
        metrics_list: list[EvaluationMetric] = []
        targets_list: set[TargetType] = set()

        for metric_name, metric_config in metrics_config.items():
            parameter_grid = metric_config.parameter_grid
            keys = list(parameter_grid.keys())
            combinations = list(itertools.product(*parameter_grid.values()))

            for combo in combinations:
                metric_kwargs = dict(zip(keys, combo, strict=False))
                metric_class: type[MetricI] = get_class(metric_config.class_path)

                for target in metric_config.target_list:
                    if target not in targets_list:
                        targets_list.add(target)

                    metric = metric_class(**metric_kwargs, **kwargs)
                    parameters = metric_kwargs.copy()
                    parameters["target"] = target

                    metrics_list.append(
                        EvaluationMetric(
                            name=metric_name,
                            parameters=parameters,
                            metric_instance=metric,
                            update_args_mapping=metric_config.update_args_mapping,
                            target=target,
                        )
                    )
        return metrics_list, targets_list

    @staticmethod
    def prefix_metrics(prefix: str, metrics: dict[str, FloatTensor1D]) -> dict[str, FloatTensor1D]:
        return {f"{prefix}/{key}": value for key, value in metrics.items()}
