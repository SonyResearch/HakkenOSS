from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from kge.common.constants import TargetType
from kge.common.entities.metric import MetricI


class MetricModel(BaseModel):
    class_path: str
    parameter_grid: dict[str, list[Any]]
    update_args_mapping: dict[str, str]
    target_list: list[TargetType]


@dataclass
class EvaluationMetric:
    name: str
    parameters: dict[str, Any]
    update_args_mapping: dict[str, str]
    metric_instance: MetricI
    target: TargetType

    def __str__(self):
        return (
            "EvaluationMetric("
            f"name={self.name}, "
            f"parameters={self.parameters}, "
            f"target={self.target}"
            ")"
        )
