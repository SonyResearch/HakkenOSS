from typing import Any

from pydantic import BaseModel

from kge.evaluator.entities import MetricModel


class KGEEvaluatorConfig(BaseModel):
    ranking_metrics: dict[str, MetricModel]
    relation_clf_metrics: dict[str, MetricModel]
    loader_kwargs: dict[str, Any]
    filter_list: list[str] | None
    enable: bool = True


class MimicKGEEvaluatorConfig(BaseModel):
    metrics: dict[str, MetricModel]
    loader_kwargs: dict[str, Any]
    enable: bool = True
