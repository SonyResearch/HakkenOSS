from typing import Any, ClassVar

from hakken_ml_toolkit.metrics.core.contracts.metric import MetricConfig, MetricI
from hakken_ml_toolkit.metrics.core.exceptions import UnknownMetricError
from hakken_ml_toolkit.metrics.impl.f1_score import F1, F1Config
from hakken_ml_toolkit.metrics.impl.hits_at_k import HitsAtK, HitsAtKConfig
from hakken_ml_toolkit.metrics.impl.mrr import MeanReciprocalRank, MRRConfig


class MetricMapper:
    METRIC_CLASS_MAP: ClassVar[dict[str, tuple[type, type]]] = {
        MeanReciprocalRank.name: (MeanReciprocalRank, MRRConfig),
        HitsAtK.name: (HitsAtK, HitsAtKConfig),
        F1.name: (F1, F1Config),
    }

    @staticmethod
    def create(
        metrics_config: dict[str, dict[str, Any]],
    ) -> dict[str, MetricI]:
        """
        Initialize the MetricMapper with a list of metric names and their configuration.

        Args:
            metrics_config (Dict[str, Dict[str, Any]]): Dictionary where the key is the metric name
            and the value is the config for that metric.
        """
        metrics: dict[str, MetricI] = {}
        for metric_name, config_dict in metrics_config.items():
            metric_class, config_class = MetricMapper.get_metric_class(metric_name)
            metric_instance = metric_class(config=config_class(**config_dict))
            metrics[metric_name] = metric_instance

        return metrics

    @staticmethod
    def get_metric_class(metric_name: str) -> tuple[type[MetricI], type[MetricConfig]]:
        """
        Given a metric name, return the corresponding metric class.

        For metrics like hits@K, return the HitsAtK class, and for MRR, return the MRR class.
        """
        metric_name = metric_name.lower()

        # Check if the metric is in the predefined dictionary
        if metric_name in MetricMapper.METRIC_CLASS_MAP:
            return MetricMapper.METRIC_CLASS_MAP[metric_name]

        # account for hits_at_X
        if "hits_at_" in metric_name:
            return MetricMapper.METRIC_CLASS_MAP[HitsAtK.name]

        # Raise an error if the metric is unknown
        raise UnknownMetricError(metric_name)
