import pytest
import torch
from torchmetrics import MeanMetric

from hakken_ml_toolkit.metrics.core.exceptions import UnknownReductionError
from hakken_ml_toolkit.metrics.impl.metrics_dict import MetricsDict, MetricsDictConfig


@pytest.fixture
def metrics_dict() -> MetricsDict:
    config = MetricsDictConfig(reduce="mean")
    return MetricsDict(config=config)


def test_metrics_dict_init(metrics_dict: MetricsDict) -> None:
    assert isinstance(metrics_dict.config, MetricsDictConfig)
    assert metrics_dict.metrics_dict == {}


def test_metrics_dict_add(metrics_dict: MetricsDict) -> None:
    metrics_dict.add("metric1")

    assert "metric1" in metrics_dict.metrics_dict
    assert isinstance(metrics_dict.metrics_dict["metric1"], MeanMetric)


def test_metrics_dict_add_invalid_reduce() -> None:
    config = MetricsDictConfig(reduce="invalid")
    metrics_dict = MetricsDict(config=config)

    with pytest.raises(UnknownReductionError):
        metrics_dict.add("metric1")


def test_metrics_dict_reset(metrics_dict: MetricsDict) -> None:
    metrics_dict.add("metric1")
    metrics_dict.update("metric1", torch.tensor([1.0]))
    metrics_dict.compute()
    metrics_dict.reset()

    assert torch.isnan(metrics_dict.metrics_dict["metric1"].compute()).item()


def test_metrics_dict_update_and_compute(metrics_dict: MetricsDict) -> None:
    metrics_dict.add("metric1")
    metrics_dict.update("metric1", torch.tensor([1.0]))
    metrics_dict.update("metric1", torch.tensor([2.0]))

    result = metrics_dict.compute()

    assert "metric1" in result
    assert result["metric1"] == 1.5  # Mean of [1.0, 2.0] is 1.5


def test_metrics_dict_get_keys(metrics_dict: MetricsDict) -> None:
    metrics_dict.add("metric1")
    metrics_dict.add("metric2")

    keys = metrics_dict.get_keys()
    assert "metric1" in keys
    assert "metric2" in keys


def test_metrics_dict_get_keys_with_filter(metrics_dict: MetricsDict) -> None:
    metrics_dict.add("metric1")
    metrics_dict.add("special_metric2")

    keys = metrics_dict.get_keys(filter="special")
    assert "special_metric2" in keys
    assert "metric1" not in keys


def test_metrics_dict_update_and_compute_sum(metrics_dict: MetricsDict) -> None:
    config = MetricsDictConfig(reduce="sum")
    metrics_dict = MetricsDict(config=config)

    metrics_dict.add("metric1")
    metrics_dict.update("metric1", torch.tensor([1.0]))
    metrics_dict.update("metric1", torch.tensor([2.0]))

    result = metrics_dict.compute()

    assert "metric1" in result
    assert result["metric1"] == 3.0  # Sum of [1.0, 2.0] is 3.0
