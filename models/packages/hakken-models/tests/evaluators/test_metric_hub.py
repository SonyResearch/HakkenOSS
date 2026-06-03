import math

import pytest
import torch

from hakken_models.evaluators.metric_bundle import MetricBundle
from hakken_models.evaluators.metric_hub import MetricHub


def _mean_rank_bundle(name: str = "mean_rank") -> MetricBundle:
    return MetricBundle(
        name=name,
        metric_class="hakken_models.evaluators.metrics.mean_rank_metric.MeanRankMetric",
        metric_kwargs={},
        input_bindings={
            "pos_scores": "pos_scores",
            "neg_scores": "neg_scores",
        },
    )


def _mse_bundle(name: str) -> MetricBundle:
    return MetricBundle(
        name=name,
        metric_class="torchmetrics.MeanSquaredError",
        metric_kwargs={},
        input_bindings={"p": "preds", "t": "target"},
    )


def test_metric_hub_empty_ok() -> None:
    MetricHub([])


def test_metric_hub_unique_names_ok() -> None:
    MetricHub([_mse_bundle("a"), _mse_bundle("b")])


def test_metric_hub_duplicate_names_raise() -> None:
    with pytest.raises(ValueError, match="duplicates:"):
        MetricHub([_mse_bundle("dup"), _mse_bundle("dup")])


def test_metric_hub_mean_rank_two_updates_global_mean() -> None:
    hub = MetricHub([_mean_rank_bundle()])
    hub.update(
        pos_scores=torch.tensor([0.5, 0.5]),
        neg_scores=torch.tensor(
            [
                [0.4, 0.6, 0.3],
                [0.1, 0.2, 0.4],
            ]
        ),
    )
    hub.update(
        pos_scores=torch.tensor([0.0]),
        neg_scores=torch.tensor([[0.0, 0.0]]),
    )
    results = hub.compute()
    assert set(results.keys()) == {"mean_rank"}
    assert results["mean_rank"].item() == pytest.approx(4.0 / 3.0)


def test_metric_hub_mean_rank_input_bindings_alias() -> None:
    bundle = MetricBundle(
        name="mr",
        metric_class="hakken_models.evaluators.metrics.mean_rank_metric.MeanRankMetric",
        metric_kwargs={},
        input_bindings={
            "batch_pos": "pos_scores",
            "batch_neg": "neg_scores",
        },
    )
    hub = MetricHub([bundle])
    hub.update(batch_pos=torch.tensor([0.0]), batch_neg=torch.tensor([[1.0]]))
    assert hub.compute()["mr"].item() == pytest.approx(2.0)


def test_metric_hub_skip_if_missing_inputs_second_bundle() -> None:
    mse_skip = MetricBundle(
        name="mse_skip",
        metric_class="torchmetrics.MeanSquaredError",
        metric_kwargs={},
        input_bindings={"p": "preds", "t": "target"},
        skip_if_missing_inputs=True,
    )
    hub = MetricHub([_mean_rank_bundle(), mse_skip])
    hub.update(pos_scores=torch.tensor([0.0]), neg_scores=torch.tensor([[1.0]]))
    out = hub.compute()
    assert "mean_rank" in out
    hub.reset()
    hub.update(
        pos_scores=torch.tensor([0.0]),
        neg_scores=torch.tensor([[1.0]]),
        preds=torch.tensor([0.0]),
        target=torch.tensor([0.0]),
    )
    out2 = hub.compute()
    assert "mse_skip" in out2


def test_metric_hub_mean_rank_compute_and_reset() -> None:
    hub = MetricHub([_mean_rank_bundle()])
    hub.update(pos_scores=torch.tensor([0.0]), neg_scores=torch.tensor([[1.0]]))
    r1 = hub.compute_and_reset()
    assert r1["mean_rank"].item() == pytest.approx(2.0)
    r2 = hub.compute()
    assert math.isinf(r2["mean_rank"].item())
