import math

import pytest
import torch

from hakken_models.evaluators.metrics.mean_rank_metric import MeanRankMetric


def test_mean_rank_single_batch_hand_computed() -> None:
    """Two rows: ranks 2 and 1 -> mean 1.5."""
    m = MeanRankMetric()
    pos = torch.tensor([0.5, 0.5])
    neg = torch.tensor(
        [
            [0.4, 0.6, 0.3],
            [0.1, 0.2, 0.4],
        ]
    )
    m.update(pos, neg)
    out = m.compute()
    assert out.shape == ()
    assert out.item() == pytest.approx(1.5)


def test_mean_rank_multiple_batches_weighted_by_count() -> None:
    m = MeanRankMetric()
    m.update(torch.tensor([0.0]), torch.tensor([[0.0, 0.0]]))
    m.update(torch.tensor([0.0, 0.0]), torch.tensor([[1.0, 0.0], [1.0, 1.0]]))
    out = m.compute()
    assert out.item() == pytest.approx((1.0 + 2.0 + 3.0) / 3.0)


def test_reset_clears_state() -> None:
    m = MeanRankMetric()
    m.update(torch.tensor([0.0]), torch.tensor([[1.0]]))
    m.reset()
    out = m.compute()
    assert math.isinf(out.item())


def test_compute_without_update_is_inf() -> None:
    m = MeanRankMetric()
    assert math.isinf(m.compute().item())


def test_non_finite_inputs_skipped() -> None:
    m = MeanRankMetric()
    m.update(torch.tensor([float("nan")]), torch.tensor([[0.0]]))
    assert math.isinf(m.compute().item())


def test_to_sets_device_for_compute_tensor() -> None:
    m = MeanRankMetric()
    m.to(torch.device("cpu"))
    m.update(torch.tensor([0.0]), torch.tensor([[1.0]]))
    out = m.compute()
    assert out.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_to_cuda_compute_on_cuda() -> None:
    m = MeanRankMetric()
    m.to("cuda")
    m.update(
        torch.tensor([0.0], device="cuda"),
        torch.tensor([[1.0]], device="cuda"),
    )
    out = m.compute()
    assert out.device.type == "cuda"
