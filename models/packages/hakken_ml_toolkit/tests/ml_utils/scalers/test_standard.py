import tempfile
from pathlib import Path

import pytest

try:
    import torch
except ImportError:
    pytest.skip("PyTorch is not installed", allow_module_level=True)


from hakken_ml_toolkit.ml_utils.extras.domain import TensorND
from hakken_ml_toolkit.ml_utils.extras.scalers import (
    StandardScaler,
    StandardScalerConfig,
)


@pytest.fixture
def sample_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=torch.float32)
    mean = torch.tensor([4.0, 5.0, 6.0])
    std = torch.tensor([3.0, 3.0, 3.0])
    return data, mean, std


@pytest.fixture
def scaler() -> StandardScaler:
    config = StandardScalerConfig(mean=0.5, std=2.0)
    return StandardScaler(config)


def test_init(scaler: StandardScaler) -> None:
    assert scaler.mean == 0.5
    assert scaler.std == 2.0


def test_fit(
    scaler: StandardScaler, sample_data: tuple[TensorND, torch.Tensor, torch.Tensor]
) -> None:
    data, expected_mean, expected_std = sample_data
    scaler.fit(data)

    assert scaler.data_mean_ is not None
    assert scaler.data_std_ is not None

    assert torch.allclose(scaler.data_mean_, expected_mean)
    assert torch.allclose(scaler.data_std_, expected_std)


def test_transform(
    scaler: StandardScaler, sample_data: tuple[TensorND, torch.Tensor, torch.Tensor]
) -> None:
    data, _, _ = sample_data
    scaler.fit(data)
    transformed = scaler.transform(data)
    expected = torch.tensor(
        [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32
    )

    expected = expected * 2.0 + 0.5  # Apply scaler's mean and std
    assert torch.allclose(transformed, expected)


def test_inverse_transform(
    scaler: StandardScaler, sample_data: tuple[TensorND, torch.Tensor, torch.Tensor]
) -> None:
    data, _, _ = sample_data
    scaler.fit(data)
    transformed = scaler.transform(data)
    inverse_transformed = scaler.inverse_transform(transformed)
    assert torch.allclose(inverse_transformed, data)


def test_fit_from_iterator(
    scaler: StandardScaler, sample_data: tuple[TensorND, torch.Tensor, torch.Tensor]
) -> None:
    data, expected_mean, expected_std = sample_data
    iterator = iter([data])
    scaler.fit_from_iterator(iterator)

    assert scaler.data_mean_ is not None
    assert scaler.data_std_ is not None

    assert torch.allclose(scaler.data_mean_, expected_mean)
    assert torch.allclose(scaler.data_std_, expected_std)


def test_to(
    scaler: StandardScaler, sample_data: tuple[TensorND, torch.Tensor, torch.Tensor]
) -> None:
    data, _, _ = sample_data
    scaler.fit(data)
    assert scaler.data_mean_ is not None
    assert scaler.data_std_ is not None
    original_device = scaler.data_mean_.device
    scaler.to("cpu")
    assert scaler.data_mean_.device == torch.device("cpu")
    assert scaler.data_std_.device == torch.device("cpu")
    scaler.to(str(original_device))


def test_save_load(
    scaler: StandardScaler, sample_data: tuple[TensorND, torch.Tensor, torch.Tensor]
) -> None:
    data, _, _ = sample_data
    scaler.fit(data)

    assert scaler.data_mean_ is not None
    assert scaler.data_std_ is not None

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        temp_path = Path(tmp.name)

    try:
        scaler.save(temp_path)
        loaded_scaler = StandardScaler.load(str(temp_path))

        assert loaded_scaler.data_mean_ is not None
        assert loaded_scaler.data_std_ is not None

        assert scaler.mean == loaded_scaler.mean
        assert scaler.std == loaded_scaler.std

        assert torch.allclose(scaler.data_mean_, loaded_scaler.data_mean_)
        assert torch.allclose(scaler.data_std_, loaded_scaler.data_std_)
    finally:
        temp_path.unlink()


def test_error_on_zero_std(scaler: StandardScaler) -> None:
    data = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=torch.float32)
    with pytest.raises(
        Exception,
        match=r"Standard deviation is 0 for some of the features. Cannot scale.",
    ):
        scaler.fit(data)


def test_error_on_unfitted_transform(
    scaler: StandardScaler, sample_data: tuple[TensorND, torch.Tensor, torch.Tensor]
) -> None:
    data, _, _ = sample_data
    with pytest.raises(Exception, match=r"Scaler is not fitted yet."):
        scaler.transform(data)


def test_error_on_unfitted_inverse_transform(
    scaler: StandardScaler, sample_data: tuple[TensorND, torch.Tensor, torch.Tensor]
) -> None:
    data, _, _ = sample_data
    with pytest.raises(Exception, match=r"Scaler is not fitted yet."):
        scaler.inverse_transform(data)
