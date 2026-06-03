from collections.abc import Iterator
from pathlib import Path

import pytest

try:
    import torch
except ImportError:
    pytest.skip("PyTorch is not installed", allow_module_level=True)


from pydantic_core import ValidationError

from hakken_ml_toolkit.ml_utils.extras.domain import TensorND
from hakken_ml_toolkit.ml_utils.extras.scalers import (
    ScalerI,
    SigmoidScaler,
    SigmoidScalerConfig,
)
from hakken_ml_toolkit.ml_utils.extras.scalers.core.values.exceptions import (
    ScalerNotFittedError,
)


@pytest.fixture
def sample_data() -> TensorND:
    return torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])


@pytest.fixture
def sample_iterator() -> Iterator[TensorND]:
    data = [
        torch.tensor([[1.0, 2.0, 3.0]]),
        torch.tensor([[4.0, 5.0, 6.0]]),
        torch.tensor([[7.0, 8.0, 9.0]]),
    ]
    return iter(data)


def test_sigmoid_scaler_init() -> None:
    config = SigmoidScalerConfig(temperature=1.0)
    scaler = SigmoidScaler(config)
    assert isinstance(scaler, ScalerI)
    assert scaler.learn_data_min is True
    assert scaler.fixed_data_min is None


def test_sigmoid_scaler_init_with_fixed_data_min() -> None:
    config = SigmoidScalerConfig(temperature=2.0, fixed_data_min=[0.0, 1.0, 2.0])
    scaler = SigmoidScaler(config)
    assert scaler.learn_data_min is False
    assert scaler.fixed_data_min == [0.0, 1.0, 2.0]


def test_sigmoid_scaler_fit(sample_data: TensorND) -> None:
    scaler = SigmoidScaler(SigmoidScalerConfig())
    scaler.fit(sample_data)
    assert torch.allclose(scaler.data_min_, torch.tensor([1.0, 2.0, 3.0]))  # type: ignore


def test_sigmoid_scaler_fit_with_fixed_data_min(sample_data: TensorND) -> None:
    config = SigmoidScalerConfig(fixed_data_min=[0.0, 1.0, 2.0])
    scaler = SigmoidScaler(config)
    scaler.fit(sample_data)
    assert torch.allclose(scaler.data_min_, torch.tensor([0.0, 1.0, 2.0]))  # type: ignore


def test_sigmoid_scaler_fit_from_iterator(sample_iterator: Iterator[TensorND]) -> None:
    scaler = SigmoidScaler(SigmoidScalerConfig())
    scaler.fit_from_iterator(sample_iterator)
    assert torch.allclose(scaler.data_min_, torch.tensor([1.0, 2.0, 3.0]))  # type: ignore


def test_sigmoid_scaler_fit_from_iterator_with_fixed_data_min(
    sample_iterator: Iterator[TensorND],
) -> None:
    config = SigmoidScalerConfig(fixed_data_min=[0.0, 1.0, 2.0])
    scaler = SigmoidScaler(config)
    scaler.fit_from_iterator(sample_iterator)
    assert torch.allclose(scaler.data_min_, torch.tensor([0.0, 1.0, 2.0]))  # type: ignore


def test_sigmoid_scaler_transform(sample_data: TensorND) -> None:
    scaler = SigmoidScaler(SigmoidScalerConfig(temperature=1.0))
    scaler.fit(sample_data)
    transformed = scaler.transform(sample_data)

    # Expected sigmoid transformation: sigmoid((data - data_min) / temperature)
    expected_z = (sample_data - torch.tensor([1.0, 2.0, 3.0])) / 1.0
    expected = torch.sigmoid(expected_z)

    assert torch.allclose(transformed, expected)


def test_sigmoid_scaler_transform_with_temperature(sample_data: TensorND) -> None:
    scaler = SigmoidScaler(SigmoidScalerConfig(temperature=2.0))
    scaler.fit(sample_data)
    transformed = scaler.transform(sample_data)

    # Expected sigmoid transformation with temperature=2.0
    expected_z = (sample_data - torch.tensor([1.0, 2.0, 3.0])) / 2.0
    expected = torch.sigmoid(expected_z)

    assert torch.allclose(transformed, expected)


def test_sigmoid_scaler_inverse_transform(sample_data: TensorND) -> None:
    scaler = SigmoidScaler(SigmoidScalerConfig())
    scaler.fit(sample_data)
    transformed = scaler.transform(sample_data)
    inverse_transformed = scaler.inverse_transform(transformed)
    assert torch.allclose(inverse_transformed, sample_data, atol=1e-6)


def test_sigmoid_scaler_inverse_transform_with_temperature(
    sample_data: TensorND,
) -> None:
    scaler = SigmoidScaler(SigmoidScalerConfig(temperature=0.5))
    scaler.fit(sample_data)
    transformed = scaler.transform(sample_data)

    inverse_transformed = scaler.inverse_transform(transformed)
    assert torch.allclose(inverse_transformed, sample_data, atol=1e-2)


def test_sigmoid_scaler_error_before_fit() -> None:
    scaler = SigmoidScaler(SigmoidScalerConfig())
    data = torch.tensor([[1.0, 2.0, 3.0]])
    with pytest.raises(ScalerNotFittedError):
        scaler.transform(data)


def test_sigmoid_scaler_error_inverse_transform_before_fit() -> None:
    scaler = SigmoidScaler(SigmoidScalerConfig())
    data = torch.tensor([[0.5, 0.7, 0.9]])
    with pytest.raises(ScalerNotFittedError):
        scaler.inverse_transform(data)


def test_sigmoid_scaler_save_load(tmp_path: Path, sample_data: TensorND) -> None:
    scaler = SigmoidScaler(SigmoidScalerConfig(temperature=1.5))
    scaler.fit(sample_data)

    save_path = tmp_path / "sigmoid_scaler.json"
    scaler.save(save_path)

    loaded_scaler = SigmoidScaler.load(str(save_path))
    assert isinstance(loaded_scaler, SigmoidScaler)
    assert loaded_scaler.config.temperature == 1.5

    assert torch.allclose(loaded_scaler.data_min_, scaler.data_min_)  # type: ignore
    assert loaded_scaler.temperature_.shape == scaler.temperature_.shape  # type: ignore
    assert torch.allclose(loaded_scaler.temperature_, scaler.temperature_)  # type: ignore

    out_orig = scaler.transform(sample_data)
    out_loaded = loaded_scaler.transform(sample_data)
    assert torch.allclose(out_orig, out_loaded)


def test_sigmoid_scaler_save_load_with_fixed_data_min(
    tmp_path: Path, sample_data: TensorND
) -> None:
    # Arrange
    config = SigmoidScalerConfig(temperature=2.0, fixed_data_min=[0.0, 1.0, 2.0])
    scaler = SigmoidScaler(config)
    scaler.fit(sample_data)

    # Act
    save_path = tmp_path / "sigmoid_scaler_fixed.json"
    scaler.save(save_path)

    loaded_scaler = SigmoidScaler.load(str(save_path))

    # Assert
    assert isinstance(loaded_scaler, SigmoidScaler)

    assert loaded_scaler.config.temperature == 2.0
    assert loaded_scaler.config.fixed_data_min == [0.0, 1.0, 2.0]

    assert torch.allclose(loaded_scaler.data_min_, scaler.data_min_)  # type: ignore
    assert loaded_scaler.temperature_.shape == scaler.temperature_.shape  # type: ignore
    assert torch.allclose(loaded_scaler.temperature_, scaler.temperature_)  # type: ignore

    assert loaded_scaler.temperature_ is not None
    assert torch.allclose(
        loaded_scaler.temperature_, torch.full_like(loaded_scaler.temperature_, 2.0)
    )

    out_orig = scaler.transform(sample_data)
    out_loaded = loaded_scaler.transform(sample_data)
    assert torch.allclose(out_orig, out_loaded)


def test_sigmoid_scaler_save_load_with_learn_temperature(
    tmp_path: Path, sample_data: TensorND
) -> None:
    # Arrange
    eps = 1e-2
    config = SigmoidScalerConfig(temperature=None, target_eps=eps)
    scaler = SigmoidScaler(config)
    scaler.fit(sample_data)

    # Act
    save_path = tmp_path / "sigmoid_scaler_fixed.json"
    scaler.save(save_path)

    loaded_scaler = SigmoidScaler.load(str(save_path))

    # Assert
    assert scaler.data_min_ is not None
    assert scaler.temperature_ is not None
    assert scaler.data_min_.shape == scaler.temperature_.shape
    assert torch.all(scaler.temperature_ > 0)

    assert isinstance(loaded_scaler, SigmoidScaler)

    assert loaded_scaler.config.temperature is None
    assert loaded_scaler.config.target_eps == eps

    assert torch.allclose(loaded_scaler.data_min_, scaler.data_min_)  # type: ignore
    assert loaded_scaler.temperature_.shape == scaler.temperature_.shape  # type: ignore
    assert torch.allclose(loaded_scaler.temperature_, scaler.temperature_)  # type: ignore

    out_before = scaler.transform(sample_data)
    out_after = loaded_scaler.transform(sample_data)
    assert torch.allclose(out_before, out_after)

    if scaler.temperature_.numel() > 1:  # type: ignore[attr-defined]
        assert torch.std(scaler.temperature_) > 0  # type: ignore[arg-type]


def test_sigmoid_scaler_to_device(sample_data: TensorND) -> None:
    scaler = SigmoidScaler(SigmoidScalerConfig())
    scaler.fit(sample_data)

    # Test moving to CPU (which should be a no-op if already on CPU)
    scaler.to("cpu")
    assert scaler.data_min_.device.type == "cpu"  # type: ignore

    # Test that it doesn't crash when data_min_ is None
    unfitted_scaler = SigmoidScaler(SigmoidScalerConfig())
    unfitted_scaler.to("cpu")  # Should not raise an error


def test_sigmoid_scaler_get_fixed_data_min() -> None:
    config = SigmoidScalerConfig(fixed_data_min=[1.0, 2.0, 3.0])
    scaler = SigmoidScaler(config)

    # Test without data
    fixed_min = scaler._get_fixed_data_min()
    expected = torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(fixed_min, expected)

    # Test with data (should expand to match data shape)
    data = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    fixed_min_expanded = scaler._get_fixed_data_min(data)
    expected_expanded = torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(fixed_min_expanded, expected_expanded)


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0, 5.0])
def test_sigmoid_scaler_different_temperatures(temperature: float, sample_data: TensorND) -> None:
    config = SigmoidScalerConfig(temperature=temperature)
    scaler = SigmoidScaler(config)
    scaler.fit(sample_data)
    transformed = scaler.transform(sample_data)
    print(transformed)

    # All values should be between 0 and 1 due to sigmoid
    assert torch.all(transformed >= 0.0)
    assert torch.all(transformed <= 1.0)

    # Test round-trip transformation
    inverse_transformed = scaler.inverse_transform(transformed)
    assert torch.allclose(inverse_transformed, sample_data, atol=1e-2)


def test_sigmoid_scaler_edge_cases() -> None:
    # Test with single value data
    single_data = torch.tensor([[5.0]])
    scaler = SigmoidScaler(SigmoidScalerConfig())
    scaler.fit(single_data)
    transformed = scaler.transform(single_data)
    expected = torch.sigmoid(torch.tensor([[0.0]]))  # (5.0 - 5.0) / 1.0 = 0.0
    assert torch.allclose(transformed, expected)

    # Test with negative data
    negative_data = torch.tensor([[-2.0, -1.0], [0.0, 1.0]])
    scaler = SigmoidScaler(SigmoidScalerConfig())
    scaler.fit(negative_data)
    transformed = scaler.transform(negative_data)
    assert torch.all(transformed >= 0.0)
    assert torch.all(transformed <= 1.0)


def test_sigmoid_scaler_fit_from_iterator_with_num_batches(
    sample_iterator: Iterator[TensorND],
) -> None:
    scaler = SigmoidScaler(SigmoidScalerConfig())
    scaler.fit_from_iterator(sample_iterator, num_batches=1)
    # Should only process first batch, so data_min should be [1.0, 2.0, 3.0]
    assert torch.allclose(scaler.data_min_, torch.tensor([1.0, 2.0, 3.0]))  # type: ignore


def test_sigmoid_scaler_config_validation() -> None:
    with pytest.raises(ValidationError):
        SigmoidScalerConfig(temperature=0.1)

    with pytest.raises(ValidationError):
        SigmoidScalerConfig(temperature=-1.0)

    # Valid temperature should work
    config = SigmoidScalerConfig(temperature=0.5)
    assert config.temperature == 0.5
