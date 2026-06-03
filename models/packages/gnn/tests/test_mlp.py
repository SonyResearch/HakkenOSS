import pytest
import torch
from hakken_ml_toolkit.ml_utils.constants import ActivationType
from torch import nn

from gnn.common.exceptions import WrongDimensionsError
from gnn.mlp import MLP, MLPConfig


# Fixtures
@pytest.fixture
def device() -> str:
    return "cpu"


@pytest.fixture
def batch_size() -> int:
    return 32


@pytest.fixture
def input_dim() -> int:
    return 10


@pytest.fixture
def hidden_dim() -> int:
    return 20


@pytest.fixture
def output_dim() -> int:
    return 5


@pytest.fixture
def sample_input(batch_size: int, input_dim: int, device: str | torch.device) -> torch.Tensor:
    return torch.randn(batch_size, input_dim, device=device)


# Test functions
@pytest.mark.parametrize(
    "num_layers,hidden_dim,expected_shapes",
    [
        (1, None, [(10, 5)]),
        (2, 20, [(10, 20), (20, 5)]),
        (3, 20, [(10, 20), (20, 20), (20, 5)]),
    ],
)
def test_layer_shapes(
    num_layers: int,
    hidden_dim: int | None,
    expected_shapes: list[tuple[int, int]],
    input_dim: int,
    output_dim: int,
    device: str,
):
    config = MLPConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        activation=ActivationType.RELU,
        device=device,
    )

    mlp = MLP(config)

    # Check number of layers
    assert len(mlp.fc_layers) == num_layers

    # Check each layer's shape
    for idx, layer in enumerate(mlp.fc_layers):
        # Get the Linear layer (first component of Sequential)
        linear_layer = next(m for m in layer.modules() if isinstance(m, nn.Linear))
        expected_in, expected_out = expected_shapes[idx]
        assert linear_layer.in_features == expected_in
        assert linear_layer.out_features == expected_out


@pytest.mark.parametrize(
    "activation,expected_activation_class",
    [
        (ActivationType.RELU, nn.ReLU),
        (ActivationType.TANH, nn.Tanh),
        (ActivationType.SIGMOID, nn.Sigmoid),
    ],
)
def test_activations(
    activation: ActivationType,
    expected_activation_class: type,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    device: str,
):
    config = MLPConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=2,
        hidden_dim=hidden_dim,
        activation=activation,
        device=device,
    )

    mlp = MLP(config)

    # Check first layer activation
    first_layer_activation = [
        m for m in mlp.fc_layers[0].modules() if isinstance(m, expected_activation_class)
    ]
    assert len(first_layer_activation) == 1


def test_forward_pass(
    sample_input: torch.Tensor,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    device: str,
):
    config = MLPConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=2,
        hidden_dim=hidden_dim,
        activation=ActivationType.RELU,
        device=device,
    )

    mlp = MLP(config)
    output = mlp(sample_input)

    assert output.shape == (sample_input.shape[0], output_dim)
    assert output.device.type == device


def test_zero_layers(input_dim: int, device: str | torch.device):
    # Test identity case (0 layers)
    config = MLPConfig(
        input_dim=input_dim,
        output_dim=input_dim,  # Must be same as input_dim
        num_layers=0,
        activation=ActivationType.RELU,
        device=device,
    )

    mlp = MLP(config)
    assert isinstance(mlp.fc_layers[0], nn.Identity)

    # Test error case when dimensions don't match
    with pytest.raises(WrongDimensionsError):
        config = MLPConfig(
            input_dim=input_dim,
            output_dim=input_dim + 1,  # Different dimension
            num_layers=0,
            activation=ActivationType.RELU,
            device=device,
        )
        MLP(config)


@pytest.mark.parametrize("use_batch_norm", [True, False])
@pytest.mark.parametrize("use_layer_norm", [True, False])
def test_normalization_layers(
    use_batch_norm: bool,
    use_layer_norm: bool,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    device: str,
):
    if use_batch_norm and use_layer_norm:
        return  # Skip invalid combination

    config = MLPConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=2,
        hidden_dim=hidden_dim,
        activation=ActivationType.RELU,
        batch_norm=use_batch_norm,
        layer_norm=use_layer_norm,
        device=device,
    )

    mlp = MLP(config)

    for layer in mlp.fc_layers:
        batch_norms = [m for m in layer.modules() if isinstance(m, nn.BatchNorm1d)]
        layer_norms = [m for m in layer.modules() if isinstance(m, nn.LayerNorm)]

        if use_batch_norm:
            assert len(batch_norms) == 1
            assert len(layer_norms) == 0
        elif use_layer_norm:
            assert len(batch_norms) == 0
            assert len(layer_norms) == 1
        else:
            assert len(batch_norms) == 0
            assert len(layer_norms) == 0


def test_dropout(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    device: str,
):
    dropout_rate = 0.5
    config = MLPConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=3,
        hidden_dim=hidden_dim,
        activation=ActivationType.RELU,
        dropout=dropout_rate,
        device=device,
    )

    mlp = MLP(config)

    # Check dropout layers exist and have correct rate
    for idx, layer in enumerate(mlp.fc_layers):
        dropouts = [m for m in layer.modules() if isinstance(m, nn.Dropout)]

        # Last layer shouldn't have dropout by default
        if idx == len(mlp.fc_layers) - 1:
            assert len(dropouts) == 0
        else:
            assert len(dropouts) == 1
            assert dropouts[0].p == dropout_rate


def test_output_activation(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    device: str,
):
    config = MLPConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=2,
        hidden_dim=hidden_dim,
        activation=ActivationType.RELU,
        use_activation_output=True,
        device=device,
    )

    mlp = MLP(config)

    # Check that last layer has activation
    last_layer_activation = [m for m in mlp.fc_layers[-1].modules() if isinstance(m, nn.ReLU)]
    assert len(last_layer_activation) == 1

    # Test without output activation
    config.use_activation_output = False
    mlp = MLP(config)

    # Check that last layer has Identity instead of activation
    last_layer_modules = list(mlp.fc_layers[-1].modules())
    assert any(isinstance(m, nn.Identity) for m in last_layer_modules)
    assert not any(isinstance(m, nn.ReLU) for m in last_layer_modules)
