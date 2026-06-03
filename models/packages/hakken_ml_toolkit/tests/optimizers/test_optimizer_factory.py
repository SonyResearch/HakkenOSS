import pytest
from torch import nn

from hakken_ml_toolkit.optimizers import OptimizerFactory
from hakken_ml_toolkit.optimizers.core.values.constants import (
    OptimizerType,
    get_random_enum,
)


@pytest.fixture
def model() -> nn.Module:
    return nn.Linear(10, 1)


@pytest.mark.parametrize("seed", list(range(10)))
def test_create_optimizer(model: nn.Module, seed: int) -> None:
    optim_type = get_random_enum(OptimizerType, seed)
    config_class = OptimizerFactory._optimizers_config[optim_type]
    config = config_class.random(seed)

    optimizer = OptimizerFactory.create(
        parameters=model.parameters(), optim_type=optim_type, config=config
    )

    target_optimizer = OptimizerFactory._optimizers[optim_type]

    assert isinstance(optimizer, target_optimizer)
