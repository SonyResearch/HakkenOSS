import pytest
from torch import nn

from hakken_ml_toolkit.optimizers import LRSchedulerFactory, OptimizerFactory
from hakken_ml_toolkit.optimizers.core.values.constants import (
    LRSchedulerType,
    OptimizerType,
    get_random_enum,
)


@pytest.fixture
def model() -> nn.Module:
    return nn.Linear(10, 1)


@pytest.mark.parametrize("seed_optim", list(range(3)))
@pytest.mark.parametrize("seed_lr", list(range(10)))
def test_lr_scheduler(model: nn.Module, seed_optim: int, seed_lr) -> None:
    model.train()
    optim_type = get_random_enum(OptimizerType, seed_optim)
    config_optim = OptimizerFactory._optimizers_config[optim_type].random(seed_optim)

    optimizer = OptimizerFactory.create(
        parameters=model.parameters(), optim_type=optim_type, config=config_optim
    )

    lr_sched_type = get_random_enum(LRSchedulerType, seed_lr)
    config = LRSchedulerFactory._lr_schedulers_config[lr_sched_type].random(seed_lr)

    lr_scheduler = LRSchedulerFactory.create(
        optimizer=optimizer, lr_scheduler_type=lr_sched_type, config=config
    )

    target_lr_sched = LRSchedulerFactory._lr_schedulers[lr_sched_type]

    assert isinstance(lr_scheduler, target_lr_sched)
