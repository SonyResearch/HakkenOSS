from typing import Any, ClassVar, TypeVar, cast

import torch

from hakken_ml_toolkit.optimizers.core.contracts.lr_scheduler import LRSchedulerProtocol
from hakken_ml_toolkit.optimizers.core.contracts.optimizer import OptimizerProtocol
from hakken_ml_toolkit.optimizers.core.entities.lr_scheduler_configs import (
    CosineAnnealingConfig,
    LRSchedulerConfig,
    ReduceLROnPlateauConfig,
)
from hakken_ml_toolkit.optimizers.core.values.constants import LRSchedulerType
from hakken_ml_toolkit.optimizers.core.values.exceptions import InvalidConfigTypeError

LRSchedulerConfigType = TypeVar("LRSchedulerConfigType", bound=LRSchedulerConfig)


# Define a type alias for the factory mapping
LRSchedulerClassMapping = dict[LRSchedulerType, type[Any]]


class LRSchedulerFactory:
    _lr_schedulers: ClassVar[LRSchedulerClassMapping] = {
        LRSchedulerType.ON_PLATEAU: torch.optim.lr_scheduler.ReduceLROnPlateau,
        LRSchedulerType.COSINE: torch.optim.lr_scheduler.CosineAnnealingLR,
    }

    _lr_schedulers_config: ClassVar[dict[LRSchedulerType, type[LRSchedulerConfig]]] = {
        LRSchedulerType.ON_PLATEAU: ReduceLROnPlateauConfig,
        LRSchedulerType.COSINE: CosineAnnealingConfig,
    }

    _config_to_lr_scheduler_type: ClassVar[dict[type[LRSchedulerConfig], LRSchedulerType]] = {
        config_class: lr_sched_type for lr_sched_type, config_class in _lr_schedulers_config.items()
    }

    @staticmethod
    def config_to_type(config: LRSchedulerConfigType) -> LRSchedulerType:
        lr_scheduler_type = LRSchedulerFactory._config_to_lr_scheduler_type.get(type(config))
        if lr_scheduler_type is None:
            scheduler_types = list(LRSchedulerFactory._config_to_lr_scheduler_type.keys())
            msg = f"Unsupported LR scheduler config type: {type(config).__name__}. "
            msg += f"Supported types: {scheduler_types}"
            raise ValueError(msg)

        return lr_scheduler_type

    @staticmethod
    def create(
        optimizer: OptimizerProtocol,
        config: LRSchedulerConfigType,
        lr_scheduler_type: LRSchedulerType | None = None,
    ) -> LRSchedulerProtocol:
        if lr_scheduler_type is None:
            lr_scheduler_type = LRSchedulerFactory.config_to_type(config)

        lr_scheduler_class = LRSchedulerFactory._lr_schedulers[lr_scheduler_type]
        config_class = LRSchedulerFactory._lr_schedulers_config[lr_scheduler_type]
        if not isinstance(config, config_class):
            raise InvalidConfigTypeError(
                expected_type=lr_scheduler_type,
                expected_class=config_class,
                received_config=config,
            )

        config_dict = config.model_dump()
        if "t_max" in config_dict:
            config_dict["T_max"] = config_dict["t_max"]
            del config_dict["t_max"]

        return cast("LRSchedulerProtocol", lr_scheduler_class(optimizer, **config_dict))
