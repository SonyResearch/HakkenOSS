from collections.abc import Iterable
from typing import ClassVar, TypeVar

import torch

from hakken_ml_toolkit.optimizers.core.contracts.optimizer import OptimizerProtocol
from hakken_ml_toolkit.optimizers.core.entities.optimizer_configs import (
    AdamConfig,
    OptimizerConfig,
)
from hakken_ml_toolkit.optimizers.core.values.constants import OptimizerType
from hakken_ml_toolkit.optimizers.core.values.exceptions import InvalidConfigTypeError

OptimizerConfigType = TypeVar("OptimizerConfigType", bound=OptimizerConfig)


class OptimizerFactory:
    _optimizers: ClassVar[dict[OptimizerType, type[OptimizerProtocol]]] = {
        OptimizerType.ADAM: torch.optim.Adam,
        OptimizerType.SGD: torch.optim.SGD,
    }

    _optimizers_config: ClassVar[dict[OptimizerType, type[OptimizerConfig]]] = {
        OptimizerType.ADAM: AdamConfig,
        OptimizerType.SGD: OptimizerConfig,
    }

    _config_to_optimizer_type: ClassVar[dict[type[OptimizerConfig], OptimizerType]] = {
        config_class: optimizer_type for optimizer_type, config_class in _optimizers_config.items()
    }

    @staticmethod
    def config_to_type(config: OptimizerConfigType) -> OptimizerType:
        optim_type = OptimizerFactory._config_to_optimizer_type.get(type(config))

        if optim_type is None:
            optimier_types = list(OptimizerFactory._config_to_optimizer_type.keys())
            msg = f"Unsupported optimizer  config type: {type(config).__name__}. "
            msg += f"Supported types: {optimier_types}"
            raise ValueError(msg)

        return optim_type

    @staticmethod
    def create(
        parameters: Iterable, optim_type: OptimizerType, config: OptimizerConfigType
    ) -> OptimizerProtocol:
        optimizer_class = OptimizerFactory._optimizers[optim_type]
        config_class = OptimizerFactory._optimizers_config[optim_type]
        if not isinstance(config, config_class):
            raise InvalidConfigTypeError(
                expected_type=optim_type,
                expected_class=config_class,
                received_config=config,
            )

        config_dict = config.model_dump()
        if "learning_rate" in config_dict:
            config_dict["lr"] = config_dict["learning_rate"]
            del config_dict["learning_rate"]
        return optimizer_class(parameters, **config_dict)
