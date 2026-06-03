import random
from typing import Literal, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound="LRSchedulerConfig")


class LRSchedulerConfig(BaseModel):
    @classmethod
    def random(cls: type[T], seed: int) -> T:
        assert isinstance(seed, int)
        raise NotImplementedError()


class ReduceLROnPlateauConfig(LRSchedulerConfig):
    mode: Literal["min", "max"] = "min"
    factor: float = 0.1
    patience: int = 10  # Number of epochs with no improvement before reducing LR
    min_lr: float = 1e-6  # Minimum learning rate

    @classmethod
    def random(cls, seed: int) -> "ReduceLROnPlateauConfig":
        rng = random.Random(seed)
        base_config = {}

        base_config.update(
            {
                "mode": rng.choice(["min", "max"]),
                "factor": rng.uniform(0.1, 0.5),
                "patience": rng.randint(5, 20),
                "min_lr": rng.uniform(1e-7, 1e-5),
            }
        )

        return cls(**base_config)  # type: ignore


class CosineAnnealingConfig(LRSchedulerConfig):
    t_max: int  # Maximum number of iterations
    eta_min: float = 0.0  # Mininum learning rate

    @classmethod
    def random(cls, seed: int) -> "CosineAnnealingConfig":
        rng = random.Random(seed)
        base_config = {}

        base_config.update(
            {
                "t_max": rng.randint(5, 20),
                "eta_min": rng.uniform(1e-7, 1e-5),
            }
        )

        return cls(**base_config)  # type: ignore
