import random

from pydantic import BaseModel


class OptimizerConfig(BaseModel):
    learning_rate: float = 0.1
    weight_decay: float = 1e-5

    @classmethod
    def random(cls, seed: int) -> "OptimizerConfig":
        rng = random.Random(seed)
        return cls(
            learning_rate=rng.uniform(0.0001, 0.1),  # Common lr range: 1e-4 to 1e-1
            weight_decay=rng.uniform(1e-6, 1e-4),  # Common weight decay range
        )


class AdamConfig(OptimizerConfig):
    betas: tuple[float, float] = (0.9, 0.999)

    @classmethod
    def random(cls, seed: int) -> "AdamConfig":
        rng = random.Random(seed)
        beta1 = rng.uniform(0.9, 0.99)
        beta2 = rng.uniform(0.999, 0.9999)

        return cls(
            learning_rate=rng.uniform(0.0001, 0.1),
            weight_decay=rng.uniform(1e-6, 1e-4),
            betas=(beta1, beta2),
        )
