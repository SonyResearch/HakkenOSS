import random
from typing import TypeVar

from strenum import StrEnum


class OptimizerType(StrEnum):
    ADAM = "adam"
    SGD = "sgd"


class LRSchedulerType(StrEnum):
    ON_PLATEAU = "on_plateau"
    COSINE = "cosine"


T = TypeVar("T", bound=StrEnum)


def get_random_enum(enum_cls: type[T], seed: int | None = None) -> T:
    """
    Return a random member of the given StrEnum class, with optional reproducibility via seed.

    Args:
        enum_cls (type[StrEnum]): The enum class to sample from.
        seed (int | None): Seed for reproducibility. If None, randomness is not seeded.

    Returns:
        StrEnum: A randomly chosen member of the enum.
    """
    rng = random.Random(seed) if seed is not None else random
    return rng.choice(list(enum_cls))
