import random
from typing import TypeVar

from strenum import StrEnum

T = TypeVar("T", bound=StrEnum)


def get_random_enum(enum_class: type[T], seed: int | None = None) -> T:
    if seed is not None:
        random.seed(seed)
    return random.choice(list(enum_class))
