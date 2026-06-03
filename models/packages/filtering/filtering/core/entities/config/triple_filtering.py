from typing import Annotated, Literal

from pydantic import BaseModel, Field

from filtering.core.values.types import TripleFilteringType


class TripleFilteringConfigBase(BaseModel):
    config_type: TripleFilteringType


class RandomTripleFilteringConfig(TripleFilteringConfigBase):
    config_type: Literal[TripleFilteringType.RANDOM] = TripleFilteringType.RANDOM

    random_seed: int | None = None


TripleFilteringConfig = Annotated[RandomTripleFilteringConfig, Field(discriminator="config_type")]
