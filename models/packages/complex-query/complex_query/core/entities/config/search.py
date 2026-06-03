from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field

from complex_query.core.values.types import SearchType


class SearchConfigBase(BaseModel):
    config_type: SearchType


class BeamSearchConfig(SearchConfigBase):
    config_type: Literal[SearchType.BEAM] = SearchType.BEAM

    batched: bool = True
    batch_size: int = 32
    beam_size: int = 5


SearchConfig: TypeAlias = Annotated[BeamSearchConfig, Field(discriminator="config_type")]
