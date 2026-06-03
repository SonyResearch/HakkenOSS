from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field

from complex_query.core.values.types import ScoreLedgerType


class ScoreLedgerConfigBase(BaseModel):
    config_type: ScoreLedgerType


class InMemoryScoreLedgerConfig(ScoreLedgerConfigBase):
    config_type: Literal[ScoreLedgerType.IN_MEMORY] = ScoreLedgerType.IN_MEMORY


ScoreLedgerConfig: TypeAlias = Annotated[
    InMemoryScoreLedgerConfig, Field(discriminator="config_type")
]
