from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field

from complex_query.core.values.types import ScoreAggregatorType


class ScoreAggregatorConfigBase(BaseModel):
    config_type: ScoreAggregatorType


class ProductScoreAggregatorConfig(ScoreAggregatorConfigBase):
    config_type: Literal[ScoreAggregatorType.PRODUCT] = ScoreAggregatorType.PRODUCT


class MinimumScoreAggregatorConfig(ScoreAggregatorConfigBase):
    config_type: Literal[ScoreAggregatorType.MINIMUM] = ScoreAggregatorType.MINIMUM


ScoreAggregatorConfig: TypeAlias = Annotated[
    ProductScoreAggregatorConfig | MinimumScoreAggregatorConfig, Field(discriminator="config_type")
]
