from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from contextualization.core.values.types import PublicationScorerType


class PublicationScorerConfigBase(BaseModel):
    config_type: PublicationScorerType


class CoveragePublicationScorerConfig(PublicationScorerConfigBase):
    config_type: Literal[PublicationScorerType.COVERAGE] = PublicationScorerType.COVERAGE


class RecencyPublicationScorerConfig(PublicationScorerConfigBase):
    config_type: Literal[PublicationScorerType.RECENCY] = PublicationScorerType.RECENCY

    use_rank: bool = True


class AggregationConfig(BaseModel):
    config: CoveragePublicationScorerConfig | RecencyPublicationScorerConfig = Field(
        discriminator="config_type"
    )
    weight: float = 1.0


class AggregatedPublicationScorerConfig(PublicationScorerConfigBase):
    config_type: Literal[PublicationScorerType.AGGREGATED] = PublicationScorerType.AGGREGATED

    aggregation_configs: list[AggregationConfig]

    @field_validator("aggregation_configs", mode="after")
    @classmethod
    def check_empty(cls, aggregation_configs: list[AggregationConfig]) -> list[AggregationConfig]:
        if not aggregation_configs:
            raise ValueError("should have at least 1 aggregation config")
        return aggregation_configs


PublicationScorerConfig = Annotated[
    CoveragePublicationScorerConfig
    | RecencyPublicationScorerConfig
    | AggregatedPublicationScorerConfig,
    Field(discriminator="config_type"),
]
