from typing import Annotated, Literal

from pydantic import BaseModel, Field

from simple_query.link_predictor.values.types import LinkPredictorType


class LinkPredictorConfigBase(BaseModel):
    config_type: LinkPredictorType


class ApiBasedLinkPredictorConfig(LinkPredictorConfigBase):
    config_type: Literal[LinkPredictorType.API_BASED] = LinkPredictorType.API_BASED

    url: str


class RandomLinkPredictorConfig(LinkPredictorConfigBase):
    config_type: Literal[LinkPredictorType.RANDOM] = LinkPredictorType.RANDOM

    seed: int | None = None


LinkPredictorConfig = Annotated[
    ApiBasedLinkPredictorConfig | RandomLinkPredictorConfig, Field(discriminator="config_type")
]
