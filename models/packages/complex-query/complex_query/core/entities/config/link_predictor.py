from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field

from complex_query.core.values.types import LinkPredictorType


class LinkPredictorConfigBase(BaseModel):
    config_type: LinkPredictorType


class ApiBasedLinkPredictorConfig(LinkPredictorConfigBase):
    config_type: Literal[LinkPredictorType.API_BASED] = LinkPredictorType.API_BASED

    url: str = "http://localhost:8000/core-model/kge/predict"


class RandomLinkPredictorConfig(LinkPredictorConfigBase):
    config_type: Literal[LinkPredictorType.RANDOM] = LinkPredictorType.RANDOM

    seed: int | None = None


LinkPredictorConfig: TypeAlias = Annotated[
    ApiBasedLinkPredictorConfig | RandomLinkPredictorConfig, Field(discriminator="config_type")
]
