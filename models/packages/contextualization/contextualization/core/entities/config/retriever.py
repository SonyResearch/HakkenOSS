from typing import Annotated, Literal

from pydantic import BaseModel, Field

from contextualization.core.values.types import RetrieverType


class RetrieverConfigBase(BaseModel):
    config_type: RetrieverType


class LookupRetrieverConfig(RetrieverConfigBase):
    config_type: Literal[RetrieverType.LOOKUP] = RetrieverType.LOOKUP


RetrieverConfig = Annotated[LookupRetrieverConfig, Field(discriminator="config_type")]
