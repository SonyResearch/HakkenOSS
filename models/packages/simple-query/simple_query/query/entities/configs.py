from typing import Annotated, Literal

from pydantic import BaseModel, Field

from simple_query.query.values.types import QueryingType


class QueryingConfigBase(BaseModel):
    config_type: QueryingType


class SimpleQueryingConfig(QueryingConfigBase):
    config_type: Literal[QueryingType.SIMPLE] = QueryingType.SIMPLE


QueryingConfig = Annotated[SimpleQueryingConfig, Field(discriminator="config_type")]
