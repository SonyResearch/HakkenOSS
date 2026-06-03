from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field

from complex_query.core.values.types import KGType


class KGConfigBase(BaseModel):
    config_type: KGType


class Neo4jKGConfig(KGConfigBase):
    config_type: Literal[KGType.NEO4J] = KGType.NEO4J

    use_okta: bool = True
    username: str = ""
    password: str = ""
    base_url: str = "bolt://localhost:7687"
    output_server_url: str = "localhost:8888"


class NetworkxKGConfig(KGConfigBase):
    config_type: Literal[KGType.NETWORKX] = KGType.NETWORKX


KGConfig: TypeAlias = Annotated[
    Neo4jKGConfig | NetworkxKGConfig,
    Field(discriminator="config_type"),
]
