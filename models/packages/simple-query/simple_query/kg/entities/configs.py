from typing import Annotated, Literal

from pydantic import BaseModel, Field

from simple_query.kg.values.types import KnowledgeGraphType


class KnowledgeGraphConfigBase(BaseModel):
    config_type: KnowledgeGraphType


class Neo4jKnowledgeGraphConfig(KnowledgeGraphConfigBase):
    config_type: Literal[KnowledgeGraphType.NEO4J] = KnowledgeGraphType.NEO4J

    use_okta: bool = False
    username: str = ""
    password: str = ""
    base_url: str = "bolt://localhost:7687"
    output_server_url: str = "localhost:8888"


KnowledgeGraphConfig = Annotated[
    Neo4jKnowledgeGraphConfig,
    Field(discriminator="config_type"),
]
