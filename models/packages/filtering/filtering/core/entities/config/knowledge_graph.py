from typing import Annotated, Literal

from pydantic import BaseModel, Field, FilePath

from filtering.core.values.types import KnowledgeGraphType


class KnowledgeGraphConfigBase(BaseModel):
    config_type: KnowledgeGraphType


class NetworkXKnowledgeGraphConfig(KnowledgeGraphConfigBase):
    config_type: Literal[KnowledgeGraphType.NETWORKX] = KnowledgeGraphType.NETWORKX

    nodes_path: FilePath
    edges_path: FilePath

    node_id_column_name: str | None = "node_id"
    edge_subject_id_column_name: str = "subject_id"
    edge_object_id_column_name: str = "object_id"
    edge_year_occurrences_column_name: str = "year_occurrences"


class Neo4jKnowledgeGraphConfig(KnowledgeGraphConfigBase):
    config_type: Literal[KnowledgeGraphType.NEO4J] = KnowledgeGraphType.NEO4J

    use_okta: bool = True
    username: str = ""
    password: str = ""
    base_url: str = "bolt://localhost:7687"
    output_server_url: str = "localhost:8888"


KnowledgeGraphConfig = Annotated[
    NetworkXKnowledgeGraphConfig | Neo4jKnowledgeGraphConfig,
    Field(discriminator="config_type"),
]
