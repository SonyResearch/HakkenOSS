from typing import Annotated, Literal

from pydantic import BaseModel, Field

from contextualization.core.entities.vector_database import MilvusMetricType, VectorType
from contextualization.core.values.types import PublicationVectorDatabaseType


class PublicationVectorDatabaseConfigBase(BaseModel):
    config_type: PublicationVectorDatabaseType

    vector_type: VectorType = VectorType.FLOAT


class MilvusPublicationVectorDatabaseConfig(PublicationVectorDatabaseConfigBase):
    config_type: Literal[PublicationVectorDatabaseType.MILVUS] = (
        PublicationVectorDatabaseType.MILVUS
    )

    collection_name: str
    dimension: int
    metric_type: MilvusMetricType = MilvusMetricType.COSINE
    uri: str = "http://localhost:19530"
    user: str = ""
    password: str = ""
    db_name: str = ""
    token: str = ""
    timeout: float | None = None


class InMemoryPublicationVectorDatabaseConfig(PublicationVectorDatabaseConfigBase):
    config_type: Literal[PublicationVectorDatabaseType.IN_MEMORY] = (
        PublicationVectorDatabaseType.IN_MEMORY
    )


PublicationVectorDatabaseConfig = Annotated[
    MilvusPublicationVectorDatabaseConfig | InMemoryPublicationVectorDatabaseConfig,
    Field(discriminator="config_type"),
]
