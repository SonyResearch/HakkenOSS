from enum import Enum

from pydantic import BaseModel, ConfigDict, PositiveInt

from contextualization.core.entities.publication import PublicationId
from contextualization.core.entities.types import Vector


class Metadata(BaseModel):
    publication_id: PublicationId
    chunk_index: PositiveInt
    num_chunks: PositiveInt
    text: str | None = None


class VectorWithMetadata(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector: Vector
    metadata: Metadata


class VectorDatabaseStatistics(BaseModel):
    finished: set[PublicationId]
    partial: set[PublicationId]


class VectorType(Enum):
    FLOAT = "float"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


class MilvusMetricType(Enum):
    COSINE = "COSINE"
    L2 = "L2"
    IP = "IP"
