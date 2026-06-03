from pydantic import BaseModel

from contextualization.core.entities.retrieval import RetrievalReturnType
from contextualization.core.entities.triple import Triple


class ContextualizationRequest(BaseModel):
    triples: list[Triple]
    max_num_references: int = 10
    return_type: RetrievalReturnType
