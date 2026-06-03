from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends

from contextualization.api.entities import ContextualizationRequest
from contextualization.core.contracts.retriever import Retriever
from contextualization.core.entities.retrieval import RetrievedContext

router = APIRouter()


@router.post("/contextualize", response_model=RetrievedContext)
@inject
def contextualize(
    request: ContextualizationRequest, retriever: Retriever = Depends(Provide["retriever"])
):
    return retriever.retrieve(
        triples=request.triples,
        max_num_references=request.max_num_references,
        return_type=request.return_type,
    )
