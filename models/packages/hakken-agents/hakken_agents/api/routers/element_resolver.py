"""Element Resolver API router - ingest and search elements."""

from fastapi import APIRouter, Depends, HTTPException

from hakken_agents.api.dependencies import get_element_resolver
from hakken_agents.tools.element_resolver import ElementResolver
from hakken_agents.tools.element_resolver.schemas import (
    ElementIngestRequest,
    ElementIngestResponse,
    ElementSearchRequest,
    ElementSearchResponse,
    ElementSearchResult,
    SimilaritySearchParam,
)

router = APIRouter(prefix="/element_resolver", tags=["element_resolver"])


@router.post("/ingest", response_model=ElementIngestResponse)
def ingest_elements(
    request: ElementIngestRequest,
    resolver: ElementResolver = Depends(get_element_resolver),
) -> ElementIngestResponse:
    """Ingest elements into the vector store.

    Each element consists of free-form content and optional metadata.
    When ``no_description`` is False (default), an LLM-generated description
    is added to improve search quality. Duplicates (by deterministic UUID
    derived from metadata) are skipped idempotently.
    """
    element_dicts = [{"content": el.content, **el.metadata} for el in request.elements]

    if request.no_description:
        documents = [resolver.to_document(**ed) for ed in element_dicts]
    else:
        documents = resolver.to_documents_with_description_batch(
            element_dicts, max_concurrency=request.max_concurrency
        )

    ingested: list[str] = []
    skipped: list[str] = []
    for doc in documents:
        if resolver.exists(doc.id):
            skipped.append(doc.id)
        else:
            resolver.add(doc)
            ingested.append(doc.id)

    return ElementIngestResponse(ingested=ingested, skipped=skipped)


@router.post("/search", response_model=ElementSearchResponse)
def search_elements(
    request: ElementSearchRequest,
    resolver: ElementResolver = Depends(get_element_resolver),
) -> ElementSearchResponse:
    """Find elements similar to the query text.

    Uses semantic search over the vector store. Optional metadata filter
    supports langchain-style operators (e.g. ``$ilike``, ``$eq``).
    """
    param = SimilaritySearchParam(k=request.k, filter=request.filter, threshold=request.threshold)
    results_with_score = resolver.find_similar_elements_with_score(request.query, param)

    results = [
        ElementSearchResult(
            element_id=doc.id or "",
            score=score,
            content=doc.page_content,
            metadata=dict(doc.metadata),
        )
        for doc, score in results_with_score
    ]

    return ElementSearchResponse(results=results)


@router.get("/filter-columns")
def get_filter_columns(
    resolver: ElementResolver = Depends(get_element_resolver),
) -> list[str]:
    """Return column names that can be used for search filters."""
    return resolver.get_filter_columns()


@router.get("/elements/{element_id}")
def get_element(
    element_id: str,
    resolver: ElementResolver = Depends(get_element_resolver),
) -> dict:
    """Get a single element by its UUID."""
    try:
        doc = resolver.get(element_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return {
        "id": doc.id,
        "content": doc.page_content,
        "metadata": doc.metadata,
    }
