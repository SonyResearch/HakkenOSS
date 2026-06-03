from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from fastapi import Request, Response, status
from loguru import logger

from hakken_api_gateway.core.entities.explain import (
    ExplainLengthReq,
    ExplainLengthRes,
    ExplainReqFromUpstream,
    ExplainReqToDownstream,
    ExplainResp,
    ExplanationConfig,
)
from hakken_api_gateway.handlers.common import forward_request

if TYPE_CHECKING:
    from hakken_api_gateway.impl.database.postgres import PostgresDatabase


# --- Constants ---
DEFAULT_BATCH_SIZE = 32
DEFAULT_EXPLANATION_TYPE = "sufficient"

# --- Private Helper Functions ---


def _generate_cache_key(explainer_req: ExplainReqToDownstream) -> str:
    """
    Generates a deterministic hash based on the request content.
    Using a hash avoids issues with special characters in the triples.
    """
    # Sort keys ensures {a:1, b:2} produces same hash as {b:2, a:1}
    content_str = explainer_req.model_dump_json(warnings=False, exclude_none=True)
    return hashlib.sha256(content_str.encode("utf-8")).hexdigest()


async def _process_explanation_length(request: Request, downstream_url: str) -> Response:
    """Core logic: Validate -> Check Cache -> (Forward + Cache)"""

    explain_length_req = await _parse_and_validate_explain_length_body(request)

    response = await forward_request(request, downstream_url, explain_length_req)

    if not response.body:
        return Response(
            content="Invalid response body from downstream.",
            status_code=status.HTTP_502_BAD_GATEWAY,
        )
    try:
        ExplainLengthRes.model_validate_json(response.body)
    except Exception:
        logger.error("Invalid response body from downstream.")
        return Response(
            content="Invalid response body from downstream.",
            status_code=status.HTTP_502_BAD_GATEWAY,
        )

    return response


async def _process_explanation_logic(
    request: Request, downstream_url: str, db: PostgresDatabase
) -> Response:
    """Core logic: Validate -> Check Cache -> (Forward + Cache)"""

    # A. Validate Incoming Request
    upstream_req = await _parse_and_validate_explain_body(request)

    # B. Prepare Downstream Request Object
    downstream_req = ExplainReqToDownstream(
        triples_to_probe=[upstream_req.triple],
        explanation_configs=[
            ExplanationConfig(batch_size=DEFAULT_BATCH_SIZE, type=DEFAULT_EXPLANATION_TYPE)
        ],
    )

    # C. Cache Lookup
    cache_id = _generate_cache_key(downstream_req)

    if cached_data := _search_explanation(cache_id, db):
        logger.info("Explanation found in cache")
        # Assuming cached_data["explanations"] is stored as JSONB (dict) in DB
        return Response(
            content=json.dumps(cached_data["explanations"]),
            status_code=status.HTTP_200_OK,
            media_type="application/json",
        )

    logger.info("Explanation NOT found in cache. Generating explanation...")

    # D. Fetch from Downstream and Cache
    return await _fetch_and_cache_remote(request, downstream_url, downstream_req, cache_id, db)


async def _parse_and_validate_explain_body(request: Request) -> ExplainReqFromUpstream:
    """Reads bytes and validates Pydantic model."""
    body = await request.body()
    if not body:
        logger.error("Empty body received for explain service")
        raise ValueError("Empty body")

    return ExplainReqFromUpstream.model_validate_json(body)


async def _parse_and_validate_explain_length_body(request: Request) -> ExplainLengthReq:
    """Reads bytes and validates Pydantic model."""
    body = await request.body()
    if not body:
        logger.error("Empty body received for explain service")
        raise ValueError("Empty body")

    return ExplainLengthReq.model_validate_json(body)


async def _fetch_and_cache_remote(
    request: Request,
    downstream_url: str,
    explainer_req: ExplainReqToDownstream,
    cache_id: str,
    db: PostgresDatabase,
) -> Response:
    """Forwards request, validates response, caches result, and returns response."""

    # 1. Forward the request
    response = await forward_request(request, downstream_url, explainer_req)

    if response.body is None:
        return Response(
            content="Invalid response body from downstream", status_code=status.HTTP_502_BAD_GATEWAY
        )

    # 2. Validate and Cache (Fail-Open)
    # We use a try/except here because if caching fails, we still want
    # to return the valid answer to the user.
    try:
        validated_resp = ExplainResp.model_validate_json(response.body)
        _insert_explanation(cache_id, explainer_req, validated_resp, db)
    except Exception as e:
        logger.error(f"Background Cache Error (returning response anyway): {e}")

    return response


def _insert_explanation(
    cache_id: str,
    explainer_request: ExplainReqToDownstream,
    explainer_response: ExplainResp,
    db: PostgresDatabase,
) -> str | None:
    """Insert explanation request and response into the database."""

    # Use model_dump(mode='json') to get a python dict/list.
    # This allows the Postgres driver to handle JSONB serialization correctly.
    explanation_data = {
        "id": cache_id,
        "query": explainer_request.model_dump(mode="json"),
        "explanations": explainer_response.model_dump(mode="json"),
    }

    return db.add_explanation(explanation_data)


def _search_explanation(
    cache_id: str,
    db: PostgresDatabase,
) -> dict[str, Any] | None:
    """Search for an existing explanation in the database by ID."""
    explanation = db.get_explanation(cache_id, "explanation")

    if explanation:
        return explanation
    return None


# --- Main Handler ---
async def handle_explain_service_request(
    request: Request, downstream_url: str, db: PostgresDatabase
) -> Response:
    """
    Main entry point for the explain service. Routes requests to specific handlers.
    """
    path = request.path_params.get("path")

    match path:
        case "time":  # TODO: Call proper KGE endpoint when ready
            downstream_url += "/time"
            return Response(content="5000", status_code=status.HTTP_200_OK)
        case "length":
            downstream_url += "/explanation_length"
            return await _process_explanation_length(request, downstream_url)
        case "" | None:
            downstream_url += "/explain"
            try:
                return await _process_explanation_logic(request, downstream_url, db)
            except ValueError as e:
                return Response(content=str(e), status_code=status.HTTP_400_BAD_REQUEST)
            except Exception:
                logger.exception("Unexpected error in explanation service")
                return Response(
                    content="Internal Server Error",
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )
        case _:
            # Handle Invalid Paths
            return Response(
                content=f"Invalid path: {path}", status_code=status.HTTP_400_BAD_REQUEST
            )
