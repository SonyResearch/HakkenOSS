from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
from fastapi import HTTPException, Request, Response, status
from loguru import logger

if TYPE_CHECKING:
    from hakken_api_gateway.core.entities.explain import ExplainLengthReq, ExplainReqToDownstream
    from hakken_api_gateway.core.entities.query import QueryReqToDownstream

# Best Practice: Use a lifespan handler in your main app to close this client
# on shutdown, rather than just leaving it globally open.
client = httpx.AsyncClient(timeout=30.0)


async def forward_request(
    request: Request,
    downstream_url: str,
    payload: QueryReqToDownstream | ExplainReqToDownstream | ExplainLengthReq | None,
) -> Response:
    """
    Forward the request to the downstream service.
    """
    # 1. Prepare Headers
    # We strip 'host' (httpx sets it automatically) and 'content-length'
    # (httpx recalculates it for the new JSON body).
    forward_headers = {
        k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")
    }

    # 2. Serialize content once
    if payload is not None:
        request_content = payload.model_dump_json()
    else:
        request_content = await request.body()

    try:
        # 3. Send Request
        downstream_response = await client.request(
            method=request.method,
            url=downstream_url,
            headers=forward_headers,
            content=request_content,
            params=request.query_params,
        )
    except httpx.RequestError as exc:
        # Handle connection errors (DNS, Refused, Timeout) gracefully
        logger.info(f"Downstream request failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Downstream service unavailable: {exc!s}",
        ) from None

    # 4. Prepare Response Headers
    # Filter out hop-by-hop headers that shouldn't be forwarded
    excluded_headers = {"content-encoding", "content-length", "transfer-encoding", "connection"}
    response_headers = {
        k: v for k, v in downstream_response.headers.items() if k.lower() not in excluded_headers
    }

    return Response(
        content=downstream_response.content,
        status_code=downstream_response.status_code,
        headers=response_headers,
    )
