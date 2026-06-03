from __future__ import annotations

from fastapi import Request, Response, status

from hakken_api_gateway.handlers.common import forward_request

# Data api handler


async def handle_context_service_request(request: Request, downstream_url: str) -> Response:
    """
    Main entry point for the explain service. Routes requests to specific handlers.
    """
    path = request.path_params.get("path")

    match path:
        case "contextualize":
            downstream_url += "/contextualize"
            response = await forward_request(request, downstream_url, None)
        case _:
            # Handle Invalid Paths
            return Response(
                content=f"Invalid path: {path}", status_code=status.HTTP_400_BAD_REQUEST
            )

    return response
