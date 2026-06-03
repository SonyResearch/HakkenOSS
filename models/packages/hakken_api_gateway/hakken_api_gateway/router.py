from typing import TYPE_CHECKING

import httpx
from fastapi import APIRouter, Depends, Response, status
from loguru import logger
from slowapi import Limiter
from slowapi.util import get_remote_address

from hakken_api_gateway.api.container import ApiConfig
from hakken_api_gateway.auth.auth import verify_okta
from hakken_api_gateway.handlers.router import (
    ServiceType,
    get_or_create_user,
    handle_gateway_request,
)
from hakken_api_gateway.impl.database.postgres import PostgresDatabase

if TYPE_CHECKING:
    from fastapi import Request


from typing import TYPE_CHECKING, Annotated

from fastapi import Path

if TYPE_CHECKING:
    from fastapi import Request


limiter = Limiter(key_func=get_remote_address)
router = APIRouter()


# Dependency to manage Database Lifecycle
# This prevents creating a new connection pool for every request.
def get_db():
    config = ApiConfig()  # Assuming this is lightweight/cached
    db = PostgresDatabase(config.database_config)
    try:
        yield db
    finally:
        # TODO: await db.close()
        pass


# --- API Route ---


@router.api_route(
    "/{version}/{service}/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE"],
)
@limiter.limit("100/minute")
async def reverse_proxy(
    request: "Request",
    user_data: Annotated[dict, Depends(verify_okta)],
    db: Annotated[PostgresDatabase, Depends(get_db)],
    service: str = Path(..., description="The downstream service name (e.g., data, explain)"),
):
    """
    Reverse proxy endpoint that forwards requests to downstream services.
    """

    # 1. Validate Service Existence
    try:
        logger.info(f"Received request for service: {service}")
        service_requested = ServiceType(service)
    except ValueError:
        logger.warning(f"Invalid service requested: {service}")
        return Response(
            content=f"Service '{service}' not found.",
            status_code=status.HTTP_404_NOT_FOUND,
        )

    # 2. Process Request
    try:
        # TODO: remove "sub" condition once the hakken client is implemented
        if "sub" in user_data and "email" not in user_data:
            user_data["email"] = user_data["sub"]
        elif "email" not in user_data:
            user_data["email"] = "no_email"
            return Response(
                content="No email found, cannot continue.",
                status_code=status.HTTP_428_PRECONDITION_REQUIRED,
            )
        elif "email" in user_data:
            pass

        # Sync User with DB
        logger.info(f"Processing request for user: {user_data['email']}")
        user_data["user_id"] = await get_or_create_user(user_data, db)

        # Forward Request
        return await handle_gateway_request(
            request=request, service_type=service_requested, user_data=user_data, db=db
        )

    except httpx.RequestError as e:
        logger.error(f"Downstream connection failed: {e!r}")
        return Response(
            content="Error connecting to downstream service.",
            status_code=status.HTTP_502_BAD_GATEWAY,
        )
    except Exception:
        logger.exception("Unexpected internal error in gateway.")
        return Response(
            content="Internal Gateway Error", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
