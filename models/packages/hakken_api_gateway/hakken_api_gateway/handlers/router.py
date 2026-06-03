from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, status
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

from hakken_api_gateway.core.entities.user import UserDBModel
from hakken_api_gateway.handlers.context import handle_context_service_request
from hakken_api_gateway.handlers.data import handle_data_service_request
from hakken_api_gateway.handlers.explain import handle_explain_service_request
from hakken_api_gateway.handlers.query import handle_query_service_request
from hakken_api_gateway.handlers.validation import handle_validation_service_request

if TYPE_CHECKING:
    from fastapi import Request, Response

    from hakken_api_gateway.impl.database.postgres import PostgresDatabase


# --- Configuration ---
class Settings(BaseSettings):
    """
    Centralized configuration using Pydantic.
    Validates environment variables on startup.
    """

    context_service_url: str = "http://localhost:57802/core-model"
    data_service_url: str = "http://localhost:53446/core-model"
    explain_service_url: str = "http://localhost:49662/core-model/path_explainer"
    query_service_url: str = "http://localhost:56543/core-model/query_mode/query"
    validation_service_url: str = "http://localhost:59996/core-model/kge"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()


class ServiceType(str, Enum):
    """Enum to prevent magic string errors in service routing."""

    CONTEXT = "context"
    DATA = "data"
    EXPLAIN = "explain"
    QUERY = "query"
    VALIDATION = "validation"


# --- User Logic ---
def _create_user_model(email: str, name: str) -> UserDBModel:
    """Helper to instantiate the UserDBModel."""
    now = datetime.now()
    return UserDBModel(
        email=email,
        name=name,
        created_at=now,
        updated_at=now,
        last_login_at=now,
    )


async def get_or_create_user(user_data: dict[str, Any], db: "PostgresDatabase") -> str | None:
    """
    Retrieves a user ID if exists, or creates a new user and returns the new ID.

    Args:
        user_data: Dictionary containing 'email' and 'name'.
        db: Database connection instance.

    Returns:
        str: The User ID.
    """
    # 1. Validate Email
    user_email = user_data.get("email")
    if not user_email:
        logger.warning("No email provided in user_data.")
        return None

    # 2. Check DB
    existing_user = db.get_user(user_email)

    # If user exist
    if existing_user is not None:
        logger.info(f"User found: {user_email}")
        return str(existing_user["id"])
    # 3. Create if not found
    logger.info(f"User with email {user_email} not found. Creating new user.")

    # Fallback: Use email as name if name is missing
    user_name = user_data.get("name", user_email)
    new_user = _create_user_model(email=user_email, name=user_name)

    # db.add_user returns the new ID
    new_user_id = db.add_user(new_user)
    logger.info("New user created.")

    return str(new_user_id)


# --- Request Handling ---
async def handle_gateway_request(
    request: "Request", service_type: ServiceType, user_data: dict[str, Any], db: "PostgresDatabase"
) -> "Response":
    """
    Router that dispatches the request to the specific service handler.
    """

    # TODO: Check user quota and log request to DB
    try:
        match service_type:
            case ServiceType.CONTEXT:
                return await handle_context_service_request(request, settings.context_service_url)
            case ServiceType.DATA:
                return await handle_data_service_request(request, settings.data_service_url)
            case ServiceType.EXPLAIN:
                return await handle_explain_service_request(
                    request, settings.explain_service_url, db
                )
            case ServiceType.QUERY:
                return await handle_query_service_request(
                    request, settings.query_service_url, user_data, db
                )
            case ServiceType.VALIDATION:
                return await handle_validation_service_request(
                    request, settings.validation_service_url
                )
            case _:
                # This catches invalid enums if passed dynamically
                logger.error(f"Unknown service type: {service_type}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Service '{service_type}' is not supported.",
                )

    except Exception as e:
        logger.exception(f"Error handling request for service {service_type}")
        raise e
