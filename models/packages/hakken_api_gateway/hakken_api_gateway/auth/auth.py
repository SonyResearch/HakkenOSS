from typing import Annotated

import jwt
from dotenv import load_dotenv
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger

# --- Configuration ---
load_dotenv()
ALGORITHMS = ["RS256"]

# --- Security Schemes ---
jwt_scheme = HTTPBearer(auto_error=False)


# --- Authentication Dependency ---
async def verify_okta(
    token: Annotated[HTTPAuthorizationCredentials | None, Security(jwt_scheme)],
):
    logger.info("Verifying authentication credentials")
    if token:
        try:
            return jwt.decode(
                token.credentials,
                algorithms=ALGORITHMS,
                options={"verify_signature": False},
            )

        except jwt.PyJWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid Token: {e}",
                headers={"WWW-Authenticate": "Bearer"},
            ) from e

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication credentials are required",
    )
