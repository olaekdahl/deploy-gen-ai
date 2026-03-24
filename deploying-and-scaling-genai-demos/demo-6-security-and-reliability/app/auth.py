"""
API Key authentication module.

Instructor note:
  This demonstrates a simple but effective API key authentication scheme.
  In production you would:
    - Store hashed keys in a database, not environment variables
    - Use OAuth2 / JWT for user-level authentication
    - Implement rate limiting per key
    - Rotate keys regularly

  The key point: every GenAI endpoint should require authentication.
  Unauthenticated models are expensive to run and easy to abuse.
"""

import os
import secrets

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.logging_config import setup_logging

logger = setup_logging()

# API key is loaded from the environment; a default is generated for demos
_API_KEY = os.getenv("API_KEY", "demo-api-key-change-me-in-production")

# The client sends the key in the X-API-Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Security(api_key_header),
) -> str:
    """
    Dependency that verifies the API key from the request header.

    Returns the verified key on success; raises 401 on failure.
    Uses secrets.compare_digest to prevent timing attacks.
    """
    if api_key is None:
        logger.warning("Request missing API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Include X-API-Key header.",
        )

    if not secrets.compare_digest(api_key, _API_KEY):
        logger.warning("Invalid API key attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )

    return api_key
