"""
Demo 6 -- Security and Reliability

A hardened GenAI service demonstrating:
  - API key authentication
  - Strict input validation with injection detection
  - Retry with exponential backoff
  - Fallback responses when the primary model fails
  - Structured error responses
  - PII-aware logging (never logs raw prompts at INFO level)
  - Rate limiting via middleware
"""

import os
import uuid
from contextlib import asynccontextmanager
from collections import defaultdict
import time

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app.auth import verify_api_key
from app.inference import load_model
from app.logging_config import setup_logging
from app.models import (
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
)
from app.retry_handler import generate_with_retry

logger = setup_logging()

# ---------------------------------------------------------------------------
# Simple in-memory rate limiter for demonstration
# Instructor note: In production, use Redis-backed rate limiting or an
# API gateway (Kong, Envoy, etc.) for distributed rate limiting.
# ---------------------------------------------------------------------------
_rate_limit_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "30"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model(os.getenv("MODEL_NAME", "distilgpt2"))
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="GenAI Service (Secured)",
    version="6.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware: request ID, rate limiting, PII-safe logging
# ---------------------------------------------------------------------------
@app.middleware("http")
async def request_pipeline(request: Request, call_next):
    # Assign request ID
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id

    # Rate limiting (by API key or IP)
    client_id = request.headers.get("X-API-Key", request.client.host if request.client else "unknown")
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    _rate_limit_store[client_id] = [
        ts for ts in _rate_limit_store[client_id] if ts > window_start
    ]

    if len(_rate_limit_store[client_id]) >= RATE_LIMIT_MAX_REQUESTS:
        logger.warning("Rate limit exceeded for client %s", client_id)
        return JSONResponse(
            status_code=429,
            content={
                "error": "rate_limit_exceeded",
                "detail": f"Rate limit: {RATE_LIMIT_MAX_REQUESTS} requests per {RATE_LIMIT_WINDOW_SECONDS}s.",
                "request_id": request_id,
            },
            headers={"X-Request-ID": request_id, "Retry-After": str(RATE_LIMIT_WINDOW_SECONDS)},
        )

    _rate_limit_store[client_id].append(now)

    # PII-safe logging: log the request path but NOT the body
    # Instructor note: Never log raw prompts at INFO level in production.
    # Prompts may contain personal data, credentials, or sensitive business info.
    logger.info(
        "Request received",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
        },
    )

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# Endpoints -- all protected by API key authentication
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check (unauthenticated for load balancers)."""
    from app.inference import _model_name, _pipeline
    return HealthResponse(
        status="healthy",
        model_loaded=_pipeline is not None,
        model_name=_model_name or "not loaded",
    )


@app.post(
    "/generate",
    response_model=GenerateResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def generate(
    request: GenerateRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Generate text with authentication, retry, and fallback.

    Requires a valid API key in the X-API-Key header.
    If the primary model fails after retries, a fallback response is returned.
    """
    result = await generate_with_retry(
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    return GenerateResponse(
        prompt=request.prompt,
        generated_text=result["generated_text"],
        tokens_generated=result["tokens_generated"],
        model_name=result["model_name"],
        fallback_used=result.get("fallback_used", False),
    )


# ---------------------------------------------------------------------------
# Structured error handlers
# ---------------------------------------------------------------------------
@app.exception_handler(422)
async def validation_exception_handler(request: Request, exc):
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "detail": str(exc),
            "request_id": request_id,
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", None)
    # Log the error but NOT the request body (PII safety)
    logger.error(
        "Unhandled exception: %s",
        type(exc).__name__,
        extra={"request_id": request_id},
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": "An internal error occurred. Contact support with the request ID.",
            "request_id": request_id,
        },
    )
