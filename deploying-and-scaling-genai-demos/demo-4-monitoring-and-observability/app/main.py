"""
Demo 4 -- Monitoring and Observability

The GenAI service from previous demos, enhanced with Prometheus metrics
and a /metrics endpoint. Designed to pair with Prometheus and Grafana
via the included docker-compose stack.
"""

import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.inference import generate_text, load_model
from app.logging_config import setup_logging
from app.models import GenerateRequest, GenerateResponse, HealthResponse
from app.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    ERROR_COUNT,
    IN_FLIGHT_REQUESTS,
    SAFETY_FLAGS,
    SERVICE_INFO,
)

logger = setup_logging()

# Simple list of keywords for a basic safety flag demonstration.
# Instructor note: In production, use a proper content moderation model or API.
SAFETY_KEYWORDS = ["hack", "exploit", "attack", "password", "credentials"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    SERVICE_INFO.info({
        "model_name": os.getenv("MODEL_NAME", "distilgpt2"),
        "version": "4.0.0",
    })
    load_model(os.getenv("MODEL_NAME", "distilgpt2"))
    yield


app = FastAPI(title="GenAI Service (Monitored)", version="4.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Metrics middleware -- records request count, latency, and in-flight gauge
# ---------------------------------------------------------------------------
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    # Skip metrics for the /metrics endpoint itself
    if request.url.path == "/metrics":
        return await call_next(request)

    method = request.method
    endpoint = request.url.path
    IN_FLIGHT_REQUESTS.inc()
    start = time.perf_counter()

    try:
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        REQUEST_COUNT.labels(method=method, endpoint=endpoint,
                             status_code=response.status_code).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(elapsed)
        return response
    except Exception as exc:
        elapsed = time.perf_counter() - start
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=500).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(elapsed)
        ERROR_COUNT.labels(endpoint=endpoint, error_type=type(exc).__name__).inc()
        raise
    finally:
        IN_FLIGHT_REQUESTS.dec()


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


# ---------------------------------------------------------------------------
# Prometheus metrics endpoint
# ---------------------------------------------------------------------------
@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Expose Prometheus metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health", response_model=HealthResponse)
async def health():
    from app.inference import _model_name, _pipeline
    return HealthResponse(status="healthy", model_loaded=_pipeline is not None,
                          model_name=_model_name or "not loaded")


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    # Basic safety check for demonstration purposes
    prompt_lower = request.prompt.lower()
    for keyword in SAFETY_KEYWORDS:
        if keyword in prompt_lower:
            SAFETY_FLAGS.labels(flag_type="prompt_keyword").inc()
            logger.warning(
                "Safety flag triggered",
                extra={"request_id": "n/a", "prompt_length": len(request.prompt)},
            )
            break

    result = generate_text(request.prompt, request.max_tokens, request.temperature)

    return GenerateResponse(
        prompt=request.prompt, generated_text=result["generated_text"],
        tokens_generated=result["tokens_generated"], model_name=result["model_name"],
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", str(exc))
    ERROR_COUNT.labels(endpoint=request.url.path, error_type=type(exc).__name__).inc()
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
