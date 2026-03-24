"""
Demo 1 -- Model Packaging and API Serving

A production-style FastAPI service that loads a small generative language model
(distilgpt2) and exposes it through a validated REST API with structured logging.

Instructor note:
  This is the starting point. Walk students through:
    - How the model is loaded once at startup via the lifespan handler
    - How Pydantic validates every request automatically
    - How structured logging captures GenAI-specific fields
    - How the /docs page auto-generates from the code
"""

import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.inference import generate_text, load_model
from app.logging_config import setup_logging
from app.models import GenerateRequest, GenerateResponse, HealthResponse

logger = setup_logging()


# ---------------------------------------------------------------------------
# Lifespan: load the model once when the process starts
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup; clean up on shutdown."""
    model_name = os.getenv("MODEL_NAME", "distilgpt2")
    onnx_path = os.getenv("ONNX_MODEL_PATH")
    load_model(model_name=model_name, onnx_path=onnx_path)
    yield
    logger.info("Shutting down service")


app = FastAPI(
    title="GenAI Text Generation Service",
    description="A lightweight generative AI API for classroom demonstrations.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware: attach a unique request ID for tracing
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health_check():
    """Returns service health and model readiness."""
    from app.inference import _model_name, _pipeline

    return HealthResponse(
        status="healthy",
        model_loaded=_pipeline is not None,
        model_name=_model_name or "not loaded",
    )


@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate(request: GenerateRequest):
    """
    Generate text from a given prompt.

    The model produces a continuation of the input prompt up to ``max_tokens``
    new tokens, controlled by the ``temperature`` parameter.
    """
    result = generate_text(
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    return GenerateResponse(
        prompt=request.prompt,
        generated_text=result["generated_text"],
        tokens_generated=result["tokens_generated"],
        model_name=result["model_name"],
    )


# ---------------------------------------------------------------------------
# Global exception handler -- prevents stack traces leaking to clients
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check service logs for details."},
    )
