"""
Demo 2 -- Dockerized GenAI Service

Same FastAPI application as Demo 1, packaged for containerized deployment.
"""

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.inference import generate_text, load_model
from app.logging_config import setup_logging
from app.models import GenerateRequest, GenerateResponse, HealthResponse

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_name = os.getenv("MODEL_NAME", "distilgpt2")
    onnx_path = os.getenv("ONNX_MODEL_PATH")
    load_model(model_name=model_name, onnx_path=onnx_path)
    yield
    logger.info("Shutting down service")


app = FastAPI(
    title="GenAI Text Generation Service",
    description="Dockerized generative AI API.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health_check():
    from app.inference import _model_name, _pipeline
    return HealthResponse(
        status="healthy",
        model_loaded=_pipeline is not None,
        model_name=_model_name or "not loaded",
    )


@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate(request: GenerateRequest):
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


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."},
    )


# ---------------------------------------------------------------------------
# Serve the React UI from the /static directory (built by Vite)
# This must be mounted AFTER the API routes so /health and /generate
# take priority over the catch-all static file handler.
# ---------------------------------------------------------------------------
_static_dir = Path(__file__).resolve().parent.parent / "static"
if _static_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")
