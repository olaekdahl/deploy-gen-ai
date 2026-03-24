"""Demo 3 -- GenAI service for Kubernetes deployment."""

import os, uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.inference import generate_text, load_model
from app.logging_config import setup_logging
from app.models import GenerateRequest, GenerateResponse, HealthResponse

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model(os.getenv("MODEL_NAME", "distilgpt2"))
    yield

app = FastAPI(title="GenAI Service", version="3.0.0", lifespan=lifespan)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


@app.get("/health", response_model=HealthResponse)
async def health():
    from app.inference import _model_name, _pipeline
    return HealthResponse(status="healthy", model_loaded=_pipeline is not None,
                          model_name=_model_name or "not loaded")


# Readiness probe -- returns 503 until the model is loaded
@app.get("/ready")
async def readiness():
    from app.inference import _pipeline
    if _pipeline is None:
        return JSONResponse(status_code=503, content={"ready": False})
    return {"ready": True}


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    result = generate_text(request.prompt, request.max_tokens, request.temperature)
    return GenerateResponse(
        prompt=request.prompt, generated_text=result["generated_text"],
        tokens_generated=result["tokens_generated"], model_name=result["model_name"],
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", str(exc))
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
