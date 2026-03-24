"""
Demo 5 -- Scaling and Performance Optimization

Extends the GenAI service with:
  - Redis response caching (eliminates redundant inference)
  - Batch generation endpoint (amortizes per-request overhead)
  - Cache hit/miss tracking in responses and logs
"""

import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.cache import (
    cache_connected,
    close_cache,
    get_cached_response,
    init_cache,
    set_cached_response,
)
from app.inference import generate_batch, generate_text, load_model
from app.logging_config import setup_logging
from app.models import (
    BatchGenerateRequest,
    BatchGenerateResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
)

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model(os.getenv("MODEL_NAME", "distilgpt2"))
    await init_cache()
    yield
    await close_cache()


app = FastAPI(
    title="GenAI Service (Optimized)",
    version="5.0.0",
    lifespan=lifespan,
)


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
    return HealthResponse(
        status="healthy",
        model_loaded=_pipeline is not None,
        model_name=_model_name or "not loaded",
        cache_connected=cache_connected(),
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text with Redis caching.

    If an identical request (same prompt, max_tokens, temperature) has been
    served recently, the cached response is returned instantly.
    """
    # Check cache first
    cached = await get_cached_response(
        request.prompt, request.max_tokens, request.temperature
    )
    if cached:
        return GenerateResponse(
            prompt=request.prompt,
            generated_text=cached["generated_text"],
            tokens_generated=cached["tokens_generated"],
            model_name=cached["model_name"],
            cache_hit=True,
        )

    # Cache miss: run inference
    result = generate_text(
        request.prompt, request.max_tokens, request.temperature
    )

    # Store in cache for future requests
    await set_cached_response(
        request.prompt, request.max_tokens, request.temperature, result
    )

    return GenerateResponse(
        prompt=request.prompt,
        generated_text=result["generated_text"],
        tokens_generated=result["tokens_generated"],
        model_name=result["model_name"],
        cache_hit=False,
    )


@app.post("/generate/batch", response_model=BatchGenerateResponse)
async def generate_batch_endpoint(request: BatchGenerateRequest):
    """
    Batch generation endpoint -- process multiple prompts in one request.

    Instructor note:
      Batching is most beneficial when the model can process multiple inputs
      in parallel (GPU). On CPU, the benefit comes from amortizing
      tokenization overhead and reducing HTTP round trips.
    """
    results = generate_batch(
        request.prompts, request.max_tokens, request.temperature
    )
    responses = [
        GenerateResponse(
            prompt=prompt,
            generated_text=r["generated_text"],
            tokens_generated=r["tokens_generated"],
            model_name=r["model_name"],
            cache_hit=False,
        )
        for prompt, r in zip(request.prompts, results)
    ]
    return BatchGenerateResponse(results=responses, batch_size=len(request.prompts))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", str(exc))
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
