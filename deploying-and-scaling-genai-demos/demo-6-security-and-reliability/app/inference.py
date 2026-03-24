"""
Inference module with primary model and fallback support.

Instructor note:
  In production GenAI systems, the primary model can fail due to:
    - OOM errors (especially with long prompts)
    - Model corruption
    - Intermittent hardware failures (GPU errors)

  A fallback path ensures the service remains available in degraded mode.
  The fallback can be:
    - A smaller, more reliable model
    - A cached/pre-computed response
    - A safe default message

  This demo uses a simple template-based fallback to demonstrate the pattern
  without requiring a second model download.
"""

import time

from transformers import pipeline as hf_pipeline

from app.logging_config import setup_logging

logger = setup_logging()

_pipeline = None
_model_name: str | None = None


def load_model(model_name: str = "distilgpt2") -> None:
    global _pipeline, _model_name
    _model_name = model_name
    logger.info("Loading model: %s", model_name, extra={"model_name": model_name})
    _pipeline = hf_pipeline("text-generation", model=model_name, device=-1)
    logger.info("Model loaded", extra={"model_name": model_name})


def generate_text(prompt: str, max_tokens: int = 50, temperature: float = 0.7) -> dict:
    """Generate text using the primary model."""
    if _pipeline is None:
        raise RuntimeError("Model not loaded.")

    start = time.perf_counter()
    results = _pipeline(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        num_return_sequences=1,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    text = results[0]["generated_text"]
    new_tokens = max(len(text.split()) - len(prompt.split()), 0)

    logger.info("Generation complete", extra={
        "model_name": _model_name,
        "prompt_length": len(prompt),
        "tokens_generated": new_tokens,
        "latency_ms": round(elapsed_ms, 2),
    })
    return {
        "generated_text": text,
        "tokens_generated": new_tokens,
        "model_name": _model_name,
        "latency_ms": round(elapsed_ms, 2),
    }


def generate_fallback(prompt: str) -> dict:
    """
    Fallback generation when the primary model fails.

    Returns a safe, deterministic response that acknowledges the request
    but explains that full generation is temporarily unavailable.

    Instructor note:
      In a real system, this could:
        - Use a smaller quantized model
        - Return a cached nearest-neighbor response
        - Call a different API endpoint
      The key is that the service stays available rather than returning 500.
    """
    logger.warning(
        "Using fallback generation",
        extra={"model_name": "fallback", "prompt_length": len(prompt), "fallback_used": True},
    )
    safe_response = (
        f"{prompt} [Note: The primary model is temporarily unavailable. "
        "This is a fallback response. Please retry shortly for full generation.]"
    )
    return {
        "generated_text": safe_response,
        "tokens_generated": 0,
        "model_name": "fallback",
        "latency_ms": 0.0,
    }
