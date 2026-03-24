import os, time
from transformers import pipeline as hf_pipeline
from app.logging_config import setup_logging

logger = setup_logging()

_pipeline = None
_model_name: str | None = None


def load_model(model_name="distilgpt2"):
    global _pipeline, _model_name
    _model_name = model_name
    logger.info("Loading model: %s", model_name, extra={"model_name": model_name})
    _pipeline = hf_pipeline("text-generation", model=model_name, device=-1)
    logger.info("Model loaded", extra={"model_name": model_name})


def generate_text(prompt, max_tokens=50, temperature=0.7):
    if _pipeline is None:
        raise RuntimeError("Model not loaded.")
    start = time.perf_counter()
    results = _pipeline(
        prompt, max_new_tokens=max_tokens,
        temperature=temperature, do_sample=temperature > 0,
        num_return_sequences=1,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    text = results[0]["generated_text"]
    new_tokens = max(len(text.split()) - len(prompt.split()), 0)
    logger.info("Generation complete", extra={
        "model_name": _model_name, "prompt_length": len(prompt),
        "tokens_generated": new_tokens, "latency_ms": round(elapsed_ms, 2),
    })
    return {"generated_text": text, "tokens_generated": new_tokens,
            "model_name": _model_name, "latency_ms": round(elapsed_ms, 2)}
