"""Inference module -- model loading and text generation."""

import os
import time

from transformers import AutoTokenizer, pipeline

from app.logging_config import setup_logging

logger = setup_logging()

_onnx_available = False
try:
    from optimum.onnxruntime import ORTModelForCausalLM
    _onnx_available = True
except ImportError:
    pass

_pipeline = None
_model_name: str | None = None


def load_model(
    model_name: str = "distilgpt2",
    onnx_path: str | None = None,
) -> None:
    global _pipeline, _model_name
    _model_name = model_name
    use_onnx = os.getenv("USE_ONNX", "false").lower() == "true"

    if use_onnx and onnx_path and _onnx_available:
        logger.info("Loading ONNX model from %s", onnx_path, extra={"model_name": model_name})
        onnx_model = ORTModelForCausalLM.from_pretrained(onnx_path)
        tokenizer = AutoTokenizer.from_pretrained(onnx_path)
        _pipeline = pipeline("text-generation", model=onnx_model, tokenizer=tokenizer)
    else:
        logger.info("Loading PyTorch model: %s", model_name, extra={"model_name": model_name})
        _pipeline = pipeline("text-generation", model=model_name, device=-1)

    logger.info("Model loaded successfully", extra={"model_name": model_name})


def generate_text(
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 0.7,
) -> dict:
    if _pipeline is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    start = time.perf_counter()
    results = _pipeline(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        num_return_sequences=1,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    generated_text: str = results[0]["generated_text"]
    prompt_words = len(prompt.split())
    total_words = len(generated_text.split())
    new_tokens = max(total_words - prompt_words, 0)

    logger.info(
        "Generation complete",
        extra={
            "model_name": _model_name,
            "prompt_length": len(prompt),
            "tokens_generated": new_tokens,
            "latency_ms": round(elapsed_ms, 2),
        },
    )
    return {
        "generated_text": generated_text,
        "tokens_generated": new_tokens,
        "model_name": _model_name,
        "latency_ms": round(elapsed_ms, 2),
    }
