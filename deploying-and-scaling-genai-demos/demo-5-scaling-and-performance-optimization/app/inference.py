import time
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


def generate_batch(prompts: list[str], max_tokens=50, temperature=0.7) -> list[dict]:
    """
    Generate text for multiple prompts in a single forward pass.

    Instructor note:
      Batching amortizes per-request overhead (tokenization, memory allocation)
      across multiple inputs. The model processes all prompts simultaneously,
      which is more efficient than sequential processing -- especially on GPU.
      On CPU the benefit is smaller but still measurable for tokenization.
    """
    if _pipeline is None:
        raise RuntimeError("Model not loaded.")

    start = time.perf_counter()
    results = _pipeline(
        prompts,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        num_return_sequences=1,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    outputs = []
    for i, prompt in enumerate(prompts):
        # pipeline returns list of lists when given multiple inputs
        text = results[i][0]["generated_text"] if isinstance(results[i], list) else results[i]["generated_text"]
        new_tokens = max(len(text.split()) - len(prompt.split()), 0)
        outputs.append({
            "generated_text": text,
            "tokens_generated": new_tokens,
            "model_name": _model_name,
            "latency_ms": round(elapsed_ms / len(prompts), 2),
        })

    logger.info("Batch generation complete", extra={
        "model_name": _model_name, "batch_size": len(prompts),
        "latency_ms": round(elapsed_ms, 2),
    })
    return outputs
