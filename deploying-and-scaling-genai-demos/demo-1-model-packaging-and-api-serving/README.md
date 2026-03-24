# Demo 1 -- Model Packaging and API Serving

## What This Demo Teaches

- Loading a pre-trained generative language model (distilgpt2) at startup
- Exporting a model to ONNX format for portable, optimized inference
- Serving a GenAI model through a production-style REST API using FastAPI
- Validating requests with Pydantic models
- Structured JSON logging with GenAI-specific fields
- Clean separation of inference logic from the HTTP layer

---

## Prerequisites

- Python 3.11 or later
- pip
- ~1 GB of free disk space (for model weights and dependencies)
- Internet access (to download the model on first run)

---

## Installation

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy the environment configuration
cp .env.example .env
```

---

## Running the Service

### Option A: PyTorch Backend (default)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The model is downloaded automatically on first run (~350 MB) and cached in
`~/.cache/huggingface/`.

### Option B: ONNX Backend

First, export the model to ONNX format:

```bash
python export_model.py
```

Then start the server with ONNX enabled:

```bash
USE_ONNX=true ONNX_MODEL_PATH=./onnx_model \
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Testing the Service

### Interactive API Docs

Open http://localhost:8000/docs in a browser to explore the Swagger UI.

### Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "distilgpt2"
}
```

### Text Generation

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of artificial intelligence is",
    "max_tokens": 60,
    "temperature": 0.8
  }'
```

Expected response (output varies due to sampling):
```json
{
  "prompt": "The future of artificial intelligence is",
  "generated_text": "The future of artificial intelligence is ...",
  "tokens_generated": 42,
  "model_name": "distilgpt2"
}
```

### Validation Error

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "", "max_tokens": 9999}'
```

Returns a 422 response showing Pydantic validation errors.

### Debug Torch

```bash
cd /deploy-gen-ai/deploying-and-scaling-genai-demos/demo-1-model-packaging-and-api-serving
python debug_pytorch.py
```

---

## File Structure

```
demo-1-model-packaging-and-api-serving/
├── app/
│   ├── __init__.py          # Package marker
│   ├── main.py              # FastAPI application and endpoints
│   ├── models.py            # Pydantic request/response schemas
│   ├── inference.py         # Model loading and text generation
│   └── logging_config.py    # Structured JSON logging setup
├── export_model.py          # ONNX export script
├── requirements.txt         # Pinned Python dependencies
├── .env.example             # Environment variable template
└── README.md                # This file
```

---

## Key Teaching Talking Points

1. **Model Loading at Startup**: The model is loaded once during the FastAPI
   lifespan handler, not on every request. This avoids repeated loading latency
   and keeps memory usage predictable.

2. **ONNX Export**: ONNX provides a portable, framework-agnostic representation
   of the model. It enables inference without PyTorch and often runs faster on
   CPU thanks to graph optimizations in ONNX Runtime.

3. **Request Validation**: Pydantic enforces constraints (prompt length, token
   limits, temperature range) before the request reaches the model. This is the
   first line of defense against bad input.

4. **Structured Logging**: Every log line is valid JSON with consistent fields.
   This makes logs searchable and parseable by tools like ELK, Loki, or
   CloudWatch Logs Insights. GenAI-specific fields like `tokens_generated` and
   `latency_ms` are included for operational visibility.

5. **Request ID Tracing**: The middleware generates a UUID for every request
   (or honors one from the `X-Request-ID` header). This enables distributed
   tracing across services.

6. **Separation of Concerns**: The inference module is independent of FastAPI.
   You could reuse it in a CLI tool, batch job, or different web framework
   without changes.
