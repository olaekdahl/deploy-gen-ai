# PyTorch Docker Examples

Two runnable PyTorch examples in Docker:

1. **Simple Training** - Basic neural network training (original example)
2. **FastAPI Inference** - Train → ONNX export → Serve via FastAPI

## Project Structure

```text
deploy-gen-ai/
├── train_simple.py     # Example 1: Simple linear regression training
├── train_export.py     # Example 2: Train classifier + export to ONNX
├── inference_onnx.py   # ONNX Runtime inference module
├── app.py              # FastAPI inference server
├── requirements.txt    # Python dependencies (CPU-only)
├── Dockerfile          # Multi-purpose container (default: simple training)
├── Dockerfile.serve    # Pre-built FastAPI server (model baked in)
├── .gitignore
└── README.md
```

---

## Example 1: Simple Training (Original)

A minimal neural network that learns `y = 2x + 1` from dummy data.

### Build & Run

```bash
# Build
docker build -t pytorch-example .

# Run training
docker run --rm pytorch-example
```

### Expected Output

```text
========================================
PyTorch Simple Neural Network Training
PyTorch version: 2.2.0+cpu
Device: CPU
========================================

Model architecture:
SimpleNet(...)

Training for 50 epochs...
------------------------------
Epoch [ 10/50] | Loss: 0.2632
...
Epoch [ 50/50] | Loss: 0.0111
------------------------------
Training complete!

Test: input=1.0 → prediction=2.9876 (expected ~3.0)
```

---

## Example 2: FastAPI Inference Server

A binary classifier on 2D synthetic data, exported to ONNX, served via FastAPI.

### Option A: Pre-built Server Image (Recommended)

Model is trained during Docker build, so the image is ready to serve.

```bash
# Build (trains model during build)
docker build -f Dockerfile.serve -t pytorch-api .

# Run server
docker run --rm -p 8000:8000 pytorch-api
```

### Option B: Train at Runtime

Train and serve in one command:

```bash
# Build base image
docker build -t pytorch-example .

# Train + serve
docker run --rm -p 8000:8000 pytorch-example \
  sh -c "python train_export.py && uvicorn app:app --host 0.0.0.0 --port 8000"
```

### Test the API

**Health check:**

```bash
curl http://localhost:8000/health
```

Response:

```json
{"status":"ok","model_loaded":true,"model_path":"./artifacts/model.onnx"}
```

**Run predictions:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[-1.5, -1.5], [1.5, 1.5], [0.0, 0.0]]}'
```

Response:

```json
{
  "logits": [[2.34, -2.18], [-2.45, 2.31], [0.12, -0.08]],
  "class_indices": [0, 1, 0],
  "class_names": ["class_0", "class_1", "class_0"]
}
```

**Interactive docs:**

Open <http://localhost:8000/docs> in your browser.

---

## Just Train + Export (No Server)

```bash
docker run --rm pytorch-example python train_export.py
```

Output:

```text
==================================================
PyTorch → ONNX Training & Export Pipeline
==================================================

Dataset: 200 samples, 2D features, 2 classes

Training for 100 epochs...
----------------------------------------
Epoch [ 20/100] | Loss: 0.4521 | Acc: 89.00%
...
Epoch [100/100] | Loss: 0.0892 | Acc: 98.50%
----------------------------------------
Training complete! Final accuracy: 98.50%

Exporting to ONNX (opset 17)...
✓ ONNX model saved and validated: ./artifacts/model.onnx

Validating ONNX export...
Max absolute difference: 1.19e-07
✓ Validation passed: outputs match!
```

---

## API Reference

### `GET /health`

Health check endpoint.

**Response:**

```json
{"status": "ok", "model_loaded": true, "model_path": "./artifacts/model.onnx"}
```

### `POST /predict`

Run inference on 2D points.

**Request body:**

```json
{
  "inputs": [[x1, y1], [x2, y2], ...]
}
```

**Response:**

```json
{
  "logits": [[logit_class0, logit_class1], ...],
  "class_indices": [0, 1, ...],
  "class_names": ["class_0", "class_1", ...]
}
```

**Model behavior:**

- Points near `(-1, -1)` → `class_0`
- Points near `(+1, +1)` → `class_1`

### `GET /docs`

Interactive Swagger UI documentation.

---

## Technical Notes

### ONNX Export

- **Opset version 17**: Stable version with broad compatibility across ONNX Runtime versions
- **Dynamic batch size**: Model accepts variable batch sizes
- **Validation**: Export is verified by comparing PyTorch vs ONNX Runtime outputs

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.2.0+cpu | Training (CPU-only, ~200MB vs ~2GB for CUDA) |
| onnx | 1.15.0 | ONNX export and validation |
| onnxruntime | 1.17.0 | Fast CPU inference |
| fastapi | 0.109.0 | REST API framework |
| uvicorn | 0.27.0 | ASGI server |

### Security

- Runs as non-root user (`appuser`)
- No external network calls during inference
- Input validation via Pydantic schemas

---

## Quick Reference

| Task | Command |
|------|---------|
| Simple training | `docker run --rm pytorch-example` |
| Train + export | `docker run --rm pytorch-example python train_export.py` |
| Build API server | `docker build -f Dockerfile.serve -t pytorch-api .` |
| Run API server | `docker run --rm -p 8000:8000 pytorch-api` |
| Test health | `curl http://localhost:8000/health` |
| Test predict | `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"inputs": [[1,1],[-1,-1]]}'` |
