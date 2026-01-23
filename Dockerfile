# =============================================================================
# Multi-purpose Dockerfile for PyTorch examples
# Supports two modes:
#   1. Simple training (train_simple.py) - original example
#   2. FastAPI inference server (train + export + serve)
# =============================================================================

FROM python:3.11-slim AS base

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy requirements first (better layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY train_simple.py .
COPY train_export.py .
COPY inference_onnx.py .
COPY app.py .

# Create artifacts directory and set ownership
RUN mkdir -p /app/artifacts && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# =============================================================================
# Default: Run simple training example (original behavior)
# Override CMD to run other modes
# =============================================================================
CMD ["python", "train_simple.py"]
