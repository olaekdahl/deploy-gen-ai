# ---------------------------------------------------------------------------
# Multi-stage Dockerfile for the GenAI Text Generation Service
#
# Instructor note:
#   Walk through the stages and explain WHY each decision was made:
#
#   Stage 1 (builder):
#     - Installs build-time dependencies (compilers, headers) in an
#       isolated layer that is NOT carried into the final image.
#     - Installs Python packages into a virtual environment so they
#       can be copied cleanly.
#
#   Stage 2 (runtime):
#     - Starts from the same base image but WITHOUT build tools.
#     - Copies only the virtual environment and application code.
#     - Result: a smaller, more secure image with fewer attack surfaces.
#
#   Layer ordering:
#     - requirements.txt is copied BEFORE the application code.
#     - This means Docker can cache the expensive pip install step as
#       long as dependencies have not changed, even if app code changes.
# ---------------------------------------------------------------------------

# ========================== Stage 1: Frontend Build ==========================
FROM node:20-alpine AS frontend

WORKDIR /build
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build
# Output lands in /build/../static -> we adjust: output goes to /build/dist
# Vite config sets outDir: '../static', so from /build it writes to /static
# We'll copy from /static in the runtime stage

# ========================== Stage 2: Python Builder ==========================
FROM python:3.11-slim AS builder

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies needed to compile native extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Create a virtual environment for clean isolation
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies first (cache-friendly layer order)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# ========================== Stage 2: Runtime ==========================
FROM python:3.11-slim AS runtime

# Security: run as a non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set the working directory
WORKDIR /home/appuser/app

# Copy application code
COPY app/ ./app/

# Copy the built React UI from the frontend stage
COPY --from=frontend /static ./static/

# Switch to non-root user
USER appuser

# Pre-download the model at build time so the container starts faster.
# Remove this line if you prefer to download at runtime or mount a volume.
RUN python -c "from transformers import pipeline; pipeline('text-generation', model='distilgpt2', device=-1)"

# Expose the default API port
EXPOSE 8000

# Health check so orchestrators know when the service is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start the server with production-grade settings
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
