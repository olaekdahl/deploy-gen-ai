# Demo 2 -- Dockerized GenAI Service

## What This Demo Teaches

- Packaging a GenAI application into a Docker container
- Multi-stage Docker builds to minimize image size
- Layer ordering for efficient build caching
- Running as a non-root user for security
- Docker health checks for orchestrator readiness
- Pre-downloading models at build time vs. runtime

---

## Prerequisites

- Docker 24+ installed and running
- Completed Demo 1 (conceptual understanding of the application)
- ~4 GB of free disk space (for the Docker image with model weights)

---

## Building the Image

```bash
# Build the Docker image (this will download the model during build)
docker build -t genai-service:v2 .
```

The build takes a few minutes because it:
1. Installs Python dependencies (cached on subsequent builds if requirements.txt is unchanged)
2. Downloads the distilgpt2 model weights (~350 MB)

### Inspecting Image Size

```bash
docker images genai-service:v2
```

---

## Running the Container

```bash
# Run the container, mapping port 8000
docker run -d \
  --name genai-service \
  -p 8000:8000 \
  -e MODEL_NAME=distilgpt2 \
  genai-service:v2
```

### View Logs

```bash
docker logs -f genai-service
```

### Check Container Health

```bash
docker inspect --format='{{.State.Health.Status}}' genai-service
```

---

## Testing the Service

Once the container is running and healthy:

```bash
# Health check
curl http://localhost:8000/health

# Generate text
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Docker makes deployment easier because", "max_tokens": 40}'
```

---

## Stopping and Cleaning Up

```bash
docker stop genai-service
docker rm genai-service
```

---

## File Structure

```
demo-2-dockerized-genai-service/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   ├── inference.py
│   └── logging_config.py
├── Dockerfile              # Multi-stage build definition
├── .dockerignore           # Files excluded from the build context
├── requirements.txt
├── .env.example
└── README.md
```

---

## Key Teaching Talking Points

1. **Multi-Stage Builds**: The builder stage installs compilers and builds
   native extensions. The runtime stage copies only the resulting virtual
   environment. This reduces the final image size and removes unnecessary
   build tools from the production image.

2. **Layer Ordering**: `requirements.txt` is copied and installed BEFORE the
   application code. This means Docker reuses the cached dependency layer
   when only application code changes, saving minutes on every rebuild.

3. **Non-Root User**: The container runs as `appuser` (UID 1000), not root.
   This limits the blast radius if the container is compromised.

4. **Health Checks**: The `HEALTHCHECK` instruction tells Docker (and
   orchestrators like Kubernetes) how to determine if the service is ready
   to receive traffic. The `start-period` is generous because model loading
   can take time.

5. **Model Pre-Download**: The model is downloaded during `docker build` so
   that container startup is fast and deterministic. In production, you might
   instead mount a shared volume with pre-cached models.

6. **`.dockerignore`**: Excluding `.git`, `.venv`, `__pycache__`, and
   documentation from the build context reduces the data sent to the Docker
   daemon and prevents secrets or unnecessary files from leaking into the image.

### Discussion Questions for Students

- What happens to image size if you skip the multi-stage build?
- Why is downloading the model at build time a trade-off?
- How would you handle model updates without rebuilding the entire image?
- What security risks exist if you run the container as root?
