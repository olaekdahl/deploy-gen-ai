# Deploying and Scaling Generative AI Applications -- Demo Projects

This repository contains six progressive, self-contained demo projects for the
**Deploying and Scaling Generative AI Applications** course (GAI-2401).

Each demo builds on the concepts from the previous one, guiding students from a
local API to a production-hardened, observable, auto-scaling deployment.

---

## Demo Overview

| # | Demo | What It Teaches | Key Technologies |
|---|------|-----------------|------------------|
| 1 | [Model Packaging and API Serving](demo-1-model-packaging-and-api-serving/) | Model loading, ONNX export, FastAPI serving, validation, structured logging | FastAPI, Transformers, ONNX Runtime |
| 2 | [Dockerized GenAI Service](demo-2-dockerized-genai-service/) | Containerization, multi-stage builds, image optimization | Docker, multi-stage Dockerfile |
| 3 | [Kubernetes Deployment and Autoscaling](demo-3-kubernetes-deployment-and-autoscaling/) | Orchestration, services, HPA, resource management | Kubernetes, HPA, kubectl |
| 4 | [Monitoring and Observability](demo-4-monitoring-and-observability/) | Prometheus metrics, Grafana dashboards, GenAI-specific telemetry | Prometheus, Grafana, prometheus_client |
| 5 | [Scaling and Performance Optimization](demo-5-scaling-and-performance-optimization/) | Response caching, batching, latency/throughput trade-offs | Redis, async batching |
| 6 | [Security and Reliability](demo-6-security-and-reliability/) | API key auth, input validation, retries, fallback, PII handling | OAuth2/API keys, tenacity, Pydantic |

---

## Suggested Teaching Order

The demos are numbered in the recommended delivery sequence:

1. **Demo 1** -- Start here. Students run a GenAI API locally and understand model
   loading, serialization (ONNX export), and request/response patterns.
2. **Demo 2** -- Package the service into a Docker container. Covers multi-stage
   builds, layer ordering, and image size optimization.
3. **Demo 3** -- Deploy the container to Kubernetes. Covers Deployments, Services,
   resource limits, and Horizontal Pod Autoscaler.
4. **Demo 4** -- Add operational visibility. Integrate Prometheus metrics (including
   GenAI-specific metrics), wire up Grafana dashboards, and structure logs.
5. **Demo 5** -- Optimize performance. Add Redis response caching and request
   batching. Benchmark before-and-after latency.
6. **Demo 6** -- Harden the service. Add API key authentication, retry logic with
   backoff, fallback responses, structured error handling, and PII-aware logging.

---

## Setup Complexity

| Demo | Local Dependencies | Containers Needed | Estimated Setup Time |
|------|--------------------|-------------------|---------------------|
| 1 | Python 3.11+, pip | None | 5-10 min |
| 2 | Docker | 1 (app) | 5-10 min |
| 3 | Docker, kubectl, minikube/kind | 1+ (cluster) | 15-20 min |
| 4 | Docker, Docker Compose | 3 (app, Prometheus, Grafana) | 10-15 min |
| 5 | Docker, Docker Compose | 2 (app, Redis) | 10-15 min |
| 6 | Python 3.11+, pip | None (optional Docker) | 5-10 min |

---

## Prerequisites

All demos assume:

- **Python 3.11 or later** is installed
- **pip** is available
- **Docker** is installed for demos 2-5
- **kubectl** and a local K8s cluster (minikube or kind) for demo 3
- A machine with at least 4 GB of free RAM (for model loading)
- Internet access to download the `distilgpt2` model (~350 MB on first run)

---

## Quick Start

```bash
# Clone this repository and navigate into it
cd deploying-and-scaling-genai-demos

# Start with Demo 1
cd demo-1-model-packaging-and-api-serving
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000/docs to explore the interactive API documentation.

---

## Model Choice

All demos use **distilgpt2** from Hugging Face -- a distilled version of GPT-2
with 82M parameters. This model is:

- Small enough to run on CPU in a classroom setting
- A real generative language model (not a mock)
- Fast enough for interactive demos
- Available without authentication or API keys

The model is downloaded automatically on first run and cached locally.

---

## Repository Structure

```
deploying-and-scaling-genai-demos/
├── README.md                                            # This file
├── demo-1-model-packaging-and-api-serving/             # Local API serving
├── demo-2-dockerized-genai-service/                    # Docker packaging
├── demo-3-kubernetes-deployment-and-autoscaling/       # K8s deployment
├── demo-4-monitoring-and-observability/                # Prometheus + Grafana
├── demo-5-scaling-and-performance-optimization/        # Redis caching + batching
└── demo-6-security-and-reliability/                    # Auth, retries, fallback
```

Each demo folder is self-contained and can be used independently, though the
progression from Demo 1 through Demo 6 provides the best learning experience.
