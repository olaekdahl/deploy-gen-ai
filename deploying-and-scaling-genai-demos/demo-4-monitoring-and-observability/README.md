# Demo 4 -- Monitoring and Observability

## What This Demo Teaches

- Exposing Prometheus metrics from a FastAPI GenAI service
- Tracking standard HTTP metrics: request count, latency, error rate
- Tracking GenAI-specific metrics: token count, prompt length, safety flags, inference latency
- Running a complete Prometheus + Grafana observability stack with Docker Compose
- Auto-provisioning Grafana datasources and dashboards
- Structured logging with GenAI-specific fields
- In-flight request tracking

---

## Prerequisites

- Docker 24+ and Docker Compose
- ~4 GB of free RAM (for the model + Prometheus + Grafana)

---

## Quick Start

```bash
# Build and start the entire stack
docker compose up --build -d

# Wait for the GenAI service to load the model (~1-2 minutes)
docker compose logs -f genai-service
```

Once you see `"message": "Model loaded successfully"` in the logs, the service is ready.

---

## Accessing the Services

| Service | URL | Credentials |
|---------|-----|-------------|
| GenAI API | http://localhost:8000 | none |
| API Docs | http://localhost:8000/docs | none |
| Prometheus | http://localhost:9090 | none |
| Grafana | http://localhost:3000 | admin / admin |

---

## Testing and Generating Metrics

### Send some requests to generate metrics data

```bash
# Health check
curl http://localhost:8000/health

# Generate text
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Monitoring is important because", "max_tokens": 30}'

# Send multiple requests in a loop to populate dashboards
for i in $(seq 1 20); do
  curl -s -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Request number '"$i"' about AI observability", "max_tokens": 20}' &
done
wait
```

### Trigger a safety flag (for demonstration)

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How to hack into a system", "max_tokens": 10}'
```

This increments the `genai_safety_flags_total` counter.

### View raw Prometheus metrics

```bash
curl http://localhost:8000/metrics
```

---

## Exploring the Dashboards

1. Open Grafana at http://localhost:3000 (login: admin / admin)
2. Go to **Dashboards** in the left sidebar
3. Open the **GenAI Service Dashboard**

The dashboard includes:

| Panel | Metric | What It Shows |
|-------|--------|---------------|
| Request Rate | `genai_http_requests_total` | Requests per second by endpoint and status |
| Request Latency | `genai_http_request_duration_seconds` | p50, p95, p99 HTTP latency |
| Error Rate | `genai_errors_total` | Errors per second by type |
| Tokens Generated | `genai_tokens_generated` | Distribution of output token counts |
| Inference Latency | `genai_inference_duration_seconds` | Model-only latency (no HTTP overhead) |
| Safety Flags | `genai_safety_flags_total` | Cumulative count of flagged prompts |
| In-Flight Requests | `genai_in_flight_requests` | Currently processing requests |
| Prompt Length | `genai_prompt_length_chars` | Distribution of prompt sizes |

---

## Prometheus Queries to Try

Open http://localhost:9090 and enter these in the query box:

```promql
# Request rate over the last minute
rate(genai_http_requests_total[1m])

# Average inference latency (model only)
rate(genai_inference_duration_seconds_sum[1m]) / rate(genai_inference_duration_seconds_count[1m])

# 95th percentile HTTP latency
histogram_quantile(0.95, rate(genai_http_request_duration_seconds_bucket[1m]))

# Total safety flags
sum(genai_safety_flags_total)

# Average tokens generated per request
rate(genai_tokens_generated_sum[1m]) / rate(genai_tokens_generated_count[1m])
```

---

## Logging Guidance

The service outputs structured JSON logs. Key fields:

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | string | UTC ISO 8601 timestamp |
| `level` | string | INFO, WARNING, ERROR |
| `message` | string | Human-readable log message |
| `model_name` | string | Model identifier |
| `prompt_length` | int | Input prompt length in characters |
| `tokens_generated` | int | Output tokens produced |
| `latency_ms` | float | Inference latency in milliseconds |
| `request_id` | string | UUID for request tracing |

Example log line:

```json
{
  "timestamp": "2026-03-24T10:15:30.123456+00:00",
  "level": "INFO",
  "logger": "genai_service",
  "message": "Generation complete",
  "model_name": "distilgpt2",
  "prompt_length": 38,
  "tokens_generated": 25,
  "latency_ms": 842.31
}
```

---

## Stopping the Stack

```bash
docker compose down
```

---

## File Structure

```
demo-4-monitoring-and-observability/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app with metrics middleware
│   ├── models.py
│   ├── inference.py          # Generation with metric instrumentation
│   ├── metrics.py            # Prometheus metric definitions
│   └── logging_config.py
├── prometheus/
│   └── prometheus.yml        # Scrape configuration
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/
│   │   │   └── prometheus.yml
│   │   └── dashboards/
│   │       └── dashboard.yml
│   └── dashboards/
│       └── genai-service.json
├── docker-compose.yaml       # App + Prometheus + Grafana
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── .env.example
└── README.md
```

---

## Key Teaching Talking Points

1. **Metrics vs. Logs vs. Traces**: Metrics are aggregated numeric measurements
   (rates, distributions). Logs are event records. Traces show request flow
   across services. This demo focuses on metrics and logs; traces require
   additional tooling like Jaeger or Tempo.

2. **GenAI-Specific Metrics**: Standard HTTP metrics (request count, latency)
   are necessary but not sufficient. GenAI services need additional metrics:
   token counts (for cost tracking), prompt lengths (for capacity planning),
   inference latency (for SLA monitoring), and safety flags (for compliance).

3. **Histogram vs. Counter vs. Gauge**: Counters only go up. Gauges go up and
   down (like in-flight requests). Histograms capture distributions and let you
   compute percentiles at query time.

4. **Scrape-Based Collection**: Prometheus pulls metrics from the `/metrics`
   endpoint rather than having the application push metrics. This makes the
   application simpler and lets Prometheus control the collection frequency.

5. **Dashboard Provisioning**: Grafana dashboards are defined as JSON and
   loaded automatically. This makes dashboards reproducible and version-controlled,
   avoiding manual dashboard creation in production.
