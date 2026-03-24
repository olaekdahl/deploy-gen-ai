"""
Prometheus metrics for the GenAI service.

Instructor note:
  This module defines all Prometheus metrics in one place. Centralizing metrics
  makes it easier to maintain, discover, and document them. Each metric has a
  clear name, description, and appropriate type (counter, histogram, gauge).

  GenAI-specific metrics go beyond standard HTTP metrics:
    - Token counts help track usage and cost
    - Prompt length distributions reveal client behavior
    - Safety flag counts alert on potentially harmful content
"""

from prometheus_client import Counter, Histogram, Gauge, Info

# ---------------------------------------------------------------------------
# Standard HTTP metrics
# ---------------------------------------------------------------------------

# Total number of HTTP requests, labeled by method, endpoint, and status code
REQUEST_COUNT = Counter(
    "genai_http_requests_total",
    "Total HTTP requests",
    labelnames=["method", "endpoint", "status_code"],
)

# Request latency distribution in seconds
REQUEST_LATENCY = Histogram(
    "genai_http_request_duration_seconds",
    "HTTP request latency in seconds",
    labelnames=["method", "endpoint"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

# Total number of unhandled errors
ERROR_COUNT = Counter(
    "genai_errors_total",
    "Total unhandled errors",
    labelnames=["endpoint", "error_type"],
)

# ---------------------------------------------------------------------------
# GenAI-specific metrics
# ---------------------------------------------------------------------------

# Number of tokens generated per request
TOKENS_GENERATED = Histogram(
    "genai_tokens_generated",
    "Number of tokens generated per request",
    buckets=(5, 10, 25, 50, 100, 200, 500),
)

# Prompt length in characters
PROMPT_LENGTH = Histogram(
    "genai_prompt_length_chars",
    "Length of input prompts in characters",
    buckets=(10, 50, 100, 200, 500, 1024),
)

# Model inference latency (excluding HTTP overhead)
INFERENCE_LATENCY = Histogram(
    "genai_inference_duration_seconds",
    "Model inference latency in seconds (excluding HTTP overhead)",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# Safety flag counter -- tracks prompts flagged by content filters
SAFETY_FLAGS = Counter(
    "genai_safety_flags_total",
    "Total prompts or responses flagged by safety filters",
    labelnames=["flag_type"],
)

# Number of requests currently being processed
IN_FLIGHT_REQUESTS = Gauge(
    "genai_in_flight_requests",
    "Number of requests currently being processed",
)

# Service info
SERVICE_INFO = Info(
    "genai_service",
    "Service metadata",
)
