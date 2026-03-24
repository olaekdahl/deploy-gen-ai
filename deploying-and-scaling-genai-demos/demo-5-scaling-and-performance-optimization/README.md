# Demo 5 -- Scaling and Performance Optimization

## What This Demo Teaches

- Redis response caching to eliminate redundant inference
- Measurable latency improvement through caching
- Batch generation to amortize per-request overhead
- Before-and-after performance comparison
- Cache key design with deterministic hashing
- Graceful degradation when the cache is unavailable
- Latency vs. throughput trade-offs

---

## Prerequisites

- Docker 24+ and Docker Compose
- Python 3.11+ (for running the benchmark script)
- pip

---

## Quick Start

```bash
# Build and start the service with Redis
docker compose up --build -d

# Wait for the model to load (~1-2 minutes)
docker compose logs -f genai-service
```

---

## Testing the Service

### Basic request (cache miss)

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Caching improves performance by", "max_tokens": 30}'
```

Note `"cache_hit": false` in the response.

### Repeat the same request (cache hit)

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Caching improves performance by", "max_tokens": 30}'
```

Note `"cache_hit": true` and the much lower latency.

### Batch generation

```bash
curl -X POST http://localhost:8000/generate/batch \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": [
      "The first benefit of batching is",
      "The second benefit of batching is",
      "The third benefit of batching is"
    ],
    "max_tokens": 20
  }'
```

### Health check (includes cache status)

```bash
curl http://localhost:8000/health
```

---

## Running the Benchmark

The benchmark script demonstrates the latency improvement clearly:

```bash
pip install requests
python benchmark.py --url http://localhost:8000
```

### Expected Output

```
======================================================================
GENAI CACHING BENCHMARK
======================================================================

--- Phase 1: Cold requests (cache miss) ---
    1234.5 ms  cache_hit=False  prompt=The benefits of cloud computing in...
    ...

--- Phase 2: Warm requests (cache hit) ---
       3.2 ms  cache_hit=True   prompt=The benefits of cloud computing in...
    ...

======================================================================
RESULTS SUMMARY
======================================================================
                       Avg (ms)  Median (ms)   Min (ms)   Max (ms)
----------------------------------------------------------------------
Cold (no cache)         1150.0       1100.0      980.0     1400.0
Warm (cached)              4.5          3.8        2.5        8.0
Repeat (cached)            3.8          3.5        2.2        6.0

Cache speedup: 255.6x faster
Latency reduction: 99.6%
```

---

## How Caching Works

1. Each request is hashed into a cache key: `SHA256(prompt + max_tokens + temperature)`
2. Before running inference, the handler checks Redis for the key
3. **Cache hit**: Return the cached JSON response immediately (sub-millisecond Redis lookup)
4. **Cache miss**: Run inference, store the result in Redis with a TTL, return the result
5. Entries expire after `CACHE_TTL_SECONDS` (default 300s = 5 minutes)

### Cache Key Design

The key includes all parameters that affect output: prompt text, max_tokens, and temperature.
If any of these differ, the request gets a unique cache entry. This prevents returning
incorrect cached responses for different generation parameters.

---

## Latency vs. Throughput Trade-offs

| Technique | Helps With | Trade-off |
|-----------|-----------|-----------|
| **Caching** | Latency for repeated queries | Stale responses, memory usage |
| **Batching** | Throughput (requests/second) | Higher per-request latency, queue delay |
| **Quantization** | Model size, inference speed | Potential accuracy loss |
| **Horizontal scaling** | Total capacity | Cost, cold start time |

**Caching** gives the biggest latency win for repeated queries (100-1000x improvement).
**Batching** improves throughput but individual requests may wait for the batch to fill.

---

## Stopping the Stack

```bash
docker compose down
```

---

## File Structure

```
demo-5-scaling-and-performance-optimization/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI with cache-aware /generate and /generate/batch
│   ├── models.py            # Includes cache_hit field and batch models
│   ├── inference.py          # Single + batch generation
│   ├── cache.py             # Redis caching layer
│   └── logging_config.py
├── docker-compose.yaml       # App + Redis
├── Dockerfile
├── .dockerignore
├── benchmark.py             # Before/after latency comparison
├── requirements.txt
├── .env.example
└── README.md
```

---

## Key Teaching Talking Points

1. **Cache Hit Ratio**: In production, monitor `cache_hit=True` vs `cache_hit=False`
   in logs or metrics. A high hit ratio means the cache is effective. If the ratio
   is low, prompts are too diverse to benefit from caching.

2. **TTL Trade-off**: Short TTL = fresher responses but more cache misses.
   Long TTL = more hits but potentially stale content. The right value depends
   on how frequently the model or use case changes.

3. **Cache Invalidation**: The hardest problem in caching. This demo uses
   TTL-based expiry. In production, you might also invalidate on model updates
   or use versioned cache keys.

4. **Graceful Degradation**: If Redis goes down, the service continues to work
   without caching. This is a deliberate design choice -- the cache is an
   optimization, not a dependency.

5. **Batching Benefits**: Batching is most impactful on GPU where parallel
   processing is native. On CPU, the benefit is smaller but still meaningful
   for reducing HTTP round-trip overhead when clients have multiple prompts.
