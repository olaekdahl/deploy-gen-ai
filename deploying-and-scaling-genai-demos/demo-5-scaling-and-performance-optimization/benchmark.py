"""
Benchmark script: measure latency with and without caching.

Demonstrates the performance improvement from Redis response caching.

Usage:
    # Start the stack first
    docker compose up --build -d

    # Wait for the service to be healthy, then run:
    pip install requests
    python benchmark.py --url http://localhost:8000
"""

import argparse
import statistics
import time

import requests

# These prompts are reused to test cache hits
TEST_PROMPTS = [
    "The benefits of cloud computing include",
    "Machine learning models can be optimized by",
    "Kubernetes autoscaling works by",
    "The future of artificial intelligence is",
    "Data pipeline architecture should consider",
]


def measure_latency(url: str, prompt: str, max_tokens: int = 20) -> dict:
    """Send a request and return latency + cache_hit."""
    payload = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.7}
    start = time.perf_counter()
    resp = requests.post(f"{url}/generate", json=payload, timeout=60)
    elapsed_ms = (time.perf_counter() - start) * 1000
    data = resp.json()
    return {
        "latency_ms": round(elapsed_ms, 1),
        "cache_hit": data.get("cache_hit", False),
        "status": resp.status_code,
    }


def run_benchmark(url: str) -> None:
    print("=" * 70)
    print("GENAI CACHING BENCHMARK")
    print("=" * 70)

    # Phase 1: Cold requests (cache miss)
    print("\n--- Phase 1: Cold requests (cache miss) ---")
    cold_latencies = []
    for prompt in TEST_PROMPTS:
        result = measure_latency(url, prompt)
        cold_latencies.append(result["latency_ms"])
        print(f"  {result['latency_ms']:>8.1f} ms  cache_hit={result['cache_hit']}  "
              f"prompt={prompt[:40]}...")

    # Phase 2: Warm requests (cache hit)
    print("\n--- Phase 2: Warm requests (cache hit) ---")
    warm_latencies = []
    for prompt in TEST_PROMPTS:
        result = measure_latency(url, prompt)
        warm_latencies.append(result["latency_ms"])
        print(f"  {result['latency_ms']:>8.1f} ms  cache_hit={result['cache_hit']}  "
              f"prompt={prompt[:40]}...")

    # Phase 3: Mixed (repeat all prompts again)
    print("\n--- Phase 3: Repeated warm requests ---")
    repeat_latencies = []
    for prompt in TEST_PROMPTS:
        result = measure_latency(url, prompt)
        repeat_latencies.append(result["latency_ms"])

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'':20s} {'Avg (ms)':>10s} {'Median (ms)':>12s} {'Min (ms)':>10s} {'Max (ms)':>10s}")
    print("-" * 70)

    for label, data in [("Cold (no cache)", cold_latencies),
                         ("Warm (cached)", warm_latencies),
                         ("Repeat (cached)", repeat_latencies)]:
        print(f"{label:20s} {statistics.mean(data):>10.1f} {statistics.median(data):>12.1f} "
              f"{min(data):>10.1f} {max(data):>10.1f}")

    if warm_latencies and cold_latencies:
        avg_cold = statistics.mean(cold_latencies)
        avg_warm = statistics.mean(warm_latencies)
        if avg_cold > 0:
            speedup = avg_cold / avg_warm
            reduction_pct = ((avg_cold - avg_warm) / avg_cold) * 100
            print(f"\nCache speedup: {speedup:.1f}x faster")
            print(f"Latency reduction: {reduction_pct:.1f}%")

    # Batch test
    print("\n--- Batch Generation Test ---")
    batch_payload = {
        "prompts": TEST_PROMPTS[:4],
        "max_tokens": 20,
        "temperature": 0.7,
    }
    start = time.perf_counter()
    resp = requests.post(f"{url}/generate/batch", json=batch_payload, timeout=120)
    batch_ms = (time.perf_counter() - start) * 1000
    if resp.status_code == 200:
        data = resp.json()
        print(f"  Batch of {data['batch_size']} prompts: {batch_ms:.1f} ms total")
        print(f"  Per-prompt average: {batch_ms/data['batch_size']:.1f} ms")
        print(f"  Sequential equivalent (cold avg): {statistics.mean(cold_latencies[:4]) * 4:.1f} ms")
    else:
        print(f"  Batch request failed: {resp.status_code}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark caching performance")
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    run_benchmark(args.url)
