"""
Load test script for the GenAI Kubernetes deployment.

Sends concurrent requests to the service to trigger HPA scaling.
Run this while watching:
    kubectl get hpa -n genai-demo --watch

Usage:
    pip install requests
    python load_test.py --url http://<NODE_IP>:30080 --concurrency 10 --duration 120
"""

import argparse
import concurrent.futures
import json
import time

import requests

DEFAULT_PAYLOAD = {
    "prompt": "Kubernetes helps scale applications by",
    "max_tokens": 30,
    "temperature": 0.7,
}


def send_request(url: str, payload: dict, request_num: int) -> dict:
    """Send a single generation request and return timing info."""
    start = time.perf_counter()
    try:
        resp = requests.post(f"{url}/generate", json=payload, timeout=60)
        elapsed = (time.perf_counter() - start) * 1000
        return {
            "request": request_num,
            "status": resp.status_code,
            "latency_ms": round(elapsed, 1),
        }
    except requests.RequestException as e:
        elapsed = (time.perf_counter() - start) * 1000
        return {
            "request": request_num,
            "status": "error",
            "error": str(e),
            "latency_ms": round(elapsed, 1),
        }


def run_load_test(url: str, concurrency: int, duration_seconds: int) -> None:
    """Run concurrent requests for the specified duration."""
    print(f"Load test: {concurrency} concurrent workers for {duration_seconds}s")
    print(f"Target: {url}/generate")
    print("-" * 60)

    results = []
    request_counter = 0
    end_time = time.time() + duration_seconds

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = []
        while time.time() < end_time:
            request_counter += 1
            future = pool.submit(send_request, url, DEFAULT_PAYLOAD, request_counter)
            futures.append(future)
            # Small delay to avoid overwhelming the client
            time.sleep(0.1)

        # Collect results
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    # Summary
    print("\n" + "=" * 60)
    print("LOAD TEST SUMMARY")
    print("=" * 60)
    total = len(results)
    successes = [r for r in results if r.get("status") == 200]
    errors = [r for r in results if r.get("status") != 200]
    latencies = [r["latency_ms"] for r in successes]

    print(f"Total requests:   {total}")
    print(f"Successful:       {len(successes)}")
    print(f"Errors:           {len(errors)}")

    if latencies:
        latencies.sort()
        print(f"Avg latency:      {sum(latencies)/len(latencies):.0f} ms")
        print(f"P50 latency:      {latencies[len(latencies)//2]:.0f} ms")
        print(f"P95 latency:      {latencies[int(len(latencies)*0.95)]:.0f} ms")
        print(f"P99 latency:      {latencies[int(len(latencies)*0.99)]:.0f} ms")
        print(f"Min latency:      {latencies[0]:.0f} ms")
        print(f"Max latency:      {latencies[-1]:.0f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load test the GenAI service")
    parser.add_argument("--url", default="http://localhost:30080", help="Service base URL")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent workers")
    parser.add_argument("--duration", type=int, default=120, help="Test duration in seconds")
    args = parser.parse_args()

    run_load_test(args.url, args.concurrency, args.duration)
