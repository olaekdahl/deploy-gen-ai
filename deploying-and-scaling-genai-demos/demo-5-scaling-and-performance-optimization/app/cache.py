"""
Redis-based response cache for GenAI text generation.

Instructor note:
  Caching is one of the highest-impact optimizations for GenAI services.
  Many real-world workloads have significant prompt repetition (FAQ-style
  queries, templated prompts, re-submitted requests). Caching these
  eliminates redundant inference entirely.

  Key design decisions:
    - Cache key: hash of (prompt, max_tokens, temperature) to ensure only
      identical requests return cached responses.
    - TTL: configurable expiry so stale responses eventually refresh.
    - Graceful degradation: if Redis is unavailable, the service continues
      to work without caching.
"""

import hashlib
import json
import os

import redis.asyncio as redis

from app.logging_config import setup_logging

logger = setup_logging()

_redis_client: redis.Redis | None = None
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "300"))  # 5 minutes default


async def init_cache() -> None:
    """Initialize the async Redis connection."""
    global _redis_client
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        _redis_client = redis.from_url(redis_url, decode_responses=True)
        await _redis_client.ping()
        logger.info("Redis cache connected: %s", redis_url)
    except Exception as exc:
        logger.warning("Redis unavailable, caching disabled: %s", str(exc))
        _redis_client = None


async def close_cache() -> None:
    """Close the Redis connection."""
    global _redis_client
    if _redis_client:
        await _redis_client.aclose()
        _redis_client = None


def cache_connected() -> bool:
    """Check if the cache is connected."""
    return _redis_client is not None


def _make_cache_key(prompt: str, max_tokens: int, temperature: float) -> str:
    """
    Create a deterministic cache key from the request parameters.
    Uses SHA-256 to avoid key-length issues with long prompts.
    """
    payload = f"{prompt}|{max_tokens}|{temperature}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"genai:response:{digest}"


async def get_cached_response(
    prompt: str, max_tokens: int, temperature: float
) -> dict | None:
    """
    Look up a cached response for the given parameters.
    Returns None on cache miss or if Redis is unavailable.
    """
    if _redis_client is None:
        return None
    key = _make_cache_key(prompt, max_tokens, temperature)
    try:
        cached = await _redis_client.get(key)
        if cached:
            logger.info("Cache HIT", extra={"cache_hit": True, "prompt_length": len(prompt)})
            return json.loads(cached)
    except Exception as exc:
        logger.warning("Cache read error: %s", str(exc))
    return None


async def set_cached_response(
    prompt: str, max_tokens: int, temperature: float, response: dict
) -> None:
    """
    Store a response in the cache with a TTL.
    Fails silently if Redis is unavailable.
    """
    if _redis_client is None:
        return
    key = _make_cache_key(prompt, max_tokens, temperature)
    try:
        await _redis_client.set(key, json.dumps(response), ex=CACHE_TTL)
        logger.info("Cache SET", extra={"cache_hit": False, "prompt_length": len(prompt)})
    except Exception as exc:
        logger.warning("Cache write error: %s", str(exc))
