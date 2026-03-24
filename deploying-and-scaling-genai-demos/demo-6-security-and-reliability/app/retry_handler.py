"""
Retry handler with exponential backoff.

Instructor note:
  Retries with backoff are essential for reliability. Without backoff,
  retries can amplify failures (retry storms). The pattern here uses:
    - Exponential backoff: wait 1s, 2s, 4s between retries
    - Jitter: randomize the wait to avoid thundering herd
    - A maximum retry count to prevent infinite loops

  This module wraps the inference call so the API layer does not need
  to manage retry logic directly.
"""

import asyncio
import random

from app.inference import generate_fallback, generate_text
from app.logging_config import setup_logging

logger = setup_logging()

MAX_RETRIES = 3
BASE_DELAY_SECONDS = 1.0
MAX_DELAY_SECONDS = 8.0


async def generate_with_retry(
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 0.7,
) -> dict:
    """
    Attempt text generation with retry and fallback.

    On each failure:
      1. Log the error
      2. Wait with exponential backoff + jitter
      3. Retry up to MAX_RETRIES times
      4. If all retries fail, return a fallback response

    Returns a dict with the generation result and whether fallback was used.
    """
    last_exception: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = generate_text(prompt, max_tokens, temperature)
            result["fallback_used"] = False
            return result
        except Exception as exc:
            last_exception = exc
            delay = min(BASE_DELAY_SECONDS * (2 ** (attempt - 1)), MAX_DELAY_SECONDS)
            # Add jitter: randomize between 50% and 100% of the delay
            jitter_delay = delay * (0.5 + random.random() * 0.5)

            logger.warning(
                "Generation attempt %d/%d failed: %s. Retrying in %.1fs",
                attempt,
                MAX_RETRIES,
                str(exc),
                jitter_delay,
            )
            await asyncio.sleep(jitter_delay)

    # All retries exhausted -- use fallback
    logger.error(
        "All %d generation attempts failed. Using fallback. Last error: %s",
        MAX_RETRIES,
        str(last_exception),
    )
    result = generate_fallback(prompt)
    result["fallback_used"] = True
    return result
