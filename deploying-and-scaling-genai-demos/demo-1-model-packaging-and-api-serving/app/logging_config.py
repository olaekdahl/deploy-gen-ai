"""
Structured JSON logging configuration.

All log output is formatted as JSON to support centralized log aggregation
(e.g., ELK, CloudWatch, Loki). Each log entry includes a UTC timestamp,
log level, logger name, message, and any extra fields passed by the caller.

Instructor note:
  Structured logging is critical in production because it enables automated
  parsing, filtering, and alerting. Compare this with plain-text logging to
  show students the operational advantages.
"""

import json
import logging
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Attach exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Forward well-known extra fields for GenAI observability
        for key in (
            "request_id",
            "model_name",
            "prompt_length",
            "tokens_generated",
            "latency_ms",
        ):
            value = getattr(record, key, None)
            if value is not None:
                log_entry[key] = value

        return json.dumps(log_entry)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Create and return the application logger with JSON formatting."""
    logger = logging.getLogger("genai_service")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

    return logger
