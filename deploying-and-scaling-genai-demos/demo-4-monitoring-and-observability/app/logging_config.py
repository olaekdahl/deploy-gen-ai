import json, logging, sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    def format(self, record):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        for key in ("request_id", "model_name", "prompt_length", "tokens_generated",
                     "latency_ms", "cache_hit"):
            value = getattr(record, key, None)
            if value is not None:
                entry[key] = value
        return json.dumps(entry)


def setup_logging(level="INFO"):
    logger = logging.getLogger("genai_service")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
    return logger
