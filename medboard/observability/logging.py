"""Human-readable console and structured JSON-lines logging."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from medboard.config import Settings

LOGGER_NAMESPACE = "medboard"
JSON_LOG_FILENAME = "medboard.jsonl"


class JsonFormatter(logging.Formatter):
    """Serialize safe, explicitly supplied event context as one JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event_name", "log_message"),
            "message": record.getMessage(),
        }
        context = getattr(record, "event_context", {})
        if isinstance(context, dict):
            payload.update(context)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(settings: Settings) -> logging.Logger:
    """Configure idempotent application logging and return its root logger."""
    settings.log_directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAMESPACE)
    logger.setLevel(settings.log_level)
    logger.propagate = False
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )

    json_handler = RotatingFileHandler(
        Path(settings.log_directory) / JSON_LOG_FILENAME,
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    json_handler.setFormatter(JsonFormatter())

    logger.addHandler(console_handler)
    logger.addHandler(json_handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger within the MedBoard namespace."""
    if name == LOGGER_NAMESPACE or name.startswith(f"{LOGGER_NAMESPACE}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAMESPACE}.{name}")


def log_event(logger: logging.Logger, event: str, **context: object) -> None:
    """Emit an event with structured context; callers must never include secrets."""
    logger.info(
        event.replace("_", " "),
        extra={"event_name": event, "event_context": context},
    )
