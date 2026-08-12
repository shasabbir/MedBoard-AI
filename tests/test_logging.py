"""Tests for structured application logging."""

import json
import logging
from pathlib import Path

from medboard.config import Settings
from medboard.observability.logging import (
    JSON_LOG_FILENAME,
    get_logger,
    log_event,
    setup_logging,
)


def test_structured_event_is_written_as_json(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, log_directory=tmp_path)
    configured_logger = setup_logging(settings)
    logger = get_logger("tests")

    log_event(logger, "agent_started", run_id="run-123", agent="history")
    for handler in configured_logger.handlers:
        handler.flush()

    payload = json.loads((tmp_path / JSON_LOG_FILENAME).read_text(encoding="utf-8"))
    assert payload["event"] == "agent_started"
    assert payload["run_id"] == "run-123"
    assert payload["agent"] == "history"
    assert payload["level"] == "INFO"

    for handler in configured_logger.handlers:
        handler.close()
    configured_logger.handlers.clear()


def test_setup_logging_is_idempotent(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, log_directory=tmp_path)

    setup_logging(settings)
    configured_logger = setup_logging(settings)

    assert len(configured_logger.handlers) == 2
    assert configured_logger.propagate is False

    for handler in configured_logger.handlers:
        handler.close()
    configured_logger.handlers.clear()
    logging.shutdown()
