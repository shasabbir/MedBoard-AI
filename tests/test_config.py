"""Tests for central application settings."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from medboard.config import AppEnvironment, LLMProvider, Settings


def test_default_settings_use_safe_demo_mode() -> None:
    settings = Settings(_env_file=None)

    assert settings.demo_mode is True
    assert settings.mode_label == "DEMO"
    assert settings.llm_provider is LLMProvider.OPENAI
    assert settings.app_env is AppEnvironment.DEVELOPMENT


@pytest.mark.parametrize(
    ("provider", "expected_error"),
    [
        ("openai", "OPENAI_API_KEY and OPENAI_MODEL"),
        ("gemini", "GOOGLE_API_KEY and GEMINI_MODEL"),
    ],
)
def test_live_mode_requires_selected_provider_configuration(
    provider: str, expected_error: str
) -> None:
    with pytest.raises(ValidationError, match=expected_error):
        Settings(_env_file=None, demo_mode=False, llm_provider=provider)


def test_live_mode_accepts_complete_gemini_configuration() -> None:
    settings = Settings(
        _env_file=None,
        demo_mode=False,
        llm_provider="GEMINI",
        google_api_key="test-key",
        gemini_model="test-model",
    )

    assert settings.mode_label == "LIVE LLM"
    assert settings.llm_provider is LLMProvider.GEMINI
    assert "test-key" not in repr(settings)


def test_runtime_directories_are_created(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        database_path=tmp_path / "database" / "medboard.db",
        workflow_checkpoint_path=tmp_path / "checkpoints" / "workflow.db",
        chroma_persist_directory=tmp_path / "chroma",
        knowledge_directory=tmp_path / "knowledge",
        demo_cases_directory=tmp_path / "cases",
        log_directory=tmp_path / "logs",
    )

    settings.ensure_runtime_directories()

    assert settings.database_path.parent.is_dir()
    assert settings.workflow_checkpoint_path.parent.is_dir()
    assert settings.chroma_persist_directory.is_dir()
    assert settings.knowledge_directory.is_dir()
    assert settings.demo_cases_directory.is_dir()
    assert settings.log_directory.is_dir()


def test_invalid_revision_limit_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_revisions=4)
