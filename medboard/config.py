"""Central, environment-driven configuration for MedBoard AI."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnvironment(StrEnum):
    """Supported application environments."""

    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class LLMProvider(StrEnum):
    """Supported live LLM providers."""

    OPENAI = "openai"
    GEMINI = "gemini"


class Settings(BaseSettings):
    """Validated application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "MedBoard AI"
    app_env: AppEnvironment = AppEnvironment.DEVELOPMENT
    demo_mode: bool = True
    log_level: str = "INFO"

    llm_provider: LLMProvider = LLMProvider.OPENAI
    openai_api_key: SecretStr | None = None
    openai_model: str | None = None
    google_api_key: SecretStr | None = None
    gemini_model: str | None = None

    database_path: Path = Path("data/medboard.db")
    workflow_checkpoint_path: Path = Path("data/workflow_checkpoints.db")
    chroma_persist_directory: Path = Path("data/chroma")
    knowledge_directory: Path = Path("data/knowledge")
    demo_cases_directory: Path = Path("data/demo_cases")
    log_directory: Path = Path("logs")

    max_revisions: int = Field(default=2, ge=1, le=3)
    max_agent_retries: int = Field(default=2, ge=0, le=5)
    agent_timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    rag_top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("app_env", "llm_provider", mode="before")
    @classmethod
    def normalize_enum_values(cls, value: object) -> object:
        """Allow convenient case-insensitive enum values in environment files."""
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, value: object) -> str:
        """Normalize and validate Python logging level names."""
        normalized = str(value).strip().upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in allowed:
            allowed_values = ", ".join(sorted(allowed))
            raise ValueError(f"log_level must be one of: {allowed_values}")
        return normalized

    @model_validator(mode="after")
    def validate_live_provider_configuration(self) -> Self:
        """Require credentials and a model only when live mode is enabled."""
        if self.demo_mode:
            return self

        if self.llm_provider is LLMProvider.OPENAI:
            if not self._has_secret(self.openai_api_key) or not self.openai_model:
                raise ValueError(
                    "Live OpenAI mode requires OPENAI_API_KEY and OPENAI_MODEL"
                )
        elif not self._has_secret(self.google_api_key) or not self.gemini_model:
            raise ValueError("Live Gemini mode requires GOOGLE_API_KEY and GEMINI_MODEL")

        return self

    @property
    def mode_label(self) -> str:
        """Return the user-facing execution mode label."""
        return "DEMO" if self.demo_mode else "LIVE LLM"

    def ensure_runtime_directories(self) -> None:
        """Create only the local directories needed by application runtime state."""
        directories = {
            self.database_path.parent,
            self.workflow_checkpoint_path.parent,
            self.chroma_persist_directory,
            self.knowledge_directory,
            self.demo_cases_directory,
            self.log_directory,
        }
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _has_secret(secret: SecretStr | None) -> bool:
        return bool(secret and secret.get_secret_value().strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one validated settings object per process."""
    return Settings()
