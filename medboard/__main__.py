"""Run a lightweight bootstrap self-check with ``python -m medboard``."""

from medboard.config import get_settings
from medboard.observability.logging import get_logger, log_event, setup_logging


def main() -> None:
    """Validate settings and initialize local runtime directories and logging."""
    settings = get_settings()
    settings.ensure_runtime_directories()
    setup_logging(settings)

    logger = get_logger(__name__)
    log_event(
        logger,
        "bootstrap_ready",
        app=settings.app_name,
        environment=settings.app_env.value,
        mode=settings.mode_label,
        provider=settings.llm_provider.value,
    )
    print(f"{settings.app_name} bootstrap ready ({settings.mode_label})")


if __name__ == "__main__":
    main()
