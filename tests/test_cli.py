"""Command-line acceptance test for the bundled offline workflow."""

from pathlib import Path

import pytest

from medboard.cli import main
from medboard.config import Settings


def test_cli_runs_bundled_case_and_prints_trace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = Settings(_env_file=None, log_directory=tmp_path / "logs")

    exit_code = main(
        ["--case", "data/demo_cases/anemia.json"],
        settings=settings,
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "MODE DEMO | CASE CASE-ANEMIA-001" in output
    assert "PLAN: history, symptoms, laboratory, medication" in output
    assert "SELECTED SPECIALISTS: cardiology" in output
    assert "DIFFERENTIAL CONSIDERATIONS: 2" in output
    assert "workflow_completed" in output
    assert "9 evidence items" in output
    assert "0 errors" in output
