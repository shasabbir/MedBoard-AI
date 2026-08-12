"""Tests for capability metrics, RAG evaluation, and ablation reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medboard.evaluation.__main__ import main
from medboard.evaluation.metrics import aggregate_metrics, evaluate_case, text_matches
from medboard.evaluation.models import BenchmarkCase, SystemOutput
from medboard.evaluation.runner import run_evaluation
from medboard.models import TriageLevel


def test_capability_metrics_use_micro_averages_and_safe_empty_routing() -> None:
    label = BenchmarkCase(
        benchmark_id="BENCH-UNIT",
        case_file=Path("unused.json"),
        expected_diagnoses=["Acute neurological process"],
        expected_specialists=[],
        expected_red_flag=False,
        expected_triage=TriageLevel.ROUTINE,
        expected_missing_information=["neurological examination", "blood pressure"],
    )
    result = evaluate_case(
        label,
        SystemOutput(
            diagnoses=["Other consideration", "Acute neurological process"],
            triage_level=TriageLevel.ROUTINE,
            missing_information=["Neurological examination"],
            unsupported_claims=1,
            total_claims=2,
            tokens=100,
            agent_calls=3,
        ),
    )

    metrics = aggregate_metrics([result])

    assert metrics.top_1_recall == 0
    assert metrics.top_3_recall == 1
    assert metrics.routing_precision == 1
    assert metrics.routing_recall == 1
    assert metrics.red_flag_recall == 1
    assert metrics.red_flag_false_alarms == 0
    assert metrics.missing_information_recall == 0.5
    assert metrics.unsupported_claim_rate == 0.5
    assert text_matches("ECG", "ecg")


def test_aggregate_metrics_requires_a_case() -> None:
    with pytest.raises(ValueError, match="at least one"):
        aggregate_metrics([])


def test_offline_evaluation_runs_all_capabilities_and_ablations(
    tmp_path: Path,
) -> None:
    report = run_evaluation(Path.cwd(), workspace=tmp_path / "workspace")

    assert [item.configuration for item in report.configurations] == [
        "single_pass_proxy",
        "multi_agent_no_critic",
        "full_medboard",
    ]
    full = report.configurations[-1]
    assert full.metrics.case_count == 5
    assert full.metrics.top_3_recall == 1
    assert full.metrics.routing_precision == 1
    assert full.metrics.routing_recall == 1
    assert full.metrics.red_flag_recall == 1
    assert full.metrics.red_flag_false_alarms == 0
    assert full.metrics.triage_accuracy == 1
    assert full.metrics.missing_information_recall == 1
    assert report.rag.question_count == 4
    assert report.rag.recall_at_k == 1
    assert report.rag.citation_completeness == 1
    assert all(item.relevant_rank is not None for item in report.rag.questions)


def test_evaluation_cli_writes_machine_and_human_readable_results(
    tmp_path: Path,
) -> None:
    output = tmp_path / "results"

    assert main(["--output", str(output)]) == 0

    data = json.loads((output / "evaluation_results.json").read_text(encoding="utf-8"))
    markdown = (output / "evaluation_results.md").read_text(encoding="utf-8")
    assert data["benchmark_version"] == "1.0"
    assert "single_pass_proxy" in markdown
    assert "deterministic offline-demo results" in markdown
    assert "RAG retrieval" in markdown
