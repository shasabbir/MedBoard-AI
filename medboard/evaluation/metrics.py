"""Pure metric calculations for benchmark and ablation results."""

from __future__ import annotations

import re

from medboard.evaluation.models import (
    BenchmarkCase,
    CapabilityMetrics,
    CaseEvaluation,
    SystemOutput,
)


def evaluate_case(label: BenchmarkCase, output: SystemOutput) -> CaseEvaluation:
    ranks = {
        expected: _relevant_rank(expected, output.diagnoses)
        for expected in label.expected_diagnoses
    }
    matched_missing = [
        expected
        for expected in label.expected_missing_information
        if _relevant_rank(expected, output.missing_information) is not None
    ]
    return CaseEvaluation(
        benchmark_id=label.benchmark_id,
        expected_diagnoses=label.expected_diagnoses,
        diagnoses=output.diagnoses,
        diagnosis_ranks=ranks,
        expected_specialists=label.expected_specialists,
        selected_specialists=output.selected_specialists,
        expected_red_flag=label.expected_red_flag,
        red_flag_detected=output.red_flag_detected,
        expected_triage=label.expected_triage,
        triage_level=output.triage_level,
        expected_missing_information=label.expected_missing_information,
        matched_missing_information=matched_missing,
        unsupported_claims=output.unsupported_claims,
        total_claims=output.total_claims,
        tokens=output.tokens,
        agent_calls=output.agent_calls,
        revisions=output.revisions,
    )


def aggregate_metrics(cases: list[CaseEvaluation]) -> CapabilityMetrics:
    if not cases:
        raise ValueError("at least one case evaluation is required")
    expected_diagnoses = sum(len(case.expected_diagnoses) for case in cases)

    selected_pairs = {
        (case.benchmark_id, specialist)
        for case in cases
        for specialist in case.selected_specialists
    }
    expected_pairs = {
        (case.benchmark_id, specialist)
        for case in cases
        for specialist in case.expected_specialists
    }
    true_positives = len(selected_pairs & expected_pairs)

    red_flag_cases = [case for case in cases if case.expected_red_flag]
    true_red_flags = sum(case.red_flag_detected for case in red_flag_cases)
    false_alarms = sum(
        case.red_flag_detected for case in cases if not case.expected_red_flag
    )
    expected_missing = sum(len(case.expected_missing_information) for case in cases)
    matched_missing = sum(len(case.matched_missing_information) for case in cases)
    total_claims = sum(case.total_claims for case in cases)
    unsupported_claims = sum(case.unsupported_claims for case in cases)
    count = len(cases)

    return CapabilityMetrics(
        case_count=count,
        top_1_recall=_top_k_recall(cases, expected_diagnoses, 1),
        top_3_recall=_top_k_recall(cases, expected_diagnoses, 3),
        top_5_recall=_top_k_recall(cases, expected_diagnoses, 5),
        routing_precision=_safe_divide(true_positives, len(selected_pairs), empty=1.0),
        routing_recall=_safe_divide(true_positives, len(expected_pairs), empty=1.0),
        red_flag_recall=_safe_divide(true_red_flags, len(red_flag_cases), empty=1.0),
        red_flag_false_alarms=false_alarms,
        triage_accuracy=sum(
            case.triage_level == case.expected_triage for case in cases
        )
        / count,
        missing_information_recall=_safe_divide(
            matched_missing, expected_missing, empty=1.0
        ),
        unsupported_claim_rate=_safe_divide(
            unsupported_claims, total_claims, empty=0.0
        ),
        mean_tokens=sum(case.tokens for case in cases) / count,
        mean_agent_calls=sum(case.agent_calls for case in cases) / count,
        mean_revisions=sum(case.revisions for case in cases) / count,
    )


def text_matches(expected: str, actual: str) -> bool:
    expected_normalized = _normalize(expected)
    actual_normalized = _normalize(actual)
    return expected_normalized in actual_normalized or actual_normalized in expected_normalized


def _relevant_rank(expected: str, actual_values: list[str]) -> int | None:
    return next(
        (
            index
            for index, actual in enumerate(actual_values, start=1)
            if text_matches(expected, actual)
        ),
        None,
    )


def _top_k_recall(
    cases: list[CaseEvaluation], expected_count: int, k: int
) -> float:
    hits = sum(
        rank is not None and rank <= k
        for case in cases
        for rank in case.diagnosis_ranks.values()
    )
    return _safe_divide(hits, expected_count, empty=0.0)


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _safe_divide(numerator: int, denominator: int, *, empty: float) -> float:
    return numerator / denominator if denominator else empty
