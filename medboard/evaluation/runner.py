"""Offline benchmark runner and controlled MedBoard ablations."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from langgraph.graph.state import CompiledStateGraph
from medboard.evaluation.metrics import aggregate_metrics, evaluate_case
from medboard.evaluation.models import (
    BenchmarkCase,
    ConfigurationEvaluation,
    EvaluationReport,
    RagBenchmarkQuestion,
    RagMetrics,
    RagQuestionResult,
    SystemOutput,
)
from medboard.graph.state import create_initial_state, validate_state
from medboard.graph.workflow import build_collaboration_workflow
from medboard.models import (
    AgentStatus,
    ContractModel,
    MedicalCaseInput,
    TraceEventType,
    TriageLevel,
)
from medboard.providers import DemoModelProvider
from medboard.rag.store import KnowledgeStore
from medboard.tools.risk_rules import RiskRuleTool

BENCHMARK_VERSION = "1.0"
DEFAULT_CASES = Path("data/benchmarks/cases.json")
DEFAULT_RAG_QUESTIONS = Path("data/benchmarks/rag_questions.json")
ItemT = TypeVar("ItemT", bound=ContractModel)


def run_evaluation(
    repository_root: Path,
    *,
    cases_path: Path = DEFAULT_CASES,
    rag_questions_path: Path = DEFAULT_RAG_QUESTIONS,
    workspace: Path | None = None,
) -> EvaluationReport:
    labels = _load_list(repository_root / cases_path, BenchmarkCase)
    rag_questions = _load_list(repository_root / rag_questions_path, RagBenchmarkQuestion)
    if workspace is not None:
        workspace.mkdir(parents=True, exist_ok=True)
        return _run(repository_root, workspace, labels, rag_questions)
    default_workspace = repository_root / "data/chroma/evaluation"
    default_workspace.mkdir(parents=True, exist_ok=True)
    return _run(repository_root, default_workspace, labels, rag_questions)


def write_evaluation_report(report: EvaluationReport, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "evaluation_results.json").write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    (output_directory / "evaluation_results.md").write_text(
        render_markdown(report),
        encoding="utf-8",
    )


def render_markdown(report: EvaluationReport) -> str:
    lines = [
        "# MedBoard AI Evaluation Results",
        "",
        f"Benchmark version: `{report.benchmark_version}`",
        "",
        (
            "These are deterministic offline-demo results. The single-pass configuration "
            "is a controlled proxy, not a measured production LLM baseline."
        ),
        "",
        "## Capability and ablation metrics",
        "",
        "| Configuration | Top-1 / 3 / 5 | Routing P / R | Red flags R / FA | "
        "Triage accuracy | Missing-info recall | Unsupported claims |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in report.configurations:
        metrics = result.metrics
        lines.append(
            f"| {result.configuration} | {_pct(metrics.top_1_recall)} / "
            f"{_pct(metrics.top_3_recall)} / {_pct(metrics.top_5_recall)} | "
            f"{_pct(metrics.routing_precision)} / {_pct(metrics.routing_recall)} | "
            f"{_pct(metrics.red_flag_recall)} / {metrics.red_flag_false_alarms} | "
            f"{_pct(metrics.triage_accuracy)} | "
            f"{_pct(metrics.missing_information_recall)} | "
            f"{_pct(metrics.unsupported_claim_rate)} |"
        )
    lines.extend(
        [
            "",
            "## Efficiency metrics",
            "",
            "| Configuration | Mean approximate tokens | Mean agent calls | "
            "Mean revisions |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for result in report.configurations:
        metrics = result.metrics
        lines.append(
            f"| {result.configuration} | {metrics.mean_tokens:.1f} | "
            f"{metrics.mean_agent_calls:.1f} | {metrics.mean_revisions:.1f} |"
        )
    lines.extend(
        [
            "",
            "## RAG retrieval",
            "",
            f"- Recall@K: {_pct(report.rag.recall_at_k)}",
            f"- Mean reciprocal rank: {report.rag.mean_reciprocal_rank:.3f}",
            f"- Citation metadata completeness: {_pct(report.rag.citation_completeness)}",
            "",
            "## Metric definitions",
            "",
            "- Differential recall checks whether each labeled consideration appears "
            "within the first 1, 3, or 5 ranked outputs.",
            "- Routing precision and recall compare selected specialist-case pairs with "
            "their labeled expectations.",
            "- Unsupported-claim rate is the share of differential considerations "
            "without structured supporting-evidence IDs.",
            "- RAG relevance uses the labeled source document; citation completeness "
            "requires document, organization, section, URL, and chunk ID.",
            "",
            "## Interpretation limits",
            "",
            "- The benchmark uses small, synthetic cases and tests software behavior, "
            "not clinical efficacy.",
            "- Expected diagnoses are differential considerations, never autonomous diagnoses.",
            "- Demo token counts are approximate and demo cost is zero.",
            "- Human clinical review remains mandatory for every generated report.",
            "",
        ]
    )
    return "\n".join(lines)


def _run(
    repository_root: Path,
    workspace: Path,
    labels: list[BenchmarkCase],
    rag_questions: list[RagBenchmarkQuestion],
) -> EvaluationReport:
    store = KnowledgeStore(workspace / "chroma")
    store.ingest_directory(repository_root / "data/knowledge")
    provider = DemoModelProvider()
    no_critic_graph = build_collaboration_workflow(provider, store)
    full_graph = build_collaboration_workflow(provider, store, max_revisions=2)

    configurations: list[
        tuple[str, str, Callable[[MedicalCaseInput, str], SystemOutput]]
    ] = [
        (
            "single_pass_proxy",
            "One deterministic pass over raw case fields without agents or evidence IDs.",
            _single_pass_proxy,
        ),
        (
            "multi_agent_no_critic",
            "Parallel intake, dynamic specialists, and RAG without red-team revision.",
            lambda case, run_id: _run_graph(no_critic_graph, case, run_id),
        ),
        (
            "full_medboard",
            "Multi-agent collaboration, dynamic specialists, RAG, critic, and risk triage.",
            lambda case, run_id: _run_graph(full_graph, case, run_id),
        ),
    ]
    evaluations: list[ConfigurationEvaluation] = []
    for name, description, execute in configurations:
        case_results = [
            evaluate_case(
                label,
                execute(label.load_case(repository_root), f"EVAL-{name}-{label.benchmark_id}"),
            )
            for label in labels
        ]
        evaluations.append(
            ConfigurationEvaluation(
                configuration=name,
                description=description,
                metrics=aggregate_metrics(case_results),
                cases=case_results,
            )
        )
    return EvaluationReport(
        benchmark_version=BENCHMARK_VERSION,
        configurations=evaluations,
        rag=_evaluate_rag(store, rag_questions),
    )


def _run_graph(
    graph: CompiledStateGraph, case: MedicalCaseInput, run_id: str
) -> SystemOutput:
    result = graph.invoke(create_initial_state(case, run_id=run_id))
    snapshot = validate_state(result)
    triage = snapshot.triage_result or RiskRuleTool().assess(snapshot.to_graph_state())
    completed_agent_calls = sum(
        1
        for event in snapshot.execution_trace
        if event.event_type is TraceEventType.AGENT_COMPLETED
        and event.status is AgentStatus.COMPLETED
        and event.agent
    )
    return SystemOutput(
        diagnoses=[item.hypothesis for item in snapshot.differential_diagnoses],
        selected_specialists=snapshot.selected_specialists,
        red_flag_detected=bool(triage.red_flags),
        triage_level=triage.triage_level,
        missing_information=[
            item.information_needed for item in snapshot.missing_information
        ],
        unsupported_claims=sum(
            not item.supporting_evidence_ids for item in snapshot.differential_diagnoses
        ),
        total_claims=len(snapshot.differential_diagnoses),
        tokens=snapshot.total_tokens,
        agent_calls=completed_agent_calls,
        revisions=snapshot.revision_count,
    )


def _single_pass_proxy(case: MedicalCaseInput, run_id: str) -> SystemOutput:
    del run_id
    symptoms = {item.casefold() for item in case.symptoms}
    diagnoses: list[str] = []
    missing: list[str] = []
    labs = {item.name.casefold(): item.value for item in case.laboratory_values}
    hemoglobin = labs.get("hemoglobin")
    ferritin = labs.get("ferritin")
    if (
        isinstance(hemoglobin, int | float)
        and hemoglobin < 10
        and isinstance(ferritin, int | float)
        and ferritin < 15
    ):
        diagnoses.append("Iron deficiency anemia pattern")
        missing.extend(["reticulocyte count", "clinician assessment of blood loss"])
    if "chest pain" in symptoms:
        diagnoses.append("Acute coronary syndrome consideration")
        missing.extend(["ECG", "troponin", "blood pressure"])
    if symptoms & {"shortness of breath", "cough", "chest pain"}:
        diagnoses.append("Pulmonary or other cardiorespiratory process")
        missing.extend(["oxygen saturation", "respiratory examination"])
    if symptoms & {"confusion", "unilateral weakness", "seizure", "headache"}:
        diagnoses.append("Acute neurological process")
        missing.extend(["neurological examination", "exact time last known well"])
    if symptoms & {"fever", "cough"}:
        diagnoses.append("Infectious process")
        missing.extend(["temperature trend", "exposure history"])
    if not diagnoses:
        diagnoses.append("Other systemic or metabolic explanation")
        missing.extend(["physical examination", "additional targeted testing"])
    output_text = " ".join([*diagnoses, *missing])
    approximate_tokens = (len(case.model_dump_json()) + len(output_text) + 3) // 4
    return SystemOutput(
        diagnoses=diagnoses,
        selected_specialists=[],
        red_flag_detected=False,
        triage_level=TriageLevel.ROUTINE,
        missing_information=list(dict.fromkeys(missing)),
        unsupported_claims=len(diagnoses),
        total_claims=len(diagnoses),
        tokens=approximate_tokens,
        agent_calls=1,
    )


def _evaluate_rag(
    store: KnowledgeStore, questions: list[RagBenchmarkQuestion]
) -> RagMetrics:
    if not questions:
        raise ValueError("at least one RAG benchmark question is required")
    results: list[RagQuestionResult] = []
    for question in questions:
        retrieved = store.search(
            question.question,
            question_id=question.question_id,
            top_k=question.top_k,
        )
        documents = [item.document for item in retrieved]
        relevant_rank = next(
            (
                index
                for index, document in enumerate(documents, start=1)
                if document == question.expected_document
            ),
            None,
        )
        results.append(
            RagQuestionResult(
                question_id=question.question_id,
                expected_document=question.expected_document,
                retrieved_documents=documents,
                relevant_rank=relevant_rank,
                citations_complete=bool(retrieved)
                and all(
                    item.document
                    and item.source
                    and item.section
                    and item.source_url
                    and item.chunk_id
                    for item in retrieved
                ),
            )
        )
    count = len(results)
    return RagMetrics(
        question_count=count,
        recall_at_k=sum(item.relevant_rank is not None for item in results) / count,
        mean_reciprocal_rank=sum(
            1 / item.relevant_rank if item.relevant_rank else 0 for item in results
        )
        / count,
        citation_completeness=sum(item.citations_complete for item in results) / count,
        questions=results,
    )


def _load_list(path: Path, item_type: type[ItemT]) -> list[ItemT]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"benchmark file must contain a JSON array: {path}")
    return [item_type.model_validate(item) for item in data]


def _pct(value: float) -> str:
    return f"{value:.1%}"
