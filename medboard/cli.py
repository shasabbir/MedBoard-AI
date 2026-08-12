"""Command-line entry point for reproducible MedBoard demo cases."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from medboard.config import Settings, get_settings
from medboard.graph.state import MedicalCaseSnapshot, create_initial_state, validate_state
from medboard.graph.workflow import build_collaboration_workflow
from medboard.models import MedicalCaseInput
from medboard.observability import get_logger, log_event, setup_logging
from medboard.providers import DemoModelProvider
from medboard.rag.store import KnowledgeStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a MedBoard synthetic case")
    parser.add_argument(
        "--case",
        type=Path,
        help="Path to a JSON synthetic case (defaults to the bundled anemia case)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete validated state instead of the concise trace",
    )
    return parser


def main(argv: Sequence[str] | None = None, settings: Settings | None = None) -> int:
    args = build_parser().parse_args(argv)
    active_settings = settings or get_settings()
    active_settings.ensure_runtime_directories()
    setup_logging(active_settings)
    logger = get_logger(__name__)

    if not active_settings.demo_mode:
        raise RuntimeError("Live providers are not implemented; set DEMO_MODE=true")

    case_path = args.case or active_settings.demo_cases_directory / "anemia.json"
    case = MedicalCaseInput.model_validate_json(case_path.read_text(encoding="utf-8"))
    initial_state = create_initial_state(case)
    log_event(logger, "case_run_started", run_id=initial_state["run_id"], case_id=case.case_id)

    knowledge_store = KnowledgeStore(active_settings.chroma_persist_directory)
    knowledge_store.ingest_directory(active_settings.knowledge_directory)
    graph = build_collaboration_workflow(
        DemoModelProvider(),
        knowledge_store,
        max_revisions=active_settings.max_revisions,
    )
    result = graph.invoke(initial_state)
    snapshot = validate_state(result)

    if args.json:
        print(snapshot.model_dump_json(indent=2))
    else:
        _print_trace(snapshot)
    log_event(
        logger,
        "case_run_completed",
        run_id=snapshot.run_id,
        errors=len(snapshot.errors),
        evidence=len(snapshot.evidence),
    )
    return 1 if snapshot.errors else 0


def _print_trace(snapshot: MedicalCaseSnapshot) -> None:
    print(f"RUN {snapshot.run_id} | MODE DEMO | CASE {snapshot.case_input.case_id}")
    planned_agents = snapshot.supervisor_plan.initial_agents if snapshot.supervisor_plan else []
    print(f"PLAN: {', '.join(planned_agents)}")
    routed = ", ".join(snapshot.selected_specialists) or "none"
    print(f"SELECTED SPECIALISTS: {routed}")
    print(f"DIFFERENTIAL CONSIDERATIONS: {len(snapshot.differential_diagnoses)}")
    print(
        f"RAG: {len(snapshot.evidence_questions)} questions, "
        f"{len(snapshot.retrieved_evidence)} retrieved chunks"
    )
    critic_decision = snapshot.critic_review.decision.value if snapshot.critic_review else "none"
    triage_level = snapshot.triage_result.triage_level.value if snapshot.triage_result else "none"
    print(
        f"REVIEW: critic={critic_decision}, revisions={snapshot.revision_count}, "
        f"triage={triage_level}"
    )
    print("EXECUTION TRACE")
    for event in snapshot.execution_trace:
        duration = f" | {event.duration_ms:.2f} ms" if event.duration_ms is not None else ""
        print(
            f"- {event.timestamp.isoformat()} | {event.event_type.value} | "
            f"{event.agent or 'system'}{duration}"
        )
    print(
        f"SUMMARY: {len(snapshot.evidence)} evidence items, "
        f"{len(snapshot.agent_messages)} messages, {len(snapshot.errors)} errors, "
        f"{snapshot.total_tokens} approximate demo tokens, "
        f"${snapshot.estimated_cost:.4f} estimated cost"
    )


if __name__ == "__main__":
    raise SystemExit(main())
