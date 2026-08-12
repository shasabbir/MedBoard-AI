"""End-to-end tests for the first multi-agent workflow."""

from pathlib import Path

from medboard.graph.state import create_initial_state, validate_state
from medboard.graph.workflow import BASE_AGENT_NAMES, build_initial_workflow
from medboard.models import MedicalCaseInput, TraceEventType
from medboard.providers import DemoModelProvider


def load_demo_case() -> MedicalCaseInput:
    case_path = Path("data/demo_cases/anemia.json")
    return MedicalCaseInput.model_validate_json(case_path.read_text(encoding="utf-8"))


def test_demo_case_completes_all_parallel_base_analyses() -> None:
    workflow = build_initial_workflow(DemoModelProvider())

    result = workflow.invoke(create_initial_state(load_demo_case(), run_id="RUN-DEMO"))
    snapshot = validate_state(result)

    assert snapshot.supervisor_plan is not None
    assert snapshot.supervisor_plan.initial_agents == BASE_AGENT_NAMES
    assert snapshot.history_findings is not None
    assert snapshot.symptom_findings is not None
    assert snapshot.laboratory_findings is not None
    assert snapshot.medication_findings is not None
    assert len(snapshot.token_usage) == 5
    assert snapshot.errors == []
    assert snapshot.execution_trace[-1].event_type is TraceEventType.WORKFLOW_COMPLETED


def test_demo_case_produces_auditable_evidence_and_messages() -> None:
    workflow = build_initial_workflow(DemoModelProvider())
    result = workflow.invoke(create_initial_state(load_demo_case(), run_id="RUN-DEMO"))
    snapshot = validate_state(result)

    assert len(snapshot.evidence) == 9
    assert len(snapshot.agent_messages) == 8
    assert snapshot.laboratory_findings is not None
    assert len(snapshot.laboratory_findings.abnormal_values) == 3
    assert snapshot.laboratory_findings.important_patterns
    assert all(
        evidence_id in {item.evidence_id for item in snapshot.evidence}
        for message in snapshot.agent_messages
        for evidence_id in message.evidence_ids
    )
