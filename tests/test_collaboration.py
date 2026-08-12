"""End-to-end tests for differential reasoning and conditional specialists."""

from pathlib import Path

import pytest

from medboard.graph.state import create_initial_state, validate_state
from medboard.graph.workflow import build_collaboration_workflow
from medboard.models import MedicalCaseInput
from medboard.providers import DemoModelProvider


def load_case(name: str) -> MedicalCaseInput:
    path = Path("data/demo_cases") / f"{name}.json"
    return MedicalCaseInput.model_validate_json(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("case_name", "expected_specialists", "excluded_specialists"),
    [
        ("anemia", ["cardiology"], ["neurology", "infectious_disease"]),
        ("neurological", ["neurology"], ["cardiology", "infectious_disease"]),
        ("infectious", ["infectious_disease"], ["cardiology", "neurology"]),
    ],
)
def test_router_selects_only_evidence_justified_specialists(
    case_name: str,
    expected_specialists: list[str],
    excluded_specialists: list[str],
) -> None:
    workflow = build_collaboration_workflow(DemoModelProvider())

    result = workflow.invoke(
        create_initial_state(load_case(case_name), run_id=f"RUN-{case_name.upper()}")
    )
    snapshot = validate_state(result)

    assert snapshot.selected_specialists == expected_specialists
    assert len(snapshot.specialist_opinions) == len(expected_specialists)
    assert snapshot.routing_decisions[-1].selected_specialists == expected_specialists
    assert set(snapshot.routing_decisions[-1].reasons) == set(expected_specialists)
    assert not (set(excluded_specialists) & set(snapshot.selected_specialists))


def test_differential_produces_competing_traceable_hypotheses() -> None:
    workflow = build_collaboration_workflow(DemoModelProvider())
    result = workflow.invoke(create_initial_state(load_case("anemia"), run_id="RUN-ANEMIA"))
    snapshot = validate_state(result)
    known_evidence = {item.evidence_id for item in snapshot.evidence}

    assert len(snapshot.differential_diagnoses) >= 2
    assert snapshot.differential_analysis is not None
    assert all(item.supporting_evidence_ids for item in snapshot.differential_diagnoses)
    assert all(
        set(item.supporting_evidence_ids) <= known_evidence
        for item in snapshot.differential_diagnoses
    )
    assert all(item.confidence < 1 for item in snapshot.differential_diagnoses)


def test_specialist_challenge_becomes_auditable_contradiction() -> None:
    workflow = build_collaboration_workflow(DemoModelProvider())
    result = workflow.invoke(create_initial_state(load_case("anemia"), run_id="RUN-ANEMIA"))
    snapshot = validate_state(result)

    assert snapshot.contradictions
    contradiction = snapshot.contradictions[0]
    assert contradiction.agent_a == "differential"
    assert contradiction.agent_b == "cardiology"
    assert contradiction.hypothesis_id == "HYP-PULMONARY"
    assert contradiction.resolved is False
    assert any(
        message.message_type.value == "challenge"
        and message.sender == "cardiology"
        and message.recipient == "differential"
        for message in snapshot.agent_messages
    )


def test_router_can_fan_out_to_multiple_specialists() -> None:
    case = MedicalCaseInput(
        case_id="CASE-MULTI-001",
        chief_complaint="Fever with acute confusion",
        symptoms=["fever", "confusion"],
    )
    workflow = build_collaboration_workflow(DemoModelProvider())

    result = workflow.invoke(create_initial_state(case, run_id="RUN-MULTI"))
    snapshot = validate_state(result)

    assert snapshot.selected_specialists == ["neurology", "infectious_disease"]
    assert {item.specialist for item in snapshot.specialist_opinions} == {
        "neurology",
        "infectious_disease",
    }
    assert snapshot.execution_trace[-1].details["selected_specialists"] == [
        "neurology",
        "infectious_disease",
    ]
    routing_requests = {
        message.recipient: set(message.evidence_ids)
        for message in snapshot.agent_messages
        if message.sender == "supervisor"
        and message.message_type.value == "request"
        and message.recipient in {"neurology", "infectious_disease"}
    }
    evidence_names = {
        item.evidence_id: item.name.casefold() for item in snapshot.evidence
    }
    assert {evidence_names[item] for item in routing_requests["neurology"]} == {
        "confusion"
    }
    assert {evidence_names[item] for item in routing_requests["infectious_disease"]} == {
        "fever"
    }


def test_router_can_select_no_specialist_and_still_complete() -> None:
    case = MedicalCaseInput(
        case_id="CASE-GENERAL-001",
        chief_complaint="Reduced appetite",
        symptoms=["reduced appetite"],
    )
    workflow = build_collaboration_workflow(DemoModelProvider())

    result = workflow.invoke(create_initial_state(case, run_id="RUN-GENERAL"))
    snapshot = validate_state(result)

    assert snapshot.selected_specialists == []
    assert snapshot.specialist_opinions == []
    assert len(snapshot.differential_diagnoses) == 2
    assert snapshot.execution_trace[-1].event_type.value == "workflow_completed"
