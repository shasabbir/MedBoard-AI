"""Tests for graph-state initialization and boundary validation."""

import pytest
from pydantic import ValidationError

from medboard.graph.state import create_initial_state, validate_state
from medboard.models import (
    AgentMessage,
    Evidence,
    EvidenceType,
    FinalReport,
    HumanReview,
    HumanStatus,
    MedicalCaseInput,
    MessageType,
    TokenUsage,
    TriageLevel,
    TriageResult,
)


def synthetic_case() -> MedicalCaseInput:
    return MedicalCaseInput(
        case_id="CASE-001",
        synthetic=True,
        age=24,
        chief_complaint="Fatigue and exertional shortness of breath",
        symptoms=["fatigue", "shortness of breath"],
    )


def test_initial_state_has_complete_collection_defaults() -> None:
    state = create_initial_state(synthetic_case(), run_id="RUN-001")

    assert state["run_id"] == "RUN-001"
    assert state["evidence"] == []
    assert state["agent_messages"] == []
    assert state["execution_trace"] == []
    assert state["revision_count"] == 0
    assert state["human_review"].status.value == "not_requested"


def test_state_validates_references_and_round_trips_json() -> None:
    evidence = Evidence(
        evidence_id="EV-001",
        evidence_type=EvidenceType.SYMPTOM,
        name="fatigue",
        value=True,
        source="user_case",
    )
    message = AgentMessage(
        sender="symptoms",
        recipient="supervisor",
        message_type=MessageType.CLAIM,
        content="Fatigue is present.",
        evidence_ids=["EV-001"],
    )
    state = create_initial_state(synthetic_case(), run_id="RUN-001")
    state["evidence"] = [evidence]
    state["agent_messages"] = [message]

    snapshot = validate_state(state)
    restored = type(snapshot).model_validate_json(snapshot.model_dump_json())

    assert restored == snapshot


def test_state_rejects_dangling_evidence_reference() -> None:
    state = create_initial_state(synthetic_case(), run_id="RUN-001")
    state["agent_messages"] = [
        AgentMessage(
            sender="symptoms",
            recipient="supervisor",
            message_type=MessageType.CLAIM,
            content="Unsupported message.",
            evidence_ids=["EV-MISSING"],
        )
    ]

    with pytest.raises(ValidationError, match="unknown evidence references"):
        validate_state(state)


def test_usage_totals_are_derived_from_auditable_records() -> None:
    state = create_initial_state(synthetic_case(), run_id="RUN-001")
    state["token_usage"] = [
        TokenUsage(
            agent="history",
            provider="demo",
            model="deterministic",
            input_tokens=100,
            output_tokens=25,
            estimated_cost=0.002,
        ),
        TokenUsage(
            agent="symptoms",
            provider="demo",
            model="deterministic",
            input_tokens=80,
            output_tokens=20,
            estimated_cost=0.001,
        ),
    ]

    snapshot = validate_state(state)

    assert snapshot.total_tokens == 225
    assert snapshot.estimated_cost == pytest.approx(0.003)


def test_final_report_requires_human_approval() -> None:
    state = create_initial_state(synthetic_case(), run_id="RUN-001")
    state["final_report"] = FinalReport(
        case_summary="Synthetic case summary",
        triage=TriageResult(
            triage_level=TriageLevel.ROUTINE,
            reasoning="No deterministic red flag was triggered.",
            recommended_escalation="Use the normal clinical review pathway.",
        ),
    )

    with pytest.raises(ValidationError, match="requires explicit human approval"):
        validate_state(state)


def test_state_rejects_report_that_diverges_from_validated_analysis() -> None:
    state = create_initial_state(synthetic_case(), run_id="RUN-REPORT-DIVERGENCE")
    state["human_review"] = HumanReview(status=HumanStatus.APPROVED)
    state["triage_result"] = TriageResult(
        triage_level=TriageLevel.EMERGENCY,
        reasoning="Validated emergency assessment.",
        recommended_escalation="Immediate emergency clinical assessment is warranted.",
    )
    state["final_report"] = FinalReport(
        case_summary=state["case_input"].chief_complaint,
        triage=TriageResult(
            triage_level=TriageLevel.ROUTINE,
            reasoning="Inconsistent model-generated assessment.",
            recommended_escalation="Use the normal clinical review pathway.",
        ),
        review_priorities=["Use the normal clinical review pathway."],
    )

    with pytest.raises(ValidationError, match="diverges from validated workflow state"):
        validate_state(state)
