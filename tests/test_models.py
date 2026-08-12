"""Contract tests for evidence, claims, messages, and audit records."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from medboard.models import (
    AgentMessage,
    Claim,
    Evidence,
    EvidenceType,
    MessageType,
    REPORT_DISCLAIMER,
    FinalReport,
    TriageLevel,
    TriageResult,
    SpecialistOpinion,
)


def test_claim_references_are_deduplicated_and_confidence_is_bounded() -> None:
    claim = Claim(
        claim_id="CLM-001",
        agent="laboratory",
        statement="Anemia remains a differential consideration.",
        evidence_ids=["EV-001", "EV-001"],
        confidence=0.72,
    )

    assert claim.evidence_ids == ["EV-001"]
    with pytest.raises(ValidationError):
        claim.confidence = 1.2


def test_claim_rejects_overlapping_support_and_contradiction() -> None:
    with pytest.raises(ValidationError, match="cannot overlap"):
        Claim(
            agent="critic",
            statement="The same evidence cannot support both positions.",
            evidence_ids=["EV-001"],
            contradicting_evidence_ids=["EV-001"],
            confidence=0.5,
        )


def test_message_requires_evidence_identifiers() -> None:
    with pytest.raises(ValidationError, match="must start with 'EV-'"):
        AgentMessage(
            sender="cardiology",
            recipient="supervisor",
            message_type=MessageType.CLAIM,
            content="Cardiac pathology remains relevant.",
            evidence_ids=["not-an-evidence-id"],
        )


def test_audit_timestamps_must_be_timezone_aware() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        Evidence(
            evidence_type=EvidenceType.HISTORY,
            name="symptom onset",
            value="two days",
            source="user_case",
            created_at=datetime(2026, 1, 1),
        )


def test_contracts_forbid_unexpected_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Evidence(
            evidence_type=EvidenceType.SYMPTOM,
            name="fatigue",
            value=True,
            source="user_case",
            invented_field="not allowed",
        )


def test_contracts_reject_non_json_metadata() -> None:
    with pytest.raises(ValidationError):
        Evidence(
            evidence_type=EvidenceType.SYMPTOM,
            name="fatigue",
            value=True,
            source="user_case",
            metadata={"unsafe": object()},
        )


def test_identifiers_are_immutable() -> None:
    evidence = Evidence(
        evidence_id="EV-001",
        evidence_type=EvidenceType.SYMPTOM,
        name="fatigue",
        value=True,
        source="user_case",
    )

    with pytest.raises(ValidationError, match="Field is frozen"):
        evidence.evidence_id = "EV-002"


def test_final_report_disclaimer_cannot_be_removed() -> None:
    triage = TriageResult(
        triage_level=TriageLevel.ROUTINE,
        reasoning="No deterministic emergency rule was triggered.",
        recommended_escalation="Review through the normal clinical pathway.",
    )
    report = FinalReport(case_summary="Synthetic case summary", triage=triage)

    assert report.disclaimer == REPORT_DISCLAIMER
    with pytest.raises(ValidationError, match="cannot be changed"):
        report.disclaimer = "No disclaimer"


def test_model_json_round_trip_preserves_contract() -> None:
    message = AgentMessage(
        message_id="MSG-001",
        sender="critic",
        recipient="supervisor",
        message_type=MessageType.REVISION,
        content="Request another review.",
        evidence_ids=[],
    )

    restored = AgentMessage.model_validate_json(message.model_dump_json())

    assert restored == message
    assert restored.timestamp.tzinfo is not None


def test_specialist_cannot_support_and_challenge_same_hypothesis() -> None:
    with pytest.raises(ValidationError, match="cannot support and challenge"):
        SpecialistOpinion(
            specialist="cardiology",
            assessment="Conflicting opinion",
            supported_hypotheses=["HYP-ONE"],
            challenged_hypotheses=["HYP-ONE"],
            confidence=0.5,
        )
