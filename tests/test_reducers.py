"""Tests for concurrent graph-state reducers."""

import pytest

from medboard.graph.reducers import (
    merge_evidence,
    merge_messages,
    merge_missing_information,
    merge_unique_strings,
)
from medboard.models import (
    AgentMessage,
    Evidence,
    EvidenceType,
    MessageType,
    MissingInformationRequest,
)


def test_evidence_merge_is_ordered_and_idempotent() -> None:
    first = Evidence(
        evidence_id="EV-001",
        evidence_type=EvidenceType.SYMPTOM,
        name="fatigue",
        value=True,
        source="user_case",
    )
    second = Evidence(
        evidence_id="EV-002",
        evidence_type=EvidenceType.LAB,
        name="hemoglobin",
        value="9.1 g/dL",
        source="user_case",
    )

    merged = merge_evidence([first], [second, first])

    assert merged == [first, second]


def test_reducer_rejects_conflicting_duplicate_identifiers() -> None:
    original = Evidence(
        evidence_id="EV-SAME",
        evidence_type=EvidenceType.SYMPTOM,
        name="fatigue",
        value=True,
        source="user_case",
    )
    conflicting = Evidence(
        evidence_id="EV-SAME",
        evidence_type=EvidenceType.SYMPTOM,
        name="fatigue",
        value=False,
        source="user_case",
    )

    with pytest.raises(ValueError, match="conflicting records"):
        merge_evidence([original], [conflicting])


def test_parallel_messages_are_not_lost() -> None:
    history_message = AgentMessage(
        message_id="MSG-HISTORY",
        sender="history",
        recipient="supervisor",
        message_type=MessageType.RESPONSE,
        content="History review complete.",
    )
    lab_message = AgentMessage(
        message_id="MSG-LAB",
        sender="laboratory",
        recipient="supervisor",
        message_type=MessageType.WARNING,
        content="Hemoglobin is outside the supplied reference range.",
    )

    assert merge_messages([history_message], [lab_message]) == [
        history_message,
        lab_message,
    ]


def test_specialist_selection_merge_is_unique_and_stable() -> None:
    assert merge_unique_strings(
        ["neurology", "infectious_disease"],
        ["neurology", "cardiology"],
    ) == ["neurology", "infectious_disease", "cardiology"]


def test_missing_information_aggregates_agents_and_priority() -> None:
    differential = MissingInformationRequest(
        request_id="REQ-DIFFERENTIAL-ECG",
        information_needed="ECG",
        requested_by=["differential"],
        reason="Needed to evaluate a cardiac consideration.",
        diagnostic_utility=0.7,
        urgency=0.6,
        evidence_ids=["EV-SYMPTOM-001"],
    )
    cardiology = MissingInformationRequest(
        request_id="REQ-CARDIOLOGY-ECG",
        information_needed=" ecg ",
        requested_by=["cardiology"],
        reason="Needed for focused cardiac review.",
        diagnostic_utility=0.9,
        urgency=0.8,
        evidence_ids=["EV-SYMPTOM-002"],
    )

    merged = merge_missing_information([differential], [cardiology])

    assert len(merged) == 1
    assert merged[0].request_id == "REQ-DIFFERENTIAL-ECG"
    assert merged[0].requested_by == ["differential", "cardiology"]
    assert merged[0].diagnostic_utility == 0.9
    assert merged[0].urgency == 0.8
    assert merged[0].evidence_ids == ["EV-SYMPTOM-001", "EV-SYMPTOM-002"]


def test_resolved_missing_information_stays_resolved_when_requested_again() -> None:
    resolved = MissingInformationRequest(
        information_needed="ECG",
        requested_by=["human_review"],
        reason="Initially requested.",
        resolved=True,
        resolution="Supplied during human review.",
    )
    repeated = MissingInformationRequest(
        information_needed="ECG",
        requested_by=["cardiology"],
        reason="Requested again during reanalysis.",
    )

    merged = merge_missing_information([resolved], [repeated])

    assert len(merged) == 1
    assert merged[0].resolved is True
    assert merged[0].resolution == "Supplied during human review."
