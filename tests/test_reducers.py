"""Tests for concurrent graph-state reducers."""

import pytest

from medboard.graph.reducers import merge_evidence, merge_messages, merge_unique_strings
from medboard.models import AgentMessage, Evidence, EvidenceType, MessageType


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
