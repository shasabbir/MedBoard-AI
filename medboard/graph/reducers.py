"""Deterministic reducers for values emitted by concurrent graph branches."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import TypeVar

from medboard.models import (
    AgentError,
    AgentMessage,
    Contradiction,
    CriticReview,
    Evidence,
    EvidenceQuestion,
    MissingInformationRequest,
    RetrievedEvidence,
    SpecialistRoutingDecision,
    SpecialistOpinion,
    TokenUsage,
    TraceEvent,
)

RecordT = TypeVar("RecordT")


def merge_records(
    current: list[RecordT],
    update: list[RecordT],
    identity: Callable[[RecordT], Hashable],
) -> list[RecordT]:
    """Merge append-only records idempotently and reject conflicting duplicate IDs."""
    merged = list(current)
    positions = {identity(record): index for index, record in enumerate(merged)}
    for record in update:
        record_id = identity(record)
        existing_position = positions.get(record_id)
        if existing_position is None:
            positions[record_id] = len(merged)
            merged.append(record)
        elif merged[existing_position] != record:
            raise ValueError(f"conflicting records share identifier {record_id!r}")
    return merged


def merge_evidence(current: list[Evidence], update: list[Evidence]) -> list[Evidence]:
    return merge_records(current, update, lambda item: item.evidence_id)


def merge_evidence_questions(
    current: list[EvidenceQuestion], update: list[EvidenceQuestion]
) -> list[EvidenceQuestion]:
    return merge_records(current, update, lambda item: item.question_id)


def merge_messages(
    current: list[AgentMessage], update: list[AgentMessage]
) -> list[AgentMessage]:
    return merge_records(current, update, lambda item: item.message_id)


def merge_contradictions(
    current: list[Contradiction], update: list[Contradiction]
) -> list[Contradiction]:
    return merge_records(current, update, lambda item: item.contradiction_id)


def merge_critic_reviews(
    current: list[CriticReview], update: list[CriticReview]
) -> list[CriticReview]:
    return merge_records(current, update, lambda item: item.review_id)


def merge_missing_information(
    current: list[MissingInformationRequest], update: list[MissingInformationRequest]
) -> list[MissingInformationRequest]:
    """Aggregate semantically identical requests across independent agents."""
    merged = list(current)
    positions = {
        _information_key(request.information_needed): index
        for index, request in enumerate(merged)
    }
    for request in update:
        key = _information_key(request.information_needed)
        position = positions.get(key)
        if position is None:
            positions[key] = len(merged)
            merged.append(request)
            continue
        existing = merged[position]
        reasons = list(dict.fromkeys([existing.reason, request.reason]))
        merged[position] = existing.model_copy(
            update={
                "requested_by": list(
                    dict.fromkeys([*existing.requested_by, *request.requested_by])
                ),
                "reason": " ".join(reasons),
                "diagnostic_utility": max(
                    existing.diagnostic_utility, request.diagnostic_utility
                ),
                "urgency": max(existing.urgency, request.urgency),
                "evidence_ids": list(
                    dict.fromkeys([*existing.evidence_ids, *request.evidence_ids])
                ),
                "resolved": existing.resolved or request.resolved,
                "resolution": existing.resolution or request.resolution,
            }
        )
    return merged


def _information_key(value: str) -> str:
    return " ".join(value.casefold().split())


def merge_specialist_opinions(
    current: list[SpecialistOpinion], update: list[SpecialistOpinion]
) -> list[SpecialistOpinion]:
    return merge_records(current, update, lambda item: item.opinion_id)


def merge_routing_decisions(
    current: list[SpecialistRoutingDecision],
    update: list[SpecialistRoutingDecision],
) -> list[SpecialistRoutingDecision]:
    return merge_records(current, update, lambda item: item.routing_id)


def merge_retrieved_evidence(
    current: list[RetrievedEvidence], update: list[RetrievedEvidence]
) -> list[RetrievedEvidence]:
    return merge_records(current, update, lambda item: item.retrieval_id)


def merge_errors(current: list[AgentError], update: list[AgentError]) -> list[AgentError]:
    return merge_records(current, update, lambda item: item.error_id)


def merge_trace(current: list[TraceEvent], update: list[TraceEvent]) -> list[TraceEvent]:
    return merge_records(current, update, lambda item: item.trace_id)


def merge_token_usage(
    current: list[TokenUsage], update: list[TokenUsage]
) -> list[TokenUsage]:
    return merge_records(current, update, lambda item: item.usage_id)


def merge_unique_strings(current: list[str], update: list[str]) -> list[str]:
    """Merge routing selections without duplicates, preserving dispatch order."""
    return list(dict.fromkeys([*current, *update]))
