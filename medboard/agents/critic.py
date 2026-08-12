"""Red-team review that attempts to falsify the current differential."""

from __future__ import annotations

from medboard.agents.base import BaseAgent, StateUpdate
from medboard.graph.state import MedicalCaseState
from medboard.models import (
    AgentMessage,
    CriticDecision,
    CriticReview,
    MessageType,
    Severity,
)
from medboard.providers import StructuredModelProvider


class CriticAgent(BaseAgent):
    name = "critic"

    def __init__(
        self,
        provider: StructuredModelProvider,
        max_revisions: int = 2,
        *,
        max_retries: int = 2,
    ) -> None:
        super().__init__(provider, max_retries=max_retries)
        self.max_revisions = max_revisions

    def analyze(self, state: MedicalCaseState) -> StateUpdate:
        unresolved = [item for item in state["contradictions"] if not item.resolved]
        unsupported = [
            diagnosis.hypothesis_id
            for diagnosis in state["differential_diagnoses"]
            if not diagnosis.supporting_evidence_ids
        ]
        revision_count = state.get("revision_count", 0)
        should_revise = bool(unresolved or unsupported) and revision_count < self.max_revisions
        limit_reached = bool(unresolved or unsupported) and revision_count >= self.max_revisions
        decision = CriticDecision.REVISE if should_revise else CriticDecision.ACCEPT
        problems = [
            f"Unresolved disagreement about {item.topic}." for item in unresolved
        ]
        if limit_reached:
            problems.append(
                "Revision limit reached; unresolved issues must remain visible for human review."
            )
        severity = (
            Severity.HIGH
            if unresolved
            else (Severity.MEDIUM if unsupported else Severity.LOW)
        )
        result = self.provider.generate(
            agent=self.name,
            prompt=(
                "Attempt to falsify the differential, identify unsupported claims, premature "
                "closure, contradictions, and missing evidence."
            ),
            response_model=CriticReview,
            demo_factory=lambda: CriticReview(
                decision=decision,
                problems=problems,
                overlooked_hypotheses=[],
                unsupported_claim_ids=unsupported,
                missing_evidence=(
                    ["Evidence capable of resolving the recorded disagreement"]
                    if unresolved
                    else []
                ),
                questions=(
                    ["Has the competing interpretation been explicitly reconsidered?"]
                    if unresolved
                    else []
                ),
                severity=severity,
            ),
        )
        review = result.output
        recipient = "supervisor" if review.decision is CriticDecision.REVISE else "risk"
        return {
            "critic_review": review,
            "critic_reviews": [review],
            "agent_messages": [
                AgentMessage(
                    sender=self.name,
                    recipient=recipient,
                    message_type=(
                        MessageType.REVISION
                        if review.decision is CriticDecision.REVISE
                        else MessageType.RESPONSE
                    ),
                    content=(
                        "Revision required: " + "; ".join(review.problems)
                        if review.decision is CriticDecision.REVISE
                        else "Red-team review accepted progression to risk assessment."
                    ),
                )
            ],
            "token_usage": [result.usage],
        }
