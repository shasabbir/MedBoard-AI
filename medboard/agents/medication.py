"""Deterministic demo implementation of medication review."""

from __future__ import annotations

from medboard.agents.base import BaseAgent, StateUpdate
from medboard.graph.state import MedicalCaseState
from medboard.models import (
    AgentMessage,
    AgentOutput,
    AgentStatus,
    Evidence,
    EvidenceType,
    MedicationFindings,
    MessageType,
    MissingInformationRequest,
)


class MedicationAgent(BaseAgent):
    name = "medication"

    def analyze(self, state: MedicalCaseState) -> StateUpdate:
        case = state["case_input"]
        evidence = [
            Evidence(
                evidence_id=f"EV-MEDICATION-{index:03d}",
                evidence_type=EvidenceType.MEDICATION,
                name="reported medication",
                value=medication,
                source="user_case",
            )
            for index, medication in enumerate(case.medications, start=1)
        ]
        missing = [] if case.medications else ["complete medication and supplement list"]
        result = self.provider.generate(
            agent=self.name,
            prompt=(
                "Review supplied medicines for symptom relevance, interactions, duplicate "
                "classes, and concerns without prescribing or recommending dose changes."
            ),
            response_model=MedicationFindings,
            demo_factory=lambda: MedicationFindings(
                possible_adverse_effects=[],
                potential_interactions=[],
                duplicate_classes=[],
                review_concerns=(
                    ["Medication effects cannot be assessed until a complete list is supplied."]
                    if missing
                    else []
                ),
                output=AgentOutput(
                    agent=self.name,
                    status=AgentStatus.COMPLETED,
                    summary=(
                        "No medication-specific concern was identified from the supplied list."
                        if evidence
                        else "Medication information is incomplete and was not inferred."
                    ),
                    missing_information=missing,
                ),
            ),
        )
        update: StateUpdate = {
            "medication_findings": result.output,
            "evidence": evidence,
            "agent_messages": [
                AgentMessage(
                    sender=self.name,
                    recipient="supervisor",
                    message_type=MessageType.RESPONSE,
                    content=result.output.output.summary,
                    evidence_ids=[item.evidence_id for item in evidence],
                )
            ],
            "token_usage": [result.usage],
        }
        if missing:
            update["missing_information"] = [
                MissingInformationRequest(
                    request_id="REQ-MEDICATION-001",
                    information_needed=missing[0],
                    requested_by=[self.name],
                    reason="Medication-related explanations cannot be assessed without a list.",
                )
            ]
        return update
