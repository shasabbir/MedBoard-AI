"""Deterministic demo implementation of symptom normalization and clustering."""

from __future__ import annotations

from medboard.agents.base import BaseAgent, StateUpdate
from medboard.graph.state import MedicalCaseState
from medboard.models import (
    AgentMessage,
    AgentOutput,
    AgentStatus,
    Claim,
    Evidence,
    EvidenceType,
    MessageType,
    MissingInformationRequest,
    SymptomFindings,
)

SYMPTOM_ALIASES = {
    "my heart is racing": "palpitations",
    "breathlessness": "shortness of breath",
    "sob": "shortness of breath",
}


class SymptomAgent(BaseAgent):
    name = "symptoms"

    def analyze(self, state: MedicalCaseState) -> StateUpdate:
        case = state["case_input"]
        normalized = [
            SYMPTOM_ALIASES.get(symptom.casefold(), symptom.casefold())
            for symptom in case.symptoms
        ]
        evidence = [
            Evidence(
                evidence_id=f"EV-SYMPTOM-{index:03d}",
                evidence_type=EvidenceType.SYMPTOM,
                name=symptom,
                value=True,
                source="user_case",
            )
            for index, symptom in enumerate(normalized, start=1)
        ]
        red_flags = _red_flags(normalized)
        systems = _possible_systems(normalized)
        missing = ["symptom severity", "precise onset and duration"]
        claim = Claim(
            agent=self.name,
            statement="The reported symptoms form a clinically relevant multi-symptom cluster.",
            evidence_ids=[item.evidence_id for item in evidence],
            confidence=0.7,
        )
        result = self.provider.generate(
            agent=self.name,
            prompt="Normalize symptoms, identify clusters, involved systems, and red flags.",
            response_model=SymptomFindings,
            demo_factory=lambda: SymptomFindings(
                normalized_symptoms=normalized,
                symptom_clusters=[", ".join(normalized)] if normalized else [],
                possible_systems=systems,
                red_flags=red_flags,
                missing_symptom_information=missing,
                output=AgentOutput(
                    agent=self.name,
                    status=AgentStatus.COMPLETED,
                    summary="Symptoms were normalized and checked for configured red flags.",
                    claims=[claim] if evidence else [],
                    missing_information=missing,
                    warnings=red_flags,
                ),
            ),
        )
        return {
            "symptom_findings": result.output,
            "evidence": evidence,
            "missing_information": [
                MissingInformationRequest(
                    request_id=f"REQ-SYMPTOM-{index:03d}",
                    information_needed=item,
                    requested_by=[self.name],
                    reason="Symptom characterization affects urgency and differential reasoning.",
                )
                for index, item in enumerate(missing, start=1)
            ],
            "agent_messages": [
                AgentMessage(
                    sender=self.name,
                    recipient="supervisor",
                    message_type=(MessageType.WARNING if red_flags else MessageType.RESPONSE),
                    content=result.output.output.summary,
                    evidence_ids=[item.evidence_id for item in evidence],
                )
            ],
            "token_usage": [result.usage],
        }


def _red_flags(symptoms: list[str]) -> list[str]:
    symptom_set = set(symptoms)
    flags: list[str] = []
    if "chest pain" in symptom_set and "shortness of breath" in symptom_set:
        flags.append("Chest pain with shortness of breath requires urgent clinical assessment.")
    if "confusion" in symptom_set or "unilateral weakness" in symptom_set:
        flags.append("Acute neurological symptoms require urgent clinical assessment.")
    return flags


def _possible_systems(symptoms: list[str]) -> list[str]:
    text = " ".join(symptoms)
    systems: list[str] = []
    mapping = {
        "cardiorespiratory": ("chest", "breath", "palpitation"),
        "neurological": ("headache", "weakness", "confusion", "dizziness"),
        "systemic": ("fatigue", "fever"),
    }
    for system, terms in mapping.items():
        if any(term in text for term in terms):
            systems.append(system)
    return systems
