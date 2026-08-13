"""Independent specialist reviews selected by the supervisor router."""

from __future__ import annotations

from abc import abstractmethod

from medboard.agents.base import BaseAgent, StateUpdate
from medboard.graph.state import MedicalCaseState
from medboard.models import (
    AgentMessage,
    DifferentialDiagnosis,
    EvidenceQuestion,
    MessageType,
    MissingInformationRequest,
    SpecialistOpinion,
)


class BaseSpecialistAgent(BaseAgent):
    """Shared transport mechanics; clinical selection remains specialist-specific."""

    @abstractmethod
    def build_opinion(self, state: MedicalCaseState) -> SpecialistOpinion:
        """Produce this specialty's independent assessment."""

    def analyze(self, state: MedicalCaseState) -> StateUpdate:
        result = self.provider.generate(
            agent=self.name,
            prompt=(
                f"Independently review the shared evidence as the {self.name} specialist. "
                "Support, challenge, or add hypotheses without making a definitive diagnosis."
            ),
            context={
                "case_input": state["case_input"],
                "evidence": state["evidence"],
                "differential_diagnoses": state["differential_diagnoses"],
            },
            response_model=SpecialistOpinion,
            demo_factory=lambda: self.build_opinion(state),
        )
        opinion = result.output
        messages = [
            AgentMessage(
                sender=self.name,
                recipient="supervisor",
                message_type=MessageType.RESPONSE,
                content=opinion.assessment,
                evidence_ids=opinion.evidence_ids,
            )
        ]
        diagnoses = {item.hypothesis_id: item for item in state["differential_diagnoses"]}
        messages.extend(
            AgentMessage(
                sender=self.name,
                recipient="differential",
                message_type=MessageType.CHALLENGE,
                content=(
                    f"Challenge to '{diagnoses[hypothesis_id].hypothesis}': "
                    f"the current evidence supports an alternative interpretation."
                ),
                evidence_ids=opinion.evidence_ids,
            )
            for hypothesis_id in opinion.challenged_hypotheses
            if hypothesis_id in diagnoses
        )
        return {
            "specialist_opinions": [opinion],
            "evidence_questions": [self.build_evidence_question(opinion)],
            "missing_information": [
                MissingInformationRequest(
                    information_needed=item,
                    requested_by=[self.name],
                    reason="The selected specialist requires this information for review.",
                    diagnostic_utility=0.8,
                )
                for item in opinion.required_information
            ],
            "agent_messages": messages,
            "token_usage": [result.usage],
        }

    def build_evidence_question(self, opinion: SpecialistOpinion) -> EvidenceQuestion:
        hypothesis_ids = list(
            dict.fromkeys(
                [*opinion.supported_hypotheses, *opinion.challenged_hypotheses]
            )
        )
        return EvidenceQuestion(
            question_id=f"Q-{self.name.upper().replace('_', '-')}-001",
            asked_by=self.name,
            question=self.specialist_question(),
            hypothesis_ids=hypothesis_ids,
            evidence_ids=opinion.evidence_ids,
        )

    @abstractmethod
    def specialist_question(self) -> str:
        """Return the focused question sent to the evidence-retrieval agent."""


class CardiologyAgent(BaseSpecialistAgent):
    name = "cardiology"

    def specialist_question(self) -> str:
        return (
            "What guideline evidence supports structured evaluation of chest symptoms, "
            "dyspnea, and possible cardiac causes?"
        )

    def build_opinion(self, state: MedicalCaseState) -> SpecialistOpinion:
        diagnoses = state["differential_diagnoses"]
        symptoms = _symptoms(state)
        supported = _hypothesis_ids(diagnoses, {"coronary"})
        challenged: list[str] = []
        low_hemoglobin = _evidence_ids(state, names={"hemoglobin"}, status="low")
        if low_hemoglobin and "chest pain" not in symptoms:
            challenged = _hypothesis_ids(diagnoses, {"pulmonary"})
        evidence_ids = _evidence_ids(
            state,
            names={"chest pain", "shortness of breath", "palpitations", "hemoglobin"},
        )
        return SpecialistOpinion(
            specialist=self.name,
            assessment=(
                "Cardiac causes remain relevant, but the low hemoglobin pattern offers a "
                "competing explanation for exertional symptoms."
                if low_hemoglobin
                else "Cardiac causes require focused review based on the supplied symptoms."
            ),
            supported_hypotheses=supported,
            challenged_hypotheses=challenged,
            alternative_hypotheses=["Cardiac rhythm or structural process"],
            required_information=["ECG", "troponin", "blood pressure"],
            critical_concerns=(
                ["Chest pain with shortness of breath"]
                if {"chest pain", "shortness of breath"} <= symptoms
                else []
            ),
            evidence_ids=evidence_ids,
            confidence=0.7,
        )


class NeurologyAgent(BaseSpecialistAgent):
    name = "neurology"

    def specialist_question(self) -> str:
        return (
            "What authoritative evidence describes urgent evaluation of sudden confusion, "
            "headache, or unilateral weakness for possible stroke?"
        )

    def build_opinion(self, state: MedicalCaseState) -> SpecialistOpinion:
        symptoms = _symptoms(state)
        evidence_ids = _evidence_ids(
            state,
            names={"headache", "confusion", "unilateral weakness", "seizure", "numbness"},
        )
        return SpecialistOpinion(
            specialist=self.name,
            assessment=(
                "The focal or altered-mental-status features require urgent neurological review."
            ),
            supported_hypotheses=_hypothesis_ids(
                state["differential_diagnoses"], {"neurological"}
            ),
            alternative_hypotheses=["Vascular, seizure-related, or metabolic neurological event"],
            required_information=["neurological examination", "exact time last known well"],
            critical_concerns=(
                ["Focal neurological deficit or confusion"]
                if symptoms & {"confusion", "unilateral weakness"}
                else []
            ),
            evidence_ids=evidence_ids,
            confidence=0.84,
        )


class InfectiousDiseaseAgent(BaseSpecialistAgent):
    name = "infectious_disease"

    def specialist_question(self) -> str:
        return (
            "What authoritative evidence describes fever, cough, and inflammatory findings "
            "in respiratory infection or pneumonia assessment?"
        )

    def build_opinion(self, state: MedicalCaseState) -> SpecialistOpinion:
        evidence_ids = _evidence_ids(state, names={"fever", "cough", "wbc"})
        return SpecialistOpinion(
            specialist=self.name,
            assessment=(
                "The fever, respiratory symptoms, or inflammatory findings support an "
                "infectious consideration while source and severity remain unresolved."
            ),
            supported_hypotheses=_hypothesis_ids(
                state["differential_diagnoses"], {"infectious"}
            ),
            alternative_hypotheses=["Localized respiratory or systemic infectious process"],
            required_information=["temperature trend", "exposure and travel history"],
            critical_concerns=(
                ["Possible systemic infection requires severity assessment"]
                if "fever" in _symptoms(state)
                else []
            ),
            evidence_ids=evidence_ids,
            confidence=0.78,
        )


def _symptoms(state: MedicalCaseState) -> set[str]:
    return {
        item.name.casefold()
        for item in state["evidence"]
        if item.evidence_type.value == "symptom" and item.value is True
    }


def _hypothesis_ids(
    diagnoses: list[DifferentialDiagnosis], terms: set[str]
) -> list[str]:
    return [
        item.hypothesis_id
        for item in diagnoses
        if any(term in item.hypothesis.casefold() for term in terms)
    ]


def _evidence_ids(
    state: MedicalCaseState,
    *,
    names: set[str],
    status: str | None = None,
) -> list[str]:
    return [
        item.evidence_id
        for item in state["evidence"]
        if item.name.casefold() in names
        and (status is None or item.metadata.get("status") == status)
    ]
