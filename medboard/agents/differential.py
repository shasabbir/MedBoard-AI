"""Evidence integration into multiple competing diagnostic considerations."""

from __future__ import annotations

from medboard.agents.base import BaseAgent, StateUpdate
from medboard.graph.state import MedicalCaseState
from medboard.models import (
    AgentMessage,
    AgentOutput,
    AgentStatus,
    Claim,
    DifferentialAnalysis,
    DifferentialDiagnosis,
    Evidence,
    EvidenceQuestion,
    MessageType,
    MissingInformationRequest,
)


class DifferentialAgent(BaseAgent):
    """Integrate base-agent evidence without collapsing to one diagnosis."""

    name = "differential"

    def analyze(self, state: MedicalCaseState) -> StateUpdate:
        evidence = state["evidence"]
        symptoms = _symptoms(evidence)
        diagnoses: list[DifferentialDiagnosis] = []

        low_labs = _lab_ids_with_status(evidence, "low")
        if _has_named_evidence(evidence, {"hemoglobin"}, low_labs):
            support = [
                *_evidence_ids(evidence, {"hemoglobin", "mcv", "ferritin"}),
                *_evidence_ids(evidence, {"fatigue", "shortness of breath"}),
                *_history_ids_containing(evidence, "bleeding"),
            ]
            diagnoses.append(
                DifferentialDiagnosis(
                    hypothesis_id="HYP-ANEMIA-PATTERN",
                    hypothesis="Iron-deficiency anemia pattern",
                    supporting_evidence_ids=_unique(support),
                    confidence=0.82,
                    missing_evidence=["reticulocyte count", "clinician assessment of blood loss"],
                )
            )

        if "chest pain" in symptoms:
            diagnoses.append(
                DifferentialDiagnosis(
                    hypothesis_id="HYP-CORONARY",
                    hypothesis="Acute coronary syndrome consideration",
                    supporting_evidence_ids=_evidence_ids(
                        evidence, {"chest pain", "shortness of breath", "palpitations"}
                    ),
                    confidence=0.64,
                    missing_evidence=["ECG", "troponin", "blood pressure"],
                )
            )

        if symptoms & {"shortness of breath", "cough", "chest pain"}:
            diagnoses.append(
                DifferentialDiagnosis(
                    hypothesis_id="HYP-PULMONARY",
                    hypothesis="Pulmonary or other cardiorespiratory process",
                    supporting_evidence_ids=_evidence_ids(
                        evidence, {"shortness of breath", "cough", "chest pain"}
                    ),
                    confidence=0.43,
                    missing_evidence=["oxygen saturation", "respiratory examination"],
                )
            )

        if symptoms & {"confusion", "unilateral weakness", "seizure", "headache"}:
            diagnoses.append(
                DifferentialDiagnosis(
                    hypothesis_id="HYP-NEUROLOGICAL",
                    hypothesis="Acute neurological process",
                    supporting_evidence_ids=_evidence_ids(
                        evidence,
                        {"confusion", "unilateral weakness", "seizure", "headache"},
                    ),
                    confidence=0.76,
                    missing_evidence=["neurological examination", "exact time last known well"],
                )
            )

        infection_support = _evidence_ids(evidence, {"fever", "cough"})
        infection_support.extend(_lab_ids_with_status(evidence, "high", names={"wbc"}))
        if infection_support:
            diagnoses.append(
                DifferentialDiagnosis(
                    hypothesis_id="HYP-INFECTION",
                    hypothesis="Infectious process",
                    supporting_evidence_ids=_unique(infection_support),
                    confidence=0.68,
                    missing_evidence=["temperature trend", "exposure history"],
                )
            )

        diagnoses = _deduplicate_hypotheses(diagnoses)
        fallback_hypotheses = [
            (
                "HYP-OTHER-SYSTEMIC",
                "Other systemic or metabolic explanation",
                ["physical examination", "additional targeted testing"],
            ),
            (
                "HYP-CONTEXTUAL",
                "Medication, exposure, or contextual explanation",
                ["complete medication list", "exposure and lifestyle history"],
            ),
        ]
        for hypothesis_id, hypothesis, missing in fallback_hypotheses:
            if len(diagnoses) >= 2:
                break
            diagnoses.append(
                DifferentialDiagnosis(
                    hypothesis_id=hypothesis_id,
                    hypothesis=hypothesis,
                    supporting_evidence_ids=[
                        item.evidence_id
                        for item in evidence
                        if item.evidence_type.value in {"symptom", "lab"}
                    ],
                    confidence=0.24,
                    missing_evidence=missing,
                )
            )

        claims = [
            Claim(
                agent=self.name,
                statement=f"{diagnosis.hypothesis} should remain under consideration.",
                evidence_ids=diagnosis.supporting_evidence_ids,
                contradicting_evidence_ids=diagnosis.contradicting_evidence_ids,
                confidence=diagnosis.confidence,
            )
            for diagnosis in diagnoses
        ]
        missing_items = _unique(
            [item for diagnosis in diagnoses for item in diagnosis.missing_evidence]
        )
        result = self.provider.generate(
            agent=self.name,
            prompt=(
                "Integrate structured history, symptom, laboratory, and medication evidence "
                "into multiple competing diagnostic considerations."
            ),
            response_model=DifferentialAnalysis,
            demo_factory=lambda: DifferentialAnalysis(
                diagnoses=diagnoses,
                output=AgentOutput(
                    agent=self.name,
                    status=AgentStatus.COMPLETED,
                    summary=(
                        f"Generated {len(diagnoses)} competing considerations from shared evidence."
                    ),
                    claims=claims,
                    missing_information=missing_items,
                ),
            ),
        )
        referenced_evidence = _unique(
            [item for diagnosis in diagnoses for item in diagnosis.supporting_evidence_ids]
        )
        evidence_questions = [
            EvidenceQuestion(
                question_id=f"Q-DIFFERENTIAL-{index:03d}",
                asked_by=self.name,
                question=(
                    "What source-backed evidence is relevant when evaluating "
                    f"{diagnosis.hypothesis}?"
                ),
                hypothesis_ids=[diagnosis.hypothesis_id],
                evidence_ids=diagnosis.supporting_evidence_ids,
            )
            for index, diagnosis in enumerate(diagnoses, start=1)
        ]
        return {
            "differential_analysis": result.output,
            "differential_diagnoses": result.output.diagnoses,
            "evidence_questions": evidence_questions,
            "missing_information": [
                MissingInformationRequest(
                    request_id=f"REQ-DIFFERENTIAL-{index:03d}",
                    information_needed=item,
                    requested_by=[self.name],
                    reason="This information could distinguish competing considerations.",
                    diagnostic_utility=0.7,
                )
                for index, item in enumerate(missing_items, start=1)
            ],
            "agent_messages": [
                AgentMessage(
                    sender=self.name,
                    recipient="supervisor",
                    message_type=MessageType.CLAIM,
                    content=result.output.output.summary,
                    evidence_ids=referenced_evidence,
                )
            ],
            "token_usage": [result.usage],
        }


def _symptoms(evidence: list[Evidence]) -> set[str]:
    return {
        item.name.casefold()
        for item in evidence
        if item.evidence_type.value == "symptom" and item.value is True
    }


def _evidence_ids(evidence: list[Evidence], names: set[str]) -> list[str]:
    return [item.evidence_id for item in evidence if item.name.casefold() in names]


def _history_ids_containing(evidence: list[Evidence], term: str) -> list[str]:
    return [
        item.evidence_id
        for item in evidence
        if item.evidence_type.value == "history" and term in str(item.value).casefold()
    ]


def _lab_ids_with_status(
    evidence: list[Evidence], status: str, names: set[str] | None = None
) -> list[str]:
    return [
        item.evidence_id
        for item in evidence
        if item.evidence_type.value == "lab"
        and item.metadata.get("status") == status
        and (names is None or item.name.casefold() in names)
    ]


def _has_named_evidence(
    evidence: list[Evidence], names: set[str], evidence_ids: list[str]
) -> bool:
    allowed = set(evidence_ids)
    return any(item.evidence_id in allowed and item.name.casefold() in names for item in evidence)


def _deduplicate_hypotheses(
    diagnoses: list[DifferentialDiagnosis],
) -> list[DifferentialDiagnosis]:
    return list({item.hypothesis_id: item for item in diagnoses}.values())


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
