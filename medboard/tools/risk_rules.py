"""Deterministic red-flag rules that do not make a final diagnosis."""

from __future__ import annotations

from dataclasses import dataclass

from medboard.graph.state import MedicalCaseState
from medboard.models import TriageLevel, TriageResult


@dataclass(frozen=True, slots=True)
class RiskRule:
    name: str
    required_symptoms: frozenset[str]
    level: TriageLevel
    warning: str
    escalation: str


class RiskRuleTool:
    """Apply transparent, conservative rules to supplied structured evidence."""

    rules = (
        RiskRule(
            name="focal_neurological_emergency",
            required_symptoms=frozenset({"confusion", "unilateral weakness"}),
            level=TriageLevel.EMERGENCY,
            warning="Sudden confusion with unilateral weakness is an emergency red flag.",
            escalation="Immediate emergency clinical assessment is warranted.",
        ),
        RiskRule(
            name="acute_cardiorespiratory_emergency",
            required_symptoms=frozenset({"chest pain", "shortness of breath"}),
            level=TriageLevel.EMERGENCY,
            warning="Chest pain with shortness of breath is an emergency red flag.",
            escalation="Immediate emergency clinical assessment is warranted.",
        ),
        RiskRule(
            name="febrile_respiratory_priority",
            required_symptoms=frozenset({"fever", "cough"}),
            level=TriageLevel.PRIORITY,
            warning="Fever with cough requires timely clinical assessment.",
            escalation="Prioritized clinician review is warranted, with severity assessment.",
        ),
    )

    def assess(self, state: MedicalCaseState) -> TriageResult:
        symptoms = {
            item.name.casefold()
            for item in state["evidence"]
            if item.evidence_type.value == "symptom" and item.value is True
        }
        matched = [rule for rule in self.rules if rule.required_symptoms <= symptoms]
        severe_anemia = any(_is_markedly_low_hemoglobin(item) for item in state["evidence"])
        if severe_anemia:
            matched.append(
                RiskRule(
                    name="marked_low_hemoglobin",
                    required_symptoms=frozenset(),
                    level=TriageLevel.URGENT,
                    warning="Markedly low supplied hemoglobin is an urgent review flag.",
                    escalation="Urgent clinician assessment is warranted.",
                )
            )

        if not matched:
            return TriageResult(
                triage_level=TriageLevel.ROUTINE,
                reasoning="No configured deterministic red-flag combination was triggered.",
                recommended_escalation="Use the normal clinical review pathway.",
            )
        priority = {
            TriageLevel.ROUTINE: 0,
            TriageLevel.PRIORITY: 1,
            TriageLevel.URGENT: 2,
            TriageLevel.EMERGENCY: 3,
        }
        level = max((rule.level for rule in matched), key=priority.__getitem__)
        return TriageResult(
            triage_level=level,
            red_flags=[rule.warning for rule in matched],
            reasoning="Deterministic rules matched supplied symptoms or laboratory evidence.",
            recommended_escalation=max(
                matched, key=lambda rule: priority[rule.level]
            ).escalation,
        )


def _is_markedly_low_hemoglobin(evidence: object) -> bool:
    from medboard.models import Evidence

    if not isinstance(evidence, Evidence):
        return False
    value = evidence.value
    if not isinstance(value, dict):
        return False
    numeric_value = value.get("value")
    return (
        evidence.evidence_type.value == "lab"
        and evidence.name.casefold() == "hemoglobin"
        and evidence.metadata.get("status") == "low"
        and isinstance(numeric_value, int | float)
        and numeric_value < 9
    )
