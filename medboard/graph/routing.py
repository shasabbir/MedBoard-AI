"""Supervisor-owned deterministic specialist routing."""

from __future__ import annotations

from medboard.graph.state import MedicalCaseState
from medboard.models import (
    AgentMessage,
    AgentStatus,
    MessageType,
    SpecialistRoutingDecision,
    TraceEvent,
    TraceEventType,
)

AVAILABLE_SPECIALISTS = ("cardiology", "neurology", "infectious_disease")


class SpecialistRouter:
    """Select one, several, or no specialists from structured case evidence."""

    name = "supervisor"

    def __call__(self, state: MedicalCaseState) -> dict[str, object]:
        symptoms = {
            item.name.casefold()
            for item in state["evidence"]
            if item.evidence_type.value == "symptom" and item.value is True
        }
        reasons: dict[str, str] = {}
        evidence_by_specialist: dict[str, list[str]] = {}
        triggering_ids: list[str] = []

        cardiac_terms = symptoms & {"chest pain", "shortness of breath", "palpitations"}
        if cardiac_terms:
            reasons["cardiology"] = (
                "Cardiorespiratory symptoms warrant focused cardiac review: "
                + ", ".join(sorted(cardiac_terms))
            )
            cardiac_ids = _symptom_ids(state, cardiac_terms)
            evidence_by_specialist["cardiology"] = cardiac_ids
            triggering_ids.extend(cardiac_ids)

        neurological_terms = symptoms & {
            "headache",
            "confusion",
            "unilateral weakness",
            "seizure",
            "numbness",
        }
        if neurological_terms:
            reasons["neurology"] = (
                "Neurological symptoms warrant focused review: "
                + ", ".join(sorted(neurological_terms))
            )
            neurological_ids = _symptom_ids(state, neurological_terms)
            evidence_by_specialist["neurology"] = neurological_ids
            triggering_ids.extend(neurological_ids)

        infectious_terms = symptoms & {"fever", "cough"}
        high_wbc_ids = [
            item.evidence_id
            for item in state["evidence"]
            if item.evidence_type.value == "lab"
            and item.name.casefold() == "wbc"
            and item.metadata.get("status") == "high"
        ]
        if "fever" in infectious_terms or ("cough" in infectious_terms and high_wbc_ids):
            reasons["infectious_disease"] = (
                "Fever or cough with inflammatory evidence warrants infectious review."
            )
            infectious_ids = [*_symptom_ids(state, infectious_terms), *high_wbc_ids]
            evidence_by_specialist["infectious_disease"] = infectious_ids
            triggering_ids.extend(_symptom_ids(state, infectious_terms))
            triggering_ids.extend(high_wbc_ids)

        selected = [name for name in AVAILABLE_SPECIALISTS if name in reasons]
        decision = SpecialistRoutingDecision(
            selected_specialists=selected,
            reasons=reasons,
            evidence_ids=list(dict.fromkeys(triggering_ids)),
        )
        plan = state.get("supervisor_plan")
        updated_plan = (
            plan.model_copy(update={"selected_specialists": selected}) if plan else None
        )
        messages = [
            AgentMessage(
                sender=self.name,
                recipient=specialist,
                message_type=MessageType.REQUEST,
                content=reasons[specialist],
                evidence_ids=evidence_by_specialist[specialist],
            )
            for specialist in selected
        ]
        return {
            "selected_specialists": selected,
            "routing_decisions": [decision],
            "supervisor_plan": updated_plan,
            "agent_messages": messages,
            "execution_trace": [
                TraceEvent(
                    event_type=TraceEventType.ROUTING_DECISION,
                    agent=self.name,
                    status=AgentStatus.COMPLETED,
                    details={
                        "selected_specialists": selected,
                        "reasons": reasons,
                    },
                )
            ],
        }


def _symptom_ids(state: MedicalCaseState, symptom_names: set[str]) -> list[str]:
    return [
        item.evidence_id
        for item in state["evidence"]
        if item.evidence_type.value == "symptom" and item.name.casefold() in symptom_names
    ]
