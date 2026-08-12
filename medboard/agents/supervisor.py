"""Supervisor planning and initial case normalization."""

from __future__ import annotations

from medboard.agents.base import BaseAgent, StateUpdate
from medboard.graph.state import MedicalCaseState
from medboard.models import (
    AgentMessage,
    MessageType,
    NormalizedCase,
    SupervisorPlan,
    TriageLevel,
)

INITIAL_AGENTS = ["history", "symptoms", "laboratory", "medication"]


class SupervisorAgent(BaseAgent):
    """Create the initial investigation plan without making a diagnosis."""

    name = "supervisor"

    def analyze(self, state: MedicalCaseState) -> StateUpdate:
        case = state["case_input"]
        case_text = " ".join(
            [
                case.chief_complaint,
                case.narrative,
                *case.symptoms,
                *case.history,
            ]
        ).casefold()
        categories = _case_categories(case_text)
        priority = (
            TriageLevel.PRIORITY
            if any(term in case_text for term in ("chest pain", "weakness", "confusion"))
            else TriageLevel.ROUTINE
        )
        prompt = (
            "Create an investigation plan for the supplied educational case. "
            "Select the base analyses and describe why they are needed."
        )
        result = self.provider.generate(
            agent=self.name,
            prompt=prompt,
            response_model=SupervisorPlan,
            demo_factory=lambda: SupervisorPlan(
                case_categories=categories,
                initial_agents=INITIAL_AGENTS,
                selected_specialists=[],
                priority=priority,
                reasoning=(
                    "Independent history, symptom, laboratory, and medication reviews "
                    "are required before integrating competing explanations."
                ),
            ),
        )
        normalized = NormalizedCase(
            age=case.age,
            biological_sex=case.biological_sex,
            chief_complaint=case.chief_complaint,
            normalized_symptoms=[symptom.casefold() for symptom in case.symptoms],
            history=case.history,
            medications=case.medications,
            allergies=case.allergies,
            laboratory_values=case.laboratory_values,
        )
        messages = [
            AgentMessage(
                sender=self.name,
                recipient=agent,
                message_type=MessageType.REQUEST,
                content=f"Perform the planned {agent} analysis and return structured evidence.",
            )
            for agent in INITIAL_AGENTS
        ]
        return {
            "supervisor_plan": result.output,
            "normalized_case": normalized,
            "agent_messages": messages,
            "token_usage": [result.usage],
        }


def _case_categories(case_text: str) -> list[str]:
    categories: list[str] = []
    keyword_groups = {
        "cardiovascular": ("chest", "palpitation", "shortness of breath"),
        "neurological": ("headache", "weakness", "confusion", "seizure"),
        "infectious": ("fever", "cough", "infection", "travel"),
        "hematological": ("anemia", "fatigue", "bleeding", "pallor"),
    }
    for category, keywords in keyword_groups.items():
        if any(keyword in case_text for keyword in keywords):
            categories.append(category)
    return categories or ["general"]
