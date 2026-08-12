"""First runnable MedBoard investigation graph."""

from __future__ import annotations

from typing import Literal, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from medboard.agents.history import HistoryAgent
from medboard.agents.differential import DifferentialAgent
from medboard.agents.laboratory import LaboratoryAgent
from medboard.agents.medication import MedicationAgent
from medboard.agents.supervisor import SupervisorAgent
from medboard.agents.symptoms import SymptomAgent
from medboard.agents.specialists import (
    CardiologyAgent,
    InfectiousDiseaseAgent,
    NeurologyAgent,
)
from medboard.graph.routing import SpecialistRouter
from medboard.graph.state import MedicalCaseState, validate_state
from medboard.models import AgentStatus, TraceEvent, TraceEventType
from medboard.providers import StructuredModelProvider
from medboard.tools.contradictions import detect_specialist_contradictions

BASE_AGENT_NAMES = ["history", "symptoms", "laboratory", "medication"]


def build_initial_workflow(provider: StructuredModelProvider) -> CompiledStateGraph:
    """Compile supervisor planning and four concurrent base analyses."""
    builder = StateGraph(MedicalCaseState)
    builder.add_node("supervisor", SupervisorAgent(provider))
    builder.add_node("history", HistoryAgent(provider))
    builder.add_node("symptoms", SymptomAgent(provider))
    builder.add_node("laboratory", LaboratoryAgent(provider))
    builder.add_node("medication", MedicationAgent(provider))
    builder.add_node("complete", _complete_workflow)

    builder.add_edge(START, "supervisor")
    for agent_name in BASE_AGENT_NAMES:
        builder.add_edge("supervisor", agent_name)
    builder.add_edge(BASE_AGENT_NAMES, "complete")
    builder.add_edge("complete", END)
    return builder.compile()


def build_collaboration_workflow(
    provider: StructuredModelProvider,
) -> CompiledStateGraph:
    """Compile intake, differential reasoning, and conditional specialist review."""
    builder = StateGraph(MedicalCaseState)
    builder.add_node("supervisor", SupervisorAgent(provider))
    builder.add_node("history", HistoryAgent(provider))
    builder.add_node("symptoms", SymptomAgent(provider))
    builder.add_node("laboratory", LaboratoryAgent(provider))
    builder.add_node("medication", MedicationAgent(provider))
    builder.add_node("differential", DifferentialAgent(provider))
    builder.add_node("specialist_router", SpecialistRouter())
    builder.add_node("cardiology", CardiologyAgent(provider))
    builder.add_node("neurology", NeurologyAgent(provider))
    builder.add_node("infectious_disease", InfectiousDiseaseAgent(provider))
    builder.add_node("collaboration_complete", _complete_collaboration)

    builder.add_edge(START, "supervisor")
    for agent_name in BASE_AGENT_NAMES:
        builder.add_edge("supervisor", agent_name)
    builder.add_edge(BASE_AGENT_NAMES, "differential")
    builder.add_edge("differential", "specialist_router")
    builder.add_conditional_edges("specialist_router", _route_selected_specialists)
    for specialist in ("cardiology", "neurology", "infectious_disease"):
        builder.add_edge(specialist, "collaboration_complete")
    builder.add_edge("collaboration_complete", END)
    return builder.compile()


SpecialistRoute = Literal[
    "cardiology", "neurology", "infectious_disease", "collaboration_complete"
]


def _route_selected_specialists(state: MedicalCaseState) -> list[SpecialistRoute]:
    selected = state["selected_specialists"]
    if not selected:
        return ["collaboration_complete"]
    allowed = {"cardiology", "neurology", "infectious_disease"}
    return [
        cast(SpecialistRoute, specialist)
        for specialist in selected
        if specialist in allowed
    ]


def _complete_workflow(state: MedicalCaseState) -> dict[str, list[TraceEvent]]:
    validate_state(state)
    return {
        "execution_trace": [
            TraceEvent(
                event_type=TraceEventType.WORKFLOW_COMPLETED,
                agent="workflow",
                status=AgentStatus.COMPLETED,
                details={
                    "completed_agents": BASE_AGENT_NAMES,
                    "evidence_count": len(state["evidence"]),
                    "error_count": len(state["errors"]),
                },
            )
        ]
    }


def _complete_collaboration(state: MedicalCaseState) -> dict[str, object]:
    contradictions = detect_specialist_contradictions(
        state["differential_diagnoses"], state["specialist_opinions"]
    )
    return {
        "contradictions": contradictions,
        "execution_trace": [
            TraceEvent(
                event_type=TraceEventType.WORKFLOW_COMPLETED,
                agent="workflow",
                status=AgentStatus.COMPLETED,
                details={
                    "completed_agents": [
                        *BASE_AGENT_NAMES,
                        "differential",
                        *state["selected_specialists"],
                    ],
                    "selected_specialists": state["selected_specialists"],
                    "differential_count": len(state["differential_diagnoses"]),
                    "contradiction_count": len(contradictions),
                    "error_count": len(state["errors"]),
                },
            )
        ],
    }
