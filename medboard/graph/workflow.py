"""First runnable MedBoard investigation graph."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from medboard.agents.history import HistoryAgent
from medboard.agents.laboratory import LaboratoryAgent
from medboard.agents.medication import MedicationAgent
from medboard.agents.supervisor import SupervisorAgent
from medboard.agents.symptoms import SymptomAgent
from medboard.graph.state import MedicalCaseState, validate_state
from medboard.models import AgentStatus, TraceEvent, TraceEventType
from medboard.providers import StructuredModelProvider

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
