"""First runnable MedBoard investigation graph."""

from __future__ import annotations

from typing import Literal, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from medboard.agents.history import HistoryAgent
from medboard.agents.human_actions import (
    RetryFailedAgent,
    apply_human_information,
    apply_requested_specialist,
)
from medboard.agents.human_review import human_review_node, mark_waiting_for_human
from medboard.agents.evidence import EvidenceRetrievalAgent
from medboard.agents.critic import CriticAgent
from medboard.agents.differential import DifferentialAgent
from medboard.agents.laboratory import LaboratoryAgent
from medboard.agents.medication import MedicationAgent
from medboard.agents.supervisor import SupervisorAgent
from medboard.agents.symptoms import SymptomAgent
from medboard.agents.risk import RiskAgent
from medboard.agents.reporter import ReporterAgent
from medboard.agents.revision import revise_differential, supervisor_revision
from medboard.agents.specialists import (
    CardiologyAgent,
    InfectiousDiseaseAgent,
    NeurologyAgent,
)
from medboard.graph.routing import SpecialistRouter
from medboard.graph.state import MedicalCaseState, validate_state
from medboard.models import AgentStatus, TraceEvent, TraceEventType
from medboard.models import HumanStatus
from medboard.providers import StructuredModelProvider
from medboard.rag.store import KnowledgeStore
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
    knowledge_store: KnowledgeStore | None = None,
    max_revisions: int | None = None,
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
    if knowledge_store is not None:
        builder.add_node(
            "evidence_retrieval",
            EvidenceRetrievalAgent(provider, knowledge_store),
        )
    if max_revisions is not None:
        builder.add_node("critic", CriticAgent(provider, max_revisions))
        builder.add_node("supervisor_revision", supervisor_revision)
        builder.add_node("differential_revision", revise_differential)
        builder.add_node("risk", RiskAgent(provider))
        builder.add_node("review_complete", _complete_review)
    builder.add_node("collaboration_complete", _complete_collaboration)

    builder.add_edge(START, "supervisor")
    for agent_name in BASE_AGENT_NAMES:
        builder.add_edge("supervisor", agent_name)
    builder.add_edge(BASE_AGENT_NAMES, "differential")
    builder.add_edge("differential", "specialist_router")
    builder.add_conditional_edges(
        "specialist_router",
        lambda state: _route_selected_specialists(
            state,
            empty_route=(
                "evidence_retrieval"
                if knowledge_store is not None
                else "collaboration_complete"
            ),
        ),
    )
    for specialist in ("cardiology", "neurology", "infectious_disease"):
        builder.add_edge(
            specialist,
            "evidence_retrieval" if knowledge_store is not None else "collaboration_complete",
        )
    if knowledge_store is not None:
        builder.add_edge("evidence_retrieval", "collaboration_complete")
    if max_revisions is None:
        builder.add_edge("collaboration_complete", END)
    else:
        builder.add_edge("collaboration_complete", "critic")
        builder.add_conditional_edges("critic", _route_critic_decision)
        builder.add_edge("supervisor_revision", "differential_revision")
        builder.add_edge("differential_revision", "critic")
        builder.add_edge("risk", "review_complete")
        builder.add_edge("review_complete", END)
    return builder.compile()


def build_reviewable_workflow(
    provider: StructuredModelProvider,
    knowledge_store: KnowledgeStore,
    *,
    max_revisions: int,
    checkpointer: object,
) -> CompiledStateGraph:
    """Compile the complete review path with a durable human interrupt."""
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
    builder.add_node("evidence_retrieval", EvidenceRetrievalAgent(provider, knowledge_store))
    builder.add_node("collaboration_complete", _complete_collaboration)
    builder.add_node("critic", CriticAgent(provider, max_revisions))
    builder.add_node("supervisor_revision", supervisor_revision)
    builder.add_node("differential_revision", revise_differential)
    builder.add_node("risk", RiskAgent(provider))
    builder.add_node("human_review", human_review_node)
    builder.add_node("mark_waiting", mark_waiting_for_human)
    builder.add_node("apply_human_information", apply_human_information)
    builder.add_node("apply_requested_specialist", apply_requested_specialist)
    builder.add_node(
        "retry_failed_agent",
        RetryFailedAgent(
            {
                "history": HistoryAgent(provider),
                "symptoms": SymptomAgent(provider),
                "laboratory": LaboratoryAgent(provider),
                "medication": MedicationAgent(provider),
                "differential": DifferentialAgent(provider),
                "cardiology": CardiologyAgent(provider),
                "neurology": NeurologyAgent(provider),
                "infectious_disease": InfectiousDiseaseAgent(provider),
                "evidence_retrieval": EvidenceRetrievalAgent(provider, knowledge_store),
                "critic": CriticAgent(provider, max_revisions),
                "risk": RiskAgent(provider),
            }
        ),
    )
    builder.add_node("reporter", ReporterAgent(provider))
    builder.add_node("audit_complete", _complete_human_review)

    builder.add_edge(START, "supervisor")
    for agent_name in BASE_AGENT_NAMES:
        builder.add_edge("supervisor", agent_name)
    builder.add_edge(BASE_AGENT_NAMES, "differential")
    builder.add_edge("differential", "specialist_router")
    builder.add_conditional_edges(
        "specialist_router",
        lambda state: _route_selected_specialists(
            state, empty_route="evidence_retrieval"
        ),
    )
    for specialist in ("cardiology", "neurology", "infectious_disease"):
        builder.add_edge(specialist, "evidence_retrieval")
    builder.add_edge("evidence_retrieval", "collaboration_complete")
    builder.add_edge("collaboration_complete", "critic")
    builder.add_conditional_edges("critic", _route_critic_decision)
    builder.add_edge("supervisor_revision", "differential_revision")
    builder.add_edge("differential_revision", "critic")
    builder.add_edge("risk", "mark_waiting")
    builder.add_edge("mark_waiting", "human_review")
    builder.add_conditional_edges("human_review", _route_human_decision)
    builder.add_edge("apply_human_information", "differential")
    builder.add_conditional_edges(
        "apply_requested_specialist", _route_requested_specialist
    )
    builder.add_edge("retry_failed_agent", "differential")
    builder.add_edge("reporter", "audit_complete")
    builder.add_edge("audit_complete", END)
    return builder.compile(checkpointer=checkpointer)


SpecialistRoute = Literal[
    "cardiology",
    "neurology",
    "infectious_disease",
    "evidence_retrieval",
    "collaboration_complete",
]


def _route_selected_specialists(
    state: MedicalCaseState,
    *,
    empty_route: SpecialistRoute = "collaboration_complete",
) -> list[SpecialistRoute]:
    selected = state["selected_specialists"]
    if not selected:
        return [empty_route]
    allowed = {"cardiology", "neurology", "infectious_disease"}
    return [
        cast(SpecialistRoute, specialist)
        for specialist in selected
        if specialist in allowed
    ]


CriticRoute = Literal["supervisor_revision", "risk"]


def _route_critic_decision(state: MedicalCaseState) -> CriticRoute:
    review = state.get("critic_review")
    if review is not None and review.decision.value == "revise":
        return "supervisor_revision"
    return "risk"


HumanRoute = Literal[
    "reporter",
    "audit_complete",
    "apply_human_information",
    "supervisor_revision",
    "apply_requested_specialist",
    "retry_failed_agent",
]


def _route_human_decision(state: MedicalCaseState) -> HumanRoute:
    routes: dict[HumanStatus, HumanRoute] = {
        HumanStatus.APPROVED: "reporter",
        HumanStatus.REJECTED: "audit_complete",
        HumanStatus.MORE_INFORMATION: "apply_human_information",
        HumanStatus.REQUEST_REVISION: "supervisor_revision",
        HumanStatus.REQUEST_SPECIALIST: "apply_requested_specialist",
        HumanStatus.RETRY_FAILED_AGENT: "retry_failed_agent",
    }
    return routes[state["human_review"].status]


def _route_requested_specialist(state: MedicalCaseState) -> SpecialistRoute:
    command = state.get("human_command")
    return cast(SpecialistRoute, command.requested_specialist if command else "")


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


def _complete_review(state: MedicalCaseState) -> dict[str, object]:
    validate_state(state)
    triage = state.get("triage_result")
    critic_review = state.get("critic_review")
    return {
        "execution_trace": [
            TraceEvent(
                event_type=TraceEventType.WORKFLOW_COMPLETED,
                agent="workflow",
                status=AgentStatus.COMPLETED,
                details={
                    "revision_count": state.get("revision_count", 0),
                    "critic_decision": critic_review.decision.value if critic_review else None,
                    "triage_level": triage.triage_level.value if triage else None,
                    "unresolved_contradictions": len(
                        [item for item in state["contradictions"] if not item.resolved]
                    ),
                },
            )
        ]
    }


def _complete_human_review(state: MedicalCaseState) -> dict[str, object]:
    validate_state(state)
    return {
        "execution_trace": [
            TraceEvent(
                event_type=TraceEventType.WORKFLOW_COMPLETED,
                agent="workflow",
                status=AgentStatus.COMPLETED,
                details={
                    "human_status": state["human_review"].status.value,
                    "report_generated": state.get("final_report") is not None,
                },
            )
        ]
    }
