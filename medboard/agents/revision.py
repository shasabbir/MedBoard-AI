"""Supervisor re-plan and bounded differential revision."""

from __future__ import annotations

from medboard.graph.state import MedicalCaseState
from medboard.models import (
    AgentMessage,
    AgentStatus,
    MessageType,
    TraceEvent,
    TraceEventType,
)


def supervisor_revision(state: MedicalCaseState) -> dict[str, object]:
    """Record a targeted re-plan without rerunning unaffected intake agents."""
    next_revision = state.get("revision_count", 0) + 1
    return {
        "revision_count": next_revision,
        "agent_messages": [
            AgentMessage(
                sender="supervisor",
                recipient="differential",
                message_type=MessageType.REVISION,
                content=(
                    "Reconsider challenged hypotheses using the specialist disagreement and "
                    "retain unresolved uncertainty for human review."
                ),
            )
        ],
        "execution_trace": [
            TraceEvent(
                event_type=TraceEventType.ROUTING_DECISION,
                agent="supervisor",
                status=AgentStatus.COMPLETED,
                details={
                    "action": "targeted_differential_revision",
                    "revision_count": next_revision,
                    "rerun_agents": ["differential_revision", "critic"],
                },
            )
        ],
    }


def revise_differential(state: MedicalCaseState) -> dict[str, object]:
    """Lower confidence in explicitly challenged hypotheses and preserve alternatives."""
    challenged = {
        hypothesis_id
        for opinion in state["specialist_opinions"]
        for hypothesis_id in opinion.challenged_hypotheses
    }
    revised = [
        diagnosis.model_copy(
            update={"confidence": max(0.1, diagnosis.confidence - 0.15)}
        )
        if diagnosis.hypothesis_id in challenged
        else diagnosis
        for diagnosis in state["differential_diagnoses"]
    ]
    analysis = state.get("differential_analysis")
    revised_analysis = analysis.model_copy(update={"diagnoses": revised}) if analysis else None
    return {
        "differential_diagnoses": revised,
        "differential_analysis": revised_analysis,
        "agent_messages": [
            AgentMessage(
                sender="differential",
                recipient="critic",
                message_type=MessageType.REVISION,
                content=(
                    "Challenged considerations were revised; confidence was reduced while "
                    "the unresolved disagreement was retained for human review."
                ),
            )
        ],
    }
