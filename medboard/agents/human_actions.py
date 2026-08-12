"""Apply nonterminal human actions and route only affected analysis."""

from __future__ import annotations

from medboard.agents.base import BaseAgent
from medboard.graph.state import MedicalCaseState
from medboard.models import (
    AgentMessage,
    AgentStatus,
    Evidence,
    EvidenceType,
    HumanAction,
    MessageType,
    TraceEvent,
    TraceEventType,
)
from medboard.tools.lab_reference import LabReferenceTool


def apply_human_information(state: MedicalCaseState) -> dict[str, object]:
    """Convert added data into traceable evidence before downstream re-analysis."""
    command = state.get("human_command")
    if command is None or command.action is not HumanAction.ADD_INFORMATION:
        raise ValueError("human information node requires an add_information command")
    evidence: list[Evidence] = []
    existing_human_evidence = sum(
        item.source == "human_review" for item in state["evidence"]
    )
    lab_tool = LabReferenceTool()
    lab_values = command.added_information.get("laboratory_values", [])
    if isinstance(lab_values, list):
        from medboard.models import LabObservation

        for index, raw in enumerate(lab_values, start=1):
            observation = LabObservation.model_validate(raw)
            assessment = lab_tool.assess(
                observation, state["case_input"].biological_sex
            )
            evidence.append(
                Evidence(
                    evidence_id=(
                        f"EV-HUMAN-{existing_human_evidence + index:03d}-LAB"
                    ),
                    evidence_type=EvidenceType.LAB,
                    name=observation.name,
                    value={"value": observation.value, "unit": observation.unit},
                    source="human_review",
                    metadata={
                        "status": assessment.status.value,
                        "reference_range": assessment.reference_range,
                    },
                )
            )
    for key, value in command.added_information.items():
        if key == "laboratory_values":
            continue
        evidence.append(
            Evidence(
                evidence_id=(
                    f"EV-HUMAN-{existing_human_evidence + len(evidence) + 1:03d}"
                ),
                evidence_type=EvidenceType.HUMAN,
                name=key.replace("_", " "),
                value=value,
                source="human_review",
            )
        )
    return {
        "evidence": evidence,
        "revision_count": 0,
        "agent_messages": [
            AgentMessage(
                sender="human_reviewer",
                recipient="differential",
                message_type=MessageType.REVISION,
                content="New human-supplied evidence requires downstream re-analysis.",
                evidence_ids=[item.evidence_id for item in evidence],
            )
        ],
        "execution_trace": [
            TraceEvent(
                event_type=TraceEventType.RESUMED,
                agent="supervisor",
                status=AgentStatus.COMPLETED,
                details={
                    "action": "add_information",
                    "evidence_count": len(evidence),
                    "rerun_from": "differential",
                },
            )
        ],
    }


class RetryFailedAgent:
    """Retry only the named agent, then recompute its downstream dependants."""

    name = "retry_failed_agent"

    def __init__(self, agents: dict[str, BaseAgent]) -> None:
        self.agents = agents

    def __call__(self, state: MedicalCaseState) -> dict[str, object]:
        command = state.get("human_command")
        if command is None or command.action is not HumanAction.RETRY_FAILED_AGENT:
            raise ValueError("retry node requires a retry_failed_agent command")
        agent_name = command.failed_agent or ""
        agent = self.agents.get(agent_name)
        if agent is None:
            raise ValueError(f"agent is not retryable: {agent_name}")
        return agent(state)


def apply_requested_specialist(state: MedicalCaseState) -> dict[str, object]:
    command = state.get("human_command")
    if command is None or command.action is not HumanAction.REQUEST_SPECIALIST:
        raise ValueError("specialist node requires a request_specialist command")
    requested = command.requested_specialist or ""
    if requested not in {"cardiology", "neurology", "infectious_disease"}:
        raise ValueError(f"unsupported specialist: {requested}")
    return {
        "selected_specialists": list(
            dict.fromkeys([*state["selected_specialists"], requested])
        ),
        "agent_messages": [
            AgentMessage(
                sender="supervisor",
                recipient=requested,
                message_type=MessageType.REQUEST,
                content="Human reviewer requested an additional specialist opinion.",
            )
        ],
    }
