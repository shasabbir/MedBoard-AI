"""Apply nonterminal human actions and route only affected analysis."""

from __future__ import annotations

from typing import cast

from langgraph.types import Overwrite

from medboard.agents.base import BaseAgent
from medboard.graph.state import MedicalCaseState
from medboard.models import (
    AgentMessage,
    AgentStatus,
    Evidence,
    EvidenceType,
    HumanAction,
    HumanReview,
    HumanStatus,
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
        "historical_hypothesis_ids": list(
            dict.fromkeys(
                [
                    *state.get("historical_hypothesis_ids", []),
                    *(
                        item.hypothesis_id
                        for item in state["differential_diagnoses"]
                    ),
                ]
            )
        ),
        "specialist_opinions": Overwrite(value=[]),
        "contradictions": Overwrite(value=[]),
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
                    "rerun_from": (
                        "laboratory" if lab_values else "differential"
                    ),
                },
            )
        ],
    }


def route_added_information(state: MedicalCaseState) -> str:
    """Send new lab data through lab analysis; other facts start at integration."""
    command = state.get("human_command")
    if command is None or command.action is not HumanAction.ADD_INFORMATION:
        raise ValueError("added-information routing requires an add_information command")
    lab_values = command.added_information.get("laboratory_values")
    return (
        "laboratory_reanalysis"
        if isinstance(lab_values, list) and lab_values
        else "differential"
    )


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
        agent_state = state
        if agent_name == "reporter":
            current_review = state["human_review"]
            agent_state = cast(MedicalCaseState, dict(state))
            agent_state["human_review"] = HumanReview(
                status=HumanStatus.APPROVED,
                feedback=current_review.feedback,
                reviewer=current_review.reviewer,
            )
        update = agent(agent_state)
        if update.get("errors"):
            return update
        resolved_errors = [
            error.model_copy(
                update={
                    "resolved": True,
                    "resolution": "Manual retry completed successfully.",
                }
            )
            if error.agent == agent_name and not error.resolved
            else error
            for error in state["errors"]
        ]
        update["errors"] = Overwrite(value=resolved_errors)
        if agent_name == "reporter":
            update["human_review"] = agent_state["human_review"]
        return update


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
