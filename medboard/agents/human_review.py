"""LangGraph human interrupt and command application."""

from __future__ import annotations

from langgraph.types import interrupt

from medboard.graph.state import MedicalCaseState
from medboard.models import (
    AgentMessage,
    HumanAction,
    HumanReview,
    HumanReviewCommand,
    HumanStatus,
    MessageType,
)


def mark_waiting_for_human(state: MedicalCaseState) -> dict[str, object]:
    """Persist waiting status before entering the interrupting node."""
    previous = state["human_review"]
    return {
        "human_review": HumanReview(
            status=HumanStatus.WAITING_FOR_HUMAN,
            feedback=previous.feedback,
            reviewer=previous.reviewer,
        )
    }


def human_review_node(state: MedicalCaseState) -> dict[str, object]:
    """Pause execution and apply a validated human review command on resume."""
    triage = state.get("triage_result")
    payload = interrupt(
        {
            "run_id": state["run_id"],
            "triage": triage.model_dump(mode="json") if triage else None,
            "differential": [
                item.model_dump(mode="json") for item in state["differential_diagnoses"]
            ],
            "unresolved_contradictions": [
                item.model_dump(mode="json")
                for item in state["contradictions"]
                if not item.resolved
            ],
            "allowed_actions": [action.value for action in HumanAction],
        }
    )
    command = HumanReviewCommand.model_validate(payload)
    status_mapping = {
        HumanAction.APPROVE: HumanStatus.APPROVED,
        HumanAction.REJECT: HumanStatus.REJECTED,
        HumanAction.ADD_INFORMATION: HumanStatus.MORE_INFORMATION,
        HumanAction.REQUEST_REVISION: HumanStatus.REQUEST_REVISION,
        HumanAction.REQUEST_SPECIALIST: HumanStatus.REQUEST_SPECIALIST,
        HumanAction.RETRY_FAILED_AGENT: HumanStatus.RETRY_FAILED_AGENT,
    }
    update: dict[str, object] = {
        "human_review": HumanReview(
            status=status_mapping[command.action],
            feedback=command.feedback,
            reviewer=command.reviewer,
        ),
        "human_command": command,
        "agent_messages": [
            AgentMessage(
                sender="human_reviewer",
                recipient="supervisor",
                message_type=(
                    MessageType.RESPONSE
                    if command.action is HumanAction.APPROVE
                    else MessageType.REVISION
                ),
                content=command.feedback or f"Human action: {command.action.value}",
            )
        ],
    }
    if command.action is HumanAction.ADD_INFORMATION:
        update["human_added_information"] = command.added_information
    return update
