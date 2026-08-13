"""Apply nonterminal human actions and route only affected analysis."""

from __future__ import annotations

from collections.abc import Mapping
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
    LabObservation,
    MedicalCaseInput,
    MessageType,
    MissingInformationRequest,
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
    updated_case, observations = _updated_case_input(
        state["case_input"], command.added_information
    )
    for index, observation in enumerate(observations, start=1):
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
    structured_keys = {
        "laboratory_values",
        "symptoms",
        "history",
        "medications",
    }
    for key, value in command.added_information.items():
        if key in structured_keys:
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
        "case_input": updated_case,
        "evidence": evidence,
        "missing_information": Overwrite(
            value=_resolve_supplied_information(
                state["missing_information"], command.added_information, observations
            )
        ),
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
                    "rerun_agents": _affected_agent_names(command.added_information),
                },
            )
        ],
    }


class ReanalyzeHumanInformation:
    """Run only intake agents affected by the latest human-supplied fields."""

    name = "human_information_reanalysis"
    _list_update_fields = {
        "evidence",
        "missing_information",
        "agent_messages",
        "errors",
        "execution_trace",
        "token_usage",
    }

    def __init__(self, agents: dict[str, BaseAgent]) -> None:
        self.agents = agents

    def __call__(self, state: MedicalCaseState) -> dict[str, object]:
        command = state.get("human_command")
        if command is None or command.action is not HumanAction.ADD_INFORMATION:
            raise ValueError("reanalysis requires an add_information command")
        combined: dict[str, object] = {}
        existing_evidence_ids = {item.evidence_id for item in state["evidence"]}
        for agent_name in _affected_agent_names(command.added_information):
            update = self.agents[agent_name](state)
            for field, value in update.items():
                if field == "evidence" and isinstance(value, list):
                    value = [
                        item
                        for item in value
                        if isinstance(item, Evidence)
                        and item.evidence_id not in existing_evidence_ids
                    ]
                if field in self._list_update_fields and isinstance(value, list):
                    existing = combined.setdefault(field, [])
                    if isinstance(existing, list):
                        existing.extend(value)
                else:
                    combined[field] = value
        return combined


def _updated_case_input(
    case: MedicalCaseInput, additions: Mapping[str, object]
) -> tuple[MedicalCaseInput, list[LabObservation]]:
    payload = case.model_dump(mode="python")
    for field in ("symptoms", "history", "medications", "allergies"):
        if field not in additions:
            continue
        supplied = additions[field]
        if not isinstance(supplied, list) or not all(
            isinstance(item, str) and item.strip() for item in supplied
        ):
            raise ValueError(f"{field} must be a list of non-empty strings")
        payload[field] = list(dict.fromkeys([*payload[field], *supplied]))
    raw_labs = additions.get("laboratory_values", [])
    if not isinstance(raw_labs, list):
        raise ValueError("laboratory_values must be a list")
    observations = [LabObservation.model_validate(item) for item in raw_labs]
    payload["laboratory_values"] = [*payload["laboratory_values"], *observations]
    return MedicalCaseInput.model_validate(payload), observations


def _affected_agent_names(additions: Mapping[str, object]) -> list[str]:
    field_agents = {
        "symptoms": "symptoms",
        "history": "history",
        "allergies": "history",
        "medications": "medication",
        "laboratory_values": "laboratory",
    }
    return list(
        dict.fromkeys(
            field_agents[key]
            for key in additions
            if key in field_agents and additions[key]
        )
    )


def _resolve_supplied_information(
    requests: list[MissingInformationRequest],
    additions: Mapping[str, object],
    observations: list[LabObservation],
) -> list[MissingInformationRequest]:
    satisfied = {_information_key(key) for key in additions}
    satisfied.update(_information_key(item.name) for item in observations)
    aliases = {
        "laboratory values": "laboratory values with explicit units",
        "medications": "complete medication and supplement list",
    }
    satisfied.update(aliases[key] for key in list(satisfied) if key in aliases)
    resolved: list[MissingInformationRequest] = []
    for item in requests:
        if _information_key(item.information_needed) in satisfied:
            item = item.model_copy(
                update={
                    "resolved": True,
                    "resolution": "Supplied during human review.",
                }
            )
        resolved.append(item)
    return resolved


def _information_key(value: str) -> str:
    return " ".join(value.replace("_", " ").casefold().split())


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
