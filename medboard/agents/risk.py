"""Risk and urgency assessment, intentionally separate from diagnosis."""

from __future__ import annotations

from medboard.agents.base import BaseAgent, StateUpdate
from medboard.graph.state import MedicalCaseState
from medboard.models import (
    AgentMessage,
    AgentStatus,
    MessageType,
    TraceEvent,
    TraceEventType,
    TriageResult,
)
from medboard.providers import StructuredModelProvider
from medboard.tools.risk_rules import RiskRuleTool


TRIAGE_PRIORITY = {
    "routine": 0,
    "priority": 1,
    "urgent": 2,
    "emergency": 3,
}


class RiskAgent(BaseAgent):
    name = "risk"

    def __init__(
        self,
        provider: StructuredModelProvider,
        risk_tool: RiskRuleTool | None = None,
        *,
        max_retries: int = 2,
    ) -> None:
        super().__init__(provider, max_retries=max_retries)
        self.risk_tool = risk_tool or RiskRuleTool()

    def analyze(self, state: MedicalCaseState) -> StateUpdate:
        deterministic_result = self.risk_tool.assess(state)
        result = self.provider.generate(
            agent=self.name,
            prompt=(
                "Explain urgency from deterministic red-flag rules. Do not diagnose and do "
                "not generate treatment or a final report."
            ),
            response_model=TriageResult,
            demo_factory=lambda: deterministic_result,
        )
        triage_result = _preserve_deterministic_urgency(
            deterministic_result, result.output
        )
        return {
            "triage_result": triage_result,
            "agent_messages": [
                AgentMessage(
                    sender=self.name,
                    recipient="human_review",
                    message_type=(
                        MessageType.WARNING
                        if triage_result.red_flags
                        else MessageType.RESPONSE
                    ),
                    content=(
                        f"Triage level: {triage_result.triage_level.value}. "
                        f"{triage_result.recommended_escalation}"
                    ),
                )
            ],
            "token_usage": [result.usage],
            "execution_trace": [
                TraceEvent(
                    event_type=TraceEventType.TOOL_CALLED,
                    agent=self.name,
                    status=AgentStatus.COMPLETED,
                    details={
                        "tool": "RiskRuleTool",
                        "call_count": 1,
                        "red_flag_count": len(triage_result.red_flags),
                    },
                )
            ],
        }


def _preserve_deterministic_urgency(
    deterministic: TriageResult, model_result: TriageResult
) -> TriageResult:
    """Allow model explanation or escalation, but never weaken rule-based safety."""
    deterministic_priority = TRIAGE_PRIORITY[deterministic.triage_level.value]
    model_priority = TRIAGE_PRIORITY[model_result.triage_level.value]
    if model_priority < deterministic_priority:
        selected_level = deterministic.triage_level
        escalation = deterministic.recommended_escalation
    else:
        selected_level = model_result.triage_level
        escalation = model_result.recommended_escalation
    return model_result.model_copy(
        update={
            "triage_level": selected_level,
            "red_flags": list(
                dict.fromkeys([*deterministic.red_flags, *model_result.red_flags])
            ),
            "recommended_escalation": escalation,
        }
    )
