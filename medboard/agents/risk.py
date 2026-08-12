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


class RiskAgent(BaseAgent):
    name = "risk"

    def __init__(
        self,
        provider: StructuredModelProvider,
        risk_tool: RiskRuleTool | None = None,
    ) -> None:
        super().__init__(provider)
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
        return {
            "triage_result": result.output,
            "agent_messages": [
                AgentMessage(
                    sender=self.name,
                    recipient="human_review",
                    message_type=(
                        MessageType.WARNING
                        if result.output.red_flags
                        else MessageType.RESPONSE
                    ),
                    content=(
                        f"Triage level: {result.output.triage_level.value}. "
                        f"{result.output.recommended_escalation}"
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
                        "red_flag_count": len(result.output.red_flags),
                    },
                )
            ],
        }
