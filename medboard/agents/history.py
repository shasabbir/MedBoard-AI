"""Deterministic demo implementation of patient-history analysis."""

from __future__ import annotations

from medboard.agents.base import BaseAgent, StateUpdate
from medboard.graph.state import MedicalCaseState
from medboard.models import (
    AgentMessage,
    AgentOutput,
    AgentStatus,
    Claim,
    Evidence,
    EvidenceType,
    HistoryFindings,
    MessageType,
    MissingInformationRequest,
)


class HistoryAgent(BaseAgent):
    name = "history"

    def analyze(self, state: MedicalCaseState) -> StateUpdate:
        case = state["case_input"]
        evidence: list[Evidence] = []
        if case.age is not None:
            evidence.append(
                Evidence(
                    evidence_id="EV-HISTORY-AGE",
                    evidence_type=EvidenceType.HISTORY,
                    name="age",
                    value=case.age,
                    source="user_case",
                )
            )
        for index, item in enumerate(case.history, start=1):
            evidence.append(
                Evidence(
                    evidence_id=f"EV-HISTORY-{index:03d}",
                    evidence_type=EvidenceType.HISTORY,
                    name="reported history",
                    value=item,
                    source="user_case",
                )
            )

        risk_evidence = [
            item.evidence_id
            for item in evidence
            if any(
                term in str(item.value).casefold()
                for term in ("bleeding", "smoking", "diabetes", "hypertension")
            )
        ]
        claims = (
            [
                Claim(
                    agent=self.name,
                    statement="The supplied history contains factors relevant to the case.",
                    evidence_ids=risk_evidence,
                    confidence=0.78,
                )
            ]
            if risk_evidence
            else []
        )
        missing = ["symptom onset and progression", "family history"]
        prompt = "Extract history, risk factors, negative findings, and missing history."
        result = self.provider.generate(
            agent=self.name,
            prompt=prompt,
            response_model=HistoryFindings,
            demo_factory=lambda: HistoryFindings(
                key_history=case.history,
                risk_factors=[
                    str(item.value)
                    for item in evidence
                    if item.evidence_id in risk_evidence
                ],
                negative_findings=[],
                missing_history=missing,
                possible_relevance=[
                    "Reported risk factors may help explain the presenting symptoms."
                ]
                if risk_evidence
                else [],
                output=AgentOutput(
                    agent=self.name,
                    status=AgentStatus.COMPLETED,
                    summary="History review identified supplied context and remaining gaps.",
                    claims=claims,
                    missing_information=missing,
                ),
            ),
        )
        return {
            "history_findings": result.output,
            "evidence": evidence,
            "missing_information": [
                MissingInformationRequest(
                    request_id=f"REQ-HISTORY-{index:03d}",
                    information_needed=item,
                    requested_by=[self.name],
                    reason="This context may change interpretation of the current evidence.",
                )
                for index, item in enumerate(missing, start=1)
            ],
            "agent_messages": [
                AgentMessage(
                    sender=self.name,
                    recipient="supervisor",
                    message_type=MessageType.RESPONSE,
                    content=result.output.output.summary,
                    evidence_ids=[item.evidence_id for item in evidence],
                )
            ],
            "token_usage": [result.usage],
        }
