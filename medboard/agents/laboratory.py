"""Laboratory agent combining deterministic ranges with structured explanation."""

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
    LaboratoryFindings,
    MessageType,
    MissingInformationRequest,
    TraceEvent,
    TraceEventType,
)
from medboard.providers import StructuredModelProvider
from medboard.tools.lab_reference import LabAssessment, LabReferenceTool, LabStatus


class LaboratoryAgent(BaseAgent):
    name = "laboratory"

    def __init__(
        self,
        provider: StructuredModelProvider,
        lab_tool: LabReferenceTool | None = None,
        *,
        max_retries: int = 2,
    ) -> None:
        super().__init__(provider, max_retries=max_retries)
        self.lab_tool = lab_tool or LabReferenceTool()

    def analyze(self, state: MedicalCaseState) -> StateUpdate:
        case = state["case_input"]
        assessments = [
            self.lab_tool.assess(observation, case.biological_sex)
            for observation in case.laboratory_values
        ]
        evidence = [
            Evidence(
                evidence_id=f"EV-LAB-{index:03d}",
                evidence_type=EvidenceType.LAB,
                name=observation.name,
                value={"value": observation.value, "unit": observation.unit},
                source="user_case",
                metadata={
                    "status": assessment.status.value,
                    "reference_range": assessment.reference_range,
                },
            )
            for index, (observation, assessment) in enumerate(
                zip(case.laboratory_values, assessments, strict=True), start=1
            )
        ]
        abnormal = [
            f"{assessment.name}: {assessment.status.value} ({assessment.value} "
            f"{assessment.unit or 'unit not supplied'})"
            for assessment in assessments
            if assessment.status in {LabStatus.LOW, LabStatus.HIGH}
        ]
        warnings = [assessment.warning for assessment in assessments if assessment.warning]
        abnormal_ids = [
            item.evidence_id
            for item, assessment in zip(evidence, assessments, strict=True)
            if assessment.status in {LabStatus.LOW, LabStatus.HIGH}
        ]
        patterns = _patterns(assessments)
        claims = [
            Claim(
                agent=self.name,
                statement="One or more supplied laboratory values are outside configured ranges.",
                evidence_ids=abnormal_ids,
                confidence=0.95,
            )
        ] if abnormal_ids else []
        missing = [] if case.laboratory_values else ["laboratory values with explicit units"]
        result = self.provider.generate(
            agent=self.name,
            prompt="Interpret deterministic lab flags and explain patterns without diagnosing.",
            context={
                "case_input": case,
                "deterministic_lab_assessments": assessments,
            },
            response_model=LaboratoryFindings,
            demo_factory=lambda: LaboratoryFindings(
                abnormal_values=abnormal,
                important_patterns=patterns,
                potential_implications=[
                    "The abnormal pattern should be integrated with symptoms and history."
                ] if abnormal else [],
                missing_tests=missing,
                data_quality_warnings=warnings,
                output=AgentOutput(
                    agent=self.name,
                    status=AgentStatus.COMPLETED,
                    summary="Laboratory values were checked using explicit units and ranges.",
                    claims=claims,
                    missing_information=missing,
                    warnings=warnings,
                ),
            ),
        )
        update: StateUpdate = {
            "laboratory_findings": result.output,
            "evidence": evidence,
            "agent_messages": [
                AgentMessage(
                    sender=self.name,
                    recipient="supervisor",
                    message_type=(MessageType.WARNING if warnings else MessageType.RESPONSE),
                    content=result.output.output.summary,
                    evidence_ids=[item.evidence_id for item in evidence],
                )
            ],
            "token_usage": [result.usage],
            "execution_trace": [
                TraceEvent(
                    event_type=TraceEventType.TOOL_CALLED,
                    agent=self.name,
                    status=AgentStatus.COMPLETED,
                    details={
                        "tool": "LabReferenceTool",
                        "call_count": len(case.laboratory_values),
                    },
                )
            ],
        }
        if missing:
            update["missing_information"] = [
                MissingInformationRequest(
                    request_id="REQ-LAB-001",
                    information_needed=missing[0],
                    requested_by=[self.name],
                    reason="Laboratory interpretation requires values and explicit units.",
                )
            ]
        return update


def _patterns(assessments: list[LabAssessment]) -> list[str]:
    statuses = {
        assessment.name.casefold(): assessment.status
        for assessment in assessments
    }
    if all(
        statuses.get(name) is LabStatus.LOW
        for name in ("hemoglobin", "mcv", "ferritin")
    ):
        return ["Low hemoglobin, MCV, and ferritin occur together in the supplied data."]
    return []
