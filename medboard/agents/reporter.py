"""Generate the structured report only after explicit human approval."""

from __future__ import annotations

from medboard.agents.base import BaseAgent, StateUpdate
from medboard.graph.state import MedicalCaseState
from medboard.models import FinalReport, HumanStatus


class ReporterAgent(BaseAgent):
    name = "reporter"

    def analyze(self, state: MedicalCaseState) -> StateUpdate:
        human_review = state["human_review"]
        if human_review.status is not HumanStatus.APPROVED:
            raise ValueError("report generation requires explicit human approval")
        triage = state.get("triage_result")
        if triage is None:
            raise ValueError("report generation requires triage assessment")
        result = self.provider.generate(
            agent=self.name,
            prompt=(
                "Generate a structured decision-support case review for clinician review. "
                "Use non-definitive language and do not prescribe."
            ),
            context={
                "case_input": state["case_input"],
                "evidence": state["evidence"],
                "differential_diagnoses": state["differential_diagnoses"],
                "specialist_opinions": state["specialist_opinions"],
                "retrieved_evidence": state["retrieved_evidence"],
                "contradictions": state["contradictions"],
                "missing_information": state["missing_information"],
                "triage": triage,
                "human_review": human_review,
            },
            response_model=FinalReport,
            demo_factory=lambda: FinalReport(
                case_summary=state["case_input"].chief_complaint,
                key_findings=[
                    f"{item.name}: {item.value}" for item in state["evidence"]
                ],
                differential_considerations=state["differential_diagnoses"],
                specialist_opinions=state["specialist_opinions"],
                retrieved_evidence=state["retrieved_evidence"],
                disagreements=state["contradictions"],
                missing_information=state["missing_information"],
                triage=triage,
                review_priorities=[triage.recommended_escalation],
            ),
        )
        return {"final_report": result.output, "token_usage": [result.usage]}
