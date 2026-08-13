"""Failure, retry-limit, and retrieval-outage acceptance tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from medboard.agents.history import HistoryAgent
from medboard.graph.state import create_initial_state, validate_state
from medboard.graph.workflow import build_collaboration_workflow
from medboard.agents.risk import RiskAgent
from medboard.agents.reporter import ReporterAgent
from medboard.models import (
    ContractModel,
    AgentError,
    HumanReview,
    HumanStatus,
    MedicalCaseInput,
    RetrievedEvidence,
    TokenUsage,
    TriageLevel,
    TriageResult,
    Severity,
)
from medboard.providers import DemoModelProvider, ProviderResult
from medboard.rag.store import KnowledgeStore

OutputT = TypeVar("OutputT", bound=ContractModel)


class AlwaysFailProvider:
    provider_name = "failure-test"
    model_name = "failure-test"

    def generate(
        self,
        *,
        agent: str,
        prompt: str,
        context: object,
        response_model: type[OutputT],
        demo_factory: Callable[[], OutputT],
    ) -> ProviderResult[OutputT]:
        del agent, prompt, context, response_model, demo_factory
        raise ConnectionError("simulated provider outage")


class FailingKnowledgeStore(KnowledgeStore):
    def search(
        self, question: str, *, question_id: str, top_k: int = 5
    ) -> list[RetrievedEvidence]:
        del question, question_id, top_k
        raise ConnectionError("simulated RAG outage")


class DowngradingRiskProvider:
    provider_name = "unsafe-test"
    model_name = "unsafe-test"

    def generate(
        self,
        *,
        agent: str,
        prompt: str,
        context: object,
        response_model: type[OutputT],
        demo_factory: Callable[[], OutputT],
    ) -> ProviderResult[OutputT]:
        del prompt, context, demo_factory
        output = TriageResult(
            triage_level=TriageLevel.ROUTINE,
            reasoning="The model attempted to weaken deterministic urgency.",
            recommended_escalation="Use the normal review pathway.",
        )
        return ProviderResult(
            output=response_model.model_validate(output.model_dump()),
            usage=TokenUsage(agent=agent, provider=self.provider_name, model=self.model_name),
        )


def test_agent_failure_retries_to_limit_and_becomes_visible() -> None:
    case = MedicalCaseInput(chief_complaint="Synthetic test case")
    agent = HistoryAgent(AlwaysFailProvider())

    update = agent(create_initial_state(case, run_id="RUN-FAILURE"))

    assert len(update["errors"]) == 1
    error = update["errors"][0]
    assert error.retryable is True
    assert error.attempt == 3
    assert error.details["retry_limit"] == 2
    retry_events = [
        event
        for event in update["execution_trace"]
        if event.details.get("retry") is True
    ]
    assert len(retry_events) == 2
    assert update["execution_trace"][-1].event_type.value == "agent_failed"


def test_agent_retry_limit_is_configurable() -> None:
    case = MedicalCaseInput(chief_complaint="Synthetic test case")
    agent = HistoryAgent(AlwaysFailProvider(), max_retries=0)

    update = agent(create_initial_state(case, run_id="RUN-NO-RETRY"))

    assert update["errors"][0].attempt == 1
    assert update["errors"][0].details["retry_limit"] == 0
    assert not any(
        event.details.get("retry") is True for event in update["execution_trace"]
    )


def test_rag_outage_is_recorded_without_fabricated_retrievals(tmp_path: Path) -> None:
    store = FailingKnowledgeStore(tmp_path / "chroma")
    case = MedicalCaseInput(
        case_id="CASE-RAG-OUTAGE",
        chief_complaint="Sudden confusion and weakness",
        symptoms=["confusion", "unilateral weakness"],
    )
    graph = build_collaboration_workflow(DemoModelProvider(), store)

    snapshot = validate_state(
        graph.invoke(create_initial_state(case, run_id="RUN-RAG-OUTAGE"))
    )

    failures = [error for error in snapshot.errors if error.agent == "evidence_retrieval"]
    assert len(failures) == 1
    assert failures[0].attempt == 3
    assert snapshot.retrieved_evidence == []
    assert any(
        event.event_type.value == "agent_failed"
        and event.agent == "evidence_retrieval"
        for event in snapshot.execution_trace
    )


def test_model_cannot_downgrade_deterministic_emergency_triage() -> None:
    case = MedicalCaseInput(
        chief_complaint="Acute cardiorespiratory symptoms",
        symptoms=["chest pain", "shortness of breath"],
    )
    state = create_initial_state(case, run_id="RUN-TRIAGE-GUARD")
    # Use the symptom agent so the risk tool receives normal structured evidence.
    from medboard.agents.symptoms import SymptomAgent

    state["evidence"] = SymptomAgent(DemoModelProvider()).analyze(state)["evidence"]

    update = RiskAgent(DowngradingRiskProvider(), max_retries=0).analyze(state)

    triage = update["triage_result"]
    assert isinstance(triage, TriageResult)
    assert triage.triage_level is TriageLevel.EMERGENCY
    assert triage.red_flags
    assert (
        triage.recommended_escalation
        == "Immediate emergency clinical assessment is warranted."
    )


def test_final_report_explicitly_flags_failed_analysis() -> None:
    case = MedicalCaseInput(chief_complaint="Synthetic report failure test")
    state = create_initial_state(case, run_id="RUN-REPORT-LIMITATION")
    state["human_review"] = HumanReview(status=HumanStatus.APPROVED)
    state["triage_result"] = TriageResult(
        triage_level=TriageLevel.ROUTINE,
        reasoning="No configured red flag was triggered.",
        recommended_escalation="Use the normal clinical review pathway.",
    )
    state["errors"] = [
        AgentError(
            agent="evidence_retrieval",
            error_type="ConnectionError",
            message="simulated RAG outage",
            severity=Severity.HIGH,
        ),
        AgentError(
            agent="medication",
            error_type="ConnectionError",
            message="simulated provider outage",
            severity=Severity.HIGH,
        ),
    ]

    update = ReporterAgent(DemoModelProvider(), max_retries=0).analyze(state)

    report = update["final_report"]
    assert report is not None
    assert any("Evidence retrieval was unavailable" in item for item in report.limitations)
    assert any("Medication analysis was unavailable" in item for item in report.limitations)
