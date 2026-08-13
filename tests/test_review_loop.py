"""Tests for critic revision bounds and deterministic risk assessment."""

from pathlib import Path

from medboard.agents.critic import CriticAgent
from medboard.graph.state import create_initial_state, validate_state
from medboard.graph.workflow import build_collaboration_workflow
from medboard.models import (
    CriticDecision,
    CriticReview,
    MedicalCaseInput,
    Severity,
    TokenUsage,
    TriageLevel,
)
from medboard.providers import DemoModelProvider, ProviderResult
from medboard.rag.store import KnowledgeStore


def run_case(
    tmp_path: Path,
    case: MedicalCaseInput,
    *,
    max_revisions: int = 2,
):
    store = KnowledgeStore(tmp_path / f"chroma-{case.case_id}")
    store.ingest_directory(Path("data/knowledge"))
    workflow = build_collaboration_workflow(
        DemoModelProvider(), store, max_revisions=max_revisions
    )
    return validate_state(
        workflow.invoke(create_initial_state(case, run_id=f"RUN-{case.case_id}"))
    )


def test_critic_accepts_supported_case_without_revision(tmp_path: Path) -> None:
    case = MedicalCaseInput(
        case_id="CASE-NEURO-REVIEW",
        chief_complaint="Sudden confusion and weakness",
        symptoms=["confusion", "unilateral weakness"],
    )

    snapshot = run_case(tmp_path, case)

    assert snapshot.critic_review is not None
    assert snapshot.critic_review.decision.value == "accept"
    assert snapshot.revision_count == 0
    assert len(snapshot.critic_reviews) == 1
    assert snapshot.triage_result is not None
    assert snapshot.triage_result.triage_level is TriageLevel.EMERGENCY


def test_critic_revision_is_bounded_and_preserves_disagreement(tmp_path: Path) -> None:
    case = MedicalCaseInput.model_validate_json(
        Path("data/demo_cases/anemia.json").read_text(encoding="utf-8")
    )

    snapshot = run_case(tmp_path, case, max_revisions=2)

    assert snapshot.revision_count == 2
    assert len(snapshot.critic_reviews) == 3
    assert [review.decision.value for review in snapshot.critic_reviews] == [
        "revise",
        "revise",
        "accept",
    ]
    assert snapshot.critic_review is not None
    assert "Revision limit reached" in " ".join(snapshot.critic_review.problems)
    assert snapshot.contradictions and not snapshot.contradictions[0].resolved
    assert snapshot.triage_result is not None
    assert snapshot.triage_result.triage_level is TriageLevel.URGENT
    assert snapshot.execution_trace[-1].details["revision_count"] == 2


def test_risk_agent_does_not_generate_final_report(tmp_path: Path) -> None:
    case = MedicalCaseInput(
        case_id="CASE-INFECTIOUS-REVIEW",
        chief_complaint="Fever and cough",
        symptoms=["fever", "cough"],
    )

    snapshot = run_case(tmp_path, case)

    assert snapshot.triage_result is not None
    assert snapshot.triage_result.triage_level is TriageLevel.PRIORITY
    assert snapshot.final_report is None


def test_live_critic_response_cannot_bypass_revision_limit() -> None:
    class AlwaysReviseProvider(DemoModelProvider):
        def generate(self, **kwargs):
            if kwargs["agent"] != "critic":
                return super().generate(**kwargs)
            return ProviderResult(
                output=CriticReview(
                    decision=CriticDecision.REVISE,
                    problems=["The model requested another revision."],
                    severity=Severity.MEDIUM,
                ),
                usage=TokenUsage(
                    agent="critic",
                    provider="test",
                    model="always-revise",
                ),
            )

    state = create_initial_state(
        MedicalCaseInput(chief_complaint="Revision-limit audit"),
        run_id="RUN-CRITIC-LIMIT",
    )
    state["revision_count"] = 2

    update = CriticAgent(
        AlwaysReviseProvider(), max_revisions=2, max_retries=0
    ).analyze(state)

    review = update["critic_review"]
    assert isinstance(review, CriticReview)
    assert review.decision is CriticDecision.ACCEPT
    assert any("Revision limit reached" in item for item in review.problems)
