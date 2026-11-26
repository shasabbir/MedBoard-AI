"""End-to-end tests for differential reasoning and conditional specialists."""

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import pytest

from medboard.agents.differential import DifferentialAgent
from medboard.graph.state import create_initial_state, validate_state
from medboard.graph.workflow import build_collaboration_workflow
from medboard.models import (
    AgentOutput,
    AgentStatus,
    ContractModel,
    DifferentialAnalysis,
    DifferentialDiagnosis,
    MedicalCaseInput,
    SpecialistOpinion,
    TokenUsage,
)
from medboard.providers import DemoModelProvider, ProviderResult

OutputT = TypeVar("OutputT", bound=ContractModel)


class AlternateDifferentialProvider(DemoModelProvider):
    """Emulate a live model returning hypotheses unlike the deterministic draft."""

    def generate(
        self,
        *,
        agent: str,
        prompt: str,
        context: object,
        response_model: type[OutputT],
        demo_factory: Callable[[], OutputT],
    ) -> ProviderResult[OutputT]:
        if agent != "differential":
            return super().generate(
                agent=agent,
                prompt=prompt,
                context=context,
                response_model=response_model,
                demo_factory=demo_factory,
            )
        output = DifferentialAnalysis(
            diagnoses=[
                DifferentialDiagnosis(
                    hypothesis_id="MODEL-PRIMARY",
                    hypothesis="Model-proposed primary consideration",
                    confidence=0.6,
                    missing_evidence=["targeted primary test"],
                ),
                DifferentialDiagnosis(
                    hypothesis_id="MODEL-PRIMARY",
                    hypothesis="Model-proposed alternate consideration",
                    confidence=0.3,
                    missing_evidence=["targeted alternate test"],
                ),
            ],
            output=AgentOutput(
                agent="differential",
                status=AgentStatus.COMPLETED,
                summary="Returned two alternate considerations.",
            ),
        )
        return ProviderResult(
            output=response_model.model_validate(output.model_dump()),
            usage=TokenUsage(
                agent=agent,
                provider=self.provider_name,
                model=self.model_name,
            ),
        )


class DisplayNameSpecialistProvider(DemoModelProvider):
    """Emulate a live model returning a human-readable specialist label."""

    def generate(
        self,
        *,
        agent: str,
        prompt: str,
        context: object,
        response_model: type[OutputT],
        demo_factory: Callable[[], OutputT],
    ) -> ProviderResult[OutputT]:
        result = super().generate(
            agent=agent,
            prompt=prompt,
            context=context,
            response_model=response_model,
            demo_factory=demo_factory,
        )
        if agent != "cardiology" or not isinstance(result.output, SpecialistOpinion):
            return result
        display_output = result.output.model_copy(update={"specialist": "Cardiology"})
        return ProviderResult(
            output=response_model.model_validate(display_output.model_dump()),
            usage=result.usage,
        )


def load_case(name: str) -> MedicalCaseInput:
    path = Path("data/demo_cases") / f"{name}.json"
    return MedicalCaseInput.model_validate_json(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("case_name", "expected_specialists", "excluded_specialists"),
    [
        ("anemia", ["cardiology"], ["neurology", "infectious_disease"]),
        ("neurological", ["neurology"], ["cardiology", "infectious_disease"]),
        ("infectious", ["infectious_disease"], ["cardiology", "neurology"]),
    ],
)
def test_router_selects_only_evidence_justified_specialists(
    case_name: str,
    expected_specialists: list[str],
    excluded_specialists: list[str],
) -> None:
    workflow = build_collaboration_workflow(DemoModelProvider())

    result = workflow.invoke(
        create_initial_state(load_case(case_name), run_id=f"RUN-{case_name.upper()}")
    )
    snapshot = validate_state(result)

    assert snapshot.selected_specialists == expected_specialists
    assert len(snapshot.specialist_opinions) == len(expected_specialists)
    assert snapshot.routing_decisions[-1].selected_specialists == expected_specialists
    assert set(snapshot.routing_decisions[-1].reasons) == set(expected_specialists)
    assert not (set(excluded_specialists) & set(snapshot.selected_specialists))


def test_specialist_node_restores_canonical_identity_from_live_model_output() -> None:
    workflow = build_collaboration_workflow(DisplayNameSpecialistProvider())

    result = workflow.invoke(
        create_initial_state(load_case("cardiac"), run_id="RUN-SPECIALIST-IDENTITY")
    )
    snapshot = validate_state(result)

    assert snapshot.selected_specialists == ["cardiology"]
    assert [item.specialist for item in snapshot.specialist_opinions] == ["cardiology"]
    assert any(message.sender == "cardiology" for message in snapshot.agent_messages)


def test_duplicate_missing_information_is_aggregated_across_agents() -> None:
    workflow = build_collaboration_workflow(DemoModelProvider())
    snapshot = validate_state(
        workflow.invoke(
            create_initial_state(load_case("cardiac"), run_id="RUN-MISSING-AGGREGATE")
        )
    )

    ecg_requests = [
        request
        for request in snapshot.missing_information
        if request.information_needed.casefold() == "ecg"
    ]
    assert len(ecg_requests) == 1
    assert ecg_requests[0].requested_by == ["differential", "cardiology"]


def test_differential_produces_competing_traceable_hypotheses() -> None:
    workflow = build_collaboration_workflow(DemoModelProvider())
    result = workflow.invoke(create_initial_state(load_case("anemia"), run_id="RUN-ANEMIA"))
    snapshot = validate_state(result)
    known_evidence = {item.evidence_id for item in snapshot.evidence}

    assert len(snapshot.differential_diagnoses) >= 2
    assert snapshot.differential_analysis is not None
    assert all(item.supporting_evidence_ids for item in snapshot.differential_diagnoses)
    assert all(
        set(item.supporting_evidence_ids) <= known_evidence
        for item in snapshot.differential_diagnoses
    )
    assert all(item.confidence < 1 for item in snapshot.differential_diagnoses)


def test_differential_questions_follow_validated_provider_hypotheses() -> None:
    state = create_initial_state(
        MedicalCaseInput(chief_complaint="Synthetic live differential test"),
        run_id="RUN-LIVE-DIFFERENTIAL",
    )

    update = DifferentialAgent(AlternateDifferentialProvider()).analyze(state)

    diagnosis_ids = {
        item.hypothesis_id for item in update["differential_diagnoses"]
    }
    question_ids = {
        hypothesis_id
        for question in update["evidence_questions"]
        for hypothesis_id in question.hypothesis_ids
    }
    assert question_ids == diagnosis_ids
    assert diagnosis_ids == {"HYP-MODEL-001", "HYP-MODEL-002"}
    assert {item.information_needed for item in update["missing_information"]} == {
        "targeted primary test",
        "targeted alternate test",
    }


def test_specialist_challenge_becomes_auditable_contradiction() -> None:
    workflow = build_collaboration_workflow(DemoModelProvider())
    result = workflow.invoke(create_initial_state(load_case("anemia"), run_id="RUN-ANEMIA"))
    snapshot = validate_state(result)

    assert snapshot.contradictions
    contradiction = snapshot.contradictions[0]
    assert contradiction.agent_a == "differential"
    assert contradiction.agent_b == "cardiology"
    assert contradiction.hypothesis_id == "HYP-PULMONARY"
    assert contradiction.resolved is False
    assert any(
        message.message_type.value == "challenge"
        and message.sender == "cardiology"
        and message.recipient == "differential"
        for message in snapshot.agent_messages
    )


def test_router_can_fan_out_to_multiple_specialists() -> None:
    case = MedicalCaseInput(
        case_id="CASE-MULTI-001",
        chief_complaint="Fever with acute confusion",
        symptoms=["fever", "confusion"],
    )
    workflow = build_collaboration_workflow(DemoModelProvider())

    result = workflow.invoke(create_initial_state(case, run_id="RUN-MULTI"))
    snapshot = validate_state(result)

    assert snapshot.selected_specialists == ["neurology", "infectious_disease"]
    assert {item.specialist for item in snapshot.specialist_opinions} == {
        "neurology",
        "infectious_disease",
    }
    assert snapshot.execution_trace[-1].details["selected_specialists"] == [
        "neurology",
        "infectious_disease",
    ]
    routing_requests = {
        message.recipient: set(message.evidence_ids)
        for message in snapshot.agent_messages
        if message.sender == "supervisor"
        and message.message_type.value == "request"
        and message.recipient in {"neurology", "infectious_disease"}
    }
    evidence_names = {
        item.evidence_id: item.name.casefold() for item in snapshot.evidence
    }
    assert {evidence_names[item] for item in routing_requests["neurology"]} == {
        "confusion"
    }
    assert {evidence_names[item] for item in routing_requests["infectious_disease"]} == {
        "fever"
    }


def test_router_can_select_no_specialist_and_still_complete() -> None:
    case = MedicalCaseInput(
        case_id="CASE-GENERAL-001",
        chief_complaint="Reduced appetite",
        symptoms=["reduced appetite"],
    )
    workflow = build_collaboration_workflow(DemoModelProvider())

    result = workflow.invoke(create_initial_state(case, run_id="RUN-GENERAL"))
    snapshot = validate_state(result)

    assert snapshot.selected_specialists == []
    assert snapshot.specialist_opinions == []
    assert len(snapshot.differential_diagnoses) == 2
    assert snapshot.execution_trace[-1].event_type.value == "workflow_completed"


def test_specialist_grounds_hypothesis_names_and_increments_question_ids() -> None:
    from medboard.agents.specialists import CardiologyAgent
    from medboard.models import Evidence, EvidenceQuestion, EvidenceType

    class DescriptiveHypothesisSpecialistProvider(DemoModelProvider):
        def generate(self, **kwargs):
            if kwargs["agent"] != "cardiology":
                return super().generate(**kwargs)
            opinion = SpecialistOpinion(
                opinion_id="OPN-STATIC-001",
                specialist="Cardiology",
                assessment="Cardiac review completed.",
                supported_hypotheses=["Acute coronary syndrome consideration", "Invented"],
                challenged_hypotheses=["Pulmonary or other cardiorespiratory process"],
                evidence_ids=["EV-SYMPTOM-001", "EV-DOES-NOT-EXIST"],
                confidence=0.72,
            )
            return ProviderResult(
                output=kwargs["response_model"].model_validate(opinion.model_dump()),
                usage=TokenUsage(
                    agent="cardiology",
                    provider="test",
                    model="descriptive-specialist",
                ),
            )

    case = MedicalCaseInput(
        case_id="CASE-SPEC-GROUND",
        chief_complaint="Chest pain",
        symptoms=["chest pain"],
    )
    state = create_initial_state(case, run_id="RUN-SPEC-GROUND")
    state["evidence"] = [
        Evidence(
            evidence_id="EV-SYMPTOM-001",
            evidence_type=EvidenceType.SYMPTOM,
            name="chest pain",
            value=True,
            source="user_case",
        )
    ]
    state["differential_diagnoses"] = [
        DifferentialDiagnosis(
            hypothesis_id="HYP-CORONARY",
            hypothesis="Acute coronary syndrome consideration",
            supporting_evidence_ids=["EV-SYMPTOM-001"],
            confidence=0.64,
        ),
        DifferentialDiagnosis(
            hypothesis_id="HYP-PULMONARY",
            hypothesis="Pulmonary or other cardiorespiratory process",
            supporting_evidence_ids=["EV-SYMPTOM-001"],
            confidence=0.43,
        ),
    ]
    state["evidence_questions"] = [
        EvidenceQuestion(
            question_id="Q-CARDIOLOGY-001",
            asked_by="cardiology",
            question="Prior cardiology question",
            hypothesis_ids=["HYP-CORONARY"],
            evidence_ids=["EV-SYMPTOM-001"],
        )
    ]

    update = CardiologyAgent(DescriptiveHypothesisSpecialistProvider()).analyze(state)

    opinion = update["specialist_opinions"][0]
    assert opinion.specialist == "cardiology"
    assert opinion.opinion_id.startswith("OPN-")
    assert opinion.opinion_id != "OPN-STATIC-001"
    assert opinion.supported_hypotheses == ["HYP-CORONARY"]
    assert opinion.challenged_hypotheses == ["HYP-PULMONARY"]
    assert opinion.evidence_ids == ["EV-SYMPTOM-001"]
    question = update["evidence_questions"][0]
    assert question.question_id == "Q-CARDIOLOGY-002"
    assert question.hypothesis_ids == ["HYP-CORONARY", "HYP-PULMONARY"]
