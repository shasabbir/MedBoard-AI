"""LangGraph-compatible shared state and whole-state validation."""

from __future__ import annotations

from typing import Annotated, Any, NotRequired, TypedDict

from pydantic import Field, model_validator

from medboard.graph.reducers import (
    merge_contradictions,
    merge_critic_reviews,
    merge_errors,
    merge_evidence,
    merge_evidence_questions,
    merge_messages,
    merge_missing_information,
    merge_retrieved_evidence,
    merge_routing_decisions,
    merge_specialist_opinions,
    merge_token_usage,
    merge_trace,
)
from medboard.models import (
    AgentError,
    AgentMessage,
    ContractModel,
    Contradiction,
    CriticReview,
    DifferentialDiagnosis,
    DifferentialAnalysis,
    Evidence,
    EvidenceQuestion,
    EvidenceRetrievalAnalysis,
    FinalReport,
    HistoryFindings,
    HumanReview,
    HumanReviewCommand,
    HumanStatus,
    LaboratoryFindings,
    MedicalCaseInput,
    MedicationFindings,
    MissingInformationRequest,
    NormalizedCase,
    RetrievedEvidence,
    SpecialistOpinion,
    SpecialistRoutingDecision,
    SupervisorPlan,
    SymptomFindings,
    TokenUsage,
    TraceEvent,
    TriageResult,
    new_id,
)


class MedicalCaseState(TypedDict):
    """Shared graph state; append-only collections have concurrent reducers."""

    run_id: str
    case_input: MedicalCaseInput
    normalized_case: NotRequired[NormalizedCase | None]
    supervisor_plan: NotRequired[SupervisorPlan | None]
    history_findings: NotRequired[HistoryFindings | None]
    symptom_findings: NotRequired[SymptomFindings | None]
    laboratory_findings: NotRequired[LaboratoryFindings | None]
    medication_findings: NotRequired[MedicationFindings | None]
    evidence: Annotated[list[Evidence], merge_evidence]
    evidence_questions: Annotated[list[EvidenceQuestion], merge_evidence_questions]
    contradictions: Annotated[list[Contradiction], merge_contradictions]
    missing_information: Annotated[
        list[MissingInformationRequest], merge_missing_information
    ]
    differential_diagnoses: NotRequired[list[DifferentialDiagnosis]]
    differential_analysis: NotRequired[DifferentialAnalysis | None]
    selected_specialists: NotRequired[list[str]]
    routing_decisions: Annotated[
        list[SpecialistRoutingDecision], merge_routing_decisions
    ]
    specialist_opinions: Annotated[
        list[SpecialistOpinion], merge_specialist_opinions
    ]
    retrieved_evidence: Annotated[list[RetrievedEvidence], merge_retrieved_evidence]
    evidence_retrieval_analysis: NotRequired[EvidenceRetrievalAnalysis | None]
    critic_review: NotRequired[CriticReview | None]
    critic_reviews: Annotated[list[CriticReview], merge_critic_reviews]
    triage_result: NotRequired[TriageResult | None]
    revision_count: NotRequired[int]
    human_review: NotRequired[HumanReview]
    human_command: NotRequired[HumanReviewCommand | None]
    human_added_information: NotRequired[dict[str, Any]]
    final_report: NotRequired[FinalReport | None]
    agent_messages: Annotated[list[AgentMessage], merge_messages]
    errors: Annotated[list[AgentError], merge_errors]
    execution_trace: Annotated[list[TraceEvent], merge_trace]
    token_usage: Annotated[list[TokenUsage], merge_token_usage]


class MedicalCaseSnapshot(ContractModel):
    """Complete, serializable state used at validation and persistence boundaries."""

    run_id: str = Field(default_factory=lambda: new_id("RUN"), frozen=True)
    case_input: MedicalCaseInput
    normalized_case: NormalizedCase | None = None
    supervisor_plan: SupervisorPlan | None = None
    history_findings: HistoryFindings | None = None
    symptom_findings: SymptomFindings | None = None
    laboratory_findings: LaboratoryFindings | None = None
    medication_findings: MedicationFindings | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    evidence_questions: list[EvidenceQuestion] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    missing_information: list[MissingInformationRequest] = Field(default_factory=list)
    differential_diagnoses: list[DifferentialDiagnosis] = Field(default_factory=list)
    differential_analysis: DifferentialAnalysis | None = None
    selected_specialists: list[str] = Field(default_factory=list)
    routing_decisions: list[SpecialistRoutingDecision] = Field(default_factory=list)
    specialist_opinions: list[SpecialistOpinion] = Field(default_factory=list)
    retrieved_evidence: list[RetrievedEvidence] = Field(default_factory=list)
    evidence_retrieval_analysis: EvidenceRetrievalAnalysis | None = None
    critic_review: CriticReview | None = None
    critic_reviews: list[CriticReview] = Field(default_factory=list)
    triage_result: TriageResult | None = None
    revision_count: int = Field(default=0, ge=0, le=3)
    human_review: HumanReview = Field(default_factory=HumanReview)
    human_command: HumanReviewCommand | None = None
    human_added_information: dict[str, Any] = Field(default_factory=dict)
    final_report: FinalReport | None = None
    agent_messages: list[AgentMessage] = Field(default_factory=list)
    errors: list[AgentError] = Field(default_factory=list)
    execution_trace: list[TraceEvent] = Field(default_factory=list)
    token_usage: list[TokenUsage] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_referential_integrity(self) -> MedicalCaseSnapshot:
        known_evidence = {item.evidence_id for item in self.evidence}
        references: list[tuple[str, list[str]]] = []

        outputs = [
            self.history_findings,
            self.symptom_findings,
            self.laboratory_findings,
            self.medication_findings,
        ]
        for findings in outputs:
            if findings is not None:
                for claim in findings.output.claims:
                    references.append(
                        (
                            claim.claim_id,
                            [*claim.evidence_ids, *claim.contradicting_evidence_ids],
                        )
                    )

        for diagnosis in self.differential_diagnoses:
            references.append(
                (
                    diagnosis.hypothesis_id,
                    [
                        *diagnosis.supporting_evidence_ids,
                        *diagnosis.contradicting_evidence_ids,
                    ],
                )
            )
        if self.differential_analysis is not None:
            for claim in self.differential_analysis.output.claims:
                references.append(
                    (
                        claim.claim_id,
                        [*claim.evidence_ids, *claim.contradicting_evidence_ids],
                    )
                )
        for decision in self.routing_decisions:
            references.append((decision.routing_id, decision.evidence_ids))
        for opinion in self.specialist_opinions:
            references.append((opinion.opinion_id, opinion.evidence_ids))
        for question in self.evidence_questions:
            references.append((question.question_id, question.evidence_ids))
        for message in self.agent_messages:
            references.append((message.message_id, message.evidence_ids))
        for request in self.missing_information:
            references.append((request.request_id, request.evidence_ids))

        dangling = {
            reference_id: sorted(set(evidence_ids) - known_evidence)
            for reference_id, evidence_ids in references
            if set(evidence_ids) - known_evidence
        }
        if dangling:
            raise ValueError(f"state contains unknown evidence references: {dangling}")

        if (
            self.differential_analysis is not None
            and self.differential_analysis.diagnoses != self.differential_diagnoses
        ):
            raise ValueError("differential analysis and state diagnoses must match")
        known_hypotheses = {
            diagnosis.hypothesis_id for diagnosis in self.differential_diagnoses
        }
        for question in self.evidence_questions:
            unknown_question_hypotheses = set(question.hypothesis_ids) - known_hypotheses
            if unknown_question_hypotheses:
                raise ValueError(
                    "evidence questions reference unknown hypotheses: "
                    f"{sorted(unknown_question_hypotheses)}"
                )
        known_questions = {question.question_id for question in self.evidence_questions}
        unknown_result_questions = {
            result.question_id
            for result in self.retrieved_evidence
            if result.question_id not in known_questions
        }
        if unknown_result_questions:
            raise ValueError(
                "retrieval results reference unknown questions: "
                f"{sorted(unknown_result_questions)}"
            )
        known_retrievals = {result.retrieval_id for result in self.retrieved_evidence}
        unknown_message_retrievals = {
            retrieval_id
            for message in self.agent_messages
            for retrieval_id in message.retrieval_ids
            if retrieval_id not in known_retrievals
        }
        if unknown_message_retrievals:
            raise ValueError(
                "messages reference unknown retrieval results: "
                f"{sorted(unknown_message_retrievals)}"
            )
        if (
            self.evidence_retrieval_analysis is not None
            and self.evidence_retrieval_analysis.results != self.retrieved_evidence
        ):
            raise ValueError("retrieval analysis and state results must match")
        for opinion in self.specialist_opinions:
            if opinion.specialist not in self.selected_specialists:
                raise ValueError(
                    f"specialist opinion was not selected by the router: {opinion.specialist}"
                )
            opinion_hypotheses = {
                *opinion.supported_hypotheses,
                *opinion.challenged_hypotheses,
            }
            unknown_hypotheses = opinion_hypotheses - known_hypotheses
            if unknown_hypotheses:
                raise ValueError(
                    f"specialist opinion references unknown hypotheses: "
                    f"{sorted(unknown_hypotheses)}"
                )
        unknown_contradictions = {
            item.hypothesis_id
            for item in self.contradictions
            if item.hypothesis_id is not None
            and item.hypothesis_id not in known_hypotheses
        }
        if unknown_contradictions:
            raise ValueError(
                f"contradictions reference unknown hypotheses: "
                f"{sorted(unknown_contradictions)}"
            )
        if self.final_report is not None and self.human_review.status is not HumanStatus.APPROVED:
            raise ValueError("a final report requires explicit human approval")
        return self

    @property
    def estimated_cost(self) -> float:
        """Calculate total estimated cost from immutable usage records."""
        return sum(record.estimated_cost for record in self.token_usage)

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens from immutable usage records."""
        return sum(record.total_tokens for record in self.token_usage)

    def to_graph_state(self) -> MedicalCaseState:
        """Return a graph-ready state while preserving validated model instances."""
        return {
            field_name: getattr(self, field_name)
            for field_name in MedicalCaseState.__annotations__
        }  # type: ignore[return-value]


def create_initial_state(
    case_input: MedicalCaseInput, run_id: str | None = None
) -> MedicalCaseState:
    """Create a fully initialized graph state for predictable reducer behavior."""
    snapshot = MedicalCaseSnapshot(
        run_id=run_id or new_id("RUN"),
        case_input=case_input,
    )
    return snapshot.to_graph_state()


def validate_state(state: MedicalCaseState | dict[str, Any]) -> MedicalCaseSnapshot:
    """Validate graph state before checkpointing, persistence, or report generation."""
    clean_state = {key: value for key, value in state.items() if not key.startswith("__")}
    return MedicalCaseSnapshot.model_validate(clean_state)
