"""LangGraph-compatible shared state and whole-state validation."""

from __future__ import annotations

from typing import Annotated, Any, NotRequired, TypedDict

from pydantic import Field, model_validator

from medboard.graph.reducers import (
    merge_contradictions,
    merge_errors,
    merge_evidence,
    merge_messages,
    merge_missing_information,
    merge_retrieved_evidence,
    merge_specialist_opinions,
    merge_token_usage,
    merge_trace,
    merge_unique_strings,
)
from medboard.models import (
    AgentError,
    AgentMessage,
    ContractModel,
    Contradiction,
    CriticReview,
    DifferentialDiagnosis,
    Evidence,
    FinalReport,
    HistoryFindings,
    HumanReview,
    HumanStatus,
    LaboratoryFindings,
    MedicalCaseInput,
    MedicationFindings,
    MissingInformationRequest,
    NormalizedCase,
    RetrievedEvidence,
    SpecialistOpinion,
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
    contradictions: Annotated[list[Contradiction], merge_contradictions]
    missing_information: Annotated[
        list[MissingInformationRequest], merge_missing_information
    ]
    differential_diagnoses: NotRequired[list[DifferentialDiagnosis]]
    selected_specialists: Annotated[list[str], merge_unique_strings]
    specialist_opinions: Annotated[
        list[SpecialistOpinion], merge_specialist_opinions
    ]
    retrieved_evidence: Annotated[list[RetrievedEvidence], merge_retrieved_evidence]
    critic_review: NotRequired[CriticReview | None]
    triage_result: NotRequired[TriageResult | None]
    revision_count: NotRequired[int]
    human_review: NotRequired[HumanReview]
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
    contradictions: list[Contradiction] = Field(default_factory=list)
    missing_information: list[MissingInformationRequest] = Field(default_factory=list)
    differential_diagnoses: list[DifferentialDiagnosis] = Field(default_factory=list)
    selected_specialists: list[str] = Field(default_factory=list)
    specialist_opinions: list[SpecialistOpinion] = Field(default_factory=list)
    retrieved_evidence: list[RetrievedEvidence] = Field(default_factory=list)
    critic_review: CriticReview | None = None
    triage_result: TriageResult | None = None
    revision_count: int = Field(default=0, ge=0, le=3)
    human_review: HumanReview = Field(default_factory=HumanReview)
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
    return MedicalCaseSnapshot.model_validate(state)
