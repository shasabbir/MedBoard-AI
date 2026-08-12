"""Validated contracts for reproducible MedBoard benchmarks."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from medboard.models import ContractModel, MedicalCaseInput, TriageLevel


class BenchmarkCase(ContractModel):
    benchmark_id: str
    case_file: Path
    expected_diagnoses: list[str] = Field(min_length=1)
    expected_specialists: list[str] = Field(default_factory=list)
    expected_red_flag: bool
    expected_triage: TriageLevel
    expected_missing_information: list[str] = Field(default_factory=list)

    def load_case(self, repository_root: Path) -> MedicalCaseInput:
        path = repository_root / self.case_file
        return MedicalCaseInput.model_validate_json(path.read_text(encoding="utf-8"))


class RagBenchmarkQuestion(ContractModel):
    question_id: str
    question: str
    expected_document: str
    top_k: int = Field(default=3, ge=1, le=20)


class SystemOutput(ContractModel):
    diagnoses: list[str] = Field(default_factory=list)
    selected_specialists: list[str] = Field(default_factory=list)
    red_flag_detected: bool = False
    triage_level: TriageLevel = TriageLevel.ROUTINE
    missing_information: list[str] = Field(default_factory=list)
    unsupported_claims: int = Field(default=0, ge=0)
    total_claims: int = Field(default=0, ge=0)
    tokens: int = Field(default=0, ge=0)
    agent_calls: int = Field(default=0, ge=0)
    revisions: int = Field(default=0, ge=0)


class CaseEvaluation(ContractModel):
    benchmark_id: str
    expected_diagnoses: list[str]
    diagnoses: list[str]
    diagnosis_ranks: dict[str, int | None]
    expected_specialists: list[str]
    selected_specialists: list[str]
    expected_red_flag: bool
    red_flag_detected: bool
    expected_triage: TriageLevel
    triage_level: TriageLevel
    expected_missing_information: list[str]
    matched_missing_information: list[str]
    unsupported_claims: int = Field(ge=0)
    total_claims: int = Field(ge=0)
    tokens: int = Field(ge=0)
    agent_calls: int = Field(ge=0)
    revisions: int = Field(ge=0)


class CapabilityMetrics(ContractModel):
    case_count: int = Field(ge=1)
    top_1_recall: float = Field(ge=0, le=1)
    top_3_recall: float = Field(ge=0, le=1)
    top_5_recall: float = Field(ge=0, le=1)
    routing_precision: float = Field(ge=0, le=1)
    routing_recall: float = Field(ge=0, le=1)
    red_flag_recall: float = Field(ge=0, le=1)
    red_flag_false_alarms: int = Field(ge=0)
    triage_accuracy: float = Field(ge=0, le=1)
    missing_information_recall: float = Field(ge=0, le=1)
    unsupported_claim_rate: float = Field(ge=0, le=1)
    mean_tokens: float = Field(ge=0)
    mean_agent_calls: float = Field(ge=0)
    mean_revisions: float = Field(ge=0)


class ConfigurationEvaluation(ContractModel):
    configuration: str
    description: str
    metrics: CapabilityMetrics
    cases: list[CaseEvaluation]


class RagQuestionResult(ContractModel):
    question_id: str
    expected_document: str
    retrieved_documents: list[str]
    relevant_rank: int | None
    citations_complete: bool


class RagMetrics(ContractModel):
    question_count: int = Field(ge=1)
    recall_at_k: float = Field(ge=0, le=1)
    mean_reciprocal_rank: float = Field(ge=0, le=1)
    citation_completeness: float = Field(ge=0, le=1)
    questions: list[RagQuestionResult]


class EvaluationReport(ContractModel):
    benchmark_version: str
    deterministic_demo: bool = True
    configurations: list[ConfigurationEvaluation]
    rag: RagMetrics
