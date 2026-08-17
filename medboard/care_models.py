"""Fail-closed contracts for the clinician-supervised care-delivery roadmap.

These models define Phase 11 boundaries only. They are intentionally disconnected from the
current educational workflow and do not authorize treatment, prescribing, or patient delivery.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
import json
from typing import Self

from pydantic import ConfigDict, Field, field_validator, model_validator

from medboard.models import (
    ContractModel,
    NonEmptyString,
    Severity,
    TriageLevel,
    new_id,
    utc_now,
)


class ConsentStatus(StrEnum):
    GRANTED = "granted"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"


class CareArtifactType(StrEnum):
    CLINICAL_ASSESSMENT = "clinical_assessment"
    CARE_PLAN = "care_plan"
    MEDICATION_ORDER = "medication_order"
    PATIENT_INSTRUCTIONS = "patient_instructions"


class CareArtifactStatus(StrEnum):
    DRAFT_READY = "draft_ready"
    WAITING_FOR_CLINICIAN = "waiting_for_clinician"
    CHANGES_REQUESTED = "changes_requested"
    ANALYSIS_APPROVED = "analysis_approved"
    CARE_PLAN_AUTHORIZED = "care_plan_authorized"
    PRESCRIPTION_AUTHORIZED = "prescription_authorized"
    REJECTED = "rejected"
    EXPIRED = "expired"
    REVOKED = "revoked"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"


class TreatmentType(StrEnum):
    NON_DRUG = "non_drug"
    MEDICATION = "medication"
    TEST = "test"
    REFERRAL = "referral"
    FOLLOW_UP = "follow_up"


class SafetyCheckOutcome(StrEnum):
    PASS = "pass"
    BLOCK = "block"
    REVIEW_REQUIRED = "review_required"


class AuthorizationDecision(StrEnum):
    AUTHORIZE = "authorize"
    REJECT = "reject"


class DeliveryStatus(StrEnum):
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    ACKNOWLEDGED = "acknowledged"
    REVOKED = "revoked"


class FollowUpStatus(StrEnum):
    OPEN = "open"
    COMPLETED = "completed"
    OVERDUE = "overdue"
    ESCALATED = "escalated"
    CANCELLED = "cancelled"


class QueueStatus(StrEnum):
    UNASSIGNED = "unassigned"
    ASSIGNED = "assigned"
    IN_REVIEW = "in_review"
    COMPLETED = "completed"
    ESCALATED = "escalated"


class PatientIdentityRef(ContractModel):
    """Reference to identity-provider data; never stores identity documents."""

    patient_ref: NonEmptyString
    verified: bool = False
    verification_method: NonEmptyString | None = None
    verified_at: datetime | None = None

    @field_validator("verified_at")
    @classmethod
    def normalize_verified_at(cls, value: datetime | None) -> datetime | None:
        return _as_utc(value) if value is not None else None

    @model_validator(mode="after")
    def verified_identity_requires_auditable_method(self) -> Self:
        has_details = self.verification_method is not None and self.verified_at is not None
        if self.verified != has_details:
            raise ValueError(
                "verified identity requires both verification_method and verified_at"
            )
        return self


class ConsentRecord(ContractModel):
    consent_id: str = Field(default_factory=lambda: new_id("CNS"), frozen=True)
    patient_ref: NonEmptyString
    purpose: NonEmptyString
    status: ConsentStatus
    granted_at: datetime
    expires_at: datetime | None = None
    withdrawn_at: datetime | None = None
    withdrawal_reason: NonEmptyString | None = None

    @field_validator("granted_at", "expires_at", "withdrawn_at")
    @classmethod
    def normalize_consent_times(cls, value: datetime | None) -> datetime | None:
        return _as_utc(value) if value is not None else None

    @model_validator(mode="after")
    def validate_consent_lifecycle(self) -> Self:
        if self.expires_at is not None and self.expires_at <= self.granted_at:
            raise ValueError("consent expiry must be after grant time")
        withdrawn = self.withdrawn_at is not None and self.withdrawal_reason is not None
        if (self.status is ConsentStatus.WITHDRAWN) != withdrawn:
            raise ValueError(
                "withdrawn consent requires withdrawn_at and withdrawal_reason"
            )
        return self


class EncounterContext(ContractModel):
    encounter_id: str = Field(default_factory=lambda: new_id("ENC"), frozen=True)
    patient_ref: NonEmptyString
    consent_id: NonEmptyString
    organization_id: NonEmptyString
    jurisdiction: NonEmptyString
    patient_location: NonEmptyString
    preferred_language: NonEmptyString
    callback_route: NonEmptyString
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return _as_utc(value)


class ClinicalAssessment(ContractModel):
    assessment_id: str = Field(default_factory=lambda: new_id("ASM"), frozen=True)
    version: int = Field(default=1, ge=1)
    encounter_id: NonEmptyString
    evidence_ids: list[NonEmptyString] = Field(min_length=1)
    hypothesis_ids: list[NonEmptyString] = Field(default_factory=list)
    uncertainty: list[NonEmptyString] = Field(default_factory=list)
    missing_critical_information: list[NonEmptyString] = Field(default_factory=list)
    red_flags: list[NonEmptyString] = Field(default_factory=list)
    status: CareArtifactStatus = CareArtifactStatus.DRAFT_READY
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("evidence_ids", "hypothesis_ids")
    @classmethod
    def references_are_unique(cls, values: list[str]) -> list[str]:
        return _unique(values)

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def status_matches_assessment_lifecycle(self) -> Self:
        _ensure_status_allowed(CareArtifactType.CLINICAL_ASSESSMENT, self.status)
        return self


class TreatmentOption(ContractModel):
    option_id: str = Field(default_factory=lambda: new_id("OPT"), frozen=True)
    treatment_type: TreatmentType
    indication: NonEmptyString
    description: NonEmptyString
    evidence_ids: list[NonEmptyString] = Field(min_length=1)
    retrieval_ids: list[NonEmptyString] = Field(default_factory=list)
    source_version_ids: list[NonEmptyString] = Field(min_length=1)

    @field_validator("evidence_ids", "retrieval_ids", "source_version_ids")
    @classmethod
    def references_are_unique(cls, values: list[str]) -> list[str]:
        return _unique(values)


class CarePlanDraft(ContractModel):
    care_plan_id: str = Field(default_factory=lambda: new_id("PLAN"), frozen=True)
    version: int = Field(default=1, ge=1)
    patient_ref: NonEmptyString
    encounter_id: NonEmptyString
    assessment_id: NonEmptyString
    options: list[TreatmentOption] = Field(min_length=1)
    follow_up_window: NonEmptyString
    warning_signs: list[NonEmptyString] = Field(min_length=1)
    status: CareArtifactStatus = CareArtifactStatus.DRAFT_READY
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def status_matches_care_plan_lifecycle(self) -> Self:
        _ensure_draft_status(self.status)
        return self


class MedicationOrderDraft(ContractModel):
    """Complete medication proposal; this contract is never itself a prescription."""

    medication_order_id: str = Field(
        default_factory=lambda: new_id("MEDORD"), frozen=True
    )
    version: int = Field(default=1, ge=1)
    patient_ref: NonEmptyString
    encounter_id: NonEmptyString
    assessment_id: NonEmptyString
    care_plan_id: NonEmptyString
    indication: NonEmptyString
    evidence_ids: list[NonEmptyString] = Field(min_length=1)
    normalized_medicine_id: NonEmptyString
    medicine_name: NonEmptyString
    formulation: NonEmptyString
    strength: NonEmptyString
    dose_value: Decimal = Field(gt=0)
    dose_unit: NonEmptyString
    route: NonEmptyString
    frequency: NonEmptyString
    timing: NonEmptyString
    duration_value: int = Field(gt=0)
    duration_unit: NonEmptyString
    quantity: Decimal = Field(gt=0)
    maximum_dose_value: Decimal | None = Field(default=None, gt=0)
    maximum_dose_unit: NonEmptyString | None = None
    start_instructions: NonEmptyString
    stop_instructions: NonEmptyString
    monitoring_requirements: list[NonEmptyString] = Field(min_length=1)
    follow_up_requirements: list[NonEmptyString] = Field(min_length=1)
    patient_specific_considerations: list[NonEmptyString] = Field(default_factory=list)
    source_version_ids: list[NonEmptyString] = Field(min_length=1)
    drafted_by_agent: NonEmptyString
    status: CareArtifactStatus = CareArtifactStatus.DRAFT_READY
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("evidence_ids", "source_version_ids")
    @classmethod
    def references_are_unique(cls, values: list[str]) -> list[str]:
        return _unique(values)

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def validate_order_draft(self) -> Self:
        has_maximum = self.maximum_dose_value is not None
        if has_maximum != (self.maximum_dose_unit is not None):
            raise ValueError("maximum dose value and unit must be supplied together")
        _ensure_draft_status(self.status)
        return self


class MedicationSafetyCheck(ContractModel):
    safety_check_id: str = Field(default_factory=lambda: new_id("SAFE"), frozen=True)
    medication_order_id: NonEmptyString
    medication_order_version: int = Field(ge=1)
    check_type: NonEmptyString
    outcome: SafetyCheckOutcome
    message: NonEmptyString
    source_id: NonEmptyString
    source_version: NonEmptyString
    blocking: bool
    checked_at: datetime = Field(default_factory=utc_now)

    @field_validator("checked_at")
    @classmethod
    def normalize_checked_at(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def outcome_matches_blocking_flag(self) -> Self:
        if self.outcome is SafetyCheckOutcome.BLOCK and not self.blocking:
            raise ValueError("blocked safety checks must be blocking")
        if self.outcome is SafetyCheckOutcome.PASS and self.blocking:
            raise ValueError("passing safety checks cannot be blocking")
        return self


class ClinicianAuthorization(ContractModel):
    """Immutable authorization decision bound to one exact artifact version."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=False,
        frozen=True,
    )

    authorization_id: str = Field(default_factory=lambda: new_id("AUTH"))
    artifact_type: CareArtifactType
    artifact_id: NonEmptyString
    artifact_version: int = Field(ge=1)
    artifact_content_hash: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    patient_ref: NonEmptyString
    encounter_id: NonEmptyString
    clinician_identity: NonEmptyString
    clinician_role: NonEmptyString
    organization_id: NonEmptyString
    jurisdiction: NonEmptyString
    verified_authorization_scope: list[NonEmptyString] = Field(min_length=1)
    safety_check_ids: list[NonEmptyString] = Field(default_factory=list)
    authentication_method: NonEmptyString
    authentication_strength: NonEmptyString
    decision: AuthorizationDecision
    rationale: NonEmptyString
    clinician_edits: list[NonEmptyString] = Field(default_factory=list)
    authorized_at: datetime
    expires_at: datetime
    supersedes_authorization_id: NonEmptyString | None = None
    revoked_at: datetime | None = None
    revocation_reason: NonEmptyString | None = None

    @field_validator("authorized_at", "expires_at", "revoked_at")
    @classmethod
    def normalize_authorization_times(cls, value: datetime | None) -> datetime | None:
        return _as_utc(value) if value is not None else None

    @field_validator("verified_authorization_scope", "safety_check_ids")
    @classmethod
    def authorization_references_are_unique(cls, values: list[str]) -> list[str]:
        return _unique(values)

    @model_validator(mode="after")
    def validate_authorization_lifecycle(self) -> Self:
        if self.expires_at <= self.authorized_at:
            raise ValueError("authorization expiry must be after authorization time")
        revoked = self.revoked_at is not None and self.revocation_reason is not None
        if (self.revoked_at is not None or self.revocation_reason is not None) and not revoked:
            raise ValueError("revocation requires revoked_at and revocation_reason")
        if self.decision is AuthorizationDecision.AUTHORIZE:
            required_scope = _REQUIRED_AUTHORIZATION_SCOPE[self.artifact_type]
            if required_scope not in self.verified_authorization_scope:
                raise ValueError(
                    f"authorization requires verified scope: {required_scope}"
                )
            if (
                self.artifact_type is CareArtifactType.MEDICATION_ORDER
                and not self.safety_check_ids
            ):
                raise ValueError(
                    "medication authorization requires completed safety-check references"
                )
        return self


class PatientInstructionArtifact(ContractModel):
    """Immutable patient rendering that can contain only previously authorized content."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=False,
        frozen=True,
    )

    instruction_id: str = Field(default_factory=lambda: new_id("PTINS"))
    version: int = Field(default=1, ge=1)
    patient_ref: NonEmptyString
    encounter_id: NonEmptyString
    authorization_ids: list[NonEmptyString] = Field(min_length=1)
    authorized_artifact_hashes: list[str] = Field(min_length=1)
    medication_order_ids: list[NonEmptyString] = Field(default_factory=list)
    clinician_display_name: NonEmptyString
    organization_name: NonEmptyString
    assessment_summary: NonEmptyString
    instructions: list[NonEmptyString] = Field(min_length=1)
    warning_signs: list[NonEmptyString] = Field(min_length=1)
    follow_up_route: NonEmptyString
    authorized_at: datetime
    valid_until: datetime
    rendered_at: datetime = Field(default_factory=utc_now)

    @field_validator("authorization_ids", "medication_order_ids")
    @classmethod
    def references_are_unique(cls, values: list[str]) -> list[str]:
        return _unique(values)

    @field_validator("authorized_artifact_hashes")
    @classmethod
    def validate_hashes(cls, values: list[str]) -> list[str]:
        unique = _unique(values)
        invalid_hash = any(
            len(value) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in value)
            for value in unique
        )
        if invalid_hash:
            raise ValueError("authorized artifact hashes must be 64 hexadecimal characters")
        return unique

    @field_validator("authorized_at", "valid_until", "rendered_at")
    @classmethod
    def normalize_instruction_times(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def validate_instruction_validity(self) -> Self:
        if self.valid_until <= self.authorized_at:
            raise ValueError("patient instruction validity must follow authorization")
        if self.rendered_at < self.authorized_at:
            raise ValueError("patient instructions cannot predate authorization")
        return self


class DeliveryReceipt(ContractModel):
    receipt_id: str = Field(default_factory=lambda: new_id("DLV"), frozen=True)
    instruction_id: NonEmptyString
    instruction_version: int = Field(ge=1)
    artifact_content_hash: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    recipient_ref: NonEmptyString
    channel: NonEmptyString
    status: DeliveryStatus
    attempted_at: datetime = Field(default_factory=utc_now)
    delivered_at: datetime | None = None
    acknowledged_at: datetime | None = None
    failure_reason: NonEmptyString | None = None

    @field_validator("attempted_at", "delivered_at", "acknowledged_at")
    @classmethod
    def normalize_delivery_times(cls, value: datetime | None) -> datetime | None:
        return _as_utc(value) if value is not None else None

    @model_validator(mode="after")
    def validate_delivery_lifecycle(self) -> Self:
        if self.status in {DeliveryStatus.DELIVERED, DeliveryStatus.ACKNOWLEDGED}:
            if self.delivered_at is None:
                raise ValueError("delivered artifacts require delivered_at")
        if self.status is DeliveryStatus.ACKNOWLEDGED:
            if self.acknowledged_at is None:
                raise ValueError("acknowledged artifacts require acknowledged_at")
            if self.delivered_at is not None and self.acknowledged_at < self.delivered_at:
                raise ValueError("acknowledgement cannot predate delivery")
        if self.status is DeliveryStatus.FAILED and self.failure_reason is None:
            raise ValueError("failed delivery requires a reason")
        return self


class FollowUpTask(ContractModel):
    task_id: str = Field(default_factory=lambda: new_id("FUP"), frozen=True)
    patient_ref: NonEmptyString
    encounter_id: NonEmptyString
    instruction_id: NonEmptyString
    reason: NonEmptyString
    status: FollowUpStatus = FollowUpStatus.OPEN
    assigned_role: NonEmptyString
    created_at: datetime = Field(default_factory=utc_now)
    due_at: datetime
    completed_at: datetime | None = None

    @field_validator("created_at", "due_at", "completed_at")
    @classmethod
    def normalize_follow_up_times(cls, value: datetime | None) -> datetime | None:
        return _as_utc(value) if value is not None else None

    @model_validator(mode="after")
    def validate_follow_up_lifecycle(self) -> Self:
        if self.due_at <= self.created_at:
            raise ValueError("follow-up due time must follow creation")
        if (self.status is FollowUpStatus.COMPLETED) != (self.completed_at is not None):
            raise ValueError("completed follow-up requires completed_at")
        return self


class QueueAssignment(ContractModel):
    assignment_id: str = Field(default_factory=lambda: new_id("QUE"), frozen=True)
    encounter_id: NonEmptyString
    triage_level: TriageLevel
    status: QueueStatus = QueueStatus.UNASSIGNED
    required_role: NonEmptyString
    jurisdiction: NonEmptyString
    assigned_clinician_identity: NonEmptyString | None = None
    created_at: datetime = Field(default_factory=utc_now)
    service_level_due_at: datetime

    @field_validator("created_at", "service_level_due_at")
    @classmethod
    def normalize_queue_times(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def validate_assignment(self) -> Self:
        if self.service_level_due_at <= self.created_at:
            raise ValueError("service-level deadline must follow queue creation")
        needs_clinician = self.status in {
            QueueStatus.ASSIGNED,
            QueueStatus.IN_REVIEW,
            QueueStatus.COMPLETED,
        }
        if needs_clinician != (self.assigned_clinician_identity is not None):
            raise ValueError("assigned queue status requires a clinician identity")
        return self


class SafetyEvent(ContractModel):
    event_id: str = Field(default_factory=lambda: new_id("SEV"), frozen=True)
    encounter_id: NonEmptyString
    category: NonEmptyString
    severity: Severity
    summary: NonEmptyString
    artifact_ids: list[NonEmptyString] = Field(default_factory=list)
    detected_at: datetime = Field(default_factory=utc_now)
    requires_delivery_halt: bool = False

    @field_validator("artifact_ids")
    @classmethod
    def artifact_references_are_unique(cls, values: list[str]) -> list[str]:
        return _unique(values)

    @field_validator("detected_at")
    @classmethod
    def normalize_detected_at(cls, value: datetime) -> datetime:
        return _as_utc(value)


_TERMINAL_STATUSES = {
    CareArtifactStatus.REJECTED,
    CareArtifactStatus.EXPIRED,
    CareArtifactStatus.REVOKED,
    CareArtifactStatus.ACKNOWLEDGED,
}

_AUTHORIZATION_STATUSES = {
    CareArtifactType.CLINICAL_ASSESSMENT: {
        CareArtifactStatus.ANALYSIS_APPROVED,
    },
    CareArtifactType.CARE_PLAN: {CareArtifactStatus.CARE_PLAN_AUTHORIZED},
    CareArtifactType.MEDICATION_ORDER: {
        CareArtifactStatus.PRESCRIPTION_AUTHORIZED,
    },
    CareArtifactType.PATIENT_INSTRUCTIONS: {
        CareArtifactStatus.CARE_PLAN_AUTHORIZED,
        CareArtifactStatus.PRESCRIPTION_AUTHORIZED,
    },
}

_REQUIRED_AUTHORIZATION_SCOPE = {
    CareArtifactType.CLINICAL_ASSESSMENT: "approve_clinical_assessment",
    CareArtifactType.CARE_PLAN: "authorize_care_plan",
    CareArtifactType.MEDICATION_ORDER: "authorize_medication_order",
    CareArtifactType.PATIENT_INSTRUCTIONS: "release_patient_instructions",
}


def validate_care_transition(
    artifact_type: CareArtifactType,
    current: CareArtifactStatus,
    target: CareArtifactStatus,
) -> None:
    """Reject lifecycle shortcuts, including delivery before authorization."""
    _ensure_status_allowed(artifact_type, current)
    _ensure_status_allowed(artifact_type, target)
    if current in _TERMINAL_STATUSES:
        raise ValueError(f"terminal care status cannot transition: {current.value}")

    authorization_statuses = _AUTHORIZATION_STATUSES[artifact_type]
    allowed: dict[CareArtifactStatus, set[CareArtifactStatus]] = {
        CareArtifactStatus.DRAFT_READY: {CareArtifactStatus.WAITING_FOR_CLINICIAN},
        CareArtifactStatus.WAITING_FOR_CLINICIAN: {
            CareArtifactStatus.CHANGES_REQUESTED,
            CareArtifactStatus.REJECTED,
            *authorization_statuses,
        },
        CareArtifactStatus.CHANGES_REQUESTED: {CareArtifactStatus.DRAFT_READY},
        CareArtifactStatus.ANALYSIS_APPROVED: {
            CareArtifactStatus.EXPIRED,
            CareArtifactStatus.REVOKED,
        },
        CareArtifactStatus.CARE_PLAN_AUTHORIZED: {
            CareArtifactStatus.DELIVERED,
            CareArtifactStatus.EXPIRED,
            CareArtifactStatus.REVOKED,
        },
        CareArtifactStatus.PRESCRIPTION_AUTHORIZED: {
            CareArtifactStatus.DELIVERED,
            CareArtifactStatus.EXPIRED,
            CareArtifactStatus.REVOKED,
        },
        CareArtifactStatus.DELIVERED: {
            CareArtifactStatus.ACKNOWLEDGED,
            CareArtifactStatus.REVOKED,
        },
    }
    if target not in allowed.get(current, set()):
        raise ValueError(
            f"invalid {artifact_type.value} transition: {current.value} -> {target.value}"
        )


def artifact_content_hash(artifact: ContractModel) -> str:
    """Return a deterministic SHA-256 digest for exact-version authorization."""
    payload = json.dumps(
        artifact.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _ensure_status_allowed(
    artifact_type: CareArtifactType, status: CareArtifactStatus
) -> None:
    shared = {
        CareArtifactStatus.DRAFT_READY,
        CareArtifactStatus.WAITING_FOR_CLINICIAN,
        CareArtifactStatus.CHANGES_REQUESTED,
        CareArtifactStatus.REJECTED,
        CareArtifactStatus.EXPIRED,
        CareArtifactStatus.REVOKED,
    }
    allowed = set(shared)
    allowed.update(_AUTHORIZATION_STATUSES[artifact_type])
    if artifact_type in {
        CareArtifactType.CARE_PLAN,
        CareArtifactType.MEDICATION_ORDER,
        CareArtifactType.PATIENT_INSTRUCTIONS,
    }:
        allowed.update(
            {CareArtifactStatus.DELIVERED, CareArtifactStatus.ACKNOWLEDGED}
        )
    if status not in allowed:
        raise ValueError(
            f"{status.value} is not valid for {artifact_type.value} artifacts"
        )


def _ensure_draft_status(status: CareArtifactStatus) -> None:
    allowed = {
        CareArtifactStatus.DRAFT_READY,
        CareArtifactStatus.WAITING_FOR_CLINICIAN,
        CareArtifactStatus.CHANGES_REQUESTED,
        CareArtifactStatus.REJECTED,
        CareArtifactStatus.EXPIRED,
        CareArtifactStatus.REVOKED,
    }
    if status not in allowed:
        raise ValueError("a draft cannot represent an authorized or delivered artifact")


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
