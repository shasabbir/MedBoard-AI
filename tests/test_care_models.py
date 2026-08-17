"""Safety-foundation tests for future clinician-supervised care delivery."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from medboard.care_models import (
    AuthorizationDecision,
    CareArtifactStatus,
    CareArtifactType,
    CarePlanDraft,
    ClinicianAuthorization,
    ClinicalAssessment,
    ConsentRecord,
    ConsentStatus,
    DeliveryReceipt,
    DeliveryStatus,
    FollowUpStatus,
    FollowUpTask,
    MedicationOrderDraft,
    MedicationSafetyCheck,
    PatientIdentityRef,
    PatientInstructionArtifact,
    QueueAssignment,
    QueueStatus,
    SafetyCheckOutcome,
    TreatmentOption,
    TreatmentType,
    artifact_content_hash,
    validate_care_transition,
)
from medboard.models import TriageLevel

NOW = datetime(2026, 8, 17, 8, 0, tzinfo=UTC)


def _treatment_option() -> TreatmentOption:
    return TreatmentOption(
        treatment_type=TreatmentType.NON_DRUG,
        indication="Clinician review of a synthetic condition",
        description="Draft non-drug care option for clinician review.",
        evidence_ids=["EV-001"],
        source_version_ids=["GUIDE-SYNTHETIC-1"],
    )


def _care_plan() -> CarePlanDraft:
    return CarePlanDraft(
        care_plan_id="PLAN-001",
        patient_ref="PATIENT-001",
        encounter_id="ENC-001",
        assessment_id="ASM-001",
        options=[_treatment_option()],
        follow_up_window="Within 24 hours",
        warning_signs=["Seek urgent assessment if symptoms worsen."],
        created_at=NOW,
    )


def _medication_draft(**overrides: object) -> MedicationOrderDraft:
    values: dict[str, object] = {
        "medication_order_id": "MEDORD-001",
        "patient_ref": "PATIENT-001",
        "encounter_id": "ENC-001",
        "assessment_id": "ASM-001",
        "care_plan_id": "PLAN-001",
        "indication": "Synthetic indication for contract testing",
        "evidence_ids": ["EV-001"],
        "normalized_medicine_id": "TEST-MEDICINE-ID",
        "medicine_name": "Synthetic test medicine",
        "formulation": "Test formulation",
        "strength": "Test strength",
        "dose_value": Decimal("1"),
        "dose_unit": "test unit",
        "route": "test route",
        "frequency": "test frequency",
        "timing": "test timing",
        "duration_value": 1,
        "duration_unit": "day",
        "quantity": Decimal("1"),
        "start_instructions": "Start only after clinician authorization.",
        "stop_instructions": "Stop according to the authorized plan.",
        "monitoring_requirements": ["Clinician-defined monitoring"],
        "follow_up_requirements": ["Clinician follow-up"],
        "source_version_ids": ["FORMULARY-SYNTHETIC-1"],
        "drafted_by_agent": "treatment_planning",
        "created_at": NOW,
    }
    values.update(overrides)
    return MedicationOrderDraft(**values)


def test_verified_patient_identity_requires_auditable_details() -> None:
    with pytest.raises(ValidationError, match="verification_method"):
        PatientIdentityRef(patient_ref="PATIENT-001", verified=True)

    identity = PatientIdentityRef(
        patient_ref="PATIENT-001",
        verified=True,
        verification_method="organization identity provider",
        verified_at=NOW,
    )

    assert identity.verified_at == NOW


def test_consent_lifecycle_is_time_aware_and_consistent() -> None:
    consent = ConsentRecord(
        patient_ref="PATIENT-001",
        purpose="Synthetic care-delivery contract test",
        status=ConsentStatus.GRANTED,
        granted_at=NOW,
        expires_at=NOW + timedelta(days=1),
    )

    assert consent.status is ConsentStatus.GRANTED
    with pytest.raises(ValidationError, match="withdrawn consent"):
        ConsentRecord(
            patient_ref="PATIENT-001",
            purpose="Synthetic test",
            status=ConsentStatus.WITHDRAWN,
            granted_at=NOW,
        )


def test_assessment_status_cannot_claim_prescription_authorization() -> None:
    with pytest.raises(ValidationError, match="not valid for clinical_assessment"):
        ClinicalAssessment(
            encounter_id="ENC-001",
            evidence_ids=["EV-001"],
            status=CareArtifactStatus.PRESCRIPTION_AUTHORIZED,
            created_at=NOW,
        )


def test_care_and_medication_drafts_cannot_claim_authorization() -> None:
    with pytest.raises(ValidationError, match="draft cannot represent"):
        CarePlanDraft.model_validate(
            {
                **_care_plan().model_dump(),
                "status": CareArtifactStatus.CARE_PLAN_AUTHORIZED,
            }
        )

    with pytest.raises(ValidationError, match="draft cannot represent"):
        _medication_draft(status=CareArtifactStatus.PRESCRIPTION_AUTHORIZED)


def test_medication_draft_requires_paired_maximum_dose_value_and_unit() -> None:
    with pytest.raises(ValidationError, match="must be supplied together"):
        _medication_draft(maximum_dose_value=Decimal("2"))


def test_safety_check_cannot_hide_a_blocking_outcome() -> None:
    with pytest.raises(ValidationError, match="must be blocking"):
        MedicationSafetyCheck(
            medication_order_id="MEDORD-001",
            medication_order_version=1,
            check_type="synthetic allergy check",
            outcome=SafetyCheckOutcome.BLOCK,
            message="Synthetic blocking result",
            source_id="RULE-001",
            source_version="1",
            blocking=False,
            checked_at=NOW,
        )


def test_authorization_is_hash_bound_time_limited_and_immutable() -> None:
    plan = _care_plan()
    authorization = ClinicianAuthorization(
        artifact_type=CareArtifactType.CARE_PLAN,
        artifact_id=plan.care_plan_id,
        artifact_version=plan.version,
        artifact_content_hash=artifact_content_hash(plan),
        patient_ref=plan.patient_ref,
        encounter_id=plan.encounter_id,
        clinician_identity="WORKFORCE-001",
        clinician_role="licensed prescriber",
        organization_id="ORG-001",
        jurisdiction="TEST-JURISDICTION",
        verified_authorization_scope=["authorize_care_plan"],
        authentication_method="test identity provider",
        authentication_strength="multi-factor",
        decision=AuthorizationDecision.AUTHORIZE,
        rationale="Synthetic authorization contract test",
        authorized_at=NOW,
        expires_at=NOW + timedelta(hours=1),
    )

    assert len(authorization.artifact_content_hash) == 64
    with pytest.raises(ValidationError, match="Instance is frozen"):
        authorization.rationale = "Changed after signing"


def test_medication_authorization_requires_scope_and_safety_checks() -> None:
    draft = _medication_draft()
    common: dict[str, object] = {
        "artifact_type": CareArtifactType.MEDICATION_ORDER,
        "artifact_id": draft.medication_order_id,
        "artifact_version": draft.version,
        "artifact_content_hash": artifact_content_hash(draft),
        "patient_ref": draft.patient_ref,
        "encounter_id": draft.encounter_id,
        "clinician_identity": "WORKFORCE-001",
        "clinician_role": "licensed prescriber",
        "organization_id": "ORG-001",
        "jurisdiction": "TEST-JURISDICTION",
        "authentication_method": "test identity provider",
        "authentication_strength": "multi-factor",
        "decision": AuthorizationDecision.AUTHORIZE,
        "rationale": "Synthetic medication authorization contract test",
        "authorized_at": NOW,
        "expires_at": NOW + timedelta(hours=1),
    }

    with pytest.raises(ValidationError, match="verified scope"):
        ClinicianAuthorization(
            **common,
            verified_authorization_scope=["authorize_care_plan"],
            safety_check_ids=["SAFE-001"],
        )
    with pytest.raises(ValidationError, match="safety-check references"):
        ClinicianAuthorization(
            **common,
            verified_authorization_scope=["authorize_medication_order"],
        )


def test_artifact_hash_changes_when_clinical_content_changes() -> None:
    original = _care_plan()
    changed = original.model_copy(update={"follow_up_window": "Within 12 hours"})

    assert artifact_content_hash(original) != artifact_content_hash(changed)


def test_patient_instructions_require_authorized_hash_and_valid_time_window() -> None:
    with pytest.raises(ValidationError, match="64 hexadecimal"):
        PatientInstructionArtifact(
            patient_ref="PATIENT-001",
            encounter_id="ENC-001",
            authorization_ids=["AUTH-001"],
            authorized_artifact_hashes=["not-a-hash"],
            clinician_display_name="Synthetic clinician",
            organization_name="Synthetic organization",
            assessment_summary="Synthetic approved assessment",
            instructions=["Follow the clinician-authorized plan."],
            warning_signs=["Seek urgent help if symptoms worsen."],
            follow_up_route="Synthetic follow-up route",
            authorized_at=NOW,
            valid_until=NOW + timedelta(hours=1),
            rendered_at=NOW,
        )


def test_delivery_acknowledgement_cannot_predate_delivery() -> None:
    with pytest.raises(ValidationError, match="cannot predate delivery"):
        DeliveryReceipt(
            instruction_id="PTINS-001",
            instruction_version=1,
            artifact_content_hash="a" * 64,
            recipient_ref="PATIENT-001",
            channel="test channel",
            status=DeliveryStatus.ACKNOWLEDGED,
            attempted_at=NOW,
            delivered_at=NOW + timedelta(minutes=2),
            acknowledged_at=NOW + timedelta(minutes=1),
        )


def test_follow_up_and_queue_deadlines_must_be_actionable() -> None:
    with pytest.raises(ValidationError, match="completed_at"):
        FollowUpTask(
            patient_ref="PATIENT-001",
            encounter_id="ENC-001",
            instruction_id="PTINS-001",
            reason="Synthetic follow-up",
            status=FollowUpStatus.COMPLETED,
            assigned_role="clinician",
            created_at=NOW,
            due_at=NOW + timedelta(hours=1),
        )

    with pytest.raises(ValidationError, match="requires a clinician identity"):
        QueueAssignment(
            encounter_id="ENC-001",
            triage_level=TriageLevel.URGENT,
            status=QueueStatus.ASSIGNED,
            required_role="clinician",
            jurisdiction="TEST-JURISDICTION",
            created_at=NOW,
            service_level_due_at=NOW + timedelta(minutes=15),
        )


def test_lifecycle_requires_review_before_authorization_or_delivery() -> None:
    with pytest.raises(ValueError, match="invalid medication_order transition"):
        validate_care_transition(
            CareArtifactType.MEDICATION_ORDER,
            CareArtifactStatus.DRAFT_READY,
            CareArtifactStatus.PRESCRIPTION_AUTHORIZED,
        )

    validate_care_transition(
        CareArtifactType.MEDICATION_ORDER,
        CareArtifactStatus.DRAFT_READY,
        CareArtifactStatus.WAITING_FOR_CLINICIAN,
    )
    validate_care_transition(
        CareArtifactType.MEDICATION_ORDER,
        CareArtifactStatus.WAITING_FOR_CLINICIAN,
        CareArtifactStatus.PRESCRIPTION_AUTHORIZED,
    )
    validate_care_transition(
        CareArtifactType.MEDICATION_ORDER,
        CareArtifactStatus.PRESCRIPTION_AUTHORIZED,
        CareArtifactStatus.DELIVERED,
    )

    with pytest.raises(ValueError, match="terminal care status"):
        validate_care_transition(
            CareArtifactType.MEDICATION_ORDER,
            CareArtifactStatus.REVOKED,
            CareArtifactStatus.DELIVERED,
        )
