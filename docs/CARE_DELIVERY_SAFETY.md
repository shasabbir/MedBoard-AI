# Clinician-supervised care delivery safety foundation

Status: **Phase 11 implemented; care delivery remains disabled.**

This document defines the safety boundary for evolving MedBoard from an educational clinical
reasoning board toward a clinician-supervised remote consultation service. It is an engineering
control document, not evidence of clinical validation, regulatory clearance, or readiness for
real patients.

## Current enforced boundary

- `CARE_DELIVERY_ENABLED` defaults to `false` and configuration rejects `true`.
- The existing human-review action approves an educational report only.
- The current application does not authenticate clinicians, verify privileges, create treatment
  plans, authorize medication orders, or deliver patient instructions.
- Future care artifacts use contracts in `medboard/care_models.py`; they are deliberately not
  connected to `MedicalCaseState` or the LangGraph workflow.
- Development continues with synthetic or properly governed de-identified data.

## Intended use under development

MedBoard is intended to assist qualified remote-care teams with intake, red-flag prioritization,
case synthesis, evidence-linked draft preparation, documentation, patient education, and
follow-up. It may eventually prepare care-plan and medication-order drafts inside an authenticated
clinician workspace. A qualified, appropriately privileged clinician remains responsible for
reviewing, editing, and authorizing the exact artifact version released to a patient.

The implementation must configure and approve the following before a pilot:

- supported patient populations, conditions, languages, and encounter types;
- explicit exclusions and situations requiring examination or transfer;
- operating organization and deployment jurisdiction;
- staffed escalation destinations and expected response times;
- clinician/pharmacist qualifications and authorization scopes;
- approved clinical knowledge, medication terminology, and rule versions;
- patient consent, privacy, retention, incident, rollback, and recall policies.

## Excluded use

MedBoard must not:

- act as an autonomous doctor, prescriber, dispenser, or emergency service;
- release patient-specific diagnosis, treatment, or medicine without verified authorization;
- infer missing dose, unit, route, duration, identity, allergy, or dose-relevant context;
- convert the current optional reviewer name into a claim of authentication;
- expose internal drafts, confidence scores, or agent debate directly to patients;
- bulk approve cases or treat a large queue as evidence that clinical capacity is adequate;
- be used for real patients until the live-use gate in `plan.md` Section 97 is signed off.

When no clinician is available, the allowed behavior is limited to information collection,
deterministic red-flag screening, pre-approved general education, and routing to a staffed remote
clinician or local emergency pathway. Patient-specific prescribing stays unavailable.

## Role and authorization matrix

Permissions are backend policy decisions. Labels supplied by a browser or model are never trusted.

| Role | May prepare/update | May authorize | May deliver/revoke |
| --- | --- | --- | --- |
| Patient/intake user | Own intake, consent, acknowledgement | Nothing | Receive own authorized artifact |
| Intake staff | Demographics and intake evidence within assignment | Nothing | Nothing |
| Reviewing clinician | Assessment and care-plan draft | Clinical assessment/care plan when privileged | Request delivery; revoke within policy |
| Prescriber | Medication draft and clinician edits | Medication order when privileged in jurisdiction | Request delivery; revoke within policy |
| Pharmacist/second reviewer | Medication safety review | Countersign when policy requires | Record dispense/review disposition later |
| Clinical administrator | Assignment and policy configuration | No patient-specific clinical content | Operational recall under controlled policy |
| Clinical safety officer | Safety-event investigation | No routine patient-specific authorization | Halt delivery, revoke/recall under incident policy |
| AI agent/model | Draft structured artifacts only | Nothing | Nothing |

Phase 12 must replace these conceptual roles with authenticated workforce identities, organization
membership, verified privileges, jurisdiction, step-up authentication, and deny-by-default policy.

## Artifact lifecycle

Each artifact has its own lifecycle. “Approved” is not a universal state.

| Artifact | Review path | Authorized state |
| --- | --- | --- |
| Clinical assessment | draft → waiting → changes/reject/approve | `ANALYSIS_APPROVED` |
| Care plan | draft → waiting → changes/reject/authorize | `CARE_PLAN_AUTHORIZED` |
| Medication order | draft → safety checks → waiting → changes/reject/authorize | `PRESCRIPTION_AUTHORIZED` |
| Patient instructions | render from authorized structured artifacts only | delivery/acknowledgement tracked separately |

`validate_care_transition()` rejects shortcuts such as draft-to-authorized or
draft-to-delivered. Terminal rejected, expired, revoked, and acknowledged artifacts cannot move
back into active delivery. Draft models reject authorized or delivered statuses. A
`ClinicianAuthorization` is immutable and binds a verified scope to the artifact ID, version, and
SHA-256 content hash. Medication authorization also requires safety-check references.

Any clinical-content change creates a new content hash and requires a new authorization. Future
services must enforce the lifecycle transactionally and idempotently; the model contracts alone
are not an authorization service.

## Initial hazard analysis

| Hazard | Potential harm | Phase 11 control | Required later verification |
| --- | --- | --- | --- |
| Existing report approval mistaken for prescribing authority | Unreviewed treatment reaches patient | Care delivery hard-disabled; separate status/types | Authorization integration and bypass tests |
| Unverified reviewer identity or scope | Unauthorized clinical decision | Immutable authorization contract requires identity, role, organization, jurisdiction, scope, authentication metadata | Identity-provider and privilege-policy tests |
| Draft changed after approval | Patient receives content clinician did not review | Version and deterministic SHA-256 content hash | Transactional signing/rendering concurrency tests |
| Missing medication field or unit | Wrong medicine administration | Required structured dose/formulation/route/frequency/duration fields; no defaults | Medication-rule golden tests and clinician review |
| Blocking safety result hidden | Contraindication/allergy/interaction harm | Safety check enforces consistent blocking outcome; medication authorization requires check references | Rule engine and fail-closed integration tests |
| Patient artifact contains model-added content | Unapproved diagnosis or instruction | Immutable patient artifact contract linked to authorization hashes | Non-generative renderer equivalence tests |
| Emergency waits in routine queue | Delayed urgent care | Intended-use boundary requires deterministic bypass | End-to-end escalation and downtime drills |
| Queue exceeds clinician capacity | Delayed or superficial review | Scale goal explicitly bounded by workload and service levels | Load, staffing, ageing, and redirect tests |
| Delivery or acknowledgement is assumed | Missed treatment/follow-up | Separate delivery and follow-up contracts | Channel failure, retry, acknowledgement, overdue tests |
| Stale/poisoned clinical source | Unsafe recommendation | Source-version references required in drafts/checks | Approved-source ingestion, signing, update/rollback tests |
| Feature enabled before controls exist | Premature real-patient deployment | Settings reject `CARE_DELIVERY_ENABLED=true` | Release-gate test remains blocking through later phases |

## Phase 11 verification

Automated tests prove that:

- care delivery defaults to disabled and cannot be enabled;
- verified identity and consent records require auditable, timezone-aware data;
- care and medication drafts cannot claim authorization;
- required medication fields and paired maximum-dose value/unit are validated;
- blocking safety checks cannot be marked non-blocking;
- authorization requires the correct verified scope and medication safety-check references;
- signed records and patient instruction artifacts are immutable and hash-bound;
- content changes produce a different hash;
- delivery, acknowledgement, queue, and follow-up timestamps cannot contradict lifecycle order;
- draft-to-authorized/delivered and terminal-state transitions fail.

## Next implementation boundary

Phase 12 may connect authenticated identity and authorization policy infrastructure, but treatment
generation and patient delivery must remain disabled. The feature gate should not become
enableable until Phases 12–18 and the live-use checklist are complete.
