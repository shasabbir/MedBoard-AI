"""Unit tests for explicit disagreement detection."""

from medboard.models import DifferentialDiagnosis, SpecialistOpinion
from medboard.tools.contradictions import detect_specialist_contradictions


def test_only_explicit_specialist_challenges_create_contradictions() -> None:
    diagnoses = [
        DifferentialDiagnosis(
            hypothesis_id="HYP-ONE",
            hypothesis="First consideration",
            confidence=0.6,
        ),
        DifferentialDiagnosis(
            hypothesis_id="HYP-TWO",
            hypothesis="Second consideration",
            confidence=0.4,
        ),
    ]
    opinion = SpecialistOpinion(
        opinion_id="OPN-ONE",
        specialist="cardiology",
        assessment="The first consideration is not well supported.",
        challenged_hypotheses=["HYP-ONE"],
        confidence=0.7,
    )

    contradictions = detect_specialist_contradictions(diagnoses, [opinion])

    assert len(contradictions) == 1
    assert contradictions[0].topic == "First consideration"
    assert contradictions[0].hypothesis_id == "HYP-ONE"
