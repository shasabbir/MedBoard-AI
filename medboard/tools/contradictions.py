"""Detect explicit specialist challenges to differential hypotheses."""

from medboard.models import Contradiction, DifferentialDiagnosis, SpecialistOpinion


def detect_specialist_contradictions(
    diagnoses: list[DifferentialDiagnosis],
    opinions: list[SpecialistOpinion],
) -> list[Contradiction]:
    """Convert explicit challenge links into auditable unresolved disagreements."""
    diagnoses_by_id = {item.hypothesis_id: item for item in diagnoses}
    contradictions: list[Contradiction] = []
    for opinion in opinions:
        for hypothesis_id in opinion.challenged_hypotheses:
            diagnosis = diagnoses_by_id.get(hypothesis_id)
            if diagnosis is None:
                continue
            contradictions.append(
                Contradiction(
                    contradiction_id=f"CON-{opinion.opinion_id}-{hypothesis_id}",
                    topic=diagnosis.hypothesis,
                    agent_a=diagnosis.proposed_by,
                    agent_b=opinion.specialist,
                    hypothesis_id=hypothesis_id,
                )
            )
    return contradictions
