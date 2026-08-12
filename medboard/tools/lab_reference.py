"""Deterministic laboratory reference checks used before model explanation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from medboard.models import ContractModel, LabObservation


class LabStatus(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    UNKNOWN = "unknown"
    UNIT_MISMATCH = "unit_mismatch"
    INVALID_VALUE = "invalid_value"


class LabAssessment(ContractModel):
    name: str
    value: float | int | str
    unit: str | None
    status: LabStatus
    reference_range: str | None = None
    warning: str | None = None


@dataclass(frozen=True, slots=True)
class ReferenceRange:
    low: float
    high: float
    unit: str


class LabReferenceTool:
    """Evaluate a deliberately small, explicit set of demonstration lab ranges."""

    _ranges = {
        "hemoglobin": ReferenceRange(12.0, 16.0, "g/dL"),
        "hemoglobin_male": ReferenceRange(13.5, 17.5, "g/dL"),
        "mcv": ReferenceRange(80.0, 100.0, "fL"),
        "ferritin": ReferenceRange(15.0, 150.0, "ng/mL"),
        "wbc": ReferenceRange(4_000.0, 11_000.0, "/uL"),
        "platelets": ReferenceRange(150_000.0, 450_000.0, "/uL"),
    }
    _aliases = {
        "hb": "hemoglobin",
        "hgb": "hemoglobin",
        "mean corpuscular volume": "mcv",
        "white blood cells": "wbc",
        "white blood cell count": "wbc",
    }

    def assess(
        self, observation: LabObservation, biological_sex: str | None = None
    ) -> LabAssessment:
        normalized_name = observation.name.casefold()
        lookup_name = self._aliases.get(normalized_name, normalized_name)
        if lookup_name == "hemoglobin" and _is_male(biological_sex):
            lookup_name = "hemoglobin_male"
        reference = self._ranges.get(lookup_name)

        if reference is None:
            return LabAssessment(
                name=observation.name,
                value=observation.value,
                unit=observation.unit,
                status=LabStatus.UNKNOWN,
                warning="No configured reference range; clinician interpretation is required.",
            )
        if observation.unit is None or observation.unit.casefold() != reference.unit.casefold():
            supplied_unit = observation.unit or "not supplied"
            return LabAssessment(
                name=observation.name,
                value=observation.value,
                unit=observation.unit,
                status=LabStatus.UNIT_MISMATCH,
                reference_range=_format_range(reference),
                warning=(
                    f"Expected unit {reference.unit}; received {supplied_unit}. "
                    "No range comparison was performed."
                ),
            )
        if not isinstance(observation.value, int | float):
            return LabAssessment(
                name=observation.name,
                value=observation.value,
                unit=observation.unit,
                status=LabStatus.INVALID_VALUE,
                reference_range=_format_range(reference),
                warning="A numeric value is required for deterministic range comparison.",
            )

        if observation.value < reference.low:
            status = LabStatus.LOW
        elif observation.value > reference.high:
            status = LabStatus.HIGH
        else:
            status = LabStatus.NORMAL
        return LabAssessment(
            name=observation.name,
            value=observation.value,
            unit=observation.unit,
            status=status,
            reference_range=_format_range(reference),
        )


def _format_range(reference: ReferenceRange) -> str:
    return f"{reference.low:g}-{reference.high:g} {reference.unit}"


def _is_male(biological_sex: str | None) -> bool:
    return bool(biological_sex and biological_sex.casefold() in {"male", "m"})
