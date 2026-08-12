"""Tests for deterministic laboratory range checks."""

from medboard.models import LabObservation
from medboard.tools.lab_reference import LabReferenceTool, LabStatus


def test_lab_tool_flags_low_value_with_matching_unit() -> None:
    result = LabReferenceTool().assess(
        LabObservation(name="hemoglobin", value=8.7, unit="g/dL"),
        biological_sex="female",
    )

    assert result.status is LabStatus.LOW
    assert result.reference_range == "12-16 g/dL"
    assert result.warning is None


def test_lab_tool_never_assumes_missing_unit() -> None:
    result = LabReferenceTool().assess(
        LabObservation(name="ferritin", value=6),
    )

    assert result.status is LabStatus.UNIT_MISMATCH
    assert "No range comparison" in (result.warning or "")


def test_lab_tool_reports_unknown_analyte() -> None:
    result = LabReferenceTool().assess(
        LabObservation(name="experimental marker", value=3, unit="units"),
    )

    assert result.status is LabStatus.UNKNOWN
