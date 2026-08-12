"""Synthetic demo and custom case input widgets."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from pydantic import ValidationError

from medboard.models import LabObservation, MedicalCaseInput


def render_case_input(demo_directory: Path) -> MedicalCaseInput | None:
    st.subheader("Case input")
    source = st.radio(
        "Input source",
        ["Bundled synthetic case", "Custom synthetic case"],
        horizontal=True,
    )
    if source == "Bundled synthetic case":
        files = sorted(demo_directory.glob("*.json"))
        if not files:
            st.error("No bundled demo cases were found.")
            return None
        labels = {path.stem.replace("_", " ").title(): path for path in files}
        selected = st.selectbox("Demo case", list(labels), key="demo_case")
        case = MedicalCaseInput.model_validate_json(
            labels[selected].read_text(encoding="utf-8")
        )
        st.caption(case.chief_complaint)
        return case

    with st.form("custom_case_form"):
        age = st.number_input("Age", min_value=0, max_value=125, value=None)
        sex = st.selectbox("Biological sex (if supplied)", ["", "female", "male"])
        chief_complaint = st.text_input("Chief complaint")
        narrative = st.text_area("Case narrative")
        symptoms = st.text_input("Symptoms (comma-separated)")
        history = st.text_area("History (one item per line)")
        medications = st.text_area("Medications (one item per line)")
        labs = st.text_area(
            "Laboratory JSON",
            value='[{"name": "hemoglobin", "value": 9.1, "unit": "g/dL"}]',
            help="JSON array; every laboratory value must include an explicit unit.",
        )
        submitted = st.form_submit_button("Validate custom case")
    if not submitted:
        return None
    try:
        lab_values = [LabObservation.model_validate(item) for item in json.loads(labs)]
        case = MedicalCaseInput(
            synthetic=True,
            age=int(age) if age is not None else None,
            biological_sex=sex or None,
            chief_complaint=chief_complaint,
            narrative=narrative,
            symptoms=_csv(symptoms),
            history=_lines(history),
            medications=_lines(medications),
            laboratory_values=lab_values,
        )
    except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
        st.error(f"Case validation failed: {exc}")
        return None
    st.success("Custom case is valid and ready to run.")
    return case


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _lines(value: str) -> list[str]:
    return [item.strip() for item in value.splitlines() if item.strip()]
