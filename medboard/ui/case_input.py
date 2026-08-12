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
        st.session_state.pop("validated_custom_case", None)
        files = sorted(demo_directory.glob("*.json"))
        if not files:
            st.error("No bundled demo cases were found.")
            return None
        labels = {path.stem.replace("_", " ").title(): path for path in files}
        selected = st.selectbox("Demo case", list(labels), key="demo_case")
        case = MedicalCaseInput.model_validate_json(
            labels[selected].read_text(encoding="utf-8")
        )
        _case_preview(case)
        return case

    with st.form("custom_case_form"):
        case_title = st.text_input("Case title", value="Custom synthetic review")
        age = st.number_input("Age", min_value=0, max_value=125, value=None)
        sex = st.selectbox("Biological sex (if supplied)", ["", "female", "male"])
        chief_complaint = st.text_input("Chief complaint")
        duration = st.text_input("Symptom duration")
        narrative = st.text_area("Case narrative")
        symptoms = st.text_input("Symptoms (comma-separated)")
        history = st.text_area("History (one item per line)")
        medications = st.text_area("Medications (one item per line)")
        allergies = st.text_area("Allergies (one item per line)")
        family_history = st.text_area("Family history (one item per line)")
        lifestyle = st.text_area("Lifestyle and exposures (one item per line)")
        labs = st.text_area(
            "Laboratory JSON",
            value='[{"name": "hemoglobin", "value": 9.1, "unit": "g/dL"}]',
            help="JSON array; every laboratory value must include an explicit unit.",
        )
        submitted = st.form_submit_button("Validate custom case")
    if not submitted:
        saved = st.session_state.get("validated_custom_case")
        return MedicalCaseInput.model_validate(saved) if saved else None
    try:
        lab_values = [LabObservation.model_validate(item) for item in json.loads(labs)]
        case = MedicalCaseInput(
            synthetic=True,
            age=int(age) if age is not None else None,
            biological_sex=sex or None,
            chief_complaint=chief_complaint,
            narrative="\n".join(
                item
                for item in [
                    case_title,
                    f"Symptom duration: {duration}" if duration else "",
                    narrative,
                    *(f"Family history: {item}" for item in _lines(family_history)),
                    *(f"Lifestyle/exposure: {item}" for item in _lines(lifestyle)),
                ]
                if item
            ),
            symptoms=_csv(symptoms),
            history=_lines(history),
            medications=_lines(medications),
            allergies=_lines(allergies),
            laboratory_values=lab_values,
        )
    except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
        st.session_state.pop("validated_custom_case", None)
        st.error(f"Case validation failed: {exc}")
        return None
    st.session_state.validated_custom_case = case.model_dump(mode="json")
    st.success("Custom case is valid and ready to run.")
    return case


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _lines(value: str) -> list[str]:
    return [item.strip() for item in value.splitlines() if item.strip()]


def _case_preview(case: MedicalCaseInput) -> None:
    st.markdown(f"**{case.chief_complaint}**")
    details = st.columns(3)
    details[0].caption(f"Age: {case.age if case.age is not None else 'Not supplied'}")
    details[1].caption(f"Sex: {case.biological_sex or 'Not supplied'}")
    details[2].caption(f"Laboratory results: {len(case.laboratory_values)}")
    st.caption("Symptoms: " + (", ".join(case.symptoms) or "None supplied"))
