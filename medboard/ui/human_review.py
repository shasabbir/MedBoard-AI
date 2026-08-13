"""Human review command controls."""

from __future__ import annotations

import json

import streamlit as st
from pydantic import ValidationError

from medboard.graph.state import MedicalCaseSnapshot
from medboard.models import HumanReviewCommand


def render_human_review(snapshot: MedicalCaseSnapshot) -> HumanReviewCommand | None:
    st.subheader("Human / clinician review")
    if snapshot.human_review.status.value != "waiting_for_human":
        st.info(f"Current human-review status: {snapshot.human_review.status.value}")
        return None
    st.warning(
        "This experimental decision-support output requires a qualified healthcare "
        "professional's review."
    )
    if snapshot.triage_result:
        st.metric("Triage level", snapshot.triage_result.triage_level.value.upper())
    action = st.selectbox(
        "Decision",
        [
            "approve",
            "reject",
            "add_information",
            "request_revision",
            "request_specialist",
            "retry_failed_agent",
        ],
        key="human_action",
    )
    reviewer = st.text_input("Reviewer identifier (optional)")
    feedback = st.text_area("Feedback")
    requested_specialist = None
    failed_agent = None
    added_information: dict[str, object] = {}
    if action == "request_specialist":
        requested_specialist = st.selectbox(
            "Specialist",
            ["cardiology", "neurology", "infectious_disease"],
        )
    elif action == "retry_failed_agent":
        failed = [
            error.agent
            for error in snapshot.errors
            if error.agent and not error.resolved
        ]
        failed_agent = st.selectbox("Failed agent", failed or ["differential"])
    elif action == "add_information":
        raw = st.text_area(
            "Additional information JSON",
            value='{"blood_pressure": "120/80 mmHg"}',
        )
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                added_information = parsed
        except json.JSONDecodeError:
            st.error("Additional information must be valid JSON.")
    if not st.button("Submit human decision", type="primary"):
        return None
    try:
        return HumanReviewCommand(
            action=action,
            feedback=feedback or None,
            reviewer=reviewer or None,
            added_information=added_information,
            requested_specialist=requested_specialist,
            failed_agent=failed_agent,
        )
    except ValidationError as exc:
        st.error(f"Human decision validation failed: {exc}")
        return None
