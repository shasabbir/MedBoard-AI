"""Human review command controls."""

from __future__ import annotations

import json

import streamlit as st
from pydantic import ValidationError

from medboard.graph.state import MedicalCaseSnapshot
from medboard.models import HumanReviewCommand


def render_human_review(snapshot: MedicalCaseSnapshot) -> HumanReviewCommand | None:
    st.subheader("Human / clinician review")
    unresolved_errors = [error for error in snapshot.errors if not error.resolved]
    if unresolved_errors:
        details = "; ".join(
            f"{error.agent or 'workflow'} ({error.error_type}: {error.message})"
            for error in unresolved_errors
        )
        st.error(
            "Unresolved agent failure(s) occurred during this run: "
            f"{details}. Use 'retry_failed_agent' below to recover."
        )
    if snapshot.human_review.status.value != "waiting_for_human":
        st.info(f"Current human-review status: {snapshot.human_review.status.value}")
        return None
    st.warning(
        "This experimental decision-support output requires a qualified healthcare "
        "professional's review."
    )
    if snapshot.triage_result is None:
        st.warning(
            "Risk / triage assessment is missing. Retry the 'risk' agent via "
            "'retry_failed_agent' before approving so the final report can be generated."
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
        failed = list(
            dict.fromkeys(
                error.agent
                for error in unresolved_errors
                if error.agent
            )
        )
        failed_agent = st.selectbox("Failed agent", failed or ["differential"])
    elif action == "add_information":
        raw = st.text_area(
            "Additional information JSON",
            value='{"blood_pressure": "120/80 mmHg"}',
            help=(
                "Use symptoms, history, medications, allergies, or laboratory_values "
                "for targeted intake-agent reanalysis; other keys are retained as "
                "human evidence."
            ),
        )
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                added_information = parsed
            else:
                st.error("Additional information must be a JSON object.")
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
