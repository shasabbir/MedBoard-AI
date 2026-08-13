"""Structured final report presentation."""

import streamlit as st

from medboard.graph.state import MedicalCaseSnapshot


def render_report(snapshot: MedicalCaseSnapshot) -> None:
    st.subheader("Final report")
    report = snapshot.final_report
    if report is None:
        st.info("A final report is generated only after explicit human approval.")
        return
    st.markdown("### MedBoard AI Case Review")
    st.markdown("#### Case summary")
    st.write(report.case_summary)
    st.markdown("#### Differential considerations")
    for item in report.differential_considerations:
        st.write(f"- {item.hypothesis} — AI reasoning confidence {item.confidence:.2f}")
    st.markdown("#### Risk / triage assessment")
    st.warning(
        f"{report.triage.triage_level.value.upper()}: "
        f"{report.triage.recommended_escalation}"
    )
    if report.limitations:
        st.markdown("#### Analysis limitations")
        for limitation in report.limitations:
            st.warning(limitation)
    st.markdown("#### Disclaimer")
    st.error(report.disclaimer)
