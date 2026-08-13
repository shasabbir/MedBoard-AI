"""Structured final report presentation."""

from urllib.parse import urlparse

import streamlit as st

from medboard.graph.state import MedicalCaseSnapshot


def _render_string_list(items: list[str], empty_message: str) -> None:
    if not items:
        st.caption(empty_message)
        return
    for item in items:
        st.write(f"- {item}")


def _is_web_url(value: str | None) -> bool:
    if value is None:
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def render_report(snapshot: MedicalCaseSnapshot) -> None:
    st.subheader("Final report")
    report = snapshot.final_report
    if report is None:
        st.info("A final report is generated only after explicit human approval.")
        return

    st.markdown("### MedBoard AI Case Review")
    st.markdown("#### Case summary")
    st.write(report.case_summary)

    st.markdown("#### Key findings")
    _render_string_list(report.key_findings, "No key findings were recorded.")

    st.markdown("#### Differential considerations")
    if not report.differential_considerations:
        st.caption("No differential considerations were recorded.")
    for index, item in enumerate(report.differential_considerations, start=1):
        st.write(
            f"{index}. {item.hypothesis} — AI reasoning confidence "
            f"{item.confidence:.2f}"
        )
        if item.supporting_evidence_ids:
            st.caption(
                "Supporting evidence: " + ", ".join(item.supporting_evidence_ids)
            )
        if item.contradicting_evidence_ids:
            st.caption(
                "Contradicting evidence: "
                + ", ".join(item.contradicting_evidence_ids)
            )
        if item.missing_evidence:
            st.caption("Still needed: " + "; ".join(item.missing_evidence))

    st.markdown("#### Specialist opinions")
    if not report.specialist_opinions:
        st.caption("No specialist opinions were requested.")
    for opinion in report.specialist_opinions:
        with st.expander(
            f"{opinion.specialist.replace('_', ' ').title()} "
            f"(confidence {opinion.confidence:.2f})"
        ):
            st.write(opinion.assessment)
            if opinion.critical_concerns:
                st.write("Critical concerns: " + "; ".join(opinion.critical_concerns))
            if opinion.required_information:
                st.write(
                    "Required information: " + "; ".join(opinion.required_information)
                )
            if opinion.evidence_ids:
                st.caption("Evidence references: " + ", ".join(opinion.evidence_ids))

    st.markdown("#### Retrieved evidence")
    if not report.retrieved_evidence:
        st.caption("No external evidence was retrieved.")
    for index, evidence in enumerate(report.retrieved_evidence, start=1):
        with st.expander(
            f"{index}. {evidence.document} — {evidence.section} "
            f"(similarity {evidence.similarity_score:.2f})"
        ):
            st.write(evidence.retrieved_text)
            st.caption(
                f"Source: {evidence.source} | Chunk: {evidence.chunk_id} | "
                f"Question: {evidence.question_id}"
            )
            if _is_web_url(evidence.source_url):
                st.link_button(
                    f"Open source {index}", evidence.source_url or "", type="secondary"
                )

    st.markdown("#### Areas of disagreement")
    if not report.disagreements:
        st.caption("No agent disagreements were recorded.")
    for disagreement in report.disagreements:
        status = "resolved" if disagreement.resolved else "unresolved"
        st.write(
            f"- {disagreement.topic}: {disagreement.agent_a} and "
            f"{disagreement.agent_b} ({status})"
        )
        if disagreement.resolution:
            st.caption(f"Resolution: {disagreement.resolution}")

    st.markdown("#### Missing information")
    if not report.missing_information:
        st.caption("No missing information requests remain.")
    for request in report.missing_information:
        st.write(f"- {request.information_needed}")
        st.caption(
            f"Requested by: {', '.join(request.requested_by)} | "
            f"Urgency: {request.urgency:.2f} | Utility: "
            f"{request.diagnostic_utility:.2f} | Reason: {request.reason}"
        )

    st.markdown("#### Risk / triage assessment")
    st.warning(
        f"{report.triage.triage_level.value.upper()}: "
        f"{report.triage.recommended_escalation}"
    )
    st.write(report.triage.reasoning)
    if report.triage.red_flags:
        st.write("Red flags: " + "; ".join(report.triage.red_flags))

    st.markdown("#### Suggested clinical review priorities")
    _render_string_list(
        report.review_priorities, "No additional review priorities were recorded."
    )

    st.markdown("#### Human review status")
    review = snapshot.human_review
    st.write(f"Status: {review.status.value.replace('_', ' ').title()}")
    if review.reviewer:
        st.write(f"Reviewer: {review.reviewer}")
    if review.feedback:
        st.write(f"Feedback: {review.feedback}")

    if report.limitations:
        st.markdown("#### Analysis limitations")
        for limitation in report.limitations:
            st.warning(limitation)

    st.markdown("#### Disclaimer")
    st.error(report.disclaimer)
