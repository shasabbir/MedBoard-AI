"""Differential, disagreement, missing-information, and RAG panels."""

from __future__ import annotations

import streamlit as st

from medboard.graph.state import MedicalCaseSnapshot


def render_analysis(snapshot: MedicalCaseSnapshot) -> None:
    st.subheader("Differential considerations")
    st.caption("Confidence values are AI reasoning scores, not medical probabilities.")
    evidence = {item.evidence_id: item for item in snapshot.evidence}
    for diagnosis in snapshot.differential_diagnoses:
        with st.expander(
            f"{diagnosis.hypothesis} · AI reasoning confidence {diagnosis.confidence:.2f}"
        ):
            st.markdown("Supporting evidence")
            for evidence_id in diagnosis.supporting_evidence_ids:
                item = evidence.get(evidence_id)
                if item:
                    st.write(f"- {item.name}: {item.value} [{evidence_id}]")
            if diagnosis.contradicting_evidence_ids:
                st.markdown(
                    "Contradicting evidence: "
                    + ", ".join(diagnosis.contradicting_evidence_ids)
                )
            st.markdown("Missing evidence: " + ", ".join(diagnosis.missing_evidence))

    st.subheader("Specialist routing and opinions")
    if snapshot.routing_decisions:
        decision = snapshot.routing_decisions[-1]
        if decision.selected_specialists:
            for specialist in decision.selected_specialists:
                st.info(f"{specialist}: {decision.reasons[specialist]}")
        else:
            st.info("No specialist was selected by the evidence-driven router.")
    for opinion in snapshot.specialist_opinions:
        with st.expander(f"{opinion.specialist.replace('_', ' ').title()} opinion"):
            st.write(opinion.assessment)
            st.caption(f"AI reasoning confidence: {opinion.confidence:.2f}")
            if opinion.critical_concerns:
                st.warning("; ".join(opinion.critical_concerns))

    st.subheader("Disagreements")
    if not snapshot.contradictions:
        st.success("No explicit specialist contradiction is recorded.")
    for contradiction in snapshot.contradictions:
        st.warning(
            f"{contradiction.topic}: {contradiction.agent_a} vs "
            f"{contradiction.agent_b} · "
            f"{'resolved' if contradiction.resolved else 'unresolved'}"
        )

    st.subheader("Highest-value missing information")
    ranked = sorted(
        snapshot.missing_information,
        key=lambda item: (item.diagnostic_utility + item.urgency, len(item.requested_by)),
        reverse=True,
    )
    for request in ranked:
        st.write(
            f"- **{request.information_needed}** — {request.reason} "
            f"(requested by {', '.join(request.requested_by)})"
        )


def render_rag(snapshot: MedicalCaseSnapshot) -> None:
    st.subheader("Retrieved clinical evidence")
    question_map = {item.question_id: item for item in snapshot.evidence_questions}
    for result in snapshot.retrieved_evidence:
        question = question_map.get(result.question_id)
        with st.expander(
            f"{result.document} · {result.section} · similarity {result.similarity_score:.2f}"
        ):
            if question:
                st.caption(f"Question from {question.asked_by}: {question.question}")
            st.write(result.retrieved_text)
            st.caption(
                f"Source: {result.source} · Chunk: {result.chunk_id} · "
                f"Retrieval: {result.retrieval_id}"
            )
            if result.source_url:
                st.link_button("Open public source", result.source_url)
