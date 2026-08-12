"""Trace and communication history panels."""

from __future__ import annotations

import streamlit as st

from medboard.graph.state import MedicalCaseSnapshot


def render_trace(snapshot: MedicalCaseSnapshot) -> None:
    st.subheader("Execution trace")
    rows = [
        {
            "time": event.timestamp.isoformat(timespec="milliseconds"),
            "event": event.event_type.value,
            "agent": event.agent or "system",
            "status": event.status.value if event.status else "",
            "duration_ms": round(event.duration_ms, 2) if event.duration_ms else None,
            "details": event.details,
        }
        for event in snapshot.execution_trace
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("Errors")
    if not snapshot.errors:
        st.success("No agent errors were recorded.")
        return
    st.dataframe(
        [error.model_dump(mode="json") for error in snapshot.errors],
        use_container_width=True,
        hide_index=True,
    )


def render_messages(snapshot: MedicalCaseSnapshot) -> None:
    st.subheader("Agent communication history")
    message_types = sorted({message.message_type.value for message in snapshot.agent_messages})
    selected_types = st.multiselect(
        "Message types",
        message_types,
        default=message_types,
        key="message_type_filter",
    )
    for message in snapshot.agent_messages:
        if message.message_type.value not in selected_types:
            continue
        with st.expander(
            f"{message.sender} → {message.recipient} · {message.message_type.value}"
        ):
            st.write(message.content)
            if message.evidence_ids:
                st.caption("Evidence: " + ", ".join(message.evidence_ids))
            if message.retrieval_ids:
                st.caption("Retrieved chunks: " + ", ".join(message.retrieval_ids))
            st.caption(message.timestamp.isoformat())
