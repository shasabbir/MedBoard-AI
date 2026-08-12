"""Durable case-history browsing and deletion."""

import streamlit as st

from medboard.memory import CaseMemoryRepository


def render_memory(repository: CaseMemoryRepository) -> str | None:
    st.subheader("Case history memory")
    runs = repository.list_runs()
    if not runs:
        st.info("No persisted runs yet.")
        return None
    st.dataframe(runs, use_container_width=True, hide_index=True)
    selected = st.selectbox(
        "Open persisted run",
        [row["run_id"] for row in runs],
        key="persisted_run",
    )
    run = repository.load_run(selected)
    if run:
        st.caption(
            f"{run.case_input.case_id} · {len(run.agent_messages)} messages · "
            f"{len(run.execution_trace)} trace events"
        )
    if st.button("Delete selected case and audit history", type="secondary") and run:
        repository.delete_case(run.case_input.case_id)
        st.session_state.pop("active_run_id", None)
        st.success("Case and cascading audit history deleted.")
        st.rerun()
    return selected if st.button("Load selected run") else None
