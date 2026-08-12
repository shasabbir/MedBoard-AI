"""Interactive Streamlit dashboard orchestrating all MedBoard views."""

from __future__ import annotations

from functools import lru_cache
from uuid import uuid4

import streamlit as st

from medboard.graph.state import MedicalCaseSnapshot
from medboard.memory import CaseMemoryRepository
from medboard.ui.analysis_view import render_analysis, render_rag
from medboard.ui.case_input import render_case_input
from medboard.ui.human_review import render_human_review
from medboard.ui.memory_view import render_memory
from medboard.ui.report_view import render_report
from medboard.ui.runtime import AppRuntime
from medboard.ui.runtime import get_runtime
from medboard.ui.trace_view import render_messages, render_trace
from medboard.ui.workflow_view import render_workflow


@lru_cache(maxsize=1)
def _runtime() -> AppRuntime:
    return get_runtime()


def run_dashboard() -> None:
    st.set_page_config(
        page_title="MedBoard AI",
        page_icon="🩺",
        layout="wide",
    )
    _styles()
    runtime = _runtime()
    st.title("MedBoard AI")
    st.caption("Interactive Multi-Agent Medical Diagnostic Decision Support Board")
    st.warning(
        "Educational prototype only — not a medical device, diagnosis service, or "
        "replacement for a qualified healthcare professional."
    )

    with st.sidebar:
        st.header("Runtime")
        st.metric("Mode", runtime.settings.mode_label)
        st.metric("Knowledge chunks", runtime.knowledge_store.count)
        st.caption(f"Provider: {runtime.settings.llm_provider.value}")
        st.caption(f"Maximum critic revisions: {runtime.settings.max_revisions}")
        page = st.radio("View", ["Investigation", "Case memory"])

    if page == "Case memory":
        selected_run = render_memory(runtime.case_memory)
        if selected_run:
            st.session_state.active_run_id = selected_run
            st.rerun()
        return

    case = render_case_input(runtime.settings.demo_cases_directory)
    if case is not None and st.button("Start multi-agent investigation", type="primary"):
        run_id = f"RUN-{uuid4().hex[:12].upper()}"
        with st.spinner("The clinical reasoning board is investigating the case..."):
            result = runtime.service.start(case, run_id)
        st.session_state.active_run_id = result.snapshot.run_id
        st.success("Investigation reached human review.")

    snapshot = _active_snapshot(runtime.case_memory)
    if snapshot is None:
        st.info("Select a synthetic case and start an investigation.")
        return

    _metrics(snapshot)
    tabs = st.tabs(
        [
            "Workflow",
            "Analysis",
            "Evidence",
            "Messages",
            "Trace",
            "Human review",
            "Report",
        ]
    )
    with tabs[0]:
        render_workflow(snapshot)
    with tabs[1]:
        render_analysis(snapshot)
    with tabs[2]:
        render_rag(snapshot)
    with tabs[3]:
        render_messages(snapshot)
    with tabs[4]:
        render_trace(snapshot)
    with tabs[5]:
        command = render_human_review(snapshot)
        if command is not None:
            with st.spinner("Resuming the affected workflow path..."):
                result = runtime.service.resume(snapshot.run_id, command)
            st.session_state.active_run_id = result.snapshot.run_id
            st.success(
                "Workflow paused for review again."
                if result.interrupted
                else "Human decision applied and run completed."
            )
            st.rerun()
    with tabs[6]:
        render_report(snapshot)


def _active_snapshot(
    repository: CaseMemoryRepository,
) -> MedicalCaseSnapshot | None:
    run_id = st.session_state.get("active_run_id")
    return repository.load_run(str(run_id)) if run_id else None


def _metrics(snapshot: MedicalCaseSnapshot) -> None:
    columns = st.columns(7)
    columns[0].metric("Run", snapshot.run_id[-8:])
    columns[1].metric("Evidence", len(snapshot.evidence))
    columns[2].metric("Messages", len(snapshot.agent_messages))
    columns[3].metric("Specialists", len(snapshot.selected_specialists))
    columns[4].metric("Revisions", snapshot.revision_count)
    columns[5].metric("Tokens", snapshot.total_tokens)
    columns[6].metric("Estimated cost", f"${snapshot.estimated_cost:.4f}")
    if snapshot.triage_result:
        level = snapshot.triage_result.triage_level.value.upper()
        if level in {"URGENT", "EMERGENCY"}:
            st.error(f"Triage: {level} — {snapshot.triage_result.recommended_escalation}")
        else:
            st.info(f"Triage: {level} — {snapshot.triage_result.recommended_escalation}")


def _styles() -> None:
    st.markdown(
        """
        <style>
        .agent-card {border: 1px solid #dce4ec; border-radius: .65rem;
                     padding: .7rem; margin-bottom: .7rem; background: #f8fafc;}
        .status-completed {color: #087f5b;} .status-pending {color: #9c6f00;}
        .status-failed {color: #c92a2a;} .status-skipped {color: #64748b;}
        </style>
        """,
        unsafe_allow_html=True,
    )
