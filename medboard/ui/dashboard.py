"""Interactive Streamlit dashboard orchestrating all MedBoard views."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import lru_cache
from uuid import uuid4

import streamlit as st

from medboard.graph.state import MedicalCaseSnapshot
from medboard.memory import CaseMemoryRepository
from medboard.observability import get_logger, log_event
from medboard.ui.analysis_view import render_analysis, render_rag
from medboard.ui.case_input import render_case_input
from medboard.ui.human_review import render_human_review
from medboard.ui.memory_view import render_memory
from medboard.ui.report_view import render_report
from medboard.ui.runtime import AppRuntime
from medboard.ui.runtime import get_runtime
from medboard.ui.system_views import (
    render_knowledge_base,
    render_settings,
    render_system_logs,
)
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
    title, mode = st.columns([5, 1])
    with title:
        st.title("MedBoard AI")
        st.caption("Interactive Multi-Agent Medical Diagnostic Decision Support Board")
    with mode:
        st.markdown(
            f"<div class='mode-badge'>{runtime.settings.mode_label}</div>",
            unsafe_allow_html=True,
        )
    st.warning(
        "Educational prototype only — not a medical device, diagnosis service, or "
        "replacement for a qualified healthcare professional."
    )

    with st.sidebar:
        st.markdown("## 🩺 MedBoard AI")
        page = st.radio(
            "View",
            ["New case", "Saved cases", "Knowledge base", "System logs", "Settings"],
        )
        st.divider()
        st.caption("RUNTIME")
        st.metric("Mode", runtime.settings.mode_label)
        st.metric("Knowledge chunks", runtime.knowledge_store.count)
        st.caption(f"Provider: {runtime.settings.llm_provider.value}")
        st.caption(f"Maximum critic revisions: {runtime.settings.max_revisions}")

    if page == "Saved cases":
        selected_run = render_memory(runtime.case_memory, runtime.service)
        if selected_run:
            st.session_state.active_run_id = selected_run
            st.rerun()
        return
    if page == "Knowledge base":
        render_knowledge_base(runtime.settings, runtime.knowledge_store)
        return
    if page == "System logs":
        render_system_logs(runtime.settings)
        return
    if page == "Settings":
        render_settings(runtime.settings)
        return

    case = render_case_input(runtime.settings.demo_cases_directory)
    if case is not None and st.button("Start multi-agent investigation", type="primary"):
        run_id = f"RUN-{uuid4().hex[:12].upper()}"
        started_snapshot = _render_live_progress(
            runtime.service.start_stream(case, run_id),
            "The clinical reasoning board is investigating the case...",
        )
        log_event(
            get_logger(__name__),
            "dashboard_case_started",
            run_id=run_id,
            case_id=case.case_id,
            interrupted=(
                started_snapshot.human_review.status.value == "waiting_for_human"
            ),
        )
        st.session_state.active_run_id = started_snapshot.run_id
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
            resumed_snapshot = _render_live_progress(
                runtime.service.resume_stream(snapshot.run_id, command),
                "Resuming the affected workflow path...",
            )
            interrupted = (
                resumed_snapshot.human_review.status.value == "waiting_for_human"
            )
            log_event(
                get_logger(__name__),
                "dashboard_human_decision",
                run_id=snapshot.run_id,
                action=command.action.value,
                interrupted=interrupted,
            )
            st.session_state.active_run_id = resumed_snapshot.run_id
            st.success(
                "Workflow paused for review again."
                if interrupted
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


def _render_live_progress(
    snapshots: Iterable[MedicalCaseSnapshot], label: str
) -> MedicalCaseSnapshot:
    """Render new trace records as cumulative LangGraph states arrive."""
    status = st.status(label, expanded=True)
    seen_trace_ids: set[str] = set()
    latest: MedicalCaseSnapshot | None = None
    for latest in snapshots:
        for event in latest.execution_trace:
            if event.trace_id in seen_trace_ids:
                continue
            seen_trace_ids.add(event.trace_id)
            if event.event_type.value in {
                "agent_started",
                "agent_completed",
                "agent_failed",
                "routing_decision",
                "tool_called",
                "interrupted",
                "resumed",
            }:
                actor = (event.agent or "workflow").replace("_", " ").title()
                status.write(f"{actor}: {event.event_type.value.replace('_', ' ')}")
    if latest is None:
        status.update(label="Workflow produced no state", state="error")
        raise RuntimeError("workflow stream completed without producing state")
    status.update(label="Workflow execution trace complete", state="complete", expanded=False)
    return latest


def _metrics(snapshot: MedicalCaseSnapshot) -> None:
    tool_calls = sum(
        (
            _tool_call_count(event.details)
            for event in snapshot.execution_trace
            if event.event_type.value == "tool_called"
        ),
        start=0,
    )
    timestamps = [event.timestamp for event in snapshot.execution_trace]
    elapsed_seconds = (
        (max(timestamps) - min(timestamps)).total_seconds() if timestamps else 0
    )
    status = snapshot.human_review.status.value.replace("_", " ").title()
    first_row = st.columns(5)
    first_row[0].metric("Run ID", snapshot.run_id[-8:])
    first_row[1].metric("Status", status)
    first_row[2].metric("Model calls", len(snapshot.token_usage))
    first_row[3].metric("Execution time", f"{elapsed_seconds:.2f}s")
    first_row[4].metric("Specialists", len(snapshot.selected_specialists))
    second_row = st.columns(5)
    second_row[0].metric("Messages", len(snapshot.agent_messages))
    second_row[1].metric("Tool calls", tool_calls)
    second_row[2].metric("Revisions", snapshot.revision_count)
    second_row[3].metric("Tokens", snapshot.total_tokens)
    second_row[4].metric("Estimated cost", f"${snapshot.estimated_cost:.4f}")
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
        .block-container {padding-top: 2rem; max-width: 1500px;}
        [data-testid="stSidebar"] {border-right: 1px solid #dce4ec;}
        .mode-badge {margin-top: 1.4rem; padding: .45rem .8rem; text-align: center;
                     border: 1px solid #0f766e; border-radius: 2rem; color: #0f766e;
                     font-weight: 700; letter-spacing: .08em; font-size: .8rem;}
        .agent-card {border: 1px solid #dce4ec; border-radius: .65rem;
                     padding: .75rem; margin-bottom: .7rem; background: #f8fafc;
                     min-height: 4.3rem;}
        .status-completed {color: #087f5b;} .status-pending {color: #9c6f00;}
        .status-failed {color: #c92a2a;} .status-skipped {color: #64748b;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _tool_call_count(details: Mapping[str, object]) -> int:
    value = details.get("call_count", 1)
    return value if isinstance(value, int) else 1
