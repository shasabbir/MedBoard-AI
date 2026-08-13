"""Workflow graph and status visualization."""

from __future__ import annotations

import html

import streamlit as st

from medboard.graph.state import MedicalCaseSnapshot

GRAPH = r"""
digraph MedBoard {
  rankdir=LR;
  graph [bgcolor="transparent", pad="0.2", nodesep="0.3", ranksep="0.55"];
  node [shape=box, style="rounded,filled", fillcolor="#eff6ff", color="#2563eb",
        fontname="Arial", fontsize=10];
  edge [color="#64748b", fontname="Arial", fontsize=8];
  input [label="Synthetic Case Input"];
  supervisor [label="Supervisor Agent", fillcolor="#dbeafe"];
  history [label="History"];
  symptoms [label="Symptoms"];
  laboratory [label="Laboratory"];
  medication [label="Medication"];
  memory [label="Shared Workflow Memory", shape=cylinder, fillcolor="#f1f5f9"];
  differential [label="Differential Diagnosis"];
  router [label="Dynamic Specialist Router", shape=diamond, fillcolor="#fef3c7"];
  cardiology [label="Cardiology"];
  neurology [label="Neurology"];
  infectious_disease [label="Infectious Disease"];
  questions [label="Clinical Evidence Questions"];
  rag [label="Evidence Retrieval / RAG"];
  knowledge [label="Knowledge Memory", shape=cylinder, fillcolor="#ecfdf5"];
  critic [label="Red-Team Critic", fillcolor="#fce7f3"];
  risk [label="Risk / Triage", fillcolor="#ffedd5"];
  human [label="Human / Clinician Review", shape=diamond, fillcolor="#fef3c7"];
  report [label="Final Report Generator", fillcolor="#dcfce7"];
  cases [label="Case History Memory", shape=cylinder, fillcolor="#f1f5f9"];
  observability [label="Observability\ntrace · logs · errors · retries\ntiming · tokens · cost",
                 fillcolor="#eef2ff"];
  ui [label="Streamlit UI\ncase input · graph · messages\nmemory · evidence · report",
      fillcolor="#eef2ff"];

  input -> supervisor;
  supervisor -> history [label="dispatch"];
  supervisor -> symptoms [label="dispatch"];
  supervisor -> laboratory [label="dispatch"];
  supervisor -> medication [label="dispatch"];
  history -> memory; symptoms -> memory; laboratory -> memory; medication -> memory;
  memory -> differential;
  differential -> router;
  router -> cardiology [label="when selected"];
  router -> neurology [label="when selected"];
  router -> infectious_disease [label="when selected"];
  differential -> questions; cardiology -> questions; neurology -> questions;
  infectious_disease -> questions;
  questions -> rag; knowledge -> rag [dir=both];
  rag -> critic;
  critic -> supervisor [label="revise ≤ limit", style=dashed];
  critic -> risk [label="accept / limit"];
  risk -> human;
  human -> supervisor [label="add info / revise / retry", style=dashed];
  human -> report [label="approve"];
  human -> cases [label="reject + audit"];
  report -> cases;
  cases -> supervisor [label="prior cases", style=dashed];
  supervisor -> memory [label="checkpoints + messages", style=dashed];
  human -> memory [label="decision + feedback", style=dashed];
  supervisor -> observability [label="events", style=dashed];
  critic -> observability [label="events", style=dashed];
  risk -> observability [label="events", style=dashed];
  memory -> ui [label="live state", style=dashed];
  observability -> ui [label="telemetry", style=dashed];
  human -> ui [label="controls", dir=both, style=dashed];
}
"""


def render_workflow(snapshot: MedicalCaseSnapshot) -> None:
    st.subheader("Execution graph")
    st.graphviz_chart(GRAPH, use_container_width=True)
    st.caption(
        "Solid edges are workflow/data flow; dashed edges are feedback or history. "
        "Specialist branches run only when selected."
    )

    agents = _agent_statuses(snapshot)
    columns = st.columns(4)
    for index, (agent, status) in enumerate(agents.items()):
        with columns[index % len(columns)]:
            st.markdown(
                f"<div class='agent-card'><strong>{html.escape(agent.replace('_', ' ').title())}"
                f"</strong><br><span class='status-{status}'>{status.upper()}</span></div>",
                unsafe_allow_html=True,
            )


def _agent_statuses(snapshot: MedicalCaseSnapshot) -> dict[str, str]:
    completed = {
        event.agent
        for event in snapshot.execution_trace
        if event.event_type.value == "agent_completed" and event.agent
    }
    failed = {
        error.agent for error in snapshot.errors if error.agent and not error.resolved
    }
    statuses: dict[str, str] = {}
    for agent in [
        "supervisor",
        "history",
        "symptoms",
        "laboratory",
        "medication",
        "differential",
        "cardiology",
        "neurology",
        "infectious_disease",
        "evidence_retrieval",
        "critic",
        "risk",
        "reporter",
    ]:
        if agent in failed:
            statuses[agent] = "failed"
        elif agent in completed:
            statuses[agent] = "completed"
        elif agent in {"cardiology", "neurology", "infectious_disease"}:
            statuses[agent] = (
                "pending" if agent in snapshot.selected_specialists else "skipped"
            )
        else:
            statuses[agent] = "pending"
    return statuses
