"""Workflow graph and status visualization."""

from __future__ import annotations

import html

import streamlit as st

from medboard.graph.state import MedicalCaseSnapshot

GRAPH = """
flowchart LR
  U[Case Input] --> S[Supervisor]
  S --> H[History]
  S --> SY[Symptoms]
  S --> L[Labs]
  S --> M[Medication]
  H --> D[Differential]
  SY --> D
  L --> D
  M --> D
  D --> SR{Specialist Router}
  SR --> C[Cardiology]
  SR --> N[Neurology]
  SR --> I[Infectious Disease]
  C --> RAG[RAG]
  N --> RAG
  I --> RAG
  SR --> RAG
  RAG --> CR[Critic]
  CR -->|revise| S
  CR -->|accept| R[Risk / Triage]
  R --> HR{Human Review}
  HR -->|approve| F[Final Report]
  HR -->|revise / add info| S
  HR -->|reject| DB[(Case History)]
  F --> DB
"""


def render_workflow(snapshot: MedicalCaseSnapshot) -> None:
    st.subheader("Execution graph")
    st.code(GRAPH, language="mermaid")
    st.caption("Conditional branches run only when selected; the Mermaid source is inspectable.")

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
    failed = {error.agent for error in snapshot.errors if error.agent}
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
