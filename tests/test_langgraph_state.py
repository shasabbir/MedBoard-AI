"""Integration proof that LangGraph applies the declared concurrent reducers."""

from typing import Any

from langgraph.graph import END, START, StateGraph

from medboard.graph.state import MedicalCaseState, create_initial_state
from medboard.models import Evidence, EvidenceType, MedicalCaseInput


def test_compiled_graph_merges_parallel_evidence() -> None:
    history_evidence = Evidence(
        evidence_id="EV-HISTORY",
        evidence_type=EvidenceType.HISTORY,
        name="smoking history",
        value="none",
        source="user_case",
    )
    symptom_evidence = Evidence(
        evidence_id="EV-SYMPTOM",
        evidence_type=EvidenceType.SYMPTOM,
        name="headache",
        value=True,
        source="user_case",
    )

    def history_node(_: MedicalCaseState) -> dict[str, Any]:
        return {
            "evidence": [history_evidence],
        }

    def symptom_node(_: MedicalCaseState) -> dict[str, Any]:
        return {
            "evidence": [symptom_evidence],
        }

    builder = StateGraph(MedicalCaseState)
    builder.add_node("history", history_node)
    builder.add_node("symptoms", symptom_node)
    builder.add_edge(START, "history")
    builder.add_edge(START, "symptoms")
    builder.add_edge("history", END)
    builder.add_edge("symptoms", END)
    graph = builder.compile()
    initial_state = create_initial_state(
        MedicalCaseInput(case_id="CASE-001", chief_complaint="Headache"),
        run_id="RUN-001",
    )

    result = graph.invoke(initial_state)

    assert {item.evidence_id for item in result["evidence"]} == {
        "EV-HISTORY",
        "EV-SYMPTOM",
    }
    assert result["selected_specialists"] == []


def test_graph_package_lazily_exposes_workflow_builders() -> None:
    from medboard.graph import build_initial_workflow

    assert callable(build_initial_workflow)
