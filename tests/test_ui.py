"""Streamlit rendering tests for the dashboard and component views."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from medboard.graph.state import create_initial_state, validate_state
from medboard.graph.workflow import build_collaboration_workflow
from medboard.models import MedicalCaseInput
from medboard.providers import DemoModelProvider

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_dashboard_boots_with_safety_notice_and_case_selector() -> None:
    app = AppTest.from_file(APP_PATH).run(timeout=20)

    assert not app.exception
    assert any("MedBoard AI" in title.value for title in app.title)
    assert any("Educational prototype only" in warning.value for warning in app.warning)
    assert any(radio.label == "Input source" for radio in app.radio)
    assert any(button.label == "Start multi-agent investigation" for button in app.button)


def test_dashboard_starts_demo_case_and_renders_review_panels() -> None:
    app = AppTest.from_file(APP_PATH).run(timeout=20)
    start = next(
        button for button in app.button if button.label == "Start multi-agent investigation"
    )

    app = start.click().run(timeout=30)

    assert not app.exception
    assert any("Investigation reached human review" in item.value for item in app.success)
    assert any("Human / clinician review" in item.value for item in app.subheader)
    assert any("Execution graph" in item.value for item in app.subheader)
    assert any("Retrieved clinical evidence" in item.value for item in app.subheader)
    assert any("Agent communication history" in item.value for item in app.subheader)
    assert any(metric.label == "Triage level" for metric in app.metric)
    assert any(metric.label == "Tokens" for metric in app.metric)

    submit = next(
        button for button in app.button if button.label == "Submit human decision"
    )
    app = submit.click().run(timeout=30)

    assert not app.exception
    assert any("MedBoard AI Case Review" in item.value for item in app.markdown)


def test_analysis_snapshot_has_required_ui_data() -> None:
    case = MedicalCaseInput(
        case_id="CASE-UI-001",
        chief_complaint="Fever and cough",
        symptoms=["fever", "cough"],
    )
    workflow = build_collaboration_workflow(DemoModelProvider())
    snapshot = validate_state(
        workflow.invoke(create_initial_state(case, run_id="RUN-UI-001"))
    )

    assert snapshot.differential_diagnoses
    assert snapshot.selected_specialists == ["infectious_disease"]
    assert snapshot.agent_messages
    assert snapshot.execution_trace
