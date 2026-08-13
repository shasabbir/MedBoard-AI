"""Streamlit rendering tests for the dashboard and component views."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from medboard.config import Settings, get_settings
from medboard.graph.state import create_initial_state, validate_state
from medboard.graph.workflow import build_collaboration_workflow
from medboard.models import MedicalCaseInput
from medboard.providers import DemoModelProvider
from medboard.ui.dashboard import _runtime
from medboard.ui.system_views import _model_label, _read_json_lines
from medboard.ui.workflow_view import GRAPH

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(autouse=True)
def force_dashboard_demo_mode(monkeypatch: pytest.MonkeyPatch):
    """Keep UI tests deterministic regardless of a developer's local .env."""
    monkeypatch.setenv("DEMO_MODE", "true")
    get_settings.cache_clear()
    _runtime.cache_clear()
    yield
    _runtime.cache_clear()
    get_settings.cache_clear()


def test_dashboard_boots_with_safety_notice_and_case_selector() -> None:
    app = AppTest.from_file(APP_PATH).run(timeout=20)

    assert not app.exception
    assert any("MedBoard AI" in title.value for title in app.title)
    assert any("Educational prototype only" in warning.value for warning in app.warning)
    assert any(radio.label == "View" for radio in app.radio)
    assert any(radio.label == "Input source" for radio in app.radio)
    assert any(button.label == "Start multi-agent investigation" for button in app.button)


def test_dashboard_starts_demo_case_and_renders_review_panels() -> None:
    app = AppTest.from_file(APP_PATH).run(timeout=20)
    start = next(
        button for button in app.button if button.label == "Start multi-agent investigation"
    )

    app = start.click().run(timeout=30)

    assert not app.exception
    assert any("Workflow execution trace complete" in item.label for item in app.status)
    assert any("Investigation reached human review" in item.value for item in app.success)
    assert any("Human / clinician review" in item.value for item in app.subheader)
    assert any("Execution graph" in item.value for item in app.subheader)
    assert any("Retrieved clinical evidence" in item.value for item in app.subheader)
    assert any("Agent communication history" in item.value for item in app.subheader)
    assert any(metric.label == "Triage level" for metric in app.metric)
    assert any(metric.label == "Tokens" for metric in app.metric)
    assert any(metric.label == "Model attempts" for metric in app.metric)
    assert any(metric.label == "Tool calls" for metric in app.metric)

    submit = next(
        button for button in app.button if button.label == "Submit human decision"
    )
    app = submit.click().run(timeout=30)

    assert not app.exception
    assert any("MedBoard AI Case Review" in item.value for item in app.markdown)
    rendered_sections = {item.value for item in app.markdown}
    for section in [
        "#### Key findings",
        "#### Differential considerations",
        "#### Specialist opinions",
        "#### Retrieved evidence",
        "#### Areas of disagreement",
        "#### Missing information",
        "#### Risk / triage assessment",
        "#### Suggested clinical review priorities",
        "#### Human review status",
        "#### Disclaimer",
    ]:
        assert section in rendered_sections


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


def test_operational_view_helpers_hide_bad_log_rows(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    log.write_text(
        '{"level":"INFO","event":"started"}\nnot-json\n'
        '{"level":"ERROR","event":"failed"}\n',
        encoding="utf-8",
    )

    assert _read_json_lines(log, limit=2) == [
        {"level": "ERROR", "event": "failed"}
    ]
    assert _model_label(Settings(_env_file=None)) == "deterministic-v1"


def test_displayed_graph_includes_control_memory_and_observability() -> None:
    for label in [
        "Dynamic Specialist Router",
        "Knowledge Memory",
        "Human / Clinician Review",
        "Case History Memory",
        "Observability",
        "Streamlit UI",
    ]:
        assert label in GRAPH


def test_dashboard_secondary_pages_render() -> None:
    app = AppTest.from_file(APP_PATH).run(timeout=20)
    navigation = next(radio for radio in app.radio if radio.label == "View")

    app = navigation.set_value("Knowledge base").run(timeout=20)
    assert not app.exception
    assert any(header.value == "Knowledge base" for header in app.header)
    assert any(metric.label == "Indexed chunks" for metric in app.metric)

    navigation = next(radio for radio in app.radio if radio.label == "View")
    app = navigation.set_value("System logs").run(timeout=20)
    assert not app.exception
    assert any(header.value == "System logs" for header in app.header)

    navigation = next(radio for radio in app.radio if radio.label == "View")
    app = navigation.set_value("Settings").run(timeout=20)
    assert not app.exception
    assert any(header.value == "Runtime settings" for header in app.header)
