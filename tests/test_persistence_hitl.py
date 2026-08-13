"""Persistence and real LangGraph interrupt/resume tests."""

from pathlib import Path

from langgraph.types import Command

from medboard.graph.state import create_initial_state, validate_state
from medboard.graph.workflow import build_reviewable_workflow
from medboard.memory import CaseMemoryRepository, Database, WorkflowCheckpoint
from medboard.models import HumanReviewCommand, MedicalCaseInput
from medboard.providers import DemoModelProvider
from medboard.rag.store import KnowledgeStore
from medboard.workflow_service import WorkflowService


def build_graph(
    tmp_path: Path,
    provider: DemoModelProvider | None = None,
    *,
    max_agent_retries: int = 2,
):
    store = KnowledgeStore(tmp_path / "chroma")
    store.ingest_directory(Path("data/knowledge"))
    checkpoint = WorkflowCheckpoint(tmp_path / "workflow.db")
    graph = build_reviewable_workflow(
        provider or DemoModelProvider(),
        store,
        max_revisions=2,
        max_agent_retries=max_agent_retries,
        checkpointer=checkpoint.saver,
    )
    return graph, checkpoint


def neurological_case() -> MedicalCaseInput:
    return MedicalCaseInput(
        case_id="CASE-HITL-NEURO",
        chief_complaint="Sudden confusion and unilateral weakness",
        symptoms=["confusion", "unilateral weakness"],
    )


def test_workflow_interrupts_and_resumes_to_approved_report(tmp_path: Path) -> None:
    graph, checkpoint = build_graph(tmp_path)
    config = {"configurable": {"thread_id": "RUN-HITL-APPROVE"}}
    try:
        paused = graph.invoke(
            create_initial_state(neurological_case(), run_id="RUN-HITL-APPROVE"),
            config,
        )
        assert paused["human_review"].status.value == "waiting_for_human"
        assert graph.get_state(config).interrupts

        completed = graph.invoke(
            Command(
                resume={
                    "action": "approve",
                    "reviewer": "test-clinician",
                    "feedback": "Approved for educational review.",
                }
            ),
            config,
        )
        snapshot = validate_state(completed)

        assert snapshot.human_review.status.value == "approved"
        assert snapshot.final_report is not None
        assert snapshot.final_report.disclaimer
        assert not graph.get_state(config).interrupts
    finally:
        checkpoint.close()


def test_failed_report_generation_returns_to_review_and_can_be_retried(
    tmp_path: Path,
) -> None:
    class FailReporterOnceProvider(DemoModelProvider):
        reporter_failures = 0

        def generate(self, **kwargs):
            if kwargs["agent"] == "reporter" and self.reporter_failures == 0:
                self.reporter_failures += 1
                raise ConnectionError("simulated reporter outage")
            return super().generate(**kwargs)

    graph, checkpoint = build_graph(
        tmp_path,
        FailReporterOnceProvider(),
        max_agent_retries=0,
    )
    config = {"configurable": {"thread_id": "RUN-HITL-REPORT-RETRY"}}
    try:
        graph.invoke(
            create_initial_state(
                neurological_case(), run_id="RUN-HITL-REPORT-RETRY"
            ),
            config,
        )
        paused_again = validate_state(
            graph.invoke(Command(resume={"action": "approve"}), config)
        )

        assert paused_again.human_review.status.value == "waiting_for_human"
        assert paused_again.final_report is None
        assert graph.get_state(config).interrupts
        assert any(
            error.agent == "reporter" and not error.resolved
            for error in paused_again.errors
        )

        completed = validate_state(
            graph.invoke(
                Command(
                    resume={
                        "action": "retry_failed_agent",
                        "failed_agent": "reporter",
                    }
                ),
                config,
            )
        )

        assert completed.human_review.status.value == "approved"
        assert completed.final_report is not None
        assert not graph.get_state(config).interrupts
        assert all(
            error.resolved for error in completed.errors if error.agent == "reporter"
        )
    finally:
        checkpoint.close()


def test_rejection_is_auditable_without_report(tmp_path: Path) -> None:
    graph, checkpoint = build_graph(tmp_path)
    config = {"configurable": {"thread_id": "RUN-HITL-REJECT"}}
    try:
        graph.invoke(
            create_initial_state(neurological_case(), run_id="RUN-HITL-REJECT"),
            config,
        )
        completed = graph.invoke(
            Command(resume={"action": "reject", "feedback": "Insufficient evidence."}),
            config,
        )
        snapshot = validate_state(completed)

        assert snapshot.human_review.status.value == "rejected"
        assert snapshot.final_report is None
    finally:
        checkpoint.close()


def test_added_information_reruns_downstream_then_interrupts_again(tmp_path: Path) -> None:
    graph, checkpoint = build_graph(tmp_path)
    config = {"configurable": {"thread_id": "RUN-HITL-INFO"}}
    try:
        graph.invoke(
            create_initial_state(neurological_case(), run_id="RUN-HITL-INFO"),
            config,
        )
        paused_again = graph.invoke(
            Command(
                resume={
                    "action": "add_information",
                    "added_information": {"blood_pressure": "190/110 mmHg"},
                }
            ),
            config,
        )
        snapshot = validate_state(paused_again)

        assert snapshot.human_review.status.value == "waiting_for_human"
        assert any(item.source == "human_review" for item in snapshot.evidence)
        initial_agent_starts = [
            event
            for event in snapshot.execution_trace
            if event.event_type.value == "agent_started"
            and event.agent in {"history", "symptoms", "laboratory", "medication"}
        ]
        assert len(initial_agent_starts) == 4
        assert graph.get_state(config).interrupts
    finally:
        checkpoint.close()


def test_added_laboratory_information_refreshes_lab_analysis(tmp_path: Path) -> None:
    graph, checkpoint = build_graph(tmp_path)
    config = {"configurable": {"thread_id": "RUN-HITL-LAB"}}
    try:
        graph.invoke(
            create_initial_state(neurological_case(), run_id="RUN-HITL-LAB"),
            config,
        )
        paused_again = graph.invoke(
            Command(
                resume={
                    "action": "add_information",
                    "added_information": {
                        "laboratory_values": [
                            {"name": "hemoglobin", "value": 7.8, "unit": "g/dL"}
                        ]
                    },
                }
            ),
            config,
        )
        snapshot = validate_state(paused_again)

        assert snapshot.laboratory_findings is not None
        assert any(
            "hemoglobin: low" in item.casefold()
            for item in snapshot.laboratory_findings.abnormal_values
        )
        laboratory_starts = [
            event
            for event in snapshot.execution_trace
            if event.event_type.value == "agent_started"
            and event.agent == "laboratory"
        ]
        assert len(laboratory_starts) == 2
        assert any(
            item.hypothesis == "Iron-deficiency anemia pattern"
            for item in snapshot.differential_diagnoses
        )
        lab_request = next(
            item
            for item in snapshot.missing_information
            if item.information_needed == "laboratory values with explicit units"
        )
        assert lab_request.resolved is True
        assert lab_request.resolution == "Supplied during human review."
        assert graph.get_state(config).interrupts
    finally:
        checkpoint.close()


def test_added_symptoms_rerun_only_symptom_intake_and_change_routing(
    tmp_path: Path,
) -> None:
    graph, checkpoint = build_graph(tmp_path)
    config = {"configurable": {"thread_id": "RUN-HITL-SYMPTOMS"}}
    case = MedicalCaseInput(
        case_id="CASE-HITL-SYMPTOMS",
        chief_complaint="Nonspecific tiredness",
        symptoms=["tiredness"],
    )
    try:
        graph.invoke(
            create_initial_state(case, run_id="RUN-HITL-SYMPTOMS"),
            config,
        )
        paused_again = graph.invoke(
            Command(
                resume={
                    "action": "add_information",
                    "added_information": {"symptoms": ["fever", "cough"]},
                }
            ),
            config,
        )
        snapshot = validate_state(paused_again)

        starts = [
            event.agent
            for event in snapshot.execution_trace
            if event.event_type.value == "agent_started"
            and event.agent in {"history", "symptoms", "laboratory", "medication"}
        ]
        assert starts.count("symptoms") == 2
        assert starts.count("history") == 1
        assert starts.count("laboratory") == 1
        assert starts.count("medication") == 1
        assert snapshot.case_input.symptoms == ["tiredness", "fever", "cough"]
        assert all(
            item.source == "human_review"
            for item in snapshot.evidence
            if item.name in {"fever", "cough"}
        )
        assert "infectious_disease" in snapshot.selected_specialists
        assert graph.get_state(config).interrupts
    finally:
        checkpoint.close()


def test_complete_add_information_resume_approve_acceptance_path(
    tmp_path: Path,
) -> None:
    graph, checkpoint = build_graph(tmp_path)
    repository = CaseMemoryRepository(Database(tmp_path / "acceptance_history.db"))
    service = WorkflowService(graph, repository)
    try:
        started = service.start(neurological_case(), "RUN-ACCEPTANCE")
        assert started.interrupted

        revised = service.resume(
            "RUN-ACCEPTANCE",
            HumanReviewCommand(
                action="add_information",
                reviewer="acceptance-clinician",
                added_information={"blood_pressure": "190/110 mmHg"},
            ),
        )
        assert revised.interrupted
        assert revised.snapshot.final_report is None
        assert any(
            item.source == "human_review" for item in revised.snapshot.evidence
        )

        approved = service.resume(
            "RUN-ACCEPTANCE",
            HumanReviewCommand(
                action="approve",
                reviewer="acceptance-clinician",
                feedback="Approved after reviewing the additional information.",
            ),
        )

        assert approved.interrupted is False
        assert approved.snapshot.final_report is not None
        assert repository.run_status("RUN-ACCEPTANCE") == "approved"
        assert repository.feedback_count("RUN-ACCEPTANCE") == 2
        assert repository.load_run("RUN-ACCEPTANCE") == approved.snapshot
    finally:
        checkpoint.close()


def test_streaming_service_yields_progress_and_persists_interrupt(
    tmp_path: Path,
) -> None:
    graph, checkpoint = build_graph(tmp_path)
    repository = CaseMemoryRepository(Database(tmp_path / "stream_history.db"))
    service = WorkflowService(graph, repository, checkpoint)
    try:
        snapshots = list(
            service.start_stream(neurological_case(), "RUN-STREAM-PROGRESS")
        )

        assert len(snapshots) > 2
        assert snapshots[-1].human_review.status.value == "waiting_for_human"
        assert repository.run_status("RUN-STREAM-PROGRESS") == "waiting_for_human"
        assert repository.load_run("RUN-STREAM-PROGRESS") == snapshots[-1]
        trace_lengths = [len(snapshot.execution_trace) for snapshot in snapshots]
        assert trace_lengths == sorted(trace_lengths)
        assert trace_lengths[-1] > trace_lengths[0]
    finally:
        checkpoint.close()


def test_requested_specialist_runs_then_returns_to_human_review(tmp_path: Path) -> None:
    graph, checkpoint = build_graph(tmp_path)
    config = {"configurable": {"thread_id": "RUN-HITL-SPECIALIST"}}
    try:
        graph.invoke(
            create_initial_state(neurological_case(), run_id="RUN-HITL-SPECIALIST"),
            config,
        )
        paused_again = graph.invoke(
            Command(
                resume={
                    "action": "request_specialist",
                    "requested_specialist": "infectious_disease",
                    "feedback": "Please consider an infectious neurological process.",
                }
            ),
            config,
        )
        snapshot = validate_state(paused_again)

        assert snapshot.human_review.status.value == "waiting_for_human"
        assert snapshot.selected_specialists == ["neurology", "infectious_disease"]
        assert {item.specialist for item in snapshot.specialist_opinions} == {
            "neurology",
            "infectious_disease",
        }
        assert graph.get_state(config).interrupts
    finally:
        checkpoint.close()


def test_requested_revision_is_bounded_then_returns_to_human_review(
    tmp_path: Path,
) -> None:
    graph, checkpoint = build_graph(tmp_path)
    config = {"configurable": {"thread_id": "RUN-HITL-REVISION"}}
    try:
        graph.invoke(
            create_initial_state(neurological_case(), run_id="RUN-HITL-REVISION"),
            config,
        )
        paused_again = graph.invoke(
            Command(
                resume={
                    "action": "request_revision",
                    "feedback": "Reconsider the differential before approval.",
                }
            ),
            config,
        )
        snapshot = validate_state(paused_again)

        assert snapshot.human_review.status.value == "waiting_for_human"
        assert snapshot.revision_count == 1
        assert any(
            event.details.get("action") == "targeted_differential_revision"
            for event in snapshot.execution_trace
        )
        assert graph.get_state(config).interrupts
    finally:
        checkpoint.close()


def test_case_memory_round_trips_snapshot_messages_and_feedback(tmp_path: Path) -> None:
    graph, checkpoint = build_graph(tmp_path)
    config = {"configurable": {"thread_id": "RUN-MEMORY"}}
    repository = CaseMemoryRepository(Database(tmp_path / "case_history.db"))
    try:
        paused = graph.invoke(
            create_initial_state(neurological_case(), run_id="RUN-MEMORY"),
            config,
        )
        paused_snapshot = validate_state(paused)
        repository.save_run(paused_snapshot, "waiting_for_human")
        command = HumanReviewCommand(
            action="reject",
            reviewer="reviewer-1",
            feedback="Retain for audit only.",
        )
        repository.save_human_feedback(paused_snapshot.run_id, command)

        restored = repository.load_run("RUN-MEMORY")
        assert restored == paused_snapshot
        assert repository.run_status("RUN-MEMORY") == "waiting_for_human"
        assert repository.feedback_count("RUN-MEMORY") == 1
        assert repository.message_count("RUN-MEMORY") == len(paused_snapshot.agent_messages)
        assert repository.trace_count("RUN-MEMORY") == len(paused_snapshot.execution_trace)
    finally:
        checkpoint.close()


def test_service_resumes_same_run_after_checkpoint_process_reopen(tmp_path: Path) -> None:
    repository = CaseMemoryRepository(Database(tmp_path / "case_history.db"))
    graph, first_checkpoint = build_graph(tmp_path)
    service = WorkflowService(graph, repository)
    started = service.start(neurological_case(), "RUN-REOPEN")

    assert started.interrupted is True
    assert repository.run_status("RUN-REOPEN") == "waiting_for_human"
    first_checkpoint.close()

    reopened_store = KnowledgeStore(tmp_path / "chroma")
    reopened_checkpoint = WorkflowCheckpoint(tmp_path / "workflow.db")
    reopened_graph = build_reviewable_workflow(
        DemoModelProvider(),
        reopened_store,
        max_revisions=2,
        checkpointer=reopened_checkpoint.saver,
    )
    reopened_service = WorkflowService(reopened_graph, repository)
    try:
        completed = reopened_service.resume(
            "RUN-REOPEN",
            HumanReviewCommand(
                action="approve",
                reviewer="reviewer-2",
                feedback="Approved after reopening the process.",
            ),
        )

        assert completed.interrupted is False
        assert completed.snapshot.final_report is not None
        assert repository.run_status("RUN-REOPEN") == "approved"
        assert repository.feedback_count("RUN-REOPEN") == 1
        assert repository.load_run("RUN-REOPEN") == completed.snapshot
    finally:
        reopened_checkpoint.close()


def test_case_deletion_cascades_audit_history(tmp_path: Path) -> None:
    repository = CaseMemoryRepository(Database(tmp_path / "case_history.db"))
    graph, checkpoint = build_graph(tmp_path)
    try:
        service = WorkflowService(graph, repository)
        service.start(neurological_case(), "RUN-DELETE")

        assert repository.delete_case("CASE-HITL-NEURO") is True
        assert repository.load_run("RUN-DELETE") is None
        assert repository.list_runs() == []
    finally:
        checkpoint.close()


def test_service_deletes_case_history_and_all_workflow_checkpoints(
    tmp_path: Path,
) -> None:
    graph, checkpoint = build_graph(tmp_path)
    repository = CaseMemoryRepository(Database(tmp_path / "delete_history.db"))
    service = WorkflowService(graph, repository, checkpoint)
    try:
        service.start(neurological_case(), "RUN-DELETE-ALL-1")
        service.start(neurological_case(), "RUN-DELETE-ALL-2")

        assert service.delete_case("CASE-HITL-NEURO") is True
        assert repository.load_run("RUN-DELETE-ALL-1") is None
        assert repository.load_run("RUN-DELETE-ALL-2") is None
        for run_id in ("RUN-DELETE-ALL-1", "RUN-DELETE-ALL-2"):
            state = graph.get_state({"configurable": {"thread_id": run_id}})
            assert state.values == {}
            assert not state.interrupts
    finally:
        checkpoint.close()
