"""Operational facade joining checkpoints with durable case-history memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from medboard.graph.state import MedicalCaseSnapshot, create_initial_state, validate_state
from medboard.memory import CaseMemoryRepository
from medboard.memory import WorkflowCheckpoint
from medboard.models import HumanReviewCommand, MedicalCaseInput


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    snapshot: MedicalCaseSnapshot
    interrupted: bool


class WorkflowService:
    """Start and resume checkpointed runs while maintaining a durable audit copy."""

    def __init__(
        self,
        graph: CompiledStateGraph,
        case_memory: CaseMemoryRepository,
        checkpoint: WorkflowCheckpoint | None = None,
    ) -> None:
        self.graph = graph
        self.case_memory = case_memory
        self.checkpoint = checkpoint

    def start(self, case: MedicalCaseInput, run_id: str) -> WorkflowResult:
        config = self._config(run_id)
        result = self.graph.invoke(create_initial_state(case, run_id=run_id), config)
        return self._persist_result(result, config)

    def resume(self, run_id: str, command: HumanReviewCommand) -> WorkflowResult:
        if self.case_memory.load_run(run_id) is None:
            raise KeyError(f"unknown persisted run: {run_id}")
        config = self._config(run_id)
        if not self.graph.get_state(config).interrupts:
            raise ValueError(f"run is not waiting for human review: {run_id}")
        self.case_memory.save_human_feedback(run_id, command)
        result = self.graph.invoke(
            Command(resume=command.model_dump(mode="json")),
            config,
        )
        return self._persist_result(result, config)

    def delete_case(self, case_id: str) -> bool:
        """Delete case history and every corresponding resumable checkpoint."""
        run_ids = self.case_memory.run_ids_for_case(case_id)
        if not run_ids:
            return False
        if self.checkpoint is None:
            raise RuntimeError("complete case deletion requires the checkpoint store")
        for run_id in run_ids:
            self.checkpoint.delete_run(run_id)
        return self.case_memory.delete_case(case_id)

    def _persist_result(
        self,
        state: dict[str, Any],
        config: dict[str, dict[str, str]],
    ) -> WorkflowResult:
        snapshot = validate_state(state)
        interrupted = bool(self.graph.get_state(config).interrupts)
        status = (
            "waiting_for_human"
            if interrupted
            else snapshot.human_review.status.value
        )
        self.case_memory.save_run(snapshot, status)
        return WorkflowResult(snapshot=snapshot, interrupted=interrupted)

    @staticmethod
    def _config(run_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": run_id}}
