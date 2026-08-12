"""Common execution boundary for all MedBoard agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any

from medboard.graph.state import MedicalCaseState
from medboard.models import (
    AgentError,
    AgentStatus,
    Severity,
    TraceEvent,
    TraceEventType,
)
from medboard.providers import StructuredModelProvider

StateUpdate = dict[str, Any]


class BaseAgent(ABC):
    """Run specialized analysis behind consistent trace and failure handling."""

    name: str

    def __init__(self, provider: StructuredModelProvider) -> None:
        self.provider = provider

    def __call__(self, state: MedicalCaseState) -> StateUpdate:
        started = perf_counter()
        start_event = TraceEvent(
            event_type=TraceEventType.AGENT_STARTED,
            agent=self.name,
            status=AgentStatus.RUNNING,
        )
        try:
            update = self.analyze(state)
        except Exception as exc:  # failures become visible graph state
            duration_ms = (perf_counter() - started) * 1_000
            return {
                "errors": [
                    AgentError(
                        agent=self.name,
                        error_type=type(exc).__name__,
                        message=str(exc),
                        severity=Severity.HIGH,
                        retryable=False,
                    )
                ],
                "execution_trace": [
                    start_event,
                    TraceEvent(
                        event_type=TraceEventType.AGENT_FAILED,
                        agent=self.name,
                        status=AgentStatus.FAILED,
                        duration_ms=duration_ms,
                    ),
                ],
            }

        duration_ms = (perf_counter() - started) * 1_000
        trace = list(update.get("execution_trace", []))
        trace.extend(
            [
                start_event,
                TraceEvent(
                    event_type=TraceEventType.AGENT_COMPLETED,
                    agent=self.name,
                    status=AgentStatus.COMPLETED,
                    duration_ms=duration_ms,
                ),
            ]
        )
        update["execution_trace"] = trace
        return update

    @abstractmethod
    def analyze(self, state: MedicalCaseState) -> StateUpdate:
        """Return a validated partial graph-state update."""
