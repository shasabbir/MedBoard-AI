"""Common execution boundary for all MedBoard agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter, sleep
from typing import Any

from medboard.graph.state import MedicalCaseState
from medboard.models import (
    AgentError,
    AgentOutput,
    AgentStatus,
    Claim,
    Severity,
    TraceEvent,
    TraceEventType,
)
from medboard.providers import (
    StructuredModelProvider,
    is_retryable_provider_error,
    provider_retry_delay_seconds,
)

StateUpdate = dict[str, Any]


class BaseAgent(ABC):
    """Run specialized analysis behind consistent trace and failure handling."""

    name: str

    def __init__(
        self, provider: StructuredModelProvider, *, max_retries: int = 2
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        self.provider = provider
        self.max_retries = max_retries

    def __call__(self, state: MedicalCaseState) -> StateUpdate:
        started = perf_counter()
        start_event = TraceEvent(
            event_type=TraceEventType.AGENT_STARTED,
            agent=self.name,
            status=AgentStatus.RUNNING,
        )
        retry_events: list[TraceEvent] = []
        failures: list[Exception] = []
        update: StateUpdate | None = None
        for attempt in range(1, self.max_retries + 2):
            try:
                update = self.analyze(state)
                break
            except Exception as exc:
                failures.append(exc)
                if not is_retryable_provider_error(exc):
                    break
                if attempt <= self.max_retries:
                    retry_delay = provider_retry_delay_seconds(exc, attempt)
                    retry_events.append(
                        TraceEvent(
                            event_type=TraceEventType.AGENT_STARTED,
                            agent=self.name,
                            status=AgentStatus.RUNNING,
                            details={
                                "retry": True,
                                "attempt": attempt + 1,
                                "retry_delay_seconds": retry_delay,
                            },
                        )
                    )
                    if retry_delay:
                        sleep(retry_delay)
        if update is None:
            duration_ms = (perf_counter() - started) * 1_000
            last_error = failures[-1]
            return {
                "errors": [
                    AgentError(
                        agent=self.name,
                        error_type=type(last_error).__name__,
                        message=str(last_error),
                        severity=Severity.HIGH,
                        retryable=is_retryable_provider_error(last_error),
                        attempt=len(failures),
                        details={"retry_limit": self.max_retries},
                    )
                ],
                "execution_trace": [
                    start_event,
                    *retry_events,
                    TraceEvent(
                        event_type=TraceEventType.AGENT_FAILED,
                        agent=self.name,
                        status=AgentStatus.FAILED,
                        duration_ms=duration_ms,
                        details={
                            "attempts": len(failures),
                            "retry_limit": self.max_retries,
                        },
                    ),
                ],
            }

        duration_ms = (perf_counter() - started) * 1_000
        trace = [start_event, *retry_events, *list(update.get("execution_trace", []))]
        trace.extend(
            [
                TraceEvent(
                    event_type=TraceEventType.AGENT_COMPLETED,
                    agent=self.name,
                    status=AgentStatus.COMPLETED,
                    duration_ms=duration_ms,
                    details={"attempts": len(failures) + 1},
                ),
            ]
        )
        update["execution_trace"] = trace
        return update

    @abstractmethod
    def analyze(self, state: MedicalCaseState) -> StateUpdate:
        """Return a validated partial graph-state update."""


def ground_agent_output(
    output: AgentOutput, *, agent: str, claims: list[Claim]
) -> AgentOutput:
    """Keep model narrative while restoring deterministic control fields."""
    return output.model_copy(
        update={
            "agent": agent,
            "status": AgentStatus.COMPLETED,
            "claims": claims,
        }
    )
