"""LangGraph state and workflow definitions."""

from typing import TYPE_CHECKING

from medboard.graph.state import (
    MedicalCaseSnapshot,
    MedicalCaseState,
    create_initial_state,
    validate_state,
)

if TYPE_CHECKING:
    from medboard.graph.workflow import (
        build_collaboration_workflow,
        build_initial_workflow,
        build_reviewable_workflow,
    )

__all__ = [
    "MedicalCaseSnapshot",
    "MedicalCaseState",
    "create_initial_state",
    "validate_state",
    "build_initial_workflow",
    "build_collaboration_workflow",
    "build_reviewable_workflow",
]


def __getattr__(name: str) -> object:
    """Load workflow builders lazily to avoid agent/graph import cycles."""
    if name in {
        "build_initial_workflow",
        "build_collaboration_workflow",
        "build_reviewable_workflow",
    }:
        from medboard.graph import workflow

        return getattr(workflow, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
