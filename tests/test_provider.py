"""Tests for the deterministic provider boundary."""

import pytest

from medboard.models import SupervisorPlan
from medboard.providers import DemoModelProvider


def test_demo_provider_returns_validated_output_and_zero_cost_usage() -> None:
    provider = DemoModelProvider()
    result = provider.generate(
        agent="supervisor",
        prompt="Create a plan",
        response_model=SupervisorPlan,
        demo_factory=lambda: SupervisorPlan(
            case_categories=["general"],
            initial_agents=["history"],
            reasoning="History review is required.",
        ),
    )

    assert isinstance(result.output, SupervisorPlan)
    assert result.usage.provider == "demo"
    assert result.usage.total_tokens > 0
    assert result.usage.estimated_cost == 0


def test_demo_provider_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        DemoModelProvider().generate(
            agent="supervisor",
            prompt=" ",
            response_model=SupervisorPlan,
            demo_factory=lambda: SupervisorPlan(
                reasoning="History review is required.",
            ),
        )
