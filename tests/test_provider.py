"""Tests for the deterministic provider boundary."""

from types import SimpleNamespace

import pytest

from medboard.config import Settings
from medboard.models import SupervisorPlan
from medboard.providers import (
    DemoModelProvider,
    GeminiModelProvider,
    OpenAIModelProvider,
    build_model_provider,
)


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


def test_openai_provider_parses_pydantic_output_and_calculates_cost() -> None:
    expected = SupervisorPlan(
        case_categories=["general"],
        initial_agents=["history"],
        reasoning="Review history first.",
    )
    parse_calls: list[dict[str, object]] = []

    def parse(**kwargs: object) -> object:
        parse_calls.append(kwargs)
        return SimpleNamespace(
            output_parsed=expected,
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        )

    client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    provider = OpenAIModelProvider(
        "test-key",
        "test-model",
        client=client,
        input_cost_per_million=2,
        output_cost_per_million=4,
    )

    result = provider.generate(
        agent="supervisor",
        prompt="Create a plan",
        response_model=SupervisorPlan,
        demo_factory=lambda: expected,
    )

    assert result.output == expected
    assert result.usage.estimated_cost == pytest.approx(0.0004)
    assert parse_calls[0]["text_format"] is SupervisorPlan


def test_gemini_provider_validates_json_output_and_usage() -> None:
    expected = SupervisorPlan(
        case_categories=["general"],
        initial_agents=["history"],
        reasoning="Review history first.",
    )
    create_calls: list[dict[str, object]] = []

    def create(**kwargs: object) -> object:
        create_calls.append(kwargs)
        return SimpleNamespace(
            output_text=expected.model_dump_json(),
            usage=SimpleNamespace(input_tokens=80, output_tokens=40),
        )

    client = SimpleNamespace(interactions=SimpleNamespace(create=create))
    provider = GeminiModelProvider("test-key", "test-model", client=client)

    result = provider.generate(
        agent="supervisor",
        prompt="Create a plan",
        response_model=SupervisorPlan,
        demo_factory=lambda: expected,
    )

    assert result.output == expected
    response_format = create_calls[0]["response_format"]
    assert isinstance(response_format, dict)
    assert response_format["mime_type"] == "application/json"


def test_provider_factory_preserves_demo_default() -> None:
    assert isinstance(build_model_provider(Settings(_env_file=None)), DemoModelProvider)
