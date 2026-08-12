"""Provider contracts and the deterministic offline provider."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from medboard.models import ContractModel, TokenUsage

OutputT = TypeVar("OutputT", bound=ContractModel)


@dataclass(frozen=True, slots=True)
class ProviderResult(Generic[OutputT]):
    """A validated model response and its auditable usage record."""

    output: OutputT
    usage: TokenUsage


class StructuredModelProvider(Protocol):
    """Interface used by agents to request validated structured output."""

    provider_name: str
    model_name: str

    def generate(
        self,
        *,
        agent: str,
        prompt: str,
        response_model: type[OutputT],
        demo_factory: Callable[[], OutputT],
    ) -> ProviderResult[OutputT]: ...


class DemoModelProvider:
    """Predictable offline provider that still crosses the production contract."""

    provider_name = "demo"
    model_name = "deterministic-v1"

    def generate(
        self,
        *,
        agent: str,
        prompt: str,
        response_model: type[OutputT],
        demo_factory: Callable[[], OutputT],
    ) -> ProviderResult[OutputT]:
        if not prompt.strip():
            raise ValueError("provider prompt cannot be empty")

        output = demo_factory()
        if not isinstance(output, response_model):
            raise TypeError(
                f"{agent} produced {type(output).__name__}; "
                f"expected {response_model.__name__}"
            )

        serialized_output = output.model_dump_json()
        usage = TokenUsage(
            agent=agent,
            provider=self.provider_name,
            model=self.model_name,
            input_tokens=_estimate_tokens(prompt),
            output_tokens=_estimate_tokens(serialized_output),
            estimated_cost=0.0,
        )
        return ProviderResult(output=output, usage=usage)


def _estimate_tokens(text: str) -> int:
    """Provide a clearly approximate demo count without claiming API billing data."""
    return max(1, (len(text) + 3) // 4)
