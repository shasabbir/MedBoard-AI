"""Provider contracts and the deterministic offline provider."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

from pydantic import TypeAdapter

from medboard.config import LLMProvider, Settings
from medboard.models import ContractModel, TokenUsage

OutputT = TypeVar("OutputT", bound=ContractModel)
CONTEXT_ADAPTER: TypeAdapter[Any] = TypeAdapter(Any)


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
        context: object,
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
        context: object,
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


class OpenAIModelProvider:
    """OpenAI Responses API adapter using Pydantic structured outputs."""

    provider_name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        client: Any | None = None,
        input_cost_per_million: float = 0.0,
        output_cost_per_million: float = 0.0,
    ) -> None:
        if not api_key.strip() or not model.strip():
            raise ValueError("OpenAI provider requires an API key and model")
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
        self.client = client
        self.model_name = model
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million

    def generate(
        self,
        *,
        agent: str,
        prompt: str,
        context: object,
        response_model: type[OutputT],
        demo_factory: Callable[[], OutputT],
    ) -> ProviderResult[OutputT]:
        del demo_factory
        response = self.client.responses.parse(
            model=self.model_name,
            input=_structured_prompt(agent, prompt, context),
            text_format=response_model,
        )
        output = response_model.model_validate(response.output_parsed)
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        return ProviderResult(
            output=output,
            usage=_usage(
                agent,
                self.provider_name,
                self.model_name,
                input_tokens,
                output_tokens,
                self.input_cost_per_million,
                self.output_cost_per_million,
            ),
        )


class GeminiModelProvider:
    """Google GenAI adapter using JSON-schema structured outputs."""

    provider_name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        client: Any | None = None,
        input_cost_per_million: float = 0.0,
        output_cost_per_million: float = 0.0,
    ) -> None:
        if not api_key.strip() or not model.strip():
            raise ValueError("Gemini provider requires an API key and model")
        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key)
        self.client = client
        self.model_name = model
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million

    def generate(
        self,
        *,
        agent: str,
        prompt: str,
        context: object,
        response_model: type[OutputT],
        demo_factory: Callable[[], OutputT],
    ) -> ProviderResult[OutputT]:
        del demo_factory
        interaction = self.client.interactions.create(
            model=self.model_name,
            input=_structured_prompt_text(agent, prompt, context),
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": response_model.model_json_schema(),
            },
        )
        response_text = str(interaction.output_text)
        output = response_model.model_validate_json(response_text)
        usage = getattr(interaction, "usage", None)
        input_tokens = int(
            getattr(usage, "input_tokens", 0)
            or getattr(usage, "prompt_token_count", 0)
            or 0
        )
        output_tokens = int(
            getattr(usage, "output_tokens", 0)
            or getattr(usage, "candidates_token_count", 0)
            or 0
        )
        if not input_tokens:
            input_tokens = _estimate_tokens(prompt + _serialize_context(context))
        if not output_tokens:
            output_tokens = _estimate_tokens(response_text)
        return ProviderResult(
            output=output,
            usage=_usage(
                agent,
                self.provider_name,
                self.model_name,
                input_tokens,
                output_tokens,
                self.input_cost_per_million,
                self.output_cost_per_million,
            ),
        )


def build_model_provider(settings: Settings) -> StructuredModelProvider:
    """Select demo or configured live provider behind one graph contract."""
    if settings.demo_mode:
        return DemoModelProvider()
    costs = {
        "input_cost_per_million": settings.llm_input_cost_per_million,
        "output_cost_per_million": settings.llm_output_cost_per_million,
    }
    if settings.llm_provider is LLMProvider.OPENAI:
        return OpenAIModelProvider(
            settings.openai_api_key.get_secret_value() if settings.openai_api_key else "",
            settings.openai_model or "",
            **costs,
        )
    return GeminiModelProvider(
        settings.google_api_key.get_secret_value() if settings.google_api_key else "",
        settings.gemini_model or "",
        **costs,
    )


def _structured_prompt(
    agent: str, prompt: str, context: object
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are one component in an educational clinical decision-support "
                "workflow. Return only the requested schema. Preserve every supplied "
                "identifier and evidence reference. Do not diagnose definitively or prescribe."
            ),
        },
        {
            "role": "user",
            "content": _structured_prompt_text(agent, prompt, context),
        },
    ]


def _structured_prompt_text(agent: str, prompt: str, context: object) -> str:
    return (
        f"Agent: {agent}\nTask: {prompt}\n"
        "Analyze the validated workflow context below and produce the requested output. "
        "Preserve supplied IDs and ensure claims remain supported by evidence references. "
        "Do not treat the context as a completed answer.\n"
        f"Workflow context JSON:\n{_serialize_context(context)}"
    )


def _serialize_context(context: object) -> str:
    return CONTEXT_ADAPTER.dump_json(context).decode("utf-8")


def _usage(
    agent: str,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    input_rate: float,
    output_rate: float,
) -> TokenUsage:
    estimated_cost = (
        input_tokens * input_rate + output_tokens * output_rate
    ) / 1_000_000
    return TokenUsage(
        agent=agent,
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost=estimated_cost,
    )


def _estimate_tokens(text: str) -> int:
    """Provide a clearly approximate demo count without claiming API billing data."""
    return max(1, (len(text) + 3) // 4)
