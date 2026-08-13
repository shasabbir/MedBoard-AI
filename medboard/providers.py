"""Provider contracts and the deterministic offline provider."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from threading import Lock
from typing import Any, Generic, Protocol, TypeVar

from pydantic import TypeAdapter

from medboard.config import LLMProvider, Settings
from medboard.models import ContractModel, TokenUsage

OutputT = TypeVar("OutputT", bound=ContractModel)
CONTEXT_ADAPTER: TypeAdapter[Any] = TypeAdapter(Any)
RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 409, 425, 429})


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
        timeout_seconds: float = 60.0,
    ) -> None:
        if not api_key.strip() or not model.strip():
            raise ValueError("OpenAI provider requires an API key and model")
        if timeout_seconds <= 0:
            raise ValueError("provider timeout must be positive")
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
        self.client = client
        self.model_name = model
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self.timeout_seconds = timeout_seconds

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
            timeout=self.timeout_seconds,
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
        timeout_seconds: float = 60.0,
    ) -> None:
        if not api_key.strip() or not model.strip():
            raise ValueError("Gemini provider requires an API key and model")
        if timeout_seconds <= 0:
            raise ValueError("provider timeout must be positive")
        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key)
        self.client = client
        self.model_name = model
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self.timeout_seconds = timeout_seconds
        self._request_lock = Lock()

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
        # A shared provider serves LangGraph's parallel branches. Serializing live
        # calls avoids a burst of simultaneous requests against free-tier quotas.
        with self._request_lock:
            interaction = self.client.interactions.create(
                model=self.model_name,
                input=_structured_prompt_text(agent, prompt, context),
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": response_model.model_json_schema(),
                },
                timeout=self.timeout_seconds,
            )
        response_text = _ground_gemini_json_references(
            _gemini_response_text(interaction), context
        )
        output = response_model.model_validate_json(response_text)
        usage = getattr(interaction, "usage", None)
        input_tokens = int(
            getattr(usage, "total_input_tokens", 0)
            or getattr(usage, "input_tokens", 0)
            or getattr(usage, "prompt_token_count", 0)
            or 0
        )
        output_tokens = int(
            getattr(usage, "total_output_tokens", 0)
            or getattr(usage, "output_tokens", 0)
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
        "timeout_seconds": settings.agent_timeout_seconds,
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


def _gemini_response_text(interaction: object) -> str:
    """Read structured text from the SDK 2.x steps response."""
    for step in reversed(list(getattr(interaction, "steps", None) or [])):
        if getattr(step, "type", None) != "model_output":
            continue
        parts = [
            str(text)
            for item in (getattr(step, "content", None) or [])
            if getattr(item, "type", None) == "text"
            and (text := getattr(item, "text", None))
        ]
        if parts:
            return "".join(parts)

    # SDK 2.x currently exposes this convenience accessor as well. Retaining the
    # fallback keeps injected test clients and minor SDK response variants safe.
    output_text = getattr(interaction, "output_text", None)
    if output_text:
        return str(output_text)
    raise ValueError("Gemini response did not contain model text")


def is_retryable_provider_error(error: Exception) -> bool:
    """Return whether a provider failure may succeed without request changes."""
    status_code = _provider_status_code(error)
    if status_code is None:
        return True
    if status_code in RETRYABLE_HTTP_STATUS_CODES:
        return True
    return status_code >= 500


def provider_retry_delay_seconds(error: Exception, attempt: int) -> float:
    """Return bounded backoff for transient HTTP failures only."""
    status_code = _provider_status_code(error)
    if status_code is None or not is_retryable_provider_error(error):
        return 0.0
    retry_after = _retry_after_seconds(error)
    if retry_after is not None:
        return min(30.0, max(0.0, retry_after))
    return min(8.0, float(2 ** (attempt - 1)))


def _provider_status_code(error: Exception) -> int | None:
    for value in (
        getattr(error, "status_code", None),
        getattr(error, "code", None),
        getattr(getattr(error, "response", None), "status_code", None),
    ):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _retry_after_seconds(error: Exception) -> float | None:
    headers = getattr(getattr(error, "response", None), "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _ground_gemini_json_references(response_text: str, context: object) -> str:
    """Remove model-authored reference links that are not present in context."""
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        return response_text

    context_payload = CONTEXT_ADAPTER.dump_python(context, mode="json")
    allowed_evidence = _collect_identifier_values(context_payload, "evidence_id")
    evidence_fields = {
        "evidence_ids",
        "supporting_evidence_ids",
        "contradicting_evidence_ids",
    }

    def ground(value: object) -> object:
        if isinstance(value, list):
            return [ground(item) for item in value]
        if not isinstance(value, dict):
            return value
        grounded: dict[str, object] = {}
        for key, item in value.items():
            if key == "claims":
                # Claims and their evidence links are reconstructed by agents from
                # deterministic evidence, never accepted from model output.
                grounded[key] = []
            elif key in evidence_fields and isinstance(item, list):
                grounded[key] = [
                    reference
                    for reference in item
                    if isinstance(reference, str) and reference in allowed_evidence
                ]
            else:
                grounded[key] = ground(item)
        return grounded

    return json.dumps(ground(payload), separators=(",", ":"))


def _collect_identifier_values(value: object, field_name: str) -> set[str]:
    identifiers: set[str] = set()
    if isinstance(value, list):
        for item in value:
            identifiers.update(_collect_identifier_values(item, field_name))
    elif isinstance(value, dict):
        for key, item in value.items():
            if key == field_name and isinstance(item, str):
                identifiers.add(item)
            else:
                identifiers.update(_collect_identifier_values(item, field_name))
    return identifiers


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
