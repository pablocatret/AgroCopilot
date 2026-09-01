from __future__ import annotations

from typing import Any


def chat_temperature_kwargs(model: str | None, temperature: float | None) -> dict[str, Any]:
    """Return temperature kwargs only for chat models that accept non-default values."""
    if temperature is None:
        return {}
    normalized = (model or "").strip().lower()
    if normalized.startswith("gpt-5"):
        return {}
    return {"temperature": temperature}


def completion_token_kwargs(
    model: str | None,
    provider: str | None,
    max_tokens: int | None,
) -> dict[str, int]:
    """Return the token-limit parameter accepted by the target endpoint.

    OpenAI's GPT-5 chat endpoints reject the legacy ``max_tokens`` field and
    require ``max_completion_tokens``. OpenRouter keeps the OpenAI-compatible
    legacy field for the evaluated provider routes, so the switch must depend
    on the actual provider as well as the model family.
    """
    if not max_tokens:
        return {}
    normalized_model = (model or "").strip().lower()
    normalized_provider = (provider or "").strip().lower()
    if normalized_provider == "openai" and (
        normalized_model.startswith("gpt-5") or normalized_model.startswith("o1")
        or normalized_model.startswith("o3") or normalized_model.startswith("o4")
    ):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}


def tool_reasoning_kwargs(model: str | None, provider: str | None) -> dict[str, str]:
    """Return compatibility flags for model-specific tool endpoints."""
    normalized_model = (model or "").strip().lower()
    normalized_provider = (provider or "").strip().lower()
    if normalized_provider == "openai" and normalized_model == "gpt-5.6-luna":
        return {"reasoning_effort": "none"}
    return {}
