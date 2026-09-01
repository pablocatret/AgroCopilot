from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator
import uuid

from libs.costs.calculator import (
    CostBreakdown,
    calculate_embedding_cost,
    calculate_text_cost,
    calculate_tool_cost,
)


current_conversation_id: ContextVar[str | None] = ContextVar("cost_conversation_id", default=None)
current_agent: ContextVar[str | None] = ContextVar("cost_agent", default=None)
current_operation: ContextVar[str | None] = ContextVar("cost_operation", default=None)
current_capture: ContextVar[list[dict] | None] = ContextVar("cost_capture", default=None)


@contextmanager
def cost_context(
    *,
    conversation_id: str | None = None,
    agent: str | None = None,
    operation: str | None = None,
) -> Iterator[None]:
    tokens = []
    if conversation_id is not None:
        tokens.append((current_conversation_id, current_conversation_id.set(conversation_id)))
    if agent is not None:
        tokens.append((current_agent, current_agent.set(agent)))
    if operation is not None:
        tokens.append((current_operation, current_operation.set(operation)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def start_cost_capture():
    return current_capture.set([])


def finish_cost_capture(token) -> list[dict]:
    captured = current_capture.get() or []
    current_capture.reset(token)
    parent_capture = current_capture.get()
    if parent_capture is not None:
        parent_capture.extend(captured)
    return captured


def _event_from_breakdown(
    breakdown: CostBreakdown,
    *,
    operation: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict:
    merged_metadata = dict(breakdown.metadata or {})
    if metadata:
        merged_metadata.update(metadata)
    return {
        "id": uuid.uuid4().hex,
        "conversation_id": current_conversation_id.get(),
        "agent": current_agent.get(),
        "operation": operation or current_operation.get() or "unknown",
        "provider": breakdown.provider,
        "model": breakdown.model,
        "pricing_mode": breakdown.pricing_mode,
        "input_tokens": breakdown.input_tokens,
        "cached_input_tokens": breakdown.cached_input_tokens,
        "output_tokens": breakdown.output_tokens,
        "total_tokens": breakdown.total_tokens,
        "unit_count": breakdown.unit_count,
        "estimated": breakdown.estimated,
        "cost_usd": breakdown.cost_usd,
        "metadata": merged_metadata,
    }


def record_cost_breakdown(
    breakdown: CostBreakdown,
    *,
    operation: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict:
    event = _event_from_breakdown(breakdown, operation=operation, metadata=metadata)
    capture = current_capture.get()
    if capture is not None:
        capture.append(event)
    try:
        from backend.cost_store import cost_store
        from backend.deps import settings

        if settings.COST_TRACKING_ENABLED:
            cost_store.insert_event(event)
    except Exception:
        # Cost tracking must never break product execution.
        pass
    return event


def record_openai_chat_usage(
    model: str,
    usage: Any,
    *,
    operation: str = "chat.completions",
    metadata: dict[str, Any] | None = None,
    provider: str = "openai",
) -> dict:
    from backend.deps import settings

    breakdown = calculate_text_cost(
        model, usage, pricing_mode=settings.COST_PRICING_MODE, provider=provider
    )
    return record_cost_breakdown(breakdown, operation=operation, metadata=metadata)


def record_openai_embedding_usage(
    model: str,
    usage: Any,
    *,
    operation: str = "embeddings",
    metadata: dict[str, Any] | None = None,
    provider: str = "openai",
) -> dict:
    from backend.deps import settings

    breakdown = calculate_embedding_cost(
        model, usage, pricing_mode=settings.COST_PRICING_MODE, provider=provider
    )
    return record_cost_breakdown(breakdown, operation=operation, metadata=metadata)


def record_web_search_call(
    provider: str,
    *,
    operation: str = "web_search",
    unit_count: int = 1,
    metadata: dict[str, Any] | None = None,
) -> dict:
    from backend.deps import settings

    provider_name = (provider or "web").lower()
    price = (
        settings.OPENAI_WEB_SEARCH_COST_USD_PER_1K
        if provider_name.startswith("openai")
        else settings.WEB_SEARCH_COST_USD_PER_1K
    )
    breakdown = calculate_tool_cost(
        f"{provider_name}-search",
        unit_count=unit_count,
        unit_price_per_1k=price,
        provider=provider_name,
        pricing_mode=settings.COST_PRICING_MODE,
        estimated=price == 0,
    )
    return record_cost_breakdown(breakdown, operation=operation, metadata=metadata)


def summarize_captured_events(events: list[dict]) -> dict[str, float | int]:
    return {
        "input_tokens": sum(int(event.get("input_tokens") or 0) for event in events),
        "output_tokens": sum(int(event.get("output_tokens") or 0) for event in events),
        "total_tokens": sum(int(event.get("total_tokens") or 0) for event in events),
        "cost_usd": sum(float(event.get("cost_usd") or 0.0) for event in events),
    }
