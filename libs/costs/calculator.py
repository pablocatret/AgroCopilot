from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from libs.costs.pricing import get_model_price, normalize_model_name


PricingMode = Literal["standard", "batch", "flex", "priority"]


@dataclass(frozen=True)
class CostBreakdown:
    provider: str
    model: str
    pricing_mode: str
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    unit_count: int = 0
    cost_usd: float = 0.0
    estimated: bool = False
    metadata: dict[str, Any] | None = None


def _usage_value(usage: Any, *names: str, default: int = 0) -> int:
    for name in names:
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        if isinstance(value, int):
            return value
    return default


def _usage_details_value(usage: Any, details_name: str, value_name: str) -> int:
    details = (
        usage.get(details_name) if isinstance(usage, dict) else getattr(usage, details_name, None)
    )
    if not details:
        return 0
    if isinstance(details, dict):
        value = details.get(value_name)
    else:
        value = getattr(details, value_name, None)
    return value if isinstance(value, int) else 0


def extract_chat_usage(usage: Any) -> tuple[int, int, int, int]:
    input_tokens = _usage_value(usage, "prompt_tokens", "input_tokens")
    output_tokens = _usage_value(usage, "completion_tokens", "output_tokens")
    total_tokens = _usage_value(usage, "total_tokens", default=input_tokens + output_tokens)
    cached_tokens = _usage_details_value(usage, "prompt_tokens_details", "cached_tokens")
    return input_tokens, cached_tokens, output_tokens, total_tokens


def extract_embedding_usage(usage: Any) -> tuple[int, int]:
    input_tokens = _usage_value(usage, "prompt_tokens", "input_tokens", "total_tokens")
    total_tokens = _usage_value(usage, "total_tokens", default=input_tokens)
    return input_tokens, total_tokens


def calculate_text_cost(
    model: str, usage: Any, *, pricing_mode: PricingMode = "standard", provider: str = "openai"
) -> CostBreakdown:
    input_tokens, cached_tokens, output_tokens, total_tokens = extract_chat_usage(usage)
    normalized = normalize_model_name(model)
    price = get_model_price(normalized, "text")
    if price is None:
        return CostBreakdown(
            provider=provider,
            model=model,
            pricing_mode=pricing_mode,
            input_tokens=input_tokens,
            cached_input_tokens=cached_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated=True,
            metadata={"reason": "unknown_model", "normalized_model": normalized},
        )
    billable_input = max(0, input_tokens - cached_tokens)
    input_rate = price.input_per_million
    output_rate = price.output_per_million
    cached_rate = (
        price.cached_input_per_million if price.cached_input_per_million is not None else input_rate
    )
    if pricing_mode == "batch":
        input_rate = (
            price.batch_input_per_million
            if price.batch_input_per_million is not None
            else input_rate * 0.5
        )
        output_rate = (
            price.batch_output_per_million
            if price.batch_output_per_million is not None
            else output_rate * 0.5
        )
        cached_rate = input_rate
    cost = (
        billable_input * input_rate + cached_tokens * cached_rate + output_tokens * output_rate
    ) / 1_000_000
    return CostBreakdown(
        provider=provider,
        model=normalized,
        pricing_mode=pricing_mode,
        input_tokens=input_tokens,
        cached_input_tokens=cached_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost_usd=cost,
        estimated=False,
    )


def calculate_embedding_cost(
    model: str, usage: Any, *, pricing_mode: PricingMode = "standard", provider: str = "openai"
) -> CostBreakdown:
    input_tokens, total_tokens = extract_embedding_usage(usage)
    normalized = normalize_model_name(model)
    price = get_model_price(normalized, "embedding")
    if price is None:
        return CostBreakdown(
            provider=provider,
            model=model,
            pricing_mode=pricing_mode,
            input_tokens=input_tokens,
            total_tokens=total_tokens,
            estimated=True,
            metadata={"reason": "unknown_embedding_model", "normalized_model": normalized},
        )
    rate = (
        price.batch_input_per_million
        if pricing_mode == "batch" and price.batch_input_per_million is not None
        else price.input_per_million
    )
    return CostBreakdown(
        provider=provider,
        model=normalized,
        pricing_mode=pricing_mode,
        input_tokens=input_tokens,
        total_tokens=total_tokens,
        cost_usd=(input_tokens * rate) / 1_000_000,
        estimated=False,
    )


def calculate_tool_cost(
    tool: str,
    *,
    unit_count: int = 1,
    unit_price_per_1k: float = 0.0,
    provider: str = "web",
    pricing_mode: PricingMode = "standard",
    estimated: bool = False,
) -> CostBreakdown:
    cost = (unit_count * unit_price_per_1k) / 1000
    return CostBreakdown(
        provider=provider,
        model=tool,
        pricing_mode=pricing_mode,
        unit_count=unit_count,
        cost_usd=cost,
        estimated=estimated,
    )
