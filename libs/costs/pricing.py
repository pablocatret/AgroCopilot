from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PricingKind = Literal["text", "embedding", "tool"]

PRICING_CATALOG_VERSION = "2026-07-22"
PRICING_SOURCES = [
    "https://openai.com/api/pricing/",
    "https://platform.openai.com/docs/pricing/",
    "https://openrouter.ai/api/v1/models",
]


@dataclass(frozen=True)
class ModelPrice:
    kind: PricingKind
    input_per_million: float = 0.0
    cached_input_per_million: float | None = None
    output_per_million: float = 0.0
    batch_input_per_million: float | None = None
    batch_output_per_million: float | None = None
    unit_per_1k: float | None = None


TEXT_PRICES: dict[str, ModelPrice] = {
    "gpt-5.4": ModelPrice("text", 2.50, 0.25, 15.00),
    "gpt-5.4-mini": ModelPrice("text", 0.75, 0.075, 4.50),
    "gpt-5.4-nano": ModelPrice("text", 0.20, 0.02, 1.25),
    "gpt-5.2": ModelPrice("text", 1.75, 0.175, 14.00),
    "gpt-5.2-chat-latest": ModelPrice("text", 1.75, 0.175, 14.00),
    "gpt-5.2-pro": ModelPrice("text", 21.00, None, 168.00),
    "gpt-5.1": ModelPrice("text", 1.25, 0.125, 10.00),
    "gpt-5.1-chat-latest": ModelPrice("text", 1.25, 0.125, 10.00),
    "gpt-5": ModelPrice("text", 1.25, 0.125, 10.00),
    "gpt-5-mini": ModelPrice("text", 0.25, 0.025, 2.00),
    "gpt-5.6-luna": ModelPrice("text", 1.00, 0.10, 6.00),
    "gpt-5-nano": ModelPrice("text", 0.05, 0.005, 0.40),
    "gpt-5-pro": ModelPrice("text", 15.00, None, 120.00),
    "gpt-5-chat-latest": ModelPrice("text", 1.25, 0.125, 10.00),
    "gpt-4.1": ModelPrice("text", 2.00, 0.50, 8.00),
    "gpt-4.1-mini": ModelPrice("text", 0.40, 0.10, 1.60),
    "gpt-4.1-nano": ModelPrice("text", 0.10, 0.025, 0.40),
    "gpt-4o": ModelPrice("text", 2.50, 1.25, 10.00),
    "gpt-4o-2024-05-13": ModelPrice("text", 5.00, None, 15.00),
    "gpt-4o-mini": ModelPrice("text", 0.15, 0.075, 0.60),
    "chatgpt-4o-latest": ModelPrice("text", 5.00, None, 15.00),
    "gpt-realtime": ModelPrice("text", 4.00, 0.40, 16.00),
    "gpt-realtime-1.5": ModelPrice("text", 4.00, 0.40, 16.00),
    "gpt-realtime-mini": ModelPrice("text", 0.60, 0.06, 2.40),
    "gpt-4o-realtime-preview": ModelPrice("text", 5.00, 2.50, 20.00),
    "gpt-4o-mini-realtime-preview": ModelPrice("text", 0.60, 0.30, 2.40),
    "gpt-audio": ModelPrice("text", 2.50, None, 10.00),
    "gpt-audio-mini": ModelPrice("text", 0.60, None, 2.40),
    "gpt-4o-audio-preview": ModelPrice("text", 2.50, None, 10.00),
    "gpt-4o-mini-audio-preview": ModelPrice("text", 0.15, None, 0.60),
    "o1": ModelPrice("text", 15.00, 7.50, 60.00),
    "o1-mini": ModelPrice("text", 1.10, 0.55, 4.40),
    "o1-pro": ModelPrice("text", 150.00, None, 600.00),
    "o3": ModelPrice("text", 2.00, 0.50, 8.00),
    "o3-mini": ModelPrice("text", 1.10, 0.55, 4.40),
    "o3-pro": ModelPrice("text", 20.00, None, 80.00),
    "o3-deep-research": ModelPrice("text", 10.00, 2.50, 40.00),
    "o4-mini": ModelPrice("text", 1.10, 0.275, 4.40),
    "o4-mini-deep-research": ModelPrice("text", 2.00, 0.50, 8.00),
    "gpt-5-search-api": ModelPrice("text", 1.25, 0.125, 10.00),
    "gpt-4o-mini-search-preview": ModelPrice("text", 0.15, None, 0.60),
    "gpt-4o-search-preview": ModelPrice("text", 2.50, None, 10.00),
    "computer-use-preview": ModelPrice("text", 3.00, None, 12.00),
    "gpt-image-2": ModelPrice("text", 8.00, 2.00, 30.00),
    "gpt-image-1.5": ModelPrice("text", 8.00, 2.00, 32.00),
    "chatgpt-image-latest": ModelPrice("text", 8.00, 2.00, 32.00),
    "gpt-image-1": ModelPrice("text", 10.00, 2.50, 40.00),
    "gpt-image-1-mini": ModelPrice("text", 2.50, 0.25, 8.00),
    "gpt-3.5-turbo": ModelPrice("text", 0.50, None, 1.50),
    "gpt-3.5-turbo-0125": ModelPrice("text", 0.50, None, 1.50),
    "gpt-3.5-turbo-1106": ModelPrice("text", 1.00, None, 2.00),
    "gpt-3.5-turbo-0613": ModelPrice("text", 1.50, None, 2.00),
    "gpt-3.5-turbo-instruct": ModelPrice("text", 1.50, None, 2.00),
    "gpt-4-turbo-2024-04-09": ModelPrice("text", 10.00, None, 30.00),
    "gpt-4-0125-preview": ModelPrice("text", 10.00, None, 30.00),
    "gpt-4-1106-preview": ModelPrice("text", 10.00, None, 30.00),
    "gpt-4-0613": ModelPrice("text", 30.00, None, 60.00),
    "gpt-4-32k": ModelPrice("text", 60.00, None, 120.00),
}

EMBEDDING_PRICES: dict[str, ModelPrice] = {
    "text-embedding-3-small": ModelPrice(
        "embedding", 0.02, None, 0.0, batch_input_per_million=0.01
    ),
    "text-embedding-3-large": ModelPrice(
        "embedding", 0.13, None, 0.0, batch_input_per_million=0.065
    ),
    "text-embedding-ada-002": ModelPrice(
        "embedding", 0.10, None, 0.0, batch_input_per_million=0.05
    ),
}

TOOL_PRICES: dict[str, ModelPrice] = {
    "openai-web-search": ModelPrice("tool", unit_per_1k=10.00),
    "file-search-tool-call": ModelPrice("tool", unit_per_1k=2.50),
    "code-interpreter-1gb": ModelPrice("tool", unit_per_1k=30.00),
}

# Popular OpenRouter models (prices per million tokens, sourced from OpenRouter API 2026-06-30)
# Organized by: Ultra-cheap -> Budget -> Mid-range -> Frontier, then by provider
OPENROUTER_TEXT_PRICES: dict[str, ModelPrice] = {
    # =========================================================================
    # CHINESE ULTRA-CHEAP (best $/performance ratio)
    # =========================================================================
    "inclusionai/ling-2.6-flash": ModelPrice("text", 0.010, None, 0.030),
    "qwen/qwen-2.5-7b-instruct": ModelPrice("text", 0.040, None, 0.100),
    "qwen/qwen3-8b": ModelPrice("text", 0.050, None, 0.400),
    "z-ai/glm-4.7-flash": ModelPrice("text", 0.060, None, 0.400),
    "tencent/hy3": ModelPrice("text", 0.132, None, 0.528),
    "tencent/hy3-preview": ModelPrice("text", 0.063, None, 0.210),
    "qwen/qwen3-coder-30b-a3b-instruct": ModelPrice("text", 0.070, None, 0.270),
    "inclusionai/ring-2.6-1t": ModelPrice("text", 0.075, None, 0.625),
    "inclusionai/ling-2.6-1t": ModelPrice("text", 0.075, None, 0.625),
    "qwen/qwen3-30b-a3b-thinking-2507": ModelPrice("text", 0.080, None, 0.400),
    "qwen/qwen3-32b": ModelPrice("text", 0.080, None, 0.280),
    "qwen/qwen3-vl-8b-instruct": ModelPrice("text", 0.080, None, 0.500),

    # =========================================================================
    # CHINESE BUDGET (high value, 100K+ context)
    # =========================================================================
    "deepseek/deepseek-v4-flash": ModelPrice("text", 0.090, None, 0.180),
    "qwen/qwen3-235b-a22b-2507": ModelPrice("text", 0.090, None, 0.100),
    "qwen/qwen3-next-80b-a3b-instruct": ModelPrice("text", 0.090, None, 1.100),
    "qwen/qwen3-235b-a22b-thinking-2507": ModelPrice("text", 0.100, None, 0.100),
    "qwen/qwen3-14b": ModelPrice("text", 0.100, None, 0.240),
    "stepfun/step-3.5-flash": ModelPrice("text", 0.100, None, 0.300),
    # OpenRouter current catalog entry used by the evaluation judges.
    "stepfun/step-3.7-flash": ModelPrice("text", 0.200, None, 1.150),
    "minimax/minimax-m2.5": ModelPrice("text", 0.120, None, 0.480),
    "qwen/qwen3-vl-32b-instruct": ModelPrice("text", 0.104, None, 0.416),
    "qwen/qwen3-coder-next": ModelPrice("text", 0.110, None, 0.800),
    "qwen/qwen3-30b-a3b": ModelPrice("text", 0.120, None, 0.500),
    "qwen/qwen3-vl-30b-a3b-instruct": ModelPrice("text", 0.130, None, 0.520),
    "z-ai/glm-4.5-air": ModelPrice("text", 0.130, None, 0.850),
    "tencent/hunyuan-a13b-instruct": ModelPrice("text", 0.140, None, 0.570),
    "minimax/minimax-m2.7": ModelPrice("text", 0.180, None, 0.720),
    "qwen/qwen3-coder-flash": ModelPrice("text", 0.195, None, 0.975),

    # =========================================================================
    # CHINESE MID-RANGE (strong general-purpose)
    # =========================================================================
    "deepseek/deepseek-chat": ModelPrice("text", 0.200, None, 0.800),
    "deepseek/deepseek-chat-v3-0324": ModelPrice("text", 0.200, None, 0.770),
    "minimax/minimax-01": ModelPrice("text", 0.200, None, 1.100),
    "deepseek/deepseek-chat-v3.1": ModelPrice("text", 0.210, None, 0.790),
    "qwen/qwen3-coder": ModelPrice("text", 0.220, None, 1.800),
    "deepseek/deepseek-v3.2": ModelPrice("text", 0.229, None, 0.343),
    "minimax/minimax-m2": ModelPrice("text", 0.255, None, 1.000),
    "qwen/qwen-plus": ModelPrice("text", 0.260, None, 0.780),
    "minimax/minimax-m2.1": ModelPrice("text", 0.290, None, 0.950),
    "deepseek/deepseek-v3.2-exp": ModelPrice("text", 0.270, None, 0.410),
    "minimax/minimax-m2-her": ModelPrice("text", 0.300, None, 1.200),
    "qwen/qwen3.7-plus": ModelPrice("text", 0.320, None, 1.280),
    "qwen/qwen-2.5-72b-instruct": ModelPrice("text", 0.360, None, 0.400),
    "moonshotai/kimi-k2.5": ModelPrice("text", 0.375, None, 2.025),

    # =========================================================================
    # CHINESE HIGH-END (near-frontier performance)
    # =========================================================================
    "z-ai/glm-4.7": ModelPrice("text", 0.400, None, 1.750),
    "minimax/minimax-m1": ModelPrice("text", 0.400, None, 2.200),
    "baidu/ernie-4.5-vl-424b-a47b": ModelPrice("text", 0.420, None, 1.250),
    "z-ai/glm-4.6": ModelPrice("text", 0.430, None, 1.740),
    "deepseek/deepseek-v4-pro": ModelPrice("text", 0.435, None, 0.870),
    "minimax/minimax-m3": ModelPrice("text", 0.300, None, 1.200),
    "qwen/qwen3.6-flash": ModelPrice("text", 0.1875, None, 1.125),
      "xiaomi/mimo-v2.5-pro": ModelPrice("text", 0.435, None, 0.870),
      "xiaomi/mimo-v2.5": ModelPrice("text", 0.105, None, 0.280),
      "xiaomi/mimo-v2-flash": ModelPrice("text", 0.100, None, 0.300),
    "nvidia/nemotron-3-ultra-550b-a55b:free": ModelPrice("text", 0.0, None, 0.0),
    "qwen/qwen3-235b-a22b": ModelPrice("text", 0.455, None, 1.820),
    "deepseek/deepseek-r1-0528": ModelPrice("text", 0.500, None, 2.150),
    "moonshotai/kimi-latest": ModelPrice("text", 0.550, None, 3.200),
    "moonshotai/kimi-k2.6": ModelPrice("text", 0.550, None, 3.200),
    "moonshotai/kimi-k2": ModelPrice("text", 0.570, None, 2.300),
    "z-ai/glm-5": ModelPrice("text", 0.600, None, 1.920),
    "moonshotai/kimi-k2-thinking": ModelPrice("text", 0.600, None, 2.500),
    "z-ai/glm-4.5": ModelPrice("text", 0.600, None, 2.200),
    "z-ai/glm-4.5v": ModelPrice("text", 0.600, None, 1.800),
    "qwen/qwen3-coder-plus": ModelPrice("text", 0.650, None, 3.250),
    "deepseek/deepseek-r1": ModelPrice("text", 0.700, None, 2.500),
    "moonshotai/kimi-k2.7-code": ModelPrice("text", 0.740, None, 3.500),
    "qwen/qwen3-max": ModelPrice("text", 0.780, None, 3.900),
    "qwen/qwen3-max-thinking": ModelPrice("text", 0.780, None, 3.900),
    "deepseek/deepseek-r1-distill-llama-70b": ModelPrice("text", 0.800, None, 0.800),
    "qwen/qwen2.5-vl-72b-instruct": ModelPrice("text", 0.800, None, 1.000),

    # =========================================================================
    # CHINESE FRONTIER (state-of-the-art)
    # =========================================================================
    "z-ai/glm-5.2": ModelPrice("text", 0.940, None, 3.000),
    "z-ai/glm-5.1": ModelPrice("text", 0.975, None, 4.300),
    "qwen/qwen3.6-max-preview": ModelPrice("text", 1.040, None, 6.240),
    "z-ai/glm-5-turbo": ModelPrice("text", 1.200, None, 4.000),
    "qwen/qwen3.7-max": ModelPrice("text", 1.250, None, 3.750),

    # =========================================================================
    # WESTERN ULTRA-CHEAP (open-source, tiny models)
    # =========================================================================
    "meta-llama/llama-3.1-8b-instruct": ModelPrice("text", 0.020, None, 0.030),
    "mistralai/mistral-nemo": ModelPrice("text", 0.020, None, 0.030),
    "openai/gpt-oss-20b": ModelPrice("text", 0.029, None, 0.140),
    "openai/gpt-oss-120b": ModelPrice("text", 0.030, None, 0.180),
    "google/gemma-4-31b-it": ModelPrice("text", 0.120, None, 0.350),

    # =========================================================================
    # WESTERN BUDGET (small open-source, < $0.20/M input)
    # =========================================================================
    "google/gemma-3-4b-it": ModelPrice("text", 0.050, None, 0.100),
    "google/gemma-3-12b-it": ModelPrice("text", 0.050, None, 0.150),
    "mistralai/mistral-small-24b-instruct-2501": ModelPrice("text", 0.050, None, 0.080),
    "openai/gpt-5-nano": ModelPrice("text", 0.050, None, 0.400),
    "google/gemma-3n-e4b-it": ModelPrice("text", 0.060, None, 0.120),
    "mistralai/mistral-small-3.2-24b-instruct": ModelPrice("text", 0.075, None, 0.200),
    "openai/gpt-oss-safeguard-20b": ModelPrice("text", 0.075, None, 0.300),
    "google/gemma-3-27b-it": ModelPrice("text", 0.080, None, 0.160),
    "mistralai/ministral-3b-2512": ModelPrice("text", 0.100, None, 0.100),
    "openai/gpt-4.1-nano": ModelPrice("text", 0.100, None, 0.400),
    "google/gemini-2.5-flash-lite": ModelPrice("text", 0.100, None, 0.400),
    "meta-llama/llama-4-scout": ModelPrice("text", 0.100, None, 0.300),
    "meta-llama/llama-3.3-70b-instruct": ModelPrice("text", 0.100, None, 0.320),
    "mistralai/ministral-8b-2512": ModelPrice("text", 0.150, None, 0.150),
    "mistralai/mistral-small-2603": ModelPrice("text", 0.150, None, 0.600),
    "openai/gpt-4o-mini": ModelPrice("text", 0.150, 0.075, 0.600),
    "meta-llama/llama-4-maverick": ModelPrice("text", 0.150, None, 0.600),
    "google/gemini-2.5-flash-lite-preview-09-2025": ModelPrice("text", 0.100, None, 0.400),
    "mistralai/ministral-14b-2512": ModelPrice("text", 0.200, None, 0.200),

    # =========================================================================
    # WESTERN MID-RANGE ($0.20 - $1.00/M input)
    # =========================================================================
    "openai/gpt-5.4-nano": ModelPrice("text", 0.200, 0.020, 1.250),
    "anthropic/claude-3-haiku": ModelPrice("text", 0.250, 0.025, 1.250),
    "openai/gpt-5-mini": ModelPrice("text", 0.250, 0.025, 2.000),
    "mistralai/codestral-2508": ModelPrice("text", 0.300, None, 0.900),
    "google/gemini-2.5-flash": ModelPrice("text", 0.300, 0.030, 2.500),
    "openai/gpt-4.1-mini": ModelPrice("text", 0.400, 0.100, 1.600),
    "mistralai/devstral-2512": ModelPrice("text", 0.400, None, 2.000),
    "meta-llama/llama-3.1-70b-instruct": ModelPrice("text", 0.400, None, 0.400),
    "google/gemini-3.1-flash-lite-preview": ModelPrice("text", 0.250, None, 1.500),
    "google/gemini-3.1-flash-lite": ModelPrice("text", 0.250, None, 1.500),

    # =========================================================================
    # WESTERN HIGH-END ($1.00 - $5.00/M input)
    # =========================================================================
    "openai/gpt-5.4-mini": ModelPrice("text", 0.750, 0.075, 4.500),
    "anthropic/claude-haiku-4.5": ModelPrice("text", 1.000, 0.100, 5.000),
    "anthropic/claude-sonnet-4": ModelPrice("text", 3.000, 0.300, 15.000),
    "anthropic/claude-sonnet-4.5": ModelPrice("text", 3.000, 0.300, 15.000),
    "anthropic/claude-sonnet-4.6": ModelPrice("text", 3.000, 0.300, 15.000),
    "openai/gpt-5.1": ModelPrice("text", 1.250, 0.125, 10.000),
    "openai/gpt-5": ModelPrice("text", 1.250, 0.125, 10.000),
    "openai/gpt-4.1": ModelPrice("text", 2.000, 0.500, 8.000),
    "openai/o3": ModelPrice("text", 2.000, 0.500, 8.000),
    "openai/o4-mini": ModelPrice("text", 1.100, 0.275, 4.400),
    "openai/o3-mini": ModelPrice("text", 1.100, 0.550, 4.400),
    "openai/gpt-5.4": ModelPrice("text", 2.500, 0.250, 15.000),
    "google/gemini-2.5-pro": ModelPrice("text", 1.250, 0.125, 10.000),
    "google/gemini-3-flash-preview": ModelPrice("text", 0.500, None, 3.000),
    "x-ai/grok-4.20": ModelPrice("text", 1.250, None, 2.500),
    "x-ai/grok-4.3": ModelPrice("text", 1.250, None, 2.500),

    # =========================================================================
    # WESTERN FRONTIER ($5.00+/M input)
    # =========================================================================
    "anthropic/claude-opus-4.5": ModelPrice("text", 5.000, 0.500, 25.000),
    "anthropic/claude-opus-4.6": ModelPrice("text", 5.000, 0.500, 25.000),
    "anthropic/claude-opus-4.7": ModelPrice("text", 5.000, 0.500, 25.000),
    "anthropic/claude-opus-4.8": ModelPrice("text", 5.000, 0.500, 25.000),
    "anthropic/claude-opus-4.8-fast": ModelPrice("text", 10.000, None, 50.000),
    "anthropic/claude-fable-5": ModelPrice("text", 10.000, None, 50.000),
    "openai/gpt-5.5": ModelPrice("text", 5.000, None, 30.000),
    "openai/gpt-5.2-pro": ModelPrice("text", 21.000, None, 168.000),
    "openai/o3-pro": ModelPrice("text", 20.000, None, 80.000),
    "openai/o1": ModelPrice("text", 15.000, 7.500, 60.000),
    "openai/o1-pro": ModelPrice("text", 150.000, None, 600.000),

    # =========================================================================
    # GOOGLE GEMINI (specialized pricing)
    # =========================================================================
    "google/gemini-3.1-pro-preview": ModelPrice("text", 2.000, None, 12.000),
    "google/gemini-3.5-flash": ModelPrice("text", 1.500, None, 9.000),
}


def normalize_model_name(model: str | None) -> str:
    raw = (model or "").strip()
    if not raw:
        return ""
    # OpenRouter routing suffixes select a provider strategy, not a distinct
    # token price.  `:free` is the explicit exception and is priced at zero.
    # Keep the suffix out of the catalog lookup so `foo:nitro` resolves to
    # the published price for `foo`.
    if ":" in raw and "/" in raw:
        raw = raw.split(":", 1)[0]
    aliases = {
        "gpt-4o-latest": "chatgpt-4o-latest",
        "gpt-5.4-2026-03-05": "gpt-5.4",
    }
    if raw in aliases:
        return aliases[raw]
    # Snapshot suffixes generally keep the base model price.
    for base in sorted({*TEXT_PRICES.keys(), *EMBEDDING_PRICES.keys()}, key=len, reverse=True):
        if raw == base or raw.startswith(f"{base}-20"):
            return base
    return raw


def normalize_openrouter_model(model: str | None) -> str:
    """Normalize OpenRouter model name: keep provider/model format for catalog lookup."""
    raw = (model or "").strip()
    if not raw:
        return ""
    # OpenRouter models use provider/model format (e.g., anthropic/claude-3.5-sonnet)
    return raw.split(":", 1)[0] if ":" in raw else raw


def get_model_price(model: str | None, kind: PricingKind = "text") -> ModelPrice | None:
    if kind == "text" and (model or "").strip().lower().endswith(":free"):
        return ModelPrice("text", 0.0, 0.0, 0.0)
    normalized = normalize_model_name(model)
    if kind == "embedding":
        return EMBEDDING_PRICES.get(normalized)
    if kind == "tool":
        return TOOL_PRICES.get(normalized)
    # Check OpenAI catalog first
    price = TEXT_PRICES.get(normalized)
    if price is not None:
        return price
    # Check OpenRouter catalog
    return OPENROUTER_TEXT_PRICES.get(normalized)


def get_pricing_catalog() -> dict:
    return {
        "version": PRICING_CATALOG_VERSION,
        "sources": PRICING_SOURCES,
        "text": {name: price.__dict__ for name, price in TEXT_PRICES.items()},
        "embeddings": {name: price.__dict__ for name, price in EMBEDDING_PRICES.items()},
        "tools": {name: price.__dict__ for name, price in TOOL_PRICES.items()},
    }
