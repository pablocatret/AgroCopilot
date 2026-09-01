from types import SimpleNamespace

import pytest

from backend.cost_store import CostStore
from libs.costs.calculator import calculate_embedding_cost, calculate_text_cost, calculate_tool_cost
from libs.costs.pricing import get_model_price, normalize_model_name
from evaluation.llm_support import LLMCallTracker


def test_text_cost_uses_cached_tokens_and_output_rate():
    usage = SimpleNamespace(
        prompt_tokens=10_000,
        completion_tokens=2_000,
        total_tokens=12_000,
        prompt_tokens_details=SimpleNamespace(cached_tokens=4_000),
    )

    cost = calculate_text_cost("gpt-4.1-mini", usage)

    expected = ((6_000 * 0.40) + (4_000 * 0.10) + (2_000 * 1.60)) / 1_000_000
    assert cost.cost_usd == pytest.approx(expected)
    assert cost.cached_input_tokens == 4_000
    assert cost.estimated is False


def test_batch_text_cost_halves_standard_rates_when_no_batch_rate_is_explicit():
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 100_000, "total_tokens": 1_100_000}

    standard = calculate_text_cost("gpt-5", usage, pricing_mode="standard")
    batch = calculate_text_cost("gpt-5", usage, pricing_mode="batch")

    assert batch.cost_usd == pytest.approx(standard.cost_usd * 0.5)


def test_unknown_model_is_tracked_as_estimated_zero_cost():
    cost = calculate_text_cost("future-model-x", {"prompt_tokens": 100, "completion_tokens": 50})

    assert cost.cost_usd == 0
    assert cost.estimated is True
    assert cost.metadata["reason"] == "unknown_model"


def test_embedding_and_web_tool_costs():
    embedding = calculate_embedding_cost("text-embedding-3-small", {"prompt_tokens": 1_000_000})
    web = calculate_tool_cost(
        "openai-web-search", unit_count=3, unit_price_per_1k=10, provider="openai"
    )

    assert embedding.cost_usd == pytest.approx(0.02)
    assert web.cost_usd == pytest.approx(0.03)


def test_model_snapshot_normalization():
    assert normalize_model_name("gpt-5.4-2026-03-05") == "gpt-5.4"


@pytest.mark.parametrize(
    "model, input_rate, output_rate",
    [
        ("openai/gpt-oss-120b:nitro", 0.030, 0.180),
        ("google/gemma-4-31b-it:nitro", 0.120, 0.350),
        ("minimax/minimax-m3", 0.300, 1.200),
        ("qwen/qwen3.6-flash", 0.1875, 1.125),
        ("stepfun/step-3.7-flash", 0.200, 1.150),
    ],
)
def test_openrouter_current_prices_and_routing_suffixes(model, input_rate, output_rate):
    price = get_model_price(model)
    assert price is not None
    assert price.input_per_million == pytest.approx(input_rate)
    assert price.output_per_million == pytest.approx(output_rate)


def test_unknown_tracker_cost_is_visible_instead_of_silent():
    tracker = LLMCallTracker()
    tracker.record("gpt-5.6-luna", "openai", 100, 50, 0.0, 1.0, cost_known=False)
    assert tracker.cost_complete is False
    assert tracker.unknown_cost_calls == 1
    assert tracker.unknown_cost_models == ["gpt-5.6-luna"]


def test_tracker_separates_system_vision_and_judge_costs_and_latency():
    tracker = LLMCallTracker()
    tracker.record("m", "openrouter", 1, 1, 0.10, 100.0, operation="system.writer")
    tracker.record("v", "openrouter", 1, 1, 0.20, 200.0, operation="system.vision_ocr")
    tracker.record("j", "openrouter", 1, 1, 0.30, 300.0, operation="eval.judge")
    assert tracker.component_breakdown("cost_usd") == {
        "judge": 0.3,
        "system": 0.1,
        "vision": 0.2,
    }
    assert tracker.component_breakdown("latency_ms")["judge"] == 300.0


def test_cost_store_summarizes_by_conversation(tmp_path):
    store = CostStore(str(tmp_path / "costs.db"))
    store.insert_event(
        {
            "id": "evt1",
            "conversation_id": "conv1",
            "agent": "legal",
            "operation": "chat.completions",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "pricing_mode": "standard",
            "input_tokens": 1000,
            "output_tokens": 500,
            "total_tokens": 1500,
            "unit_count": 0,
            "estimated": False,
            "cost_usd": 0.001,
            "metadata": {"x": 1},
        }
    )
    store.insert_event(
        {
            "id": "evt2",
            "conversation_id": "conv1",
            "agent": "legal",
            "operation": "web_search.provider_call",
            "provider": "serper",
            "model": "serper-search",
            "pricing_mode": "standard",
            "unit_count": 1,
            "estimated": True,
            "cost_usd": 0.0,
            "metadata": {},
        }
    )

    summary = store.summarize_conversation("conv1")

    assert summary["total_cost_usd"] == pytest.approx(0.001)
    assert summary["total_tokens"] == 1500
    assert summary["web_calls"] == 1
    assert summary["by_agent"]["legal"]["events"] == 2
    store.close()
