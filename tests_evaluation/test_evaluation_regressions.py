from __future__ import annotations

import asyncio
import json
import os

import pytest

from evaluation.concurrency import AdaptiveConcurrency
from evaluation.config import EvalConfig
from evaluation.metrics import (
    check_routing_assertion,
    compute_answer_completeness,
    compute_execution_metrics,
)
from evaluation.reporting import BatchLock
from evaluation.schemas import CaseSpec, GoldExpectations, NormalizedOutput
from evaluation.llm_support import _decode_json_content


def _case(**kwargs) -> CaseSpec:
    return CaseSpec(case_id="regression", family="general", query="q", **kwargs)


def test_structured_fields_are_evaluable():
    case = _case(gold_expectations=GoldExpectations(must_actions=["comprobar parcela"]))
    output = NormalizedOutput(next_actions=["Comprobar la parcela"])
    metrics = compute_execution_metrics(case, output)
    assert metrics.actionability == 1.0
    assert metrics.answer_completeness == 1.0


def test_judge_json_accepts_markdown_fence():
    payload = _decode_json_content('```json\n{"scores": {"ok": 1}}\n```')
    assert payload == {"scores": {"ok": 1}}


def test_routing_requires_all_agents_in_both_assertion():
    case = _case(routing_assertion="both stac/rs_analyst and legal agents must be invoked")
    assert check_routing_assertion(case, ["legal"]) is False
    assert check_routing_assertion(case, ["stac", "rs_analyst", "legal"]) is True


def test_unsupported_context_assertion_is_not_silently_true():
    case = _case(routing_assertion="system should maintain conversation context across turns")
    assert check_routing_assertion(case, []) is False


def test_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        EvalConfig(runs_per_case=0)
    with pytest.raises(ValueError):
        EvalConfig(max_concurrent=0)
    with pytest.raises(ValueError):
        EvalConfig(budget_usd=-1)


def test_batch_lock_reclaims_stale_pid(tmp_path):
    lock = tmp_path / "batch.lock"
    lock.write_text(json.dumps({"pid": 999999999, "created_at": "old"}), encoding="utf-8")
    with BatchLock(tmp_path):
        payload = json.loads(lock.read_text(encoding="utf-8"))
        assert payload["pid"] == os.getpid()


def test_adaptive_concurrency_reduces_and_recovers():
    async def scenario():
        controller = AdaptiveConcurrency(8, maximum=8)
        await controller.acquire()
        await controller.release(saturated=True)
        assert controller.current == 4
        for _ in range(16):
            await controller.acquire()
            await controller.release()
        return controller.stats

    stats = asyncio.run(scenario())
    assert stats.reductions == 1
    assert stats.current >= 4
