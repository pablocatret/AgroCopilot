from __future__ import annotations

import json
import uuid
from pathlib import Path

from evaluation.export import build_export_rows, export_csv
from evaluation.stats import compute_full_correlations


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _make_judge_payload(overall: int, factual: int, domain: int, actionability: int) -> dict:
    base = {
        "schema_version": "2.0",
        "factual_correctness": {"score": factual, "rationale": ""},
        "domain_accuracy": {"score": domain, "rationale": ""},
        "responsible_action_quality": {"score": 3, "rationale": ""},
        "actionability": {"score": actionability, "rationale": ""},
        "decision_support_quality": {"score": 3, "rationale": ""},
        "evidence_utilization": {"score": 3, "rationale": ""},
        "transparent_confidence": {"score": 3, "rationale": ""},
        "case_personalization": {"score": 3, "rationale": ""},
        "practical_value": {"score": 3, "rationale": ""},
        "overall_quality": {"score": overall, "rationale": ""},
        "judge_confidence": 0.8,
        "perceived_difficulty": 2,
        "gold_concepts_coverage": 0.7,
        "gold_actions_coverage": 0.6,
        "gold_facts_coverage": 0.5,
        "forbidden_claims_violated": ["x"],
    }
    return base


def _make_artifact(case_id: str, model: str, run_idx: int, family: str, difficulty: str, timestamp: str) -> dict:
    return {
        "schema_version": "2.0",
        "run_id": f"run-{case_id}-{run_idx}",
        "case_id": case_id,
        "model": model,
        "input_query": f"Query {case_id}",
        "timestamp_iso": timestamp,
        "family": family,
        "difficulty": difficulty,
        "run_idx": run_idx,
        "errors": [],
    }


def _make_metrics(latency: float, cost: float, actionability: float, overall: int) -> dict:
    return {
        "schema_version": "2.0",
        "success": True,
        "latency_ms": latency,
        "estimated_cost_usd": cost,
        "model_calls": 3,
        "token_prompt_total": 120,
        "token_completion_total": 45,
        "answer_completeness": 0.0,
        "forbidden_claim_rate": 0.1,
        "overclaim_count": 1.0,
        "actionability": actionability,
        "clarification_detected": False,
        "agents_invoked": ["organizer", "writer"],
        "agents_ok": 2,
        "agents_error": 0,
        "route_observed": ["organizer", "writer"],
        "routing_score": 1.0,
        "routing_assertion_pass": True,
    }


def _seed_batch_dir(base: Path) -> Path:
    batch_dir = base / "batch_123"
    _write_json(batch_dir / "schema_version.json", {"version": "2.0"})
    _write_json(batch_dir / "batch_summary.json", {"batch_id": "batch_123", "schema_version": "2.0"})
    _write_json(batch_dir / "eval_config.json", {"models": {}, "judges": []})

    for idx, case_id in enumerate(["case_001", "case_002", "case_003"], start=1):
        case_dir = batch_dir / case_id
        _write_json(
            case_dir / "artifact.json",
            _make_artifact(case_id, "gpt-5-mini", 0, "diagnosis", "medium", f"2026-07-12T10:2{idx}:00.000Z"),
        )
        _write_json(case_dir / "metrics.json", _make_metrics(1000.0 + idx, 0.01 * idx, 0.5 + idx * 0.1, 4))
        _write_json(case_dir / "output.json", {"schema_version": "2.0", "parse_status": "ok", "message_md": "ok"})
        for judge_name in ("mimo-v2.5", "glm-5.2"):
            _write_json(
                case_dir / "judges" / judge_name / "judge_metrics.json",
                _make_judge_payload(3 + idx, 2 + idx, 3 + idx, 2 + idx),
            )

    return batch_dir


def test_export_csv_and_rows():
    tmp_dir = Path.cwd() / "evaluation" / "results" / f"evaluation_export_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    batch_dir = _seed_batch_dir(tmp_dir)
    output_path = batch_dir / "aggregate.csv"

    rows = build_export_rows(batch_dir)
    assert len(rows) == 6
    assert {row["judge_name"] for row in rows} == {"mimo-v2.5", "glm-5.2"}

    export_csv(batch_dir, output_path)
    assert output_path.exists()

    csv_text = output_path.read_text(encoding="utf-8")
    assert "schema_version" in csv_text
    assert "overall_quality" in csv_text


def test_compute_full_correlations():
    tmp_dir = Path.cwd() / "evaluation" / "results" / f"evaluation_export_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    batch_dir = _seed_batch_dir(tmp_dir)
    rows = build_export_rows(batch_dir)
    correlations = compute_full_correlations(rows)

    assert "inter_judge" in correlations
    assert "cost_quality" in correlations
    assert correlations["n_rows"] == 6
    assert {
        "mimo-v2.5_vs_glm-5.2",
        "glm-5.2_vs_mimo-v2.5",
    } & set(correlations["inter_judge"].keys())
