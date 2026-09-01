"""Export plano de resultados de evaluación a CSV."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from evaluation.schemas import SCHEMA_VERSION


JUDGE_DIMENSIONS = [
    "factual_correctness",
    "domain_accuracy",
    "responsible_action_quality",
    "actionability",
    "decision_support_quality",
    "evidence_utilization",
    "transparent_confidence",
    "case_personalization",
    "practical_value",
]


def build_export_rows(results_dir: Path) -> list[dict[str, Any]]:
    """Construye las filas planas del CSV a partir de un directorio de batch."""
    results_dir = Path(results_dir)
    summary = _load_json(results_dir / "batch_summary.json")
    batch_id = summary.get("batch_id", results_dir.name)
    schema_version = _load_schema_version(results_dir, summary)

    rows: list[dict[str, Any]] = []
    for artifact_dir in sorted(results_dir.iterdir()):
        if not artifact_dir.is_dir():
            continue
        artifact_path = artifact_dir / "artifact.json"
        metrics_path = artifact_dir / "metrics.json"
        if not artifact_path.exists() or not metrics_path.exists():
            continue

        artifact = _load_json(artifact_path)
        metrics = _load_json(metrics_path)
        judges_dir = artifact_dir / "judges"

        judge_entries: list[tuple[str, dict[str, Any]]] = []
        if judges_dir.is_dir():
            for judge_dir in sorted(judges_dir.iterdir()):
                if not judge_dir.is_dir():
                    continue
                judge_metrics_path = judge_dir / "judge_metrics.json"
                if not judge_metrics_path.exists():
                    continue
                judge_data = _load_json(judge_metrics_path)
                if isinstance(judge_data, dict) and "error" not in judge_data:
                    judge_entries.append((judge_dir.name, judge_data))
        else:
            legacy_path = artifact_dir / "judge_metrics.json"
            if legacy_path.exists():
                judge_data = _load_json(legacy_path)
                if isinstance(judge_data, dict) and "error" not in judge_data:
                    judge_entries.append(("default", judge_data))

        for judge_name, judge_data in judge_entries:
            row = {
                "batch_id": batch_id,
                "schema_version": schema_version,
                "case_id": artifact.get("case_id", artifact_dir.name),
                "family": artifact.get("family", ""),
                "difficulty": artifact.get("difficulty", ""),
                "model": artifact.get("model", ""),
                "judge_name": judge_name,
                "run_idx": artifact.get("run_idx", 0),
                "overall_quality": _judge_score(judge_data, "overall_quality"),
                "factual_correctness": _judge_score(judge_data, "factual_correctness"),
                "domain_accuracy": _judge_score(judge_data, "domain_accuracy"),
                "responsible_action_quality": _judge_score(judge_data, "responsible_action_quality"),
                "actionability_judge": _judge_score(judge_data, "actionability"),
                "decision_support_quality": _judge_score(judge_data, "decision_support_quality"),
                "evidence_utilization": _judge_score(judge_data, "evidence_utilization"),
                "transparent_confidence": _judge_score(judge_data, "transparent_confidence"),
                "case_personalization": _judge_score(judge_data, "case_personalization"),
                "practical_value": _judge_score(judge_data, "practical_value"),
                "judge_confidence": _as_float(judge_data.get("judge_confidence")),
                "perceived_difficulty": _as_float(judge_data.get("perceived_difficulty")),
                "gold_concepts_coverage": _as_float(judge_data.get("gold_concepts_coverage")),
                "gold_actions_coverage": _as_float(judge_data.get("gold_actions_coverage")),
                "gold_facts_coverage": _as_float(judge_data.get("gold_facts_coverage")),
                "forbidden_claims_violated": _forbidden_count(judge_data),
                "forbidden_claim_rate": _as_float(metrics.get("forbidden_claim_rate")),
                "overclaim_count": _as_float(metrics.get("overclaim_count")),
                "actionability_det": _as_float(metrics.get("actionability")),
                "actionability_structured": _as_float(metrics.get("actionability_structured")),
                "actionability_visible": _as_float(metrics.get("actionability_visible")),
                "answer_completeness": _as_float(metrics.get("answer_completeness")),
                "clarification_detected": bool(metrics.get("clarification_detected", False)),
                "latency_ms": _as_float(metrics.get("latency_ms")),
                "system_latency_ms": _as_float(metrics.get("system_latency_ms")),
                "judge_latency_ms": _as_float(metrics.get("judge_latency_ms")),
                "task_wall_latency_ms": _as_float(metrics.get("task_wall_latency_ms")),
                "queue_latency_ms": _as_float(metrics.get("queue_latency_ms")),
                "cost_usd": _as_float(metrics.get("estimated_cost_usd")),
                "system_cost_usd": _as_float(metrics.get("system_cost_usd")),
                "vision_cost_usd": _as_float(metrics.get("vision_cost_usd")),
                "judge_cost_usd": _as_float(metrics.get("judge_cost_usd")),
                "model_calls": int(metrics.get("model_calls", 0) or 0),
                "success": bool(metrics.get("success", False)),
                "agents_invoked": json.dumps(metrics.get("agents_invoked", []), ensure_ascii=False),
                "agents_planned": json.dumps(metrics.get("agents_planned", []), ensure_ascii=False),
                "agents_failed": json.dumps(metrics.get("agents_failed", []), ensure_ascii=False),
                "agents_extra": json.dumps(metrics.get("agents_extra", []), ensure_ascii=False),
                "required_agents_missing": json.dumps(metrics.get("required_agents_missing", []), ensure_ascii=False),
                "visual_evidence_status": metrics.get("visual_evidence_status", "not_required"),
                "visual_evidence_used": bool(metrics.get("visual_evidence_used", False)),
                "routing_score": _as_float(metrics.get("routing_score")),
                "routing_precision": _as_float(metrics.get("routing_precision")),
                "routing_recall": _as_float(metrics.get("routing_recall")),
                "routing_order_score": _as_float(metrics.get("routing_order_score")),
                "routing_assertion_pass": bool(metrics.get("routing_assertion_pass", True)),
                "execution_status": metrics.get("execution_status", ""),
                "failure_reason": metrics.get("failure_reason", ""),
                "timestamp": artifact.get("timestamp_iso", ""),
            }
            rows.append(row)

    rows.sort(key=lambda row: (
        str(row.get("case_id", "")),
        str(row.get("model", "")),
        str(row.get("judge_name", "")),
        int(row.get("run_idx", 0) or 0),
    ))
    return rows


def export_csv(results_dir: Path, output_path: Path) -> Path:
    """Escribe un CSV plano a partir de los artefactos persistidos."""
    rows = build_export_rows(results_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "batch_id",
        "schema_version",
        "case_id",
        "family",
        "difficulty",
        "model",
        "judge_name",
        "run_idx",
        "overall_quality",
        "factual_correctness",
        "domain_accuracy",
        "responsible_action_quality",
        "actionability_judge",
        "decision_support_quality",
        "evidence_utilization",
        "transparent_confidence",
        "case_personalization",
        "practical_value",
        "judge_confidence",
        "perceived_difficulty",
        "gold_concepts_coverage",
        "gold_actions_coverage",
        "gold_facts_coverage",
        "forbidden_claims_violated",
        "forbidden_claim_rate",
        "overclaim_count",
        "actionability_det",
        "actionability_structured",
        "actionability_visible",
        "answer_completeness",
        "clarification_detected",
        "latency_ms",
        "system_latency_ms",
        "judge_latency_ms",
        "task_wall_latency_ms",
        "queue_latency_ms",
        "cost_usd",
        "system_cost_usd",
        "vision_cost_usd",
        "judge_cost_usd",
        "model_calls",
        "success",
        "agents_invoked",
        "agents_planned",
        "agents_failed",
        "agents_extra",
        "required_agents_missing",
        "visual_evidence_status",
        "visual_evidence_used",
        "routing_score",
        "routing_precision",
        "routing_recall",
        "routing_order_score",
        "routing_assertion_pass",
        "execution_status",
        "failure_reason",
        "timestamp",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    return output_path


def _load_schema_version(results_dir: Path, summary: dict[str, Any]) -> str:
    version_path = results_dir / "schema_version.json"
    if version_path.exists():
        data = _load_json(version_path)
        version = data.get("version")
        if isinstance(version, str) and version:
            return version
    version = summary.get("schema_version")
    if isinstance(version, str) and version:
        return version
    return SCHEMA_VERSION


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _judge_score(judge_data: dict[str, Any], field_name: str) -> int:
    data = judge_data.get(field_name)
    if isinstance(data, dict):
        score = data.get("score")
        if isinstance(score, (int, float)):
            return int(score)
    return 0


def _forbidden_count(judge_data: dict[str, Any]) -> int:
    value = judge_data.get("forbidden_claims_violated", [])
    if isinstance(value, list):
        return len(value)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
