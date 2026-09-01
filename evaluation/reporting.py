"""Generacion de reportes para comparacion de modelos.

Analiza resultados de batch y genera reportes JSON + HTML
con estadisticas por modelo, routing de agentes, y graficas.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from loguru import logger


class BatchLock:
    """Process lock preventing concurrent writes to one results directory."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.path = self.root / "batch.lock"

    def __enter__(self) -> "BatchLock":
        self.root.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                pid = int(payload.get("pid", 0))
                if pid and _pid_is_alive(pid):
                    raise RuntimeError(f"batch directory is already locked by pid {pid}")
            except RuntimeError:
                raise
            except (OSError, ValueError, json.JSONDecodeError):
                pass
            self.path.unlink(missing_ok=True)
        try:
            with self.path.open("x", encoding="utf-8") as handle:
                json.dump({"pid": os.getpid(), "created_at": time.time()}, handle)
        except FileExistsError as exc:
            raise RuntimeError("batch directory is already locked") from exc
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.path.unlink(missing_ok=True)


def _pid_is_alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def generate_report(
    results_dir: Path,
    output_dir: str,
    fmt: str = "both",
) -> Path:
    """Genera un reporte desde resultados de evaluacion."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = _load_summary(results_dir)
    artifacts = _load_artifacts(results_dir)

    if not artifacts:
        raise ValueError(f"No se encontraron artefactos en {results_dir}")

    analysis = _analyze_results(summary, artifacts)

    report_path = out / "report.json"
    report_path.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if fmt in ("html", "both"):
        html_path = out / "report.html"
        html_content = _generate_html_report(analysis)
        html_path.write_text(html_content, encoding="utf-8")
        logger.info(f"Reporte HTML generado: {html_path}")

    _generate_charts(analysis, out)

    return report_path


def _load_summary(results_dir: Path) -> dict[str, Any]:
    summary_path = results_dir / "batch_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return {}


def _load_artifacts(results_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    for case_dir in sorted(results_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        artifact: dict[str, Any] = {"case_id": case_dir.name}

        for file_name in ["output.json", "metrics.json", "artifact.json", "routing.json"]:
            file_path = case_dir / file_name
            if file_path.exists():
                artifact[file_name.replace(".json", "")] = json.loads(
                    file_path.read_text(encoding="utf-8")
                )

        # Cargar jueces multi-judge (judges/ subdirectorio)
        judges_dir = case_dir / "judges"
        if judges_dir.is_dir():
            judge_results = {}
            for jdir in judges_dir.iterdir():
                if jdir.is_dir():
                    jm_path = jdir / "judge_metrics.json"
                    if jm_path.exists():
                        jm_data = json.loads(jm_path.read_text(encoding="utf-8"))
                        if "error" not in jm_data:
                            judge_results[jdir.name] = jm_data
            if judge_results:
                artifact["judge_results"] = judge_results
                # Legacy: primer juez
                first_judge = next(iter(judge_results.values()), None)
                if first_judge:
                    artifact["judge_metrics"] = first_judge
        else:
            # Fallback: archivo legacy judge_metrics.json
            jm_path = case_dir / "judge_metrics.json"
            if jm_path.exists():
                artifact["judge_metrics"] = json.loads(
                    jm_path.read_text(encoding="utf-8")
                )

        if len(artifact) > 1:
            artifacts.append(artifact)

    return artifacts


def _analyze_results(
    summary: dict[str, Any], artifacts: list[dict[str, Any]]
) -> dict[str, Any]:
    """Analiza resultados agrupando por modelo, con desglose por juez."""
    by_model: dict[str, list[dict]] = {}
    for a in artifacts:
        model = "unknown"
        if "artifact" in a and isinstance(a["artifact"], dict):
            model = a["artifact"].get("model", "unknown")
        by_model.setdefault(model, []).append(a)

    # Recopilar todos los nombres de jueces
    all_judge_names: set[str] = set()
    for a in artifacts:
        jr = a.get("judge_results") or {}
        all_judge_names.update(jr.keys())

    model_stats = {}
    for model, arts in by_model.items():
        latencies: list[float] = []
        costs: list[float] = []
        concepts_cov: list[float] = []
        actions_cov: list[float] = []
        facts_cov: list[float] = []
        agent_invocations: dict[str, int] = {}

        # Legacy judge scores
        judge_scores: list[float] = []

        # Multi-judge scores
        judge_breakdown: dict[str, list[float]] = {jn: [] for jn in all_judge_names}

        for a in arts:
            # Multi-judge
            jr = a.get("judge_results") or {}
            for jname, jm_data in jr.items():
                if isinstance(jm_data, dict):
                    oq = jm_data.get("overall_quality") or {}
                    score = oq.get("score") if isinstance(oq, dict) else None
                    if isinstance(score, (int, float)):
                        judge_breakdown.setdefault(jname, []).append(score)

            # Legacy
            jm = a.get("judge_metrics") or {}
            if isinstance(jm, dict):
                oq = jm.get("overall_quality") or {}
                score = oq.get("score") if isinstance(oq, dict) else None
                if isinstance(score, (int, float)):
                    judge_scores.append(score)

                for cov_key, cov_list in [
                    ("gold_concepts_coverage", concepts_cov),
                    ("gold_actions_coverage", actions_cov),
                    ("gold_facts_coverage", facts_cov),
                ]:
                    val = jm.get(cov_key)
                    if isinstance(val, (int, float)):
                        cov_list.append(val)

            m = a.get("metrics") or {}
            if isinstance(m, dict):
                latency = m.get("latency_ms")
                if isinstance(latency, (int, float)):
                    latencies.append(latency)
                cost = m.get("estimated_cost_usd")
                if isinstance(cost, (int, float)):
                    costs.append(cost)

            routing = a.get("routing") or {}
            if isinstance(routing, dict):
                agents = routing.get("agents", {})
                for agent_name in agents:
                    agent_invocations[agent_name] = agent_invocations.get(agent_name, 0) + 1

        # Agregación por juez
        judge_stats = {}
        for jname in sorted(all_judge_names):
            scores = judge_breakdown.get(jname, [])
            judge_stats[jname] = {
                "overall_mean": _safe_mean(scores),
                "overall_std": _safe_std(scores),
                "n_scored": len(scores),
            }

        model_stats[model] = {
            "n_cases": len(arts),
            "judge_overall_mean": _safe_mean(judge_scores),
            "judge_overall_std": _safe_std(judge_scores),
            "judges": judge_stats,
            "concepts_coverage_mean": _safe_mean(concepts_cov),
            "actions_coverage_mean": _safe_mean(actions_cov),
            "facts_coverage_mean": _safe_mean(facts_cov),
            "latency_mean_ms": _safe_mean(latencies),
            "cost_mean_usd": _safe_mean(costs),
            "system_latency_mean_ms": _safe_mean([
                a.get("metrics", {}).get("system_latency_ms", 0)
                for a in arts if isinstance(a.get("metrics"), dict)
            ]),
            "judge_latency_mean_ms": _safe_mean([
                a.get("metrics", {}).get("judge_latency_ms", 0)
                for a in arts if isinstance(a.get("metrics"), dict)
            ]),
            "task_wall_latency_mean_ms": _safe_mean([
                a.get("metrics", {}).get("task_wall_latency_ms", 0)
                for a in arts if isinstance(a.get("metrics"), dict)
            ]),
            "system_cost_total_usd": round(sum(
                a.get("metrics", {}).get("system_cost_usd", 0)
                for a in arts if isinstance(a.get("metrics"), dict)
            ), 6),
            "vision_cost_total_usd": round(sum(
                a.get("metrics", {}).get("vision_cost_usd", 0)
                for a in arts if isinstance(a.get("metrics"), dict)
            ), 6),
            "judge_cost_total_usd": round(sum(
                a.get("metrics", {}).get("judge_cost_usd", 0)
                for a in arts if isinstance(a.get("metrics"), dict)
            ), 6),
            "total_cost_usd": sum(costs),
            "agent_invocations": agent_invocations,
            "unique_agents": list(agent_invocations.keys()),
        }

    # Agregar dimension scores por modelo (desglose por juez)
    for model, arts in by_model.items():
        dim_scores: dict[str, list[float]] = {}
        judge_dim_scores: dict[str, dict[str, list[float]]] = {}

        for a in arts:
            # Multi-judge dimensions
            jr = a.get("judge_results") or {}
            for jname, jm_data in jr.items():
                if not isinstance(jm_data, dict):
                    continue
                for dim in [
                    "factual_correctness", "domain_accuracy", "responsible_action_quality",
                    "actionability", "decision_support_quality", "evidence_utilization",
                    "transparent_confidence", "case_personalization", "practical_value",
                ]:
                    raw_dim = jm_data.get(dim)
                    val = raw_dim.get("score") if isinstance(raw_dim, dict) else jm_data.get(f"judge_{dim}")
                    if isinstance(val, (int, float)):
                        dim_scores.setdefault(dim, []).append(val)
                        judge_dim_scores.setdefault(jname, {}).setdefault(dim, []).append(val)

            # Legacy dimensions
            jm = a.get("judge_metrics") or {}
            if not isinstance(jm, dict):
                continue
            for dim in [
                "factual_correctness", "domain_accuracy", "responsible_action_quality",
                "actionability", "decision_support_quality", "evidence_utilization",
                "transparent_confidence", "case_personalization", "practical_value",
            ]:
                val = jm.get(f"judge_{dim}")
                if isinstance(val, (int, float)):
                    dim_scores.setdefault(dim, []).append(val)

        model_stats[model]["dimension_scores"] = {
            dim: _safe_mean(vals) for dim, vals in dim_scores.items()
        }
        model_stats[model]["judge_dimension_scores"] = {
            jname: {dim: _safe_mean(vals) for dim, vals in dims.items()}
            for jname, dims in judge_dim_scores.items()
        }

    return {
        "report_type": "AgroCopilot Model Comparison",
        "summary": summary,
        "model_stats": model_stats,
        "total_cases": len(artifacts),
        "total_cost_usd": summary.get("total_cost_usd", 0),
        "judges_used": summary.get("judges_used", []),
    }


def _safe_mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return round(variance ** 0.5, 3)


def _generate_html_report(analysis: dict[str, Any]) -> str:
    """Genera reporte HTML para comparacion de modelos."""
    model_stats = analysis.get("model_stats", {})
    total = analysis.get("total_cases", 0)
    cost = analysis.get("total_cost_usd", 0)

    rows = ""
    for model, stats in model_stats.items():
        dim_scores = stats.get("dimension_scores", {})
        dims_html = " ".join(
            f'<span class="dim" title="{d}">{v:.1f}</span>'
            for d, v in dim_scores.items()
        )
        rows += f"""
        <tr>
            <td>{model}</td>
            <td>{stats['n_cases']}</td>
            <td>{stats['judge_overall_mean']:.2f} ± {stats['judge_overall_std']:.2f}</td>
            <td>{stats['concepts_coverage_mean']:.2f}</td>
            <td>{stats['actions_coverage_mean']:.2f}</td>
            <td>{stats['latency_mean_ms']:.0f}ms</td>
            <td>{stats['system_latency_mean_ms']:.0f}ms / {stats['judge_latency_mean_ms']:.0f}ms</td>
            <td>${stats['system_cost_total_usd']:.4f} / ${stats['judge_cost_total_usd']:.4f}</td>
            <td>{dims_html}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>AgroCopilot - Comparacion de Modelos</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; }}
        h1 {{ color: #2d5016; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; font-size: 0.9em; }}
        th, td {{ border: 1px solid #ddd; padding: 0.6rem; text-align: left; }}
        th {{ background-color: #4a7c2e; color: white; }}
        tr:nth-child(even) {{ background-color: #f5f5f5; }}
        .summary {{ background: #f0f7e6; padding: 1rem; border-radius: 8px; margin-bottom: 2rem; }}
        .dim {{ display: inline-block; padding: 2px 4px; margin: 1px; background: #e8f5e9; border-radius: 3px; font-size: 0.8em; }}
    </style>
</head>
<body>
    <h1>AgroCopilot - Comparacion de Modelos</h1>
    <div class="summary">
        <strong>Total casos:</strong> {total} |
        <strong>Coste total:</strong> ${cost:.4f} |
        <strong>Modelos:</strong> {len(model_stats)}
    </div>
    <h2>Resultados por Modelo</h2>
    <table>
        <tr>
            <th>Modelo</th>
            <th>Casos</th>
            <th>Judge Overall</th>
            <th>Cobertura Conceptos</th>
            <th>Cobertura Acciones</th>
            <th>Latencia Sistema</th>
            <th>Latencia Sistema / Jueces</th>
            <th>Coste Sistema / Jueces</th>
            <th>Dimensiones</th>
        </tr>
        {rows}
    </table>
</body>
</html>"""


def _generate_charts(analysis: dict[str, Any], output_dir: Path) -> None:
    """Genera graficas si matplotlib esta disponible."""
    try:
        from evaluation.charts import create_radar_chart, create_forest_plot
    except ImportError:
        return

    model_stats = analysis.get("model_stats", {})
    if not model_stats:
        return

    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    # Radar chart
    radar_data: dict[str, list[float]] = {}
    for model, stats in model_stats.items():
        dims = stats.get("dimension_scores", {})
        if dims:
            radar_data[model] = list(dims.values())
    if radar_data:
        create_radar_chart(radar_data, str(charts_dir / "radar.png"))

    # Forest plot
    forest_data: dict[str, dict[str, float]] = {}
    for model, stats in model_stats.items():
        mean = stats.get("judge_overall_mean", 3.0)
        std = stats.get("judge_overall_std", 0.5)
        forest_data[model] = {
            "mean": mean,
            "ci_low": mean - 1.96 * std,
            "ci_high": mean + 1.96 * std,
        }
    if forest_data:
        create_forest_plot(forest_data, str(charts_dir / "forest.png"))
