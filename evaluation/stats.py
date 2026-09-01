"""Auxiliary analytics for evaluation exports.

The dissertation benchmark's canonical inferential analysis lives in
``evaluation/analyze_benchmark_statistics.py``. This module retains
descriptive, correlation and Pareto helpers used by reporting and tests.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from scipy import stats as scipy_stats


# ── Intervalos de confianza bootstrap ────────────────────────────────


def bootstrap_ci(
    values: list[float],
    *,
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Calcula media e IC95% por bootstrap.

    Args:
        values: Lista de valores numéricos.
        n_bootstrap: Número de muestras bootstrap.
        ci_level: Nivel de confianza (default 0.95).
        seed: Semilla para reproducibilidad.

    Returns:
        Tupla (media, ci_low, ci_high).
    """
    if not values:
        return (0.0, 0.0, 0.0)
    if len(values) == 1:
        v = values[0]
        return (v, v, v)

    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / len(sample))
    means.sort()

    alpha = (1 - ci_level) / 2
    idx_low = int(math.floor(alpha * n_bootstrap))
    idx_high = int(math.floor((1 - alpha) * n_bootstrap)) - 1
    idx_low = max(0, min(idx_low, n_bootstrap - 1))
    idx_high = max(0, min(idx_high, n_bootstrap - 1))

    mean_val = sum(values) / len(values)
    return (mean_val, means[idx_low], means[idx_high])


# ── Test t pareado ──────────────────────────────────────────────────


@dataclass
class PairedTTest:
    """Resultado de un test t pareado."""

    n_pairs: int
    mean_delta: float
    std_delta: float
    se_delta: float
    t_statistic: float
    p_value: float
    ci95_low: float
    ci95_high: float
    cohens_d: float
    interpretation: str = ""


def paired_t_test(
    values_a: list[float],
    values_b: list[float],
) -> PairedTTest:
    """Test t pareado para comparar dos sistemas.

    H0: media(A - B) = 0 (no hay diferencia)
    H1: media(A - B) ≠ 0 (hay diferencia)

    Args:
        values_a: Métricas del sistema A.
        values_b: Métricas del sistema B (mismo orden, mismos casos).

    Returns:
        PairedTTest con resultados.
    """
    n = min(len(values_a), len(values_b))
    if n < 2:
        return PairedTTest(
            n_pairs=n,
            mean_delta=0.0,
            std_delta=0.0,
            se_delta=0.0,
            t_statistic=0.0,
            p_value=1.0,
            ci95_low=0.0,
            ci95_high=0.0,
            cohens_d=0.0,
            interpretation="Insuficiente para test (n < 2)",
        )

    deltas = [values_a[i] - values_b[i] for i in range(n)]
    mean_d = sum(deltas) / n
    var_d = sum((d - mean_d) ** 2 for d in deltas) / (n - 1)
    std_d = math.sqrt(var_d) if var_d > 0 else 0.0
    se_d = std_d / math.sqrt(n) if n > 0 else 0.0
    t_stat = mean_d / se_d if se_d > 0 else 0.0

    # Aproximación de p-value usando distribución t de Student (two-tailed)
    df = n - 1
    p_value = 2.0 * (1.0 - scipy_stats.t.cdf(abs(t_stat), df))

    # IC95% del delta
    t_critical_val = scipy_stats.t.ppf(0.975, df)
    ci_low = mean_d - t_critical_val * se_d
    ci_high = mean_d + t_critical_val * se_d

    # Cohen's d
    cohens_d = mean_d / std_d if std_d > 0 else 0.0

    # Interpretación del tamaño del efecto
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        effect = "despreciable"
    elif abs_d < 0.5:
        effect = "pequeño"
    elif abs_d < 0.8:
        effect = "moderado"
    else:
        effect = "grande"

    interpretation = f"Efecto {effect} (d={cohens_d:.2f}), p={p_value:.4f}"

    return PairedTTest(
        n_pairs=n,
        mean_delta=mean_d,
        std_delta=std_d,
        se_delta=se_d,
        t_statistic=t_stat,
        p_value=p_value,
        ci95_low=ci_low,
        ci95_high=ci_high,
        cohens_d=cohens_d,
        interpretation=interpretation,
    )


# ── Correlaciones ────────────────────────────────────────────────────


@dataclass
class CorrelationResult:
    """Resultado de una correlación entre dos métricas."""

    metric_a: str
    metric_b: str
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    interpretation: str = ""


def correlation_analysis(
    data: dict[str, list[float]],
) -> list[CorrelationResult]:
    """Calcula correlaciones de Pearson y Spearman entre todas las métricas.

    Args:
        data: Dict metric_name -> list de valores por caso.

    Returns:
        Lista de resultados de correlación.
    """
    metric_names = [k for k, v in data.items() if len(v) >= 3]
    results: list[CorrelationResult] = []

    for i in range(len(metric_names)):
        for j in range(i + 1, len(metric_names)):
            name_a = metric_names[i]
            name_b = metric_names[j]
            vals_a = data[name_a]
            vals_b = data[name_b]
            n = min(len(vals_a), len(vals_b))
            if n < 3:
                continue

            pearson_r, pearson_p = _pearson_correlation(vals_a[:n], vals_b[:n])
            spearman_rho, spearman_p = _spearman_correlation(vals_a[:n], vals_b[:n])

            abs_r = abs(pearson_r)
            if abs_r >= 0.7:
                interp = "alta"
            elif abs_r >= 0.4:
                interp = "moderada"
            else:
                interp = "baja"

            results.append(
                CorrelationResult(
                    metric_a=name_a,
                    metric_b=name_b,
                    pearson_r=pearson_r,
                    pearson_p=pearson_p,
                    spearman_rho=spearman_rho,
                    spearman_p=spearman_p,
                    interpretation=interp,
                )
            )

    return results


def compute_full_correlations(csv_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Calcula correlaciones ampliadas a partir del CSV plano de evaluación."""
    rows = [_normalize_row(row) for row in csv_data]
    rows = [row for row in rows if row.get("case_id") and row.get("model") and row.get("judge_name")]
    by_experiment: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["case_id"]), str(row["model"]), int(row.get("run_idx", 0)))
        by_experiment.setdefault(key, []).append(row)
    experimental_rows: list[dict[str, Any]] = []
    for group in by_experiment.values():
        representative = dict(group[0])
        judge_scores = [_as_float(row.get("overall_quality")) for row in group]
        representative["overall_quality"] = sum(judge_scores) / len(judge_scores)
        experimental_rows.append(representative)

    grouped: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["case_id"]), str(row["model"]), int(row.get("run_idx", 0)))
        grouped.setdefault(key, {})[str(row["judge_name"])] = row

    judge_names = sorted({str(row["judge_name"]) for row in rows})
    dimensions = [
        "factual_correctness",
        "domain_accuracy",
        "responsible_action_quality",
        "actionability_judge",
        "decision_support_quality",
        "evidence_utilization",
        "transparent_confidence",
        "case_personalization",
        "practical_value",
        "overall_quality",
    ]

    inter_judge: dict[str, Any] = {}
    for i, judge_a in enumerate(judge_names):
        for judge_b in judge_names[i + 1 :]:
            xs: list[float] = []
            ys: list[float] = []
            for pair_rows in grouped.values():
                row_a = pair_rows.get(judge_a)
                row_b = pair_rows.get(judge_b)
                if not row_a or not row_b:
                    continue
                xs.append(_as_float(row_a.get("overall_quality")))
                ys.append(_as_float(row_b.get("overall_quality")))
            bundle = _correlation_bundle(xs, ys)
            bundle["mean_a"] = _safe_mean(xs)
            bundle["mean_b"] = _safe_mean(ys)
            bundle["mean_delta"] = round(bundle["mean_a"] - bundle["mean_b"], 4)
            bundle["bias_direction"] = _bias_direction(bundle["mean_delta"])
            inter_judge[f"{judge_a}_vs_{judge_b}"] = bundle

    inter_dimension: dict[str, Any] = {}
    for judge_name in judge_names:
        judge_rows = [row for row in rows if str(row["judge_name"]) == judge_name]
        per_dim: dict[str, Any] = {}
        for i, dim_a in enumerate(dimensions):
            for dim_b in dimensions[i + 1 :]:
                xs = [_as_float(row.get(dim_a)) for row in judge_rows]
                ys = [_as_float(row.get(dim_b)) for row in judge_rows]
                per_dim[f"{dim_a}_vs_{dim_b}"] = _correlation_bundle(xs, ys)
        inter_dimension[judge_name] = per_dim

    deterministic_vs_judge = {
        "global": {
            metric: _correlation_bundle(
                [_as_float(row.get(metric)) for row in experimental_rows],
                [_as_float(row.get("overall_quality")) for row in experimental_rows],
            )
            for metric in ("forbidden_claim_rate", "actionability_det", "overclaim_count")
        }
    }

    cost_quality = {
        "global": {
            "cost_usd_vs_overall_quality": _correlation_bundle(
                [_as_float(row.get("cost_usd")) for row in experimental_rows],
                [_as_float(row.get("overall_quality")) for row in experimental_rows],
            ),
            "latency_ms_vs_overall_quality": _correlation_bundle(
                [_as_float(row.get("latency_ms")) for row in experimental_rows],
                [_as_float(row.get("overall_quality")) for row in experimental_rows],
            ),
        },
        "by_model": {},
    }
    for model in sorted({str(row["model"]) for row in rows}):
        model_rows = [row for row in experimental_rows if str(row["model"]) == model]
        cost_quality["by_model"][model] = {
            "cost_usd_vs_overall_quality": _correlation_bundle(
                [_as_float(row.get("cost_usd")) for row in model_rows],
                [_as_float(row.get("overall_quality")) for row in model_rows],
            ),
            "latency_ms_vs_overall_quality": _correlation_bundle(
                [_as_float(row.get("latency_ms")) for row in model_rows],
                [_as_float(row.get("overall_quality")) for row in model_rows],
            ),
        }

    semantic_vs_dimensions = {
        "global": {
            "gold_concepts_coverage_vs_factual_correctness": _correlation_bundle(
                [_as_float(row.get("gold_concepts_coverage")) for row in experimental_rows],
                [_as_float(row.get("factual_correctness")) for row in experimental_rows],
            ),
            "gold_actions_coverage_vs_actionability": _correlation_bundle(
                [_as_float(row.get("gold_actions_coverage")) for row in experimental_rows],
                [_as_float(row.get("actionability_judge")) for row in experimental_rows],
            ),
            "gold_facts_coverage_vs_domain_accuracy": _correlation_bundle(
                [_as_float(row.get("gold_facts_coverage")) for row in experimental_rows],
                [_as_float(row.get("domain_accuracy")) for row in experimental_rows],
            ),
        },
        "by_judge": {},
    }
    for judge_name in judge_names:
        judge_rows = [row for row in rows if str(row["judge_name"]) == judge_name]
        semantic_vs_dimensions["by_judge"][judge_name] = {
            "gold_concepts_coverage_vs_factual_correctness": _correlation_bundle(
                [_as_float(row.get("gold_concepts_coverage")) for row in judge_rows],
                [_as_float(row.get("factual_correctness")) for row in judge_rows],
            ),
            "gold_actions_coverage_vs_actionability": _correlation_bundle(
                [_as_float(row.get("gold_actions_coverage")) for row in judge_rows],
                [_as_float(row.get("actionability_judge")) for row in judge_rows],
            ),
            "gold_facts_coverage_vs_domain_accuracy": _correlation_bundle(
                [_as_float(row.get("gold_facts_coverage")) for row in judge_rows],
                [_as_float(row.get("domain_accuracy")) for row in judge_rows],
            ),
        }

    return {
        "inter_judge": inter_judge,
        "inter_dimension": inter_dimension,
        "deterministic_vs_judge": deterministic_vs_judge,
        "cost_quality": cost_quality,
        "semantic_vs_dimensions": semantic_vs_dimensions,
        "n_rows": len(rows),
    }


def _pearson_correlation(
    x: list[float], y: list[float]
) -> tuple[float, float]:
    """Correlación de Pearson con p-value aproximado."""
    n = len(x)
    if n < 3:
        return (0.0, 1.0)

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    ss_xy = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    ss_xx = sum((x[i] - mean_x) ** 2 for i in range(n))
    ss_yy = sum((y[i] - mean_y) ** 2 for i in range(n))

    if ss_xx == 0 or ss_yy == 0:
        return (0.0, 1.0)

    r = ss_xy / math.sqrt(ss_xx * ss_yy)

    # p-value usando scipy
    if abs(r) >= 1.0:
        p = 0.0
    else:
        t_stat = r * math.sqrt((n - 2) / (1 - r * r))
        p = 2.0 * (1.0 - scipy_stats.t.cdf(abs(t_stat), n - 2))

    return (r, p)


def _spearman_correlation(
    x: list[float], y: list[float]
) -> tuple[float, float]:
    """Correlación de Spearman con p-value aproximado."""
    n = len(x)
    if n < 3:
        return (0.0, 1.0)

    rank_x = _rankdata(x)
    rank_y = _rankdata(y)

    return _pearson_correlation(rank_x, rank_y)


def _correlation_bundle(x: list[float], y: list[float]) -> dict[str, Any]:
    n = min(len(x), len(y))
    if n < 3:
        return {
            "pearson_r": 0.0,
            "pearson_p": 1.0,
            "spearman_rho": 0.0,
            "spearman_p": 1.0,
            "interpretation": "insuficiente",
            "n": n,
        }

    pearson_r, pearson_p = _pearson_correlation(x[:n], y[:n])
    spearman_rho, spearman_p = _spearman_correlation(x[:n], y[:n])
    return {
        "pearson_r": round(pearson_r, 4),
        "pearson_p": round(pearson_p, 6),
        "spearman_rho": round(spearman_rho, 4),
        "spearman_p": round(spearman_p, 6),
        "interpretation": _interpretation(abs(pearson_r)),
        "n": n,
    }


def _interpretation(abs_r: float) -> str:
    if abs_r >= 0.7:
        return "alta"
    if abs_r >= 0.4:
        return "moderada"
    return "baja"


def _bias_direction(delta: float) -> str:
    if delta > 0:
        return "judge_a_more_generous"
    if delta < 0:
        return "judge_b_more_generous"
    return "balanced"


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    for key, value in list(normalized.items()):
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.lower() in {"true", "false"}:
                normalized[key] = stripped.lower() == "true"
                continue
            try:
                if "." in stripped or "e" in stripped.lower():
                    normalized[key] = float(stripped)
                else:
                    normalized[key] = int(stripped)
            except ValueError:
                normalized[key] = value
    return normalized


def _safe_mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _rankdata(values: list[float]) -> list[float]:
    """Assigns ranks to values (average rank for ties)."""
    n = len(values)
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and math.isclose(indexed[j + 1][1], indexed[j][1], rel_tol=1e-9):
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


# ── Frontera de Pareto ──────────────────────────────────────────────


@dataclass
class ParetoPoint:
    """Punto en la frontera de Pareto."""

    name: str
    cost: float
    quality: float
    is_pareto_optimal: bool = False


def pareto_frontier(
    points: list[tuple[str, float, float]],
) -> list[ParetoPoint]:
    """Identifica puntos de Pareto en el espacio coste-calidad.

    Un punto es Pareto-optimal si no existe otro punto con:
    - menor coste Y mayor o igual calidad, O
    - igual coste Y mayor calidad

    Args:
        points: Lista de (nombre, coste, calidad).

    Returns:
        Lista de ParetoPoint con flag is_pareto_optimal.
    """
    result = []
    for name, cost, quality in points:
        result.append(ParetoPoint(name=name, cost=cost, quality=quality))

    for p in result:
        dominated = False
        for q in result:
            if q.name == p.name:
                continue
            # q domina a p si: q tiene <= coste Y >= calidad (al menos uno estricto)
            if q.cost <= p.cost and q.quality >= p.quality:
                if q.cost < p.cost or q.quality > p.quality:
                    dominated = True
                    break
        p.is_pareto_optimal = not dominated

    return result
