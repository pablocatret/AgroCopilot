"""Reproducible statistical analysis for a completed benchmark.

The experimental unit is (case_id, model, run_idx). Judge rows are kept
separate for agreement analysis and averaged only for model comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


JUDGES = ("mimo-v2.5", "hy3-preview", "step-3.7-flash")
MODEL_METRICS = (
    "answer_completeness",
    "forbidden_claim_rate",
    "actionability_det",
    "routing_score",
    "routing_precision",
    "routing_recall",
)
JUDGE_METRICS = (
    "factual_correctness",
    "evidence_utilization",
    "actionability_judge",
    "practical_value",
)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else float("nan")


def sample_sd(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.stdev(values) if len(values) > 1 else 0.0


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    position = (len(values) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (position - low)


def wilson_ci(successes: int, trials: int, z: float = 1.959963984540054) -> dict:
    if trials == 0:
        return {"rate": None, "ci95_low": None, "ci95_high": None}
    rate = successes / trials
    denominator = 1 + z**2 / trials
    centre = (rate + z**2 / (2 * trials)) / denominator
    margin = z * math.sqrt((rate * (1 - rate) + z**2 / (4 * trials)) / trials) / denominator
    return {"rate": rate, "ci95_low": max(0.0, centre - margin), "ci95_high": min(1.0, centre + margin)}


def bootstrap_mean_ci(values: list[float], iterations: int, rng: random.Random) -> dict:
    if not values:
        return {"mean": None, "ci95_low": None, "ci95_high": None}
    estimates = []
    n = len(values)
    for _ in range(iterations):
        estimates.append(mean(values[rng.randrange(n)] for _ in range(n)))
    return {
        "mean": mean(values),
        "ci95_low": percentile(estimates, 0.025),
        "ci95_high": percentile(estimates, 0.975),
    }


def wilcoxon_paired(x: list[float], y: list[float]) -> dict:
    """Use SciPy's exact method when possible, asymptotic otherwise."""
    try:
        from scipy.stats import wilcoxon
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("Este análisis requiere scipy") from exc
    differences = [a - b for a, b in zip(x, y)]
    nonzero = [d for d in differences if d != 0]
    if not nonzero:
        return {"statistic": 0.0, "p_value": 1.0, "n_nonzero": 0}
    method = "exact" if len(nonzero) <= 25 and len(set(abs(d) for d in nonzero)) == len(nonzero) else "approx"
    result = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", method=method)
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue), "n_nonzero": len(nonzero), "method": method}


def friedman_global(model_values: dict[str, list[float]], models: list[str]) -> dict:
    try:
        from scipy.stats import friedmanchisquare
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("Este análisis requiere scipy") from exc
    result = friedmanchisquare(*(model_values[model] for model in models))
    return {"n_cases": len(model_values[models[0]]), "n_models": len(models), "statistic": float(result.statistic), "p_value": float(result.pvalue)}


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    m = len(ordered)
    for index, (name, p_value) in enumerate(ordered):
        value = min(1.0, (m - index) * p_value)
        value = max(value, running)
        adjusted[name] = value
        running = value
    return adjusted


def icc_2_1(matrix: list[list[float]]) -> float | None:
    """Two-way random-effects, absolute-agreement ICC(2,1)."""
    n = len(matrix)
    k = len(matrix[0]) if matrix else 0
    if n < 2 or k < 2:
        return None
    grand = mean(value for row in matrix for value in row)
    subject_means = [mean(row) for row in matrix]
    rater_means = [mean(matrix[i][j] for i in range(n)) for j in range(k)]
    ss_subject = k * sum((value - grand) ** 2 for value in subject_means)
    ss_rater = n * sum((value - grand) ** 2 for value in rater_means)
    ss_total = sum((value - grand) ** 2 for row in matrix for value in row)
    ss_error = ss_total - ss_subject - ss_rater
    ms_subject = ss_subject / (n - 1)
    ms_rater = ss_rater / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denominator = ms_subject + (k - 1) * ms_error + k * (ms_rater - ms_error) / n
    return (ms_subject - ms_error) / denominator if denominator else None


def kendall_w(matrix: list[list[float]]) -> float | None:
    """Kendall's coefficient of concordance for ordinal judge scores."""
    n = len(matrix)
    k = len(matrix[0]) if matrix else 0
    if n < 2 or k < 2:
        return None
    # Rank each judge across subjects, then sum ranks per subject.
    rank_sums = [0.0] * n
    tie_correction = 0.0
    for judge_index in range(k):
        values = [matrix[subject_index][judge_index] for subject_index in range(n)]
        for subject_index, value in enumerate(values):
            rank_sums[subject_index] += 1 + sum(other < value for other in values) + (sum(other == value for other in values) - 1) / 2
        for value in set(values):
            tie_size = values.count(value)
            tie_correction += tie_size**3 - tie_size
    average = mean(rank_sums)
    s = sum((value - average) ** 2 for value in rank_sums)
    denominator = k * k * (n**3 - n) - k * tie_correction
    return (12 * s) / denominator if denominator else None


def paired_effect(x: list[float], y: list[float]) -> dict:
    differences = [a - b for a, b in zip(x, y)]
    positive = sum(d > 0 for d in differences)
    negative = sum(d < 0 for d in differences)
    nonzero = positive + negative
    return {
        "mean_difference": mean(differences),
        "sd_difference": sample_sd(differences),
        "cohen_dz": mean(differences) / sample_sd(differences) if sample_sd(differences) else None,
        "paired_cliffs_delta": (positive - negative) / nonzero if nonzero else 0.0,
        "positive": positive,
        "negative": negative,
        "ties": len(differences) - nonzero,
    }


def grouped_summary(tasks: dict, by_task: dict, field: str) -> dict:
    result = {}
    groups = sorted({row.get(field, "unknown") for row in tasks.values()})
    for group in groups:
        keys = [key for key, row in tasks.items() if row.get(field, "unknown") == group]
        judge_values = [numeric(row, "overall_quality") for key in keys for row in by_task[key]]
        deterministic = [tasks[key] for key in keys]
        success = sum(row.get("success") == "True" for row in deterministic)
        result[group] = {
            "n_cases": len({key[0] for key in keys}),
            "n_tasks": len(keys),
            "overall_mean": mean(judge_values),
            "completeness_mean": mean(numeric(row, "answer_completeness") for row in deterministic),
            "success_count": success,
            "success_rate": success / len(keys) if keys else None,
        }
    return result


def load_rows(input_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(input_dir.rglob("aggregate.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No se encontraron aggregate.csv en {input_dir}")
    return rows


def load_parser_summary(input_dir: Path) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)
    for path in sorted(input_dir.glob("batch_*/report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        for name, value in (report.get("summary", {}).get("parser_summary", {}) or {}).items():
            totals[name] += int(value)
    return dict(sorted(totals.items()))


def task_key(row: dict) -> tuple[str, str, str]:
    return row["case_id"], row["model"], row["run_idx"]


def numeric(row: dict, field: str) -> float:
    value = row.get(field, "")
    return float(value) if value not in (None, "") else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("evaluation/results/benchmark_final"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()
    output = args.output or args.input / "statistical_analysis"
    output.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.input)
    by_task = defaultdict(list)
    for row in rows:
        by_task[task_key(row)].append(row)
    if any(len(group) != len(JUDGES) for group in by_task.values()):
        raise RuntimeError("Each experimental unit must have exactly three judges")
    tasks = {key: group[0] for key, group in by_task.items()}
    models = sorted({key[1] for key in tasks})
    rng = random.Random(args.seed)

    model_summary = {}
    model_values = {}
    for model in models:
        keys = sorted(key for key in tasks if key[1] == model)
        judge_mean = [mean(numeric(row, "overall_quality") for row in by_task[key]) for key in keys]
        model_values[model] = judge_mean
        summary = {
            "n_tasks": len(keys),
            "overall_quality": {
                **bootstrap_mean_ci(judge_mean, args.bootstrap, rng),
                "sd": sample_sd(judge_mean),
                "median": statistics.median(judge_mean),
                "iqr": percentile(judge_mean, 0.75) - percentile(judge_mean, 0.25),
            },
        }
        for metric in MODEL_METRICS:
            values = [numeric(tasks[key], metric) for key in keys]
            values = [value for value in values if not math.isnan(value)]
            summary[metric] = {**bootstrap_mean_ci(values, args.bootstrap, rng), "sd": sample_sd(values)}
        for metric in JUDGE_METRICS:
            values = [numeric(row, metric) for key in keys for row in by_task[key]]
            summary[metric] = {"mean": mean(values), "sd": sample_sd(values), "median": statistics.median(values), "iqr": percentile(values, 0.75) - percentile(values, 0.25)}
        success = [numeric(tasks[key], "success") if tasks[key].get("success") in ("0", "1") else (1.0 if tasks[key].get("success") == "True" else 0.0) for key in keys]
        success_ci = bootstrap_mean_ci(success, args.bootstrap, rng)
        summary["success_rate"] = {**success_ci, "wilson": wilson_ci(sum(success), len(success))}
        model_summary[model] = summary

    pairwise = {}
    raw_p = {}
    for index, left in enumerate(models):
        for right in models[index + 1 :]:
            left_by_case = {key[0]: value for key, value in zip(sorted(key for key in tasks if key[1] == left), model_values[left])}
            right_by_case = {key[0]: value for key, value in zip(sorted(key for key in tasks if key[1] == right), model_values[right])}
            common = sorted(set(left_by_case) & set(right_by_case))
            x = [left_by_case[case] for case in common]
            y = [right_by_case[case] for case in common]
            test = wilcoxon_paired(x, y)
            key_name = f"{left}__vs__{right}"
            raw_p[key_name] = test["p_value"]
            pairwise[key_name] = {"left": left, "right": right, "n_cases": len(common), **paired_effect(x, y), **test}
    adjusted = holm_adjust(raw_p)
    for key_name, value in pairwise.items():
        value["holm_p_value"] = adjusted[key_name]
        value["significant_holm_alpha_0_05"] = adjusted[key_name] < 0.05

    global_test = friedman_global(model_values, models)

    matrix = []
    for key in sorted(tasks):
        scores = {row["judge_name"]: numeric(row, "overall_quality") for row in by_task[key]}
        matrix.append([scores[judge] for judge in JUDGES])
    agreement = {"n_tasks": len(matrix), "judges": list(JUDGES), "icc_2_1_absolute_agreement": icc_2_1(matrix), "kendall_w": kendall_w(matrix)}

    result = {
        "analysis_version": "1.0",
        "experimental_unit": "(case_id, model, run_idx)",
        "judge_rows": len(rows),
        "tasks": len(tasks),
        "models": models,
        "parser_summary_by_llm_call": load_parser_summary(args.input),
        "bootstrap_iterations": args.bootstrap,
        "random_seed": args.seed,
        "model_summary": model_summary,
        "family_summary": grouped_summary(tasks, by_task, "family"),
        "difficulty_summary": grouped_summary(tasks, by_task, "difficulty"),
        "pairwise_overall_wilcoxon_holm": pairwise,
        "global_friedman_overall": global_test,
        "inter_judge_agreement": agreement,
        "notes": [
            "Deterministic metrics use one row per task.",
            "Model comparisons are paired by case.",
            "Bootstrap intervals resample cases, not judges.",
            "With one run per case, between-run variance is not estimated.",
        ],
    }
    (output / "statistics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "model_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["model", "n_tasks", "overall_mean", "overall_median", "overall_iqr", "overall_sd", "overall_ci95_low", "overall_ci95_high", "success_rate", "success_bootstrap_ci95_low", "success_bootstrap_ci95_high", "success_wilson_ci95_low", "success_wilson_ci95_high"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model, summary in model_summary.items():
            writer.writerow({"model": model, "n_tasks": summary["n_tasks"], "overall_mean": summary["overall_quality"]["mean"], "overall_median": summary["overall_quality"]["median"], "overall_iqr": summary["overall_quality"]["iqr"], "overall_sd": summary["overall_quality"]["sd"], "overall_ci95_low": summary["overall_quality"]["ci95_low"], "overall_ci95_high": summary["overall_quality"]["ci95_high"], "success_rate": summary["success_rate"]["mean"], "success_bootstrap_ci95_low": summary["success_rate"]["ci95_low"], "success_bootstrap_ci95_high": summary["success_rate"]["ci95_high"], "success_wilson_ci95_low": summary["success_rate"]["wilson"]["ci95_low"], "success_wilson_ci95_high": summary["success_rate"]["wilson"]["ci95_high"]})
    print(f"Analysis written to: {output}")
    print(f"Tasks: {len(tasks)} | judge rows: {len(rows)} | models: {len(models)}")
    print(f"ICC(2,1): {agreement['icc_2_1_absolute_agreement']:.4f} | Kendall W: {agreement['kendall_w']:.4f}")


if __name__ == "__main__":
    main()
