"""Tests for evaluation/stats.py — statistics functions."""
import math

from evaluation.stats import (
    bootstrap_ci,
    correlation_analysis,
    paired_t_test,
    pareto_frontier,
)


class TestBootstrapCI:
    def test_empty(self):
        mean, lo, hi = bootstrap_ci([])
        assert mean == 0.0

    def test_single_value(self):
        mean, lo, hi = bootstrap_ci([5.0])
        assert mean == 5.0
        assert lo == 5.0
        assert hi == 5.0

    def test_returns_tuple(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean, lo, hi = bootstrap_ci(values, n_bootstrap=500, seed=42)
        assert abs(mean - 3.0) < 0.01
        assert lo < mean < hi

    def test_reproducible_with_seed(self):
        values = [1.0, 2.0, 3.0]
        r1 = bootstrap_ci(values, seed=123)
        r2 = bootstrap_ci(values, seed=123)
        assert r1 == r2


class TestPairedTTest:
    def test_identical_values(self):
        result = paired_t_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert result.mean_delta == 0.0
        assert result.p_value > 0.05

    def test_different_values(self):
        a = [5.0, 4.5, 5.5, 4.8, 5.2]
        b = [1.0, 1.5, 0.8, 1.2, 1.1]
        result = paired_t_test(a, b)
        assert result.mean_delta > 0
        assert result.p_value < 0.05
        assert result.cohens_d > 0

    def test_too_few_pairs(self):
        result = paired_t_test([1.0], [2.0])
        assert result.n_pairs == 1
        assert "Insuficiente" in result.interpretation


class TestCorrelationAnalysis:
    def test_perfect_correlation(self):
        data = {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0],
        }
        results = correlation_analysis(data)
        assert len(results) == 1
        assert results[0].pearson_r > 0.99
        assert results[0].interpretation == "alta"

    def test_no_correlation(self):
        data = {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [5.0, 3.0, 1.0, 4.0, 2.0],
        }
        results = correlation_analysis(data)
        assert len(results) == 1
        assert abs(results[0].pearson_r) <= 0.5


class TestParetoFrontier:
    def test_all_pareto(self):
        # Each point has unique cost and quality, no dominance
        points = [("A", 1.0, 5.0), ("B", 2.0, 4.0), ("C", 3.0, 3.0)]
        result = pareto_frontier(points)
        # A is cheapest + best → dominates all → only A is pareto
        # B is dominated by A, C is dominated by A
        pareto_names = [p.name for p in result if p.is_pareto_optimal]
        assert "A" in pareto_names
        assert len(pareto_names) == 1

    def test_one_dominated(self):
        points = [("A", 1.0, 5.0), ("B", 2.0, 4.0), ("C", 1.5, 3.0)]
        result = pareto_frontier(points)
        # B is dominated by A (lower cost, higher quality)
        # C is dominated by A (lower cost, higher quality)
        a = next(p for p in result if p.name == "A")
        b = next(p for p in result if p.name == "B")
        c = next(p for p in result if p.name == "C")
        assert a.is_pareto_optimal
        assert not b.is_pareto_optimal
        assert not c.is_pareto_optimal
