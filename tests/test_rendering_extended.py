import numpy as np
import pytest

from libs.rendering import _compute_hotspots


def test_compute_hotspots_detects_high_cluster():
    values = np.concatenate([np.full(90, 0.5), np.full(10, 2.0)])
    result = _compute_hotspots(values, mean=0.64, std=0.38)
    assert result["high_count"] > 0
    assert result["cluster_detected"] is True


def test_compute_hotspots_detects_low_cluster():
    values = np.concatenate([np.full(90, 0.5), np.full(10, -1.0)])
    result = _compute_hotspots(values, mean=0.35, std=0.45)
    assert result["low_count"] > 0
    assert result["cluster_detected"] is True


def test_compute_hotspots_returns_no_cluster_for_uniform_data():
    values = np.full(100, 0.5)
    result = _compute_hotspots(values, mean=0.5, std=0.0)
    assert result["cluster_detected"] is False
    assert result["high_count"] == 0
    assert result["low_count"] == 0


def test_compute_hotspots_handles_small_sample():
    values = np.array([0.1, 0.2, 0.3])
    result = _compute_hotspots(values, mean=0.2, std=0.08)
    assert result["cluster_detected"] is False


def test_compute_hotspots_returns_zero_counts_for_empty():
    values = np.array([])
    result = _compute_hotspots(values, mean=0.0, std=0.0)
    assert result["high_count"] == 0
    assert result["low_count"] == 0
