import pytest
from unittest.mock import patch, MagicMock

from libs.meteo import fetch_meteo_context, _simple_precipitation_index, _daily_mean
from libs.schemas import MeteoContext


def test_daily_mean_returns_average_of_valid_values():
    assert _daily_mean([10.0, 20.0, None, 30.0]) == 20.0


def test_daily_mean_returns_none_for_empty():
    assert _daily_mean([]) is None


def test_daily_mean_returns_none_for_all_none():
    assert _daily_mean([None, None]) is None


def test_pii_returns_minus_one_for_all_zero_precip():
    assert _simple_precipitation_index([0.0, 0.0, 0.0]) == -1.0


def test_pii_returns_zero_for_constant_precip():
    pii = _simple_precipitation_index([5.0, 5.0, 5.0, 5.0, 5.0])
    assert pii is not None
    assert pii == 0.0


def test_pii_returns_negative_for_variable_precip():
    pii = _simple_precipitation_index([0.0, 10.0, 0.0, 10.0, 0.0])
    assert pii is not None
    assert pii < 0


def test_pii_returns_none_for_too_few_values():
    assert _simple_precipitation_index([5.0]) is None


def test_pii_clamps_to_minus_three():
    pii = _simple_precipitation_index([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert pii is not None
    assert pii >= -3.0


@patch("libs.meteo.httpx.Client")
def test_fetch_meteo_context_returns_meteo_context(mock_client_cls):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "daily": {
            "temperature_2m_max": [25.0, 28.0, 30.0],
            "temperature_2m_min": [12.0, 14.0, 16.0],
            "precipitation_sum": [0.0, 5.0, 10.0],
        }
    }
    mock_resp.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.get.return_value = mock_resp
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client_cls.return_value = mock_client

    result = fetch_meteo_context(
        bbox=[-3.5, 37.0, -3.0, 37.5],
        start_date="2026-01-01",
        end_date="2026-01-03",
    )

    assert isinstance(result, MeteoContext)
    assert result.total_precip_mm == 15.0
    assert result.period_start == "2026-01-01"
    assert result.period_end == "2026-01-03"
    assert result.max_temp_c == 30.0
    assert result.min_temp_c == 12.0


@patch("libs.meteo.httpx.Client")
def test_fetch_meteo_context_normalizes_iso_timestamps(mock_client_cls):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "daily": {
            "temperature_2m_max": [25.0],
            "temperature_2m_min": [12.0],
            "precipitation_sum": [0.0],
        }
    }
    mock_resp.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.get.return_value = mock_resp
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client_cls.return_value = mock_client

    result = fetch_meteo_context(
        bbox=[-3.5, 37.0, -3.0, 37.5],
        start_date="2026-01-01T10:30:00Z",
        end_date="2026-01-03T09:00:00Z",
    )

    params = mock_client.get.call_args[1]["params"]
    assert params["start_date"] == "2026-01-01"
    assert params["end_date"] == "2026-01-03"
    assert result is not None
    assert result.period_start == "2026-01-01"
    assert result.period_end == "2026-01-03"


@patch("libs.meteo.httpx.Client")
def test_fetch_meteo_context_returns_none_on_error(mock_client_cls):
    mock_client = MagicMock()
    mock_client.get.side_effect = Exception("network error")
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client_cls.return_value = mock_client

    result = fetch_meteo_context(
        bbox=[-3.5, 37.0, -3.0, 37.5],
        start_date="2026-01-01",
        end_date="2026-01-03",
    )

    assert result is None


@patch("libs.meteo.httpx.Client")
def test_fetch_meteo_context_uses_center_of_bbox(mock_client_cls):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "daily": {
            "temperature_2m_max": [20.0],
            "temperature_2m_min": [10.0],
            "precipitation_sum": [5.0],
        }
    }
    mock_resp.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.get.return_value = mock_resp
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client_cls.return_value = mock_client

    fetch_meteo_context(
        bbox=[-3.5, 37.0, -3.0, 37.5],
        start_date="2026-01-01",
        end_date="2026-01-01",
    )

    call_args = mock_client.get.call_args
    params = call_args[1]["params"] if "params" in call_args[1] else call_args[0][1]
    assert params["latitude"] == pytest.approx(37.25, abs=0.01)
    assert params["longitude"] == pytest.approx(-3.25, abs=0.01)


def test_pii_ignores_negative_precipitation_values():
    result = _simple_precipitation_index([-5.0, -3.0, -1.0])
    assert result is None


def test_pii_mixes_negative_and_positive():
    result = _simple_precipitation_index([-2.0, 10.0, 5.0, 0.0])
    assert result is not None
    assert result <= 0
