from __future__ import annotations

from typing import Optional

import httpx
from loguru import logger

from libs.schemas import MeteoContext


_OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _daily_mean(values: list[float]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def _daily_total(values: list[float]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    return sum(valid) if valid else None


def _simple_precipitation_index(precip_values: list[float]) -> Optional[float]:
    """Indice de irregularidad pluviometrica basado en coeficiente de variacion.

    No es un SPI verdadero (requiere 30+ anios de datos climatologicos).
    Usa CV como proxy: CV alto = precipitacion irregular = sequia relativa.
    Rango: 0 (regular) a -3 (muy irregular/seco).
    """
    valid = [v for v in precip_values if v is not None and v >= 0]
    if len(valid) < 2:
        return None
    mean = sum(valid) / len(valid)
    if mean < 1e-6:
        return -1.0 if all(v < 0.1 for v in valid) else 0.0
    variance = sum((v - mean) ** 2 for v in valid) / (len(valid) - 1)
    std = variance ** 0.5
    cv = std / mean
    return round(max(-3.0, min(0.0, -cv * 1.5)), 2)


def _normalize_date(value: str) -> Optional[str]:
    try:
        return value[:10] if len(value) >= 10 else None
    except TypeError:
        return None


def _build_meteo_params(
    bbox: list[float],
    start_date: str,
    end_date: str,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
) -> Optional[dict[str, float | str]]:
    start_norm = _normalize_date(start_date)
    end_norm = _normalize_date(end_date)
    if not start_norm or not end_norm:
        return None

    if lat is None and lon is None:
        center_lat = (bbox[1] + bbox[3]) / 2
        center_lon = (bbox[0] + bbox[2]) / 2
    else:
        center_lat = lat if lat is not None else (bbox[1] + bbox[3]) / 2
        center_lon = lon if lon is not None else (bbox[0] + bbox[2]) / 2

    return {
        "latitude": round(center_lat, 4),
        "longitude": round(center_lon, 4),
        "start_date": start_norm,
        "end_date": end_norm,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "auto",
    }


def _parse_meteo_context(data: dict, *, start_norm: str, end_norm: str) -> Optional[MeteoContext]:
    daily = data.get("daily", {})
    if not daily:
        return None

    temps_max = daily.get("temperature_2m_max", [])
    temps_min = daily.get("temperature_2m_min", [])
    precip = daily.get("precipitation_sum", [])

    total_precip = _daily_total(precip)
    avg_temp = _daily_mean(
        [(lo + hi) / 2 for lo, hi in zip(temps_min, temps_max) if lo is not None and hi is not None]
        or []
    )
    valid_min = [v for v in temps_min if v is not None]
    valid_max = [v for v in temps_max if v is not None]
    min_temp_val = min(valid_min) if valid_min else None
    max_temp_val = max(valid_max) if valid_max else None

    pii = _simple_precipitation_index(precip)

    return MeteoContext(
        total_precip_mm=round(total_precip, 1) if total_precip is not None else None,
        avg_temp_c=round(avg_temp, 1) if avg_temp is not None else None,
        max_temp_c=round(max_temp_val, 1) if max_temp_val is not None else None,
        min_temp_c=round(min_temp_val, 1) if min_temp_val is not None else None,
        precipitation_irregularity_index=pii,
        period_start=start_norm,
        period_end=end_norm,
    )


async def fetch_meteo_context_async(
    bbox: list[float],
    start_date: str,
    end_date: str,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
) -> Optional[MeteoContext]:
    params = _build_meteo_params(bbox, start_date, end_date, lat=lat, lon=lon)
    if not params:
        return None
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(_OPEN_METEO_ARCHIVE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning({"event": "meteo_fetch_error", "msg": str(exc)})
        return None
    return _parse_meteo_context(
        data,
        start_norm=str(params["start_date"]),
        end_norm=str(params["end_date"]),
    )


def fetch_meteo_context(
    bbox: list[float],
    start_date: str,
    end_date: str,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
) -> Optional[MeteoContext]:
    """
    Obtiene contexto meteorologico de Open-Meteo.

    Args:
        bbox: [minLon, minLat, maxLon, maxLat]
        start_date: 'YYYY-MM-DD'
        end_date: 'YYYY-MM-DD'
        lat: latitud del punto central (usa bbox si no se proporciona)
        lon: longitud del punto central (usa bbox si no se proporciona)

    Returns:
        MeteoContext o None si falla.
    """
    params = _build_meteo_params(bbox, start_date, end_date, lat=lat, lon=lon)
    if not params:
        return None

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(_OPEN_METEO_ARCHIVE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning({"event": "meteo_fetch_error", "msg": str(exc)})
        return None

    return _parse_meteo_context(
        data,
        start_norm=str(params["start_date"]),
        end_norm=str(params["end_date"]),
    )
