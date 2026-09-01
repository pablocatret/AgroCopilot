# libs/geo.py
from __future__ import annotations

import json
import tempfile
from typing import List, Optional

import diskcache as dcache
import httpx
from loguru import logger

from backend.deps import settings

try:
    _cache = dcache.Cache("./.cache/geocoding")
except Exception:
    fallback_dir = tempfile.mkdtemp(prefix="geocoding-cache-")
    _cache = dcache.Cache(fallback_dir)


def _parse_viewbox(s: str | None) -> Optional[List[float]]:
    if not s:
        return None
    try:
        parts = [float(x) for x in s.split(",")]
        if len(parts) == 4:
            return parts
    except Exception:
        pass
    return None


async def geocode_bbox(place: str) -> Optional[List[float]]:
    """
    Devuelve [minLon, minLat, maxLon, maxLat] o None.
    Prioriza Nominatim; soporta Mapbox si hay token.
    Cachea por (provider, place, country_bias, lang, viewbox).
    """
    place_norm = place.strip()
    if not place_norm:
        return None

    lang = "es"
    country = settings.GEOCODER_COUNTRY_BIAS.strip() if settings.GEOCODER_COUNTRY_BIAS else None
    viewbox = _parse_viewbox(settings.GEOCODER_VIEWBOX)
    key = json.dumps(
        {"p": place_norm, "prov": settings.GEOCODER, "c": country, "l": lang, "vb": viewbox}
    )

    if key in _cache:
        return _cache.get(key)

    prov = settings.GEOCODER.upper()
    try:
        if prov == "MAPBOX" and settings.GEOCODER_MAPBOX_TOKEN:
            bbox = await _mapbox_geocode(place_norm, lang, country)
        else:
            # Default: Nominatim
            bbox = await _nominatim_geocode(place_norm, lang, country, viewbox)
    except Exception as e:
        logger.warning({"event": "geocode_error", "msg": str(e)})
        bbox = None

    if bbox:
        bbox = _expand_small_bbox(bbox)
        _cache.set(key, bbox, expire=60 * 60 * 24 * 30)  # 30 días
    return bbox


def _expand_small_bbox(bbox: List[float], min_size: float = 0.002) -> List[float]:
    """Si el bbox es muy pequeño, lo expande lo justo para evitar AOI degeneradas."""
    min_lon, min_lat, max_lon, max_lat = bbox
    width = max_lon - min_lon
    height = max_lat - min_lat
    if width >= min_size and height >= min_size:
        return bbox
    expand_w = max(0, (min_size - width) / 2)
    expand_h = max(0, (min_size - height) / 2)
    return [min_lon - expand_w, min_lat - expand_h, max_lon + expand_w, max_lat + expand_h]


async def _nominatim_geocode(
    place: str, lang: str, country: str | None, viewbox: List[float] | None
):
    params = {
        "q": place,
        "format": "jsonv2",
        "polygon_geojson": 0,
        "addressdetails": 0,
        "limit": 3,
    }
    if country:
        params["countrycodes"] = country.lower()
    if viewbox:
        params["viewbox"] = ",".join(str(x) for x in viewbox)
        params["bounded"] = 1

    headers = {
        "User-Agent": f"AgroCopilot/1.0 ({settings.GEOCODER_EMAIL or 'contact@local'})",
        "Accept-Language": lang,
    }
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(settings.GEOCODER_URL, params=params, headers=headers)
        r.raise_for_status()
        data = r.json()

    if not data:
        return None

    # Selección: prioriza administraciones (type=administrative, class=boundary)
    def score(item):
        cls = item.get("class", "")
        typ = item.get("type", "")
        imp = item.get("importance", 0.0) or 0.0
        bonus = 1.0 if (cls == "boundary" and typ in {"administrative", "political"}) else 0.0
        return imp + bonus

    best = sorted(data, key=score, reverse=True)[0]
    try:
        south, north, west, east = map(float, best["boundingbox"])
        return [west, south, east, north]
    except Exception:
        return None


async def _mapbox_geocode(place: str, lang: str, country: str | None):
    token = settings.GEOCODER_MAPBOX_TOKEN
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{place}.json"
    params = {
        "access_token": token,
        "language": lang,
        "limit": 3,
        "types": "region,place,locality,neighborhood,district,postcode",
    }
    if country:
        params["country"] = country.lower()
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    feats = data.get("features") or []
    if not feats:
        return None
    # Mapbox bbox ya viene en [minLon, minLat, maxLon, maxLat]
    for f in feats:
        bbox = f.get("bbox")
        if bbox and len(bbox) == 4:
            return [float(x) for x in bbox]
    # sin bbox → crea una caja alrededor del centro, adaptada al tipo de lugar
    center = feats[0].get("center")
    place_type = feats[0].get("place_type", [""])[0] if feats[0].get("place_type") else ""
    if center and len(center) == 2:
        lon, lat = map(float, center)
        half_sizes = {
            "neighborhood": 0.01,
            "locality": 0.015,
            "place": 0.02,
            "district": 0.03,
            "region": 0.1,
        }
        half = half_sizes.get(place_type, 0.015)
        return [lon - half, lat - half, lon + half, lat + half]
    return None
