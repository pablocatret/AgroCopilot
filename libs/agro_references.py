from __future__ import annotations

from typing import Dict, Optional, Tuple


AGRO_THRESHOLDS: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {
    "trigo": {
        "siembra": {"NDVI": (0.15, 0.35)},
        "nascencia": {"NDVI": (0.15, 0.35)},
        "macollaje": {"NDVI": (0.35, 0.60)},
        "hinchazon": {"NDVI": (0.40, 0.65)},
        "espigazon": {"NDVI": (0.45, 0.70)},
        "madurez": {"NDVI": (0.30, 0.50)},
        "cosecha": {"NDVI": (0.20, 0.40)},
    },
    "cebada": {
        "siembra": {"NDVI": (0.15, 0.35)},
        "macollaje": {"NDVI": (0.35, 0.55)},
        "espigazon": {"NDVI": (0.40, 0.65)},
        "cosecha": {"NDVI": (0.20, 0.40)},
    },
    "avena": {
        "siembra": {"NDVI": (0.15, 0.35)},
        "macollaje": {"NDVI": (0.35, 0.55)},
        "espigazon": {"NDVI": (0.40, 0.65)},
        "cosecha": {"NDVI": (0.20, 0.40)},
    },
    "centeno": {
        "siembra": {"NDVI": (0.15, 0.35)},
        "macollaje": {"NDVI": (0.35, 0.55)},
        "espigazon": {"NDVI": (0.40, 0.65)},
        "cosecha": {"NDVI": (0.20, 0.40)},
    },
    "maiz": {
        "siembra": {"NDVI": (0.10, 0.30)},
        "nascencia": {"NDVI": (0.10, 0.30)},
        "vegetativo": {"NDVI": (0.40, 0.70)},
        "crecimiento": {"NDVI": (0.45, 0.75)},
        "llenado": {"NDVI": (0.40, 0.70)},
        "maduracion": {"NDVI": (0.25, 0.50)},
        "cosecha": {"NDVI": (0.15, 0.35)},
    },
    "soja": {
        "siembra": {"NDVI": (0.10, 0.30)},
        "emergencia": {"NDVI": (0.10, 0.30)},
        "vegetativo": {"NDVI": (0.40, 0.70)},
        "floracion": {"NDVI": (0.50, 0.75)},
        "llenado": {"NDVI": (0.45, 0.70)},
        "maduracion": {"NDVI": (0.25, 0.50)},
        "cosecha": {"NDVI": (0.15, 0.35)},
    },
    "girasol": {
        "siembra": {"NDVI": (0.10, 0.30)},
        "vegetativo": {"NDVI": (0.35, 0.60)},
        "crecimiento": {"NDVI": (0.40, 0.65)},
        "floracion": {"NDVI": (0.35, 0.60)},
        "maduracion": {"NDVI": (0.20, 0.45)},
        "cosecha": {"NDVI": (0.15, 0.35)},
    },
    "olivo": {
        "invierno": {"NDVI": (0.20, 0.40)},
        "primavera": {"NDVI": (0.30, 0.55)},
        "floracion": {"NDVI": (0.35, 0.55)},
        "cuajado": {"NDVI": (0.30, 0.50)},
        "engorde": {"NDVI": (0.35, 0.55)},
        "cosecha": {"NDVI": (0.20, 0.40)},
    },
    "vid": {
        "dormancia": {"NDVI": (0.08, 0.20)},
        "brotacion": {"NDVI": (0.15, 0.35)},
        "floracion": {"NDVI": (0.35, 0.60)},
        "cuajado": {"NDVI": (0.35, 0.60)},
        "envero": {"NDVI": (0.25, 0.50)},
        "maduracion": {"NDVI": (0.15, 0.40)},
        "cosecha": {"NDVI": (0.10, 0.30)},
    },
    "cafe": {
        "vegetativo": {"NDVI": (0.40, 0.70)},
        "crecimiento": {"NDVI": (0.45, 0.75)},
        "floracion": {"NDVI": (0.40, 0.65)},
        "cosecha": {"NDVI": (0.30, 0.55)},
    },
    "pasto": {
        "crecimiento": {"NDVI": (0.30, 0.60)},
        "maduracion": {"NDVI": (0.20, 0.45)},
    },
}

_CROP_SYNONYMS: Dict[str, str] = {
    "café": "cafe",
    "maíz": "maiz",
    "maize": "maiz",
    "girasoles": "girasol",
    "soya": "soja",
    "uva": "vid",
    "viñedo": "vid",
    "viñedo": "vid",
    "pradera": "pasto",
    "forraje": "pasto",
}

_STAGE_SYNONYMS: Dict[str, str] = {
    "floración": "floracion",
    "cosech": "cosecha",
    "maduración": "maduracion",
    "nascencia": "siembra",
}


def _normalize_crop(crop_type: str) -> str:
    low = crop_type.lower().strip()
    return _CROP_SYNONYMS.get(low, low)


def _normalize_stage(growth_stage: str) -> str:
    low = growth_stage.lower().strip()
    return _STAGE_SYNONYMS.get(low, low)


def get_threshold_context(
    crop_type: str,
    growth_stage: str,
    index_name: str,
    current_mean: float,
) -> Optional["ThresholdContext"]:
    """Compara el valor actual contra el rango de referencia agronomica.

    Returns None if no reference is available for the given crop/stage/index.
    """
    from libs.schemas import ThresholdContext

    crop = _normalize_crop(crop_type)
    stage = _normalize_stage(growth_stage)
    index_upper = (index_name or "").upper()

    crop_refs = AGRO_THRESHOLDS.get(crop)
    if not crop_refs:
        return None

    stage_refs = crop_refs.get(stage)
    if not stage_refs:
        return None

    ref_range = stage_refs.get(index_upper)
    if not ref_range:
        return None

    ref_min, ref_max = ref_range
    if current_mean < ref_min:
        status = "below"
        message = (
            f"{index_upper} de {_fmt(current_mean)} por debajo del rango esperado "
            f"({_fmt(ref_min)}-{_fmt(ref_max)}) para {crop_type} en {growth_stage}."
        )
    elif current_mean > ref_max:
        status = "above"
        message = (
            f"{index_upper} de {_fmt(current_mean)} por encima del rango esperado "
            f"({_fmt(ref_min)}-{_fmt(ref_max)}) para {crop_type} en {growth_stage}."
        )
    else:
        status = "normal"
        message = (
            f"{index_upper} de {_fmt(current_mean)} dentro del rango esperado "
            f"({_fmt(ref_min)}-{_fmt(ref_max)}) para {crop_type} en {growth_stage}."
        )

    return ThresholdContext(
        reference_range=ref_range,
        status=status,
        message=message,
    )


def _fmt(value: float) -> str:
    return f"{value:.2f}"
