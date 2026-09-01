from __future__ import annotations

import ast
import base64
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    from rio_tiler.colormap import cmap
    from rio_tiler.io import Reader
    from rio_tiler.models import ImageData
except (ImportError, OSError):  # pragma: no cover - math helpers remain testable without rio-tiler.
    cmap = None
    Reader = None
    ImageData = Any


_ALLOWED_BIN_OPS = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.divide,
}
_ALLOWED_UNARY_OPS = {
    ast.UAdd: lambda value: value,
    ast.USub: np.negative,
}
DEFAULT_EXCLUDED_SCL_CLASSES = (0, 1, 3, 6, 8, 9, 10, 11)
ESA_WORLDCOVER_CLASSES = {
    10: "arbolado",
    20: "matorral",
    30: "herbaceo",
    40: "cultivo",
    50: "construido",
    60: "suelo desnudo",
    70: "nieve/hielo",
    80: "agua",
    90: "humedal",
    95: "manglar",
    100: "musgo/liquen",
}


def _to_base64_string(img_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(img_bytes).decode("utf-8")


def _calculate_cumulative_min_max(data: np.ndarray) -> Tuple[float, float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return (0.0, 1.0)
    return float(np.nanpercentile(finite, 5)), float(np.nanpercentile(finite, 95))


def _resize_if_needed(img: ImageData, max_size: int = 1024) -> ImageData:
    width = getattr(img, "width", None)
    height = getattr(img, "height", None)
    if not width or not height:
        return img
    if width <= max_size and height <= max_size:
        return img
    if width >= height:
        new_w = max_size
        new_h = max(1, int(height * (max_size / width)))
    else:
        new_h = max_size
        new_w = max(1, int(width * (max_size / height)))
    return img.resize(width=new_w, height=new_h)


def _resize_array_nearest(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if values.shape == shape:
        return values
    target_h, target_w = shape
    source_h, source_w = values.shape
    row_idx = np.clip(np.round(np.linspace(0, source_h - 1, target_h)).astype(int), 0, source_h - 1)
    col_idx = np.clip(np.round(np.linspace(0, source_w - 1, target_w)).astype(int), 0, source_w - 1)
    return values[row_idx][:, col_idx]


def _require_rio_tiler() -> None:
    if Reader is None or cmap is None:
        raise RuntimeError("rio-tiler not installed; cannot render STAC products.")


def render_preview(
    url: str,
    colormap: str = "viridis",
    rescale: Optional[Tuple[float, float]] = None,
    bbox: Optional[Iterable[float]] = None,
) -> str:
    """Genera un PNG base64 de la primera banda."""
    _require_rio_tiler()
    with Reader(url) as cog:
        if bbox:
            img: ImageData = cog.part(bbox=list(bbox), bounds_crs="epsg:4326")
        else:
            img: ImageData = cog.preview(max_size=1024)

    img = _resize_if_needed(img, max_size=1024)
    if not rescale:
        rescale = _calculate_cumulative_min_max(img.data)
    img.rescale(in_range=(rescale,))
    buffer = img.render(img_format="PNG", colormap=cmap.get(colormap))
    return _to_base64_string(buffer)


def _load_previews_as_arrays(
    band_urls: Dict[str, str],
    bbox: Optional[Iterable[float]] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    masks: Dict[str, np.ndarray] = {}
    if Reader is None:
        raise RuntimeError("rio-tiler not installed; cannot load STAC bands.")

    for name, url in band_urls.items():
        with Reader(url) as cog:
            if bbox:
                img = cog.part(bbox=list(bbox), bounds_crs="epsg:4326")
            else:
                img = cog.preview(max_size=1024)

        img = _resize_if_needed(img, max_size=1024)
        arrays[name] = img.data[0].astype("float32")
        masks[name] = img.mask

    if not arrays:
        raise ValueError("Could not load bands")

    # Sentinel-2 combines 10 m bands (for example B08) with 20 m bands
    # (for example B11). Align every band and mask before evaluating an index.
    target_shape = max(
        (array.shape for array in arrays.values()),
        key=lambda shape: shape[0] * shape[1],
    )
    combined_mask: Optional[np.ndarray] = None
    for name, array in list(arrays.items()):
        arrays[name] = _resize_array_nearest(array, target_shape)
        mask = _resize_array_nearest(masks[name], target_shape)
        combined_mask = mask if combined_mask is None else combined_mask & mask

    if combined_mask is None:
        raise ValueError("Could not load band masks")
    return arrays, combined_mask


def _evaluate_index_expression(expression: str, ctx: Dict[str, np.ndarray]) -> np.ndarray:
    def _eval(node: ast.AST) -> np.ndarray | float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.BinOp):
            op = _ALLOWED_BIN_OPS.get(type(node.op))
            if op is None:
                raise ValueError("Operador no permitido en indice espectral.")
            return op(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = _ALLOWED_UNARY_OPS.get(type(node.op))
            if op is None:
                raise ValueError("Operador unario no permitido en indice espectral.")
            return op(_eval(node.operand))
        if isinstance(node, ast.Name):
            if node.id not in ctx:
                raise ValueError(f"Banda desconocida en indice espectral: {node.id}")
            return ctx[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Expresion de indice espectral no permitida.")

    tree = ast.parse(expression, mode="eval")
    return np.asarray(_eval(tree), dtype="float32")


def _index_context(arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    ctx: Dict[str, np.ndarray] = {}
    for key, arr in arrays.items():
        clean = key.lower().replace("_", "")
        ctx[clean] = arr

        import re

        match = re.search(r"b0?(\d+)", clean)
        if match:
            num = match.group(1)
            num_nozero = num.lstrip("0") or num
            ctx[num] = arr
            ctx[num_nozero] = arr
            ctx[f"b{num}"] = arr
            ctx[f"b{num_nozero}"] = arr
    return ctx


def _evaluate_masked_index(
    band_urls: Dict[str, str],
    index_expression: str,
    bbox: Optional[Iterable[float]] = None,
    quality_mask_url: Optional[str] = None,
    excluded_quality_classes: Sequence[int] = DEFAULT_EXCLUDED_SCL_CLASSES,
) -> tuple[np.ndarray, np.ndarray]:
    arrays, mask = _load_previews_as_arrays(band_urls, bbox=bbox)
    if quality_mask_url:
        mask = _apply_quality_mask(
            mask,
            quality_mask_url=quality_mask_url,
            bbox=bbox,
            excluded_classes=excluded_quality_classes,
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        expr_val = _evaluate_index_expression(index_expression, _index_context(arrays))
    return expr_val, mask


def _apply_quality_mask(
    base_mask: np.ndarray,
    *,
    quality_mask_url: str,
    bbox: Optional[Iterable[float]],
    excluded_classes: Sequence[int],
) -> np.ndarray:
    if Reader is None:
        raise RuntimeError("rio-tiler not installed; cannot apply SCL mask.")
    with Reader(quality_mask_url) as cog:
        if bbox:
            img = cog.part(bbox=list(bbox), bounds_crs="epsg:4326")
        else:
            img = cog.preview(max_size=1024)
    img = _resize_if_needed(img, max_size=1024)
    scl = _resize_array_nearest(img.data[0], base_mask.shape)
    scl_mask = _resize_array_nearest(img.mask, base_mask.shape)
    valid_scl = (scl_mask > 0) & ~np.isin(scl, list(excluded_classes))
    return np.where((base_mask > 0) & valid_scl, 255, 0).astype("uint8")


def render_spectral_index(
    band_urls: Dict[str, str],
    index_expression: str,
    colormap: str = "rdylgn",
    bbox: Optional[Iterable[float]] = None,
    quality_mask_url: Optional[str] = None,
    excluded_quality_classes: Sequence[int] = DEFAULT_EXCLUDED_SCL_CLASSES,
) -> str:
    """Evalua un indice espectral y devuelve un PNG base64."""
    _require_rio_tiler()
    expr_val, mask = _evaluate_masked_index(
        band_urls,
        index_expression,
        bbox=bbox,
        quality_mask_url=quality_mask_url,
        excluded_quality_classes=excluded_quality_classes,
    )
    if expr_val.ndim == 2:
        expr_val = np.expand_dims(expr_val, axis=0)

    try:
        result_img = ImageData(expr_val, mask)
    except TypeError:
        result_img = ImageData(expr_val)
    result_img.rescale(in_range=(_calculate_cumulative_min_max(result_img.data),))
    buffer = result_img.render(img_format="PNG", colormap=cmap.get(colormap))
    return _to_base64_string(buffer)


def spectral_index_statistics(
    band_urls: Dict[str, str],
    index_expression: str,
    bbox: Optional[Iterable[float]] = None,
    quality_mask_url: Optional[str] = None,
    excluded_quality_classes: Sequence[int] = DEFAULT_EXCLUDED_SCL_CLASSES,
) -> Dict[str, Any]:
    """Evalua un indice espectral y devuelve estadisticas robustas dentro del AOI."""
    values, mask_size = _masked_index_values(
        band_urls,
        index_expression,
        bbox=bbox,
        quality_mask_url=quality_mask_url,
        excluded_quality_classes=excluded_quality_classes,
    )
    if values.size == 0:
        return {"valid_pixels": 0}
    return {
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "valid_pixels": int(values.size),
        "masked_pixels": int(mask_size - values.size),
        "quality_mask_applied": bool(quality_mask_url),
    }


def _masked_index_values(
    band_urls: Dict[str, str],
    index_expression: str,
    bbox: Optional[Iterable[float]] = None,
    quality_mask_url: Optional[str] = None,
    excluded_quality_classes: Sequence[int] = DEFAULT_EXCLUDED_SCL_CLASSES,
) -> tuple["np.ndarray", int]:
    index, mask = _evaluate_masked_index(
        band_urls,
        index_expression,
        bbox=bbox,
        quality_mask_url=quality_mask_url,
        excluded_quality_classes=excluded_quality_classes,
    )
    if index.ndim == 3:
        index = index[0]
    valid = (mask > 0) & np.isfinite(index)
    values = index[valid]
    return values, int(mask.size)


def spectral_index_statistics_extended(
    band_urls: Dict[str, str],
    index_expression: str,
    bbox: Optional[Iterable[float]] = None,
    quality_mask_url: Optional[str] = None,
    excluded_quality_classes: Sequence[int] = DEFAULT_EXCLUDED_SCL_CLASSES,
) -> Dict[str, Any]:
    """Estadisticas extendidas: agrega percentiles, CV y zonas extremas."""
    values, mask_size = _masked_index_values(
        band_urls,
        index_expression,
        bbox=bbox,
        quality_mask_url=quality_mask_url,
        excluded_quality_classes=excluded_quality_classes,
    )
    if values.size == 0:
        return {"valid_pixels": 0}
    mean_val = float(np.nanmean(values))
    std_val = float(np.nanstd(values))
    return {
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "mean": mean_val,
        "std": std_val,
        "valid_pixels": int(values.size),
        "masked_pixels": int(mask_size - values.size),
        "quality_mask_applied": bool(quality_mask_url),
        "percentile_2": float(np.nanpercentile(values, 2)),
        "percentile_98": float(np.nanpercentile(values, 98)),
        "cv": abs(std_val / mean_val) if abs(mean_val) > 1e-9 else 0.0,
        "hotspots": _compute_hotspots(values, mean_val, std_val),
    }


def _compute_hotspots(values: "np.ndarray", mean: float, std: float, threshold_sigma: float = 1.5) -> Dict[str, Any]:
    """Resume la proporcion de zonas extremas, no clusters espaciales reales."""
    if std < 1e-9 or values.size < 10:
        return {"high_count": 0, "low_count": 0, "high_pct": 0.0, "low_pct": 0.0, "cluster_detected": False}
    high_mask = values > mean + threshold_sigma * std
    low_mask = values < mean - threshold_sigma * std
    high_count = int(np.sum(high_mask))
    low_count = int(np.sum(low_mask))
    total = values.size
    return {
        "high_count": high_count,
        "low_count": low_count,
        "high_pct": round(high_count / total * 100, 1),
        "low_pct": round(low_count / total * 100, 1),
        "cluster_detected": high_count > total * 0.05 or low_count > total * 0.05,
    }


def band_statistics(url: str, bbox: Optional[Iterable[float]] = None) -> Dict[str, Any]:
    """Devuelve estadisticas robustas de una banda, con el mismo contrato que los indices."""
    _require_rio_tiler()
    with Reader(url) as cog:
        if bbox:
            img = cog.part(bbox=list(bbox), bounds_crs="epsg:4326")
        else:
            img = cog.preview(max_size=1024)

    img = _resize_if_needed(img, max_size=1024)
    data = img.data[0].astype("float32")
    valid = (img.mask > 0) & np.isfinite(data)
    values = data[valid]
    if values.size == 0:
        return {"valid_pixels": 0}
    return {
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "valid_pixels": int(values.size),
        "masked_pixels": int(img.mask.size - values.size),
        "quality_mask_applied": False,
    }


def categorical_statistics(url: str, bbox: Optional[Iterable[float]] = None) -> Dict[str, Any]:
    """Cuenta clases categoricas WorldCover dentro del recorte."""
    if Reader is None:
        raise RuntimeError("rio-tiler not installed; cannot compute categorical classes.")
    with Reader(url) as cog:
        if bbox:
            img = cog.part(bbox=list(bbox), bounds_crs="epsg:4326")
        else:
            img = cog.preview(max_size=1024)

    img = _resize_if_needed(img, max_size=1024)
    data = img.data[0]
    valid = (img.mask > 0) & np.isfinite(data) & (data > 0)
    values = data[valid].astype("int32")
    if values.size == 0:
        return {"valid_pixels": 0, "masked_pixels": int(img.mask.size), "class_stats": []}

    codes, counts = np.unique(values, return_counts=True)
    total = int(values.size)
    class_stats = []
    for code, count in sorted(zip(codes, counts), key=lambda item: (-int(item[1]), int(item[0]))):
        code_int = int(code)
        pixels = int(count)
        class_stats.append(
            {
                "code": code_int,
                "label": ESA_WORLDCOVER_CLASSES.get(code_int, f"clase {code_int}"),
                "pixels": pixels,
                "percent": round((pixels / total) * 100, 1),
            }
        )
    return {
        "valid_pixels": total,
        "masked_pixels": int(img.mask.size - total),
        "class_stats": class_stats,
        "quality_mask_applied": False,
    }


def render_spectral_index_difference(
    previous_band_urls: Dict[str, str],
    current_band_urls: Dict[str, str],
    index_expression: str,
    colormap: str = "rdylgn",
    bbox: Optional[Iterable[float]] = None,
    previous_quality_mask_url: Optional[str] = None,
    current_quality_mask_url: Optional[str] = None,
    excluded_quality_classes: Sequence[int] = DEFAULT_EXCLUDED_SCL_CLASSES,
) -> str:
    """Renderiza un mapa diferencial current - previous para un indice espectral."""
    _require_rio_tiler()
    previous, previous_mask = _evaluate_masked_index(
        previous_band_urls,
        index_expression,
        bbox=bbox,
        quality_mask_url=previous_quality_mask_url,
        excluded_quality_classes=excluded_quality_classes,
    )
    current, current_mask = _evaluate_masked_index(
        current_band_urls,
        index_expression,
        bbox=bbox,
        quality_mask_url=current_quality_mask_url,
        excluded_quality_classes=excluded_quality_classes,
    )
    if previous.ndim == 3:
        previous = previous[0]
    if current.ndim == 3:
        current = current[0]
    if previous.shape != current.shape:
        current = _resize_array_nearest(current, previous.shape)
        current_mask = _resize_array_nearest(current_mask, previous_mask.shape)
    combined_mask = np.where((previous_mask > 0) & (current_mask > 0), 255, 0).astype("uint8")
    with np.errstate(invalid="ignore"):
        delta = current - previous
    valid = delta[(combined_mask > 0) & np.isfinite(delta)]
    if valid.size:
        spread = float(max(abs(np.nanpercentile(valid, 2)), abs(np.nanpercentile(valid, 98)), 0.05))
    else:
        spread = 0.2
    delta = np.expand_dims(delta.astype("float32"), axis=0)
    try:
        result_img = ImageData(delta, combined_mask)
    except TypeError:
        result_img = ImageData(delta)
    result_img.rescale(in_range=((-spread, spread),))
    buffer = result_img.render(img_format="PNG", colormap=cmap.get(colormap))
    return _to_base64_string(buffer)


def get_region_statistics(url: str, bbox: Iterable[float]) -> Dict[str, float]:
    """Calcula estadisticas basicas de una banda dentro de un bbox WGS84."""
    if Reader is None:
        raise RuntimeError("rio-tiler not installed; cannot compute STAC statistics.")
    with Reader(url) as cog:
        img = cog.part(bbox=list(bbox), bounds_crs="epsg:4326")

    data = img.data[0]
    masked_data = np.ma.array(data, mask=(img.mask == 0))
    return {
        "min": float(np.ma.min(masked_data)),
        "max": float(np.ma.max(masked_data)),
        "mean": float(np.ma.mean(masked_data)),
        "std": float(np.ma.std(masked_data)),
        "count": int(masked_data.count()),
    }
