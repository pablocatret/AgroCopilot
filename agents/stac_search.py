from __future__ import annotations

import asyncio
import functools
import json
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

try:
    import planetary_computer
    import pystac_client
except (
    ImportError,
    OSError,
    PermissionError,
):  # pragma: no cover - exercised in minimal test environments.
    planetary_computer = None
    pystac_client = None

from backend.deps import settings
from agents.base import _build_client, _message_field
from libs.costs.tracker import record_openai_chat_usage
from libs.robust_json import JsonParseError, parse_json_content
from libs.context_engineering import (
    summarize_attachments,
    summarize_case_history,
    summarize_memory_context,
    summarize_monitoring_signal,
    summarize_observations,
    summarize_temporal_focus,
)
from libs import rendering
from libs.prompts import compose_system_prompt, render_prompt
from libs.geo import geocode_bbox
from libs.temporal_selection import (
    PREFERRED_MIN_GAP_DAYS,
    TemporalWindow,
    build_temporal_windows,
    dedupe_items_by_id,
    detect_recent_override,
    detect_temporal_intent,
    expand_window,
    select_temporal_pair,
)
from agents.base import BaseAgent
from libs.schemas import (
    AgentInput,
    AgentRefs,
    RemoteSensingStats,
    SceneQuality,
    StacAgentOutput,
    StacAsset,
    StacItem,
    StacResults,
    TemporalComparisonContract,
    TemporalSelection,
    TemporalStrategySettings,
)


class STACSearchInput(BaseModel):
    """Parámetros estructurados para describir una búsqueda STAC."""

    rationale: str = Field(..., description="Motivación de la selección de colecciones y bandas.")
    collections: List[str] = Field(
        ...,
        description="IDs de colecciones STAC exactos (ej. sentinel-2-l2a, sentinel-1-rtc, esa-worldcover).",
    )
    bbox: List[float] = Field(..., description="[minLon, minLat, maxLon, maxLat] en WGS84.")
    datetime: Optional[str] = Field(
        None,
        description="Intervalo ISO 8601 con separador '/'. Opcional para colecciones estáticas (WorldCover).",
    )
    max_cloud_cover: int = Field(
        default=20, description="Porcentaje máximo de nubes (0-100). Solo aplica a ópticos."
    )
    assets_filter: Optional[List[str]] = Field(
        default=None,
        description="Lista de assets. SI el usuario pide un índice o dato técnico, lista las bandas necesarias (ej. ['vv','vh'] para Radar).",
    )
    limit: int = Field(
        default=5,
        description="Número EXACTO de imágenes. Si el usuario usa singular ('una imagen'), DEBES poner 1. Prioriza reciente.",
    )
    temporal_strategy: str = Field(
        default="auto",
        description="auto|recent_pair|monitoring_window|seasonal_baseline|annual_baseline|long_term_change",
    )
    target_gap_days: Optional[int] = Field(
        default=None,
        description="Separación temporal objetivo cuando se necesite una comparación más robusta.",
    )
    target_dates: Optional[List[str]] = Field(
        default=None,
        description=(
            "Fechas objetivo ISO 8601 (solo fecha) para cada escena de la comparación. "
            "Ejemplo: ['2020-05-15', '2025-05-15'] para comparar mayo de cada año. "
            "Cuando se usa, el sistema busca ventanas centradas en cada fecha y "
            "expande progresivamente si no hay resultados."
        ),
    )


class StacItemResult(BaseModel):
    """Representación agnóstica y tipada del resultado STAC."""

    id: str
    collection: str
    datetime: str
    bbox: List[float] = Field(default_factory=list)
    assets: Dict[str, str] = Field(default_factory=dict)
    properties: Dict[str, Any] = Field(default_factory=dict)
    cloud_cover: Optional[float] = None
    product_type: Optional[str] = None
    product_label: Optional[str] = None
    index_name: Optional[str] = None
    index_stats: Optional[Dict[str, Any]] = None
    quality: Optional[Dict[str, Any]] = None
    change_preview_href: Optional[str] = None


class STACToolkit:
    CATALOG_CHAIN = [
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        "https://earth-search.aws.element84.com/v1",
        "https://catalogue.dataspace.copernicus.eu/stac",
    ]

    def __init__(self) -> None:
        catalog_url = (
            getattr(settings, "STAC_API_URL", "") or ""
        ).strip()
        self.catalog_url = catalog_url
        self.client = None
        if planetary_computer and pystac_client:
            urls_to_try = [catalog_url] if catalog_url else []
            urls_to_try.extend(u for u in self.CATALOG_CHAIN if u != catalog_url)
            for url in urls_to_try:
                try:
                    self.client = pystac_client.Client.open(
                        url, modifier=planetary_computer.sign_inplace,
                    )
                    self.catalog_url = url
                    logger.info("stac.catalog_connected", url=url)
                    break
                except Exception as exc:
                    logger.debug("stac.catalog_failed", url=url, error=str(exc))
                    continue
            if self.client is None:
                logger.warning("stac.all_catalogs_failed")

    @staticmethod
    def _validate_bbox(bbox: List[float]) -> tuple[List[float], list[str]]:
        """Valida y corrige bbox. Devuelve (bbox_corregido, warnings)."""
        warnings: list[str] = []
        if len(bbox) != 4:
            raise ValueError(
                f"bbox debe tener 4 elementos [min_lon, min_lat, max_lon, max_lat]; recibido {len(bbox)}"
            )
        min_lon, min_lat, max_lon, max_lat = bbox
        if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
            warnings.append(f"Coordenadas lon fuera de rango: [{min_lon}, {max_lon}]")
        if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
            warnings.append(f"Coordenadas lat fuera de rango: [{min_lat}, {max_lat}]")
        if min_lon > max_lon:
            min_lon, max_lon = max_lon, min_lon
            warnings.append("min_lon > max_lon; invertidos")
        if min_lat > max_lat:
            min_lat, max_lat = max_lat, min_lat
            warnings.append("min_lat > max_lat; invertidos")
        width = max_lon - min_lon
        height = max_lat - min_lat
        area_deg2 = width * height
        if area_deg2 < 0.0001:
            warnings.append(f"bbox muy pequeño ({area_deg2:.6f}°²); expandiendo")
            min_size = 0.01
            expand_w = max(0, (min_size - width) / 2)
            expand_h = max(0, (min_size - height) / 2)
            min_lon -= expand_w
            max_lon += expand_w
            min_lat -= expand_h
            max_lat += expand_h
        elif area_deg2 > 100:
            warnings.append(f"bbox muy grande ({area_deg2:.1f}°²); puede retornar muchos resultados")
        return [min_lon, min_lat, max_lon, max_lat], warnings

    async def search_images(self, params: STACSearchInput) -> List[StacItemResult]:
        logger.info(
            "stac.search",
            collections=params.collections,
            datetime=params.datetime,
            max_cloud=params.max_cloud_cover,
            limit=params.limit,
        )
        if self.client is None:
            logger.warning("stac.client_unavailable")
            return []

        effective_limit = max(1, min(params.limit or 5, 20))

        try:
            validated_bbox, bbox_warnings = self._validate_bbox(params.bbox)
        except ValueError as exc:
            logger.warning("stac.invalid_bbox", error=str(exc))
            return []
        for w in bbox_warnings:
            logger.info("stac.bbox_warning", warning=w)

        # --- LÓGICA INTELIGENTE DE FILTROS ---
        # Solo aplicamos filtro de nubes a colecciones ópticas conocidas
        optical_collections = {
            "sentinel-2-l2a",
            "landsat-c2-l2",
            "landsat-8-c2-l2",
            "landsat-9-c2-l2",
        }
        is_optical = all(c in optical_collections for c in params.collections)

        search_params = {
            "collections": params.collections,
            "bbox": validated_bbox,
            "limit": effective_limit,
            "max_items": effective_limit,
            "sortby": [{"field": "properties.datetime", "direction": "desc"}],
        }

        if params.datetime:
            search_params["datetime"] = params.datetime

        if is_optical:
            # Filtro estándar de nubes para ópticos
            search_params["query"] = {"eo:cloud_cover": {"lt": params.max_cloud_cover}}

        # Ejecutar búsqueda
        search = self.client.search(**search_params)

        item_collection = []
        try:
            for item in search.items():
                item_collection.append(item)
                if len(item_collection) >= effective_limit:
                    break
        except Exception as exc:
            logger.warning("stac.search_with_sort_failed", error=str(exc))
            search_params.pop("sortby", None)
            search = self.client.search(**search_params)
            item_collection = list(search.items())[:effective_limit]

        results: List[StacItemResult] = []

        # --- PARALELISMO REAL PARA THUMBNAILS ---
        # 1. Recolectar tareas
        tasks = []
        for item in item_collection:
            requested = list(params.assets_filter or []) or self._default_requested_assets(item)
            # Siempre intentamos previsualizar
            task = self._auto_thumbnail(item, requested, preview_bbox=params.bbox)
            tasks.append(task)

        # 2. Ejecutar todas a la vez
        previews = await asyncio.gather(*tasks)
        await self._attach_change_preview(item_collection, previews, params.bbox)

        # 3. Ensamblar resultados
        for item, preview_bundle in zip(item_collection, previews):
            preview_infos = (
                preview_bundle if isinstance(preview_bundle, list) else [preview_bundle or {}]
            )
            for preview_info in preview_infos:
                results.append(
                    self._item_result_from_preview(
                        item,
                        preview_info or {},
                        list(params.assets_filter or []) or self._default_requested_assets(item),
                    )
                )
        return results

    def _item_result_from_preview(
        self,
        item: Any,
        preview_info: Dict[str, Any],
        requested: List[str],
    ) -> StacItemResult:
        signed_assets: Dict[str, str] = {}
        thumb_b64 = preview_info.get("thumbnail")

        # Asegurar assets de preview básicos
        for preview_key in ("thumbnail", "rendered_preview", "overview"):
            if preview_key in item.assets and preview_key not in requested:
                requested.append(preview_key)

        for key in requested:
            asset = item.assets.get(key)
            if asset:
                signed_assets[key] = asset.href

        # Inyectar nuestro thumbnail generado (si existe) o el original
        if thumb_b64:
            signed_assets["thumbnail"] = thumb_b64
        elif "thumbnail" in item.assets:
            signed_assets["thumbnail"] = item.assets["thumbnail"].href

        dt_iso = ""
        if item.datetime:
            dt_iso = item.datetime if isinstance(item.datetime, str) else item.datetime.isoformat()

        index_name = preview_info.get("index_name")
        result_id = item.id
        if preview_info.get("product_type") == "spectral_index" and index_name:
            result_id = f"{item.id}::{index_name}"

        return StacItemResult(
            id=result_id,
            collection=item.collection_id,
            datetime=dt_iso,
            bbox=item.bbox or [],
            assets=signed_assets,
            properties=item.properties or {},
            cloud_cover=self._cloud_cover(item.properties or {}),
            product_type=preview_info.get("product_type"),
            product_label=preview_info.get("product_label"),
            index_name=index_name,
            index_stats=preview_info.get("index_stats"),
            quality=self._scene_quality(item.properties or {}, preview_info),
            change_preview_href=preview_info.get("change_preview_href"),
        )

    async def get_catalog_context(self) -> str:
        # Contexto expandido para el LLM
        return (
            "AVAILABLE COLLECTIONS (Use IDs strictly):\n"
            "- 'sentinel-2-l2a': Optico 10m. (B02, B03, B04, B08, B11, SCL). Util: NDVI vigor, NDWI agua/encharcamiento, NDMI humedad/canopia, visual.\n"
            "- 'landsat-c2-l2': Optico 30m + Termico. (SR_B3, SR_B4, SR_B5, SR_B6, ST_B10). Util: NDVI, NDWI, NDMI, temperatura (LST), historico.\n"
            "- 'sentinel-1-rtc': Radar (SAR). (vv, vh). Útil: Inundaciones, Nubes, Estructura. NO tiene filtro de nubes.\n"
            "- 'esa-worldcover': Mapa de cobertura del suelo (10m). Banda: 'map'. Útil: Uso de suelo (Urbano, Cultivo, Agua).\n"
            "- 'cop-dem-glo-30': Elevación (DEM). Banda: 'data'.\n"
        )

    async def _auto_thumbnail(
        self,
        item,
        requested_assets: List[str],
        preview_bbox: Optional[List[float]] = None,
    ) -> Optional[Dict[str, Any] | List[Dict[str, Any]]]:
        """
        Lógica polimórfica para generar thumbnails según el tipo de dato.
        """
        norm_req = self._normalize_set(requested_assets)
        # Importante: mirar qué tiene el item realmente, no solo lo pedido
        avail_assets = item.assets or {}
        norm_avail = self._normalize_set(list(avail_assets.keys()))

        # Unimos lo pedido con lo disponible para decidir qué pintar
        # (Si el usuario no pidió nada específico, assets_filter es None, pero queremos ser proactivos)
        bands = norm_req | norm_avail
        item_bbox = preview_bbox or getattr(item, "bbox", None)

        # Definición de Aliases
        green = {"b03", "sr_b3", "b3", "green"}
        red = {"b04", "sr_b4", "b4", "red"}
        nir = {"b08", "sr_b5", "b5", "nir"}
        swir = {"b11", "sr_b6", "b6", "swir", "swir1"}
        scene_class = {"scl", "sceneclassification", "scene_classification"}
        thermal = {"st_b10", "b10", "sr_b10", "thermal"}
        radar = {"vv", "vh"}
        landcover = {"map", "classification"}

        # 1. Indices opticos ligeros.
        optical_products: list[Dict[str, Any]] = []
        optical_specs = [
            ("NDVI", "NDVI recortado", {"b4": red, "b8": nir}, "(b8 - b4) / (b8 + b4)", "rdylgn"),
            ("NDWI", "NDWI recortado", {"b3": green, "b8": nir}, "(b3 - b8) / (b3 + b8)", "brbg"),
            (
                "NDMI",
                "NDMI recortado",
                {"b8": nir, "b11": swir},
                "(b8 - b11) / (b8 + b11)",
                "rdylbu",
            ),
        ]
        scl = self._find_asset_by_aliases(item, scene_class)
        for index_name, product_label, required_bands, expr, colormap in optical_specs:
            if not all(norm_req & aliases for aliases in required_bands.values()):
                continue
            band_urls = {
                key: self._find_asset_by_aliases(item, aliases)
                for key, aliases in required_bands.items()
            }
            if not all(band_urls.values()):
                continue
            clean_band_urls = {key: value for key, value in band_urls.items() if value}
            optical_products.append(
                {
                    "thumbnail": await self._render_index_async(
                        clean_band_urls,
                        expr,
                        colormap,
                        bbox=item_bbox,
                        quality_mask_url=scl,
                    ),
                    "product_type": "spectral_index",
                    "product_label": product_label,
                    "index_name": index_name,
                    "index_stats": await self._index_stats_async(
                        clean_band_urls,
                        expr,
                        bbox=item_bbox,
                        quality_mask_url=scl,
                    ),
                    "quality_mask": "SCL" if scl else None,
                }
            )
        if optical_products:
            return optical_products[0] if len(optical_products) == 1 else optical_products

        # 2. Térmico (Landsat)
        if bands & thermal:
            b10 = self._find_asset_by_aliases(item, thermal)
            if b10:
                return {
                    "thumbnail": await self._render_preview_async(
                        b10, colormap="magma", bbox=item_bbox
                    ),
                    "product_type": "thermal",
                    "product_label": "Vista termica recortada",
                }

        # 3. Radar (Sentinel-1)
        # Renderizamos VV por defecto en escala de grises o magma
        if bands & radar:
            # Preferencia: VV > VH
            vv = self._find_asset_by_aliases(item, {"vv"})
            vh = self._find_asset_by_aliases(item, {"vh"})
            if vv and vh and {"vv", "vh"}.issubset(norm_req):
                band_urls = {"vv": vv, "vh": vh}
                return {
                    "thumbnail": await self._render_index_async(
                        band_urls,
                        "vh / vv",
                        "viridis",
                        bbox=item_bbox,
                    ),
                    "product_type": "radar",
                    "product_label": "Radar Sentinel-1 ratio VH/VV recortado",
                    "index_name": "S1_VH_VV_RATIO",
                    "index_stats": await self._index_stats_async(
                        band_urls,
                        "vh / vv",
                        bbox=item_bbox,
                    ),
                }
            target = vv or vh
            if target:
                metric = "S1_VV" if target == vv else "S1_VH"
                polarization = "VV" if target == vv else "VH"
                # Radar suele necesitar un rescale agresivo o logarítmico,
                # pero render_preview calcula min/max automático al 2-98% que suele funcionar bien.
                return {
                    "thumbnail": await self._render_preview_async(
                        target, colormap="inferno", bbox=item_bbox
                    ),
                    "product_type": "radar",
                    "product_label": f"Radar Sentinel-1 {polarization} recortado",
                    "index_name": metric,
                    "index_stats": await self._band_stats_async(target, bbox=item_bbox),
                }

        # 4. Land Cover (WorldCover)
        if bands & landcover:
            lc = self._find_asset_by_aliases(item, landcover)
            if lc:
                # 'tab20' es bueno para categorías discretas
                return {
                    "thumbnail": await self._render_preview_async(
                        lc, colormap="tab20", bbox=item_bbox
                    ),
                    "product_type": "landcover",
                    "product_label": "ESA WorldCover recortado",
                    "index_name": "ESA_WORLDCOVER",
                    "index_stats": await self._categorical_stats_async(lc, bbox=item_bbox),
                }

        return None

    def _cloud_cover(self, properties: Dict[str, Any]) -> Optional[float]:
        value = properties.get("eo:cloud_cover")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _scene_quality(
        self, properties: Dict[str, Any], preview_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        cloud = self._cloud_cover(properties)
        reasons: List[str] = []
        if cloud is None:
            label = "desconocida"
            reasons.append("La coleccion no informa cobertura de nubes.")
        elif cloud <= 15:
            label = "alta"
            reasons.append(f"Cobertura de nubes baja ({cloud:.1f}%).")
        elif cloud <= 35:
            label = "media"
            reasons.append(f"Cobertura de nubes moderada ({cloud:.1f}%).")
        else:
            label = "baja"
            reasons.append(f"Cobertura de nubes alta ({cloud:.1f}%).")
        index_stats = preview_info.get("index_stats") or {}
        index_name = preview_info.get("index_name")
        if (
            index_name
            and index_name != "ESA_WORLDCOVER"
            and int(index_stats.get("valid_pixels") or 0) <= 0
        ):
            label = "baja"
            reasons.append("No se pudieron calcular estadisticas del indice.")
        if preview_info.get("quality_mask") == "SCL":
            reasons.append(
                "Mascara SCL aplicada para excluir nubes, sombras, agua, nieve y nodata."
            )
        return {"label": label, "cloud_cover": cloud, "reasons": reasons}

    def _find_asset_by_aliases(self, item, aliases: set[str]) -> Optional[str]:
        for k, asset in (item.assets or {}).items():
            norm = k.lower()
            flat = norm.replace("_", "")
            if norm in aliases or flat in aliases:
                return asset.href
        return None

    def _normalize_set(self, names: List[str]) -> set[str]:
        out: set[str] = set()
        for n in names:
            norm = n.lower()
            out.add(norm)
            out.add(norm.replace("_", ""))
        return out

    def _default_requested_assets(self, item: Any) -> List[str]:
        assets = item.assets or {}
        names = list(assets.keys())
        norm_avail = self._normalize_set(names)
        red = {"b04", "sr_b4", "b4", "red"}
        nir = {"b08", "sr_b5", "b5", "nir"}
        scene_class = {"scl", "sceneclassification", "scene_classification"}
        thermal = {"st_b10", "b10", "sr_b10", "thermal"}
        radar = {"vv", "vh"}
        landcover = {"map", "classification"}

        def matching(aliases: set[str]) -> List[str]:
            return [
                name
                for name in names
                if name.lower() in aliases or name.lower().replace("_", "") in aliases
            ]

        if norm_avail & red and norm_avail & nir:
            return matching(red)[:1] + matching(nir)[:1] + matching(scene_class)[:1]
        if norm_avail & radar:
            return matching(radar) or names
        if norm_avail & landcover:
            return matching(landcover)[:1]
        if norm_avail & thermal:
            return matching(thermal)[:1]
        return names

    async def _render_preview_async(
        self, url: str, colormap: str = "viridis", bbox: Optional[List[float]] = None
    ) -> Optional[str]:
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                None,
                functools.partial(
                    rendering.render_preview, url, colormap=colormap, rescale=None, bbox=bbox
                ),
            )
        except Exception as exc:
            logger.warning("stac.preview_failed", error=str(exc))
            return None

    async def _render_index_async(
        self,
        band_urls: Dict[str, str],
        expr: str,
        colormap: str,
        bbox: Optional[List[float]] = None,
        quality_mask_url: Optional[str] = None,
    ) -> Optional[str]:
        loop = asyncio.get_running_loop()
        try:
            func = functools.partial(
                rendering.render_spectral_index,
                band_urls,
                expr,
                colormap,
                bbox=bbox,
                quality_mask_url=quality_mask_url,
            )
            return await loop.run_in_executor(None, func)
        except Exception as exc:
            logger.warning("stac.index_failed", error=str(exc))
            return None

    async def _index_stats_async(
        self,
        band_urls: Dict[str, str],
        expr: str,
        bbox: Optional[List[float]] = None,
        quality_mask_url: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        try:
            func = functools.partial(
                rendering.spectral_index_statistics_extended,
                band_urls,
                expr,
                bbox=bbox,
                quality_mask_url=quality_mask_url,
            )
            return await loop.run_in_executor(None, func)
        except Exception as exc:
            logger.warning("stac.index_stats_failed", error=str(exc))
            return None

    async def _band_stats_async(
        self,
        url: str,
        bbox: Optional[List[float]] = None,
    ) -> Optional[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        try:
            func = functools.partial(rendering.band_statistics, url, bbox=bbox)
            return await loop.run_in_executor(None, func)
        except Exception as exc:
            logger.warning("stac.band_stats_failed", error=str(exc))
            return None

    async def _categorical_stats_async(
        self,
        url: str,
        bbox: Optional[List[float]] = None,
    ) -> Optional[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        try:
            func = functools.partial(rendering.categorical_statistics, url, bbox=bbox)
            return await loop.run_in_executor(None, func)
        except Exception as exc:
            logger.warning("stac.categorical_stats_failed", error=str(exc))
            return None

    async def _render_difference_async(
        self,
        previous_band_urls: Dict[str, str],
        current_band_urls: Dict[str, str],
        bbox: Optional[List[float]] = None,
        previous_quality_mask_url: Optional[str] = None,
        current_quality_mask_url: Optional[str] = None,
        index_expression: str = "(b8 - b4) / (b8 + b4)",
        colormap: str = "rdylgn",
    ) -> Optional[str]:
        loop = asyncio.get_running_loop()
        try:
            func = functools.partial(
                rendering.render_spectral_index_difference,
                previous_band_urls,
                current_band_urls,
                index_expression,
                colormap,
                bbox=bbox,
                previous_quality_mask_url=previous_quality_mask_url,
                current_quality_mask_url=current_quality_mask_url,
            )
            return await loop.run_in_executor(None, func)
        except Exception as exc:
            logger.warning("stac.difference_failed", error=str(exc))
            return None

    async def _attach_change_preview(
        self,
        items: List[Any],
        previews: List[Optional[Dict[str, Any]]],
        bbox: List[float],
    ) -> None:
        if len(items) < 2 or not previews:
            return
        current_idx = 0
        previous_idx = len(items) - 1
        current = items[current_idx]
        previous = items[previous_idx]
        if getattr(current, "collection_id", None) != getattr(previous, "collection_id", None):
            return

        current_preview = previews[current_idx]
        target_index = None
        if isinstance(current_preview, list):
            target = next(
                (
                    preview
                    for preview in current_preview
                    if isinstance(preview, dict) and preview.get("index_name") in ("NDVI", "NDWI", "NDMI")
                ),
                None,
            )
            if target is None and current_preview:
                first = current_preview[0]
                if isinstance(first, dict):
                    target = first
            if target is not None:
                target_index = target.get("index_name")

        scene_class = {"scl", "sceneclassification", "scene_classification"}

        if target_index == "NDWI":
            band_a_aliases = {"b03", "sr_b3", "b3", "green"}
            band_b_aliases = {"b08", "sr_b5", "b5", "nir"}
            diff_expr = "(b3 - b8) / (b3 + b8)"
            diff_cmap = "brbg"
        elif target_index == "NDMI":
            band_a_aliases = {"b08", "sr_b5", "b5", "nir"}
            band_b_aliases = {"b11", "sr_b6", "b6", "swir"}
            diff_expr = "(b8 - b11) / (b8 + b11)"
            diff_cmap = "rdylbu"
        else:
            band_a_aliases = {"b04", "sr_b4", "b4", "red"}
            band_b_aliases = {"b08", "sr_b5", "b5", "nir"}
            diff_expr = "(b8 - b4) / (b8 + b4)"
            diff_cmap = "rdylgn"

        previous_a = self._find_asset_by_aliases(previous, band_a_aliases)
        previous_b = self._find_asset_by_aliases(previous, band_b_aliases)
        current_a = self._find_asset_by_aliases(current, band_a_aliases)
        current_b = self._find_asset_by_aliases(current, band_b_aliases)
        if not all([previous_a, previous_b, current_a, current_b]):
            return

        if target_index == "NDWI":
            prev_bands = {"b03": previous_a, "b08": previous_b}
            curr_bands = {"b03": current_a, "b08": current_b}
        elif target_index == "NDMI":
            prev_bands = {"b08": previous_a, "b11": previous_b}
            curr_bands = {"b08": current_a, "b11": current_b}
        else:
            prev_bands = {"b04": previous_a, "b08": previous_b}
            curr_bands = {"b04": current_a, "b08": current_b}

        diff = await self._render_difference_async(
            prev_bands,
            curr_bands,
            bbox=bbox,
            previous_quality_mask_url=self._find_asset_by_aliases(previous, scene_class),
            current_quality_mask_url=self._find_asset_by_aliases(current, scene_class),
            index_expression=diff_expr,
            colormap=diff_cmap,
        )
        if diff and isinstance(current_preview, list) and target is not None:
            target["change_preview_href"] = diff

    async def inspect_region(
        self, asset_url: str, bbox: List[float], band_name: Optional[str] = None
    ) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        func = functools.partial(rendering.get_region_statistics, asset_url, bbox)
        return await loop.run_in_executor(None, func)


class StacSearchAgent(BaseAgent):
    name = "stac"
    output_model = StacAgentOutput
    _provider_key = "LLM_PROVIDER_STAC"

    def __init__(self) -> None:
        super().__init__()
        provider = settings.resolve_provider("LLM_PROVIDER_STAC")
        self._client = (
            _build_client(provider)
            if settings.OPENAI_API_KEY and not settings.DISABLE_EXTERNALS
            else None
        )
        self.model = settings.resolve_openai_model(
            "OPENAI_MODEL_STAC",
            "OPENAI_MODEL_ORGANIZER",
        )
        self.toolkit = STACToolkit()

    async def _run(self, user_query: AgentInput) -> StacAgentOutput:
        query_str = user_query.query if hasattr(user_query, "query") else str(user_query)
        catalog_ctx = await self.toolkit.get_catalog_context()

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_satellite_images",
                    "description": "Busca imágenes en Planetary Computer.",
                    "parameters": STACSearchInput.model_json_schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "geocode_place",
                    "description": "Obtiene coordenadas de un lugar.",
                    "parameters": {
                        "type": "object",
                        "properties": {"place_name": {"type": "string"}},
                        "required": ["place_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "inspect_region",
                    "description": "Calcula estadísticas de píxeles.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "asset_url": {"type": "string"},
                            "bbox": {"type": "array", "items": {"type": "number"}},
                            "band_name": {"type": "string", "nullable": True},
                        },
                        "required": ["asset_url", "bbox"],
                    },
                },
            },
        ]

        messages = [
            {
                "role": "system",
                "content": compose_system_prompt(
                    agent_name="stac",
                    body=render_prompt("stac_system.txt"),
                    output_contract="Do not respond with free text. Use tool calls exclusively.",
                ),
            },
            {
                "role": "user",
                "content": render_prompt(
                    "stac_user.txt",
                    query=query_str,
                    mission=str((user_query.context or {}).get("mission") or ""),
                    decision_mode=user_query.decision_mode,
                    memory_summary=summarize_memory_context(
                        str(user_query.context.get("user_memory", "") or "")
                    ),
                    attachments_summary=summarize_attachments(user_query.attachments),
                    case_history_summary=summarize_case_history(
                        user_query.context.get("case_history", [])
                    ),
                    observations_summary=summarize_observations(
                        user_query.context.get("observations", [])
                    ),
                    temporal_focus=summarize_temporal_focus(
                        user_query.context.get("observations", [])
                    ),
                    monitoring_summary=summarize_monitoring_signal(
                        query_str,
                        user_query.context.get("observations", []),
                        user_query.context.get("case_history", []),
                        language=user_query.language,
                    ),
                    catalog_context=catalog_ctx,
                ),
            },
        ]

        logger.info("stac.agent.start", query=query_str)

        client = self._client
        if settings.DISABLE_EXTERNALS or client is None:
            return StacAgentOutput(
                agent=self.name,
                summary="STAC deshabilitado en modo sin externos.",
                refs=AgentRefs(),
                data=StacResults(items=[]),
            )
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            if getattr(response, "usage", None) is not None:
                record_openai_chat_usage(
                    self.model, response.usage, operation="stac.tool_selection",
                    provider=self.provider,
                )
        except Exception as exc:
            logger.exception("stac.agent.llm_error", error=str(exc))
            return StacAgentOutput(
                agent=self.name, summary="Error STAC.", refs=AgentRefs(), data=StacResults(items=[])
            )

        msg = response.choices[0].message
        tool_calls = _message_field(msg, "tool_calls")

        if not tool_calls:
            logger.warning("stac.agent.no_tool_call", content=_message_field(msg, "content", ""))
            return StacAgentOutput(
                agent=self.name,
                summary="Sin herramientas STAC.",
                refs=AgentRefs(),
                data=StacResults(items=[]),
            )

        messages.append(msg)
        should_continue, search_args, search_result = await self._handle_tool_calls(
            messages, tool_calls, user_query
        )
        if should_continue:
            return await self._continue_with_tools(
                messages,
                tools,
                user_query,
                fallback_result=search_result,
            )
        if search_args is not None:
            return await self._execute_search(search_args, user_query)

        return StacAgentOutput(
            agent=self.name,
            summary="Sin resultados STAC.",
            refs=AgentRefs(),
            data=StacResults(items=[]),
        )

    async def _continue_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        user_query: AgentInput,
        fallback_result: Optional[StacAgentOutput] = None,
        depth: int = 0,
    ) -> StacAgentOutput:
        client = self._client
        if settings.DISABLE_EXTERNALS or client is None:
            return StacAgentOutput(
                agent=self.name,
                summary="STAC deshabilitado en modo sin externos.",
                refs=AgentRefs(),
                data=StacResults(items=[]),
            )
        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
        )
        if getattr(response, "usage", None) is not None:
            record_openai_chat_usage(self.model, response.usage, operation="stac.tool_continuation",
                                     provider=self.provider)
        msg = response.choices[0].message
        tool_calls = _message_field(msg, "tool_calls")
        if not tool_calls:
            logger.warning("stac.agent.no_tool_post_geocode", content=_message_field(msg, "content", ""))
            if fallback_result is not None:
                return fallback_result
            return StacAgentOutput(
                agent=self.name,
                summary="Sin respuesta STAC.",
                refs=AgentRefs(),
                data=StacResults(items=[]),
            )
        messages.append(msg)
        should_continue, search_args, search_result = await self._handle_tool_calls(
            messages, tool_calls, user_query
        )
        next_fallback = search_result or fallback_result
        if should_continue:
            if depth >= 4:
                logger.warning("stac.tool_continuation_depth_limit", depth=depth)
                return next_fallback or StacAgentOutput(
                    agent=self.name,
                    summary="Limite de iteraciones STAC alcanzado.",
                    refs=AgentRefs(),
                    data=StacResults(items=[]),
                )
            return await self._continue_with_tools(
                messages,
                tools,
                user_query,
                fallback_result=next_fallback,
                depth=depth + 1,
            )
        if search_args is not None:
            return await self._execute_search(search_args, user_query)
        if next_fallback is not None:
            return next_fallback
        return StacAgentOutput(
            agent=self.name,
            summary="Sin resultados STAC.",
            refs=AgentRefs(),
            data=StacResults(items=[]),
        )

    async def _handle_tool_calls(
        self,
        messages: List[Dict[str, Any]],
        tool_calls: List[Any],
        user_query: AgentInput,
    ) -> tuple[bool, Optional[Dict[str, Any]], Optional[StacAgentOutput]]:
        should_continue = False
        pending_search_args: Optional[Dict[str, Any]] = None
        search_result: Optional[StacAgentOutput] = None

        for tool_call in tool_calls:
            function = _message_field(tool_call, "function", {})
            fn_name = _message_field(function, "name", "")
            try:
                args = self._parse_args(_message_field(function, "arguments"))
            except ValueError as exc:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": _message_field(tool_call, "id", ""),
                        "content": json.dumps({"error": str(exc)}),
                    }
                )
                should_continue = True
                continue

            if fn_name == "geocode_place":
                place = args.get("place_name", "").strip()
                if not place:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": _message_field(tool_call, "id", ""),
                            "content": json.dumps({"bbox": None, "status": "invalid_args"}),
                        }
                    )
                    should_continue = True
                    continue
                bbox = await geocode_bbox(place)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": _message_field(tool_call, "id", ""),
                        "content": json.dumps(
                            {"bbox": bbox, "status": "ok" if bbox else "not_found"}
                        ),
                    }
                )
                should_continue = True
                continue

            if fn_name == "inspect_region":
                stats = await self._execute_inspect(args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": _message_field(tool_call, "id", ""),
                        "content": json.dumps({"stats": stats}),
                    }
                )
                should_continue = True
                continue

            if fn_name == "search_satellite_images":
                if pending_search_args is None:
                    pending_search_args = args
                if should_continue:
                    current_result = await self._execute_search(args, user_query)
                    if search_result is None:
                        search_result = current_result
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": _message_field(tool_call, "id", ""),
                            "content": json.dumps(
                                {
                                    "summary": current_result.summary,
                                    "status": current_result.status,
                                    "data": (
                                        current_result.data.model_dump()
                                        if hasattr(current_result.data, "model_dump")
                                        else {}
                                    ),
                                }
                            ),
                        }
                    )
                continue

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": _message_field(tool_call, "id", ""),
                    "content": json.dumps(
                        {
                            "error": f"Tool '{fn_name}' not supported by STAC agent",
                            "status": "unsupported_tool",
                        }
                    ),
                }
            )
            should_continue = True

        return should_continue, pending_search_args, search_result

    async def _execute_search(
        self, args: Dict[str, Any], user_query: AgentInput
    ) -> StacAgentOutput:
        try:
            params = STACSearchInput(**args)
        except Exception as exc:
            logger.error("stac.agent.invalid_args", error=str(exc), args=args)
            return StacAgentOutput(
                agent=self.name,
                summary="Invalid STAC arguments.",
                refs=AgentRefs(),
                data=StacResults(items=[]),
            )
        temporal_mode = self._should_plan_temporal_search(params, user_query)
        force_recent_pair = detect_recent_override(user_query.query)
        preferred_gap_days = self._preferred_gap_days(params, force_recent_pair=force_recent_pair)
        target_dates = getattr(params, "target_dates", None)
        if not target_dates and temporal_mode:
            from libs.temporal_selection import detect_target_months
            target_dates = detect_target_months(user_query.query)
        query_windows: list[TemporalWindow] = []
        if temporal_mode:
            query_windows = build_temporal_windows(
                params.datetime,
                preferred_min_gap_days=preferred_gap_days,
                force_recent_pair=force_recent_pair,
                per_query_limit=max(2, min(params.limit, 3)),
                target_dates=target_dates,
            )
        if query_windows:
            results = await self._run_temporal_search_plan(params, query_windows)
            if not results:
                expanded_windows = [expand_window(w, extra_days=15) for w in query_windows]
                results = await self._run_temporal_search_plan(params, expanded_windows)
                if not results:
                    expanded_windows2 = [expand_window(w, extra_days=30) for w in query_windows]
                    results = await self._run_temporal_search_plan(params, expanded_windows2)
        else:
            results = await self.toolkit.search_images(params)

        ordered_results = sorted(
            dedupe_items_by_id(results),
            key=lambda item: item.datetime or "",
            reverse=True,
        )
        selection = None
        temporal_contract = None
        if temporal_mode:
            pair = select_temporal_pair(
                ordered_results,
                preferred_min_gap_days=preferred_gap_days,
                force_recent_pair=force_recent_pair,
                target_dates=target_dates,
            )
            if pair is not None:
                selection = TemporalSelection(
                    previous_item_id=pair.previous.id,
                    current_item_id=pair.current.id,
                    rationale=pair.rationale,
                    strategy=self._selection_strategy_label(params, len(query_windows) > 1),
                    preferred_min_gap_days=pair.preferred_min_gap_days,
                    actual_gap_days=pair.actual_gap_days,
                    used_multi_window_search=len(query_windows) > 1,
                    query_windows=[window.datetime_range for window in query_windows],
                )
                temporal_contract = TemporalComparisonContract(
                    strategy_settings=TemporalStrategySettings(
                        strategy=str(getattr(params, "temporal_strategy", "auto") or "auto").lower(),
                        target_gap_days=getattr(params, "target_gap_days", None),
                        force_same_collection=True,
                        force_same_index=True,
                        reasoning=getattr(params, "rationale", "") or "",
                    ),
                    previous_item_id=pair.previous.id,
                    current_item_id=pair.current.id,
                    previous_datetime=pair.previous.datetime,
                    current_datetime=pair.current.datetime,
                    collection=getattr(pair.current, "collection", None)
                    or getattr(pair.previous, "collection", None),
                    index_name=getattr(pair.current, "index_name", None)
                    or getattr(pair.previous, "index_name", None),
                    rationale=pair.rationale,
                    preferred_min_gap_days=pair.preferred_min_gap_days,
                    actual_gap_days=pair.actual_gap_days,
                    used_multi_window_search=len(query_windows) > 1,
                    query_windows=[window.datetime_range for window in query_windows],
                )
        if selection is not None:
            selected_ids = {
                selection.previous_item_id,
                selection.current_item_id,
            }
            exposed_results = [item for item in ordered_results if item.id in selected_ids]
            if len(exposed_results) < params.limit:
                for item in ordered_results:
                    if item.id in selected_ids:
                        continue
                    exposed_results.append(item)
                    if len(exposed_results) >= params.limit:
                        break
        else:
            exposed_results = ordered_results[: params.limit]
        data = self._results_to_schema(
            exposed_results,
            temporal_selection=selection,
            temporal_contract=temporal_contract,
        )
        return StacAgentOutput(
            agent=self.name,
            summary="Resultados STAC generados." if data.items else "Sin items STAC.",
            refs=AgentRefs(),
            data=data,
        )

    async def _run_temporal_search_plan(
        self,
        params: STACSearchInput,
        query_windows: List[TemporalWindow],
    ) -> List[StacItemResult]:
        scoped_params = [
            params.model_copy(
                update={
                    "datetime": window.datetime_range,
                    "limit": window.limit,
                }
            )
            for window in query_windows
        ]
        batches = await asyncio.gather(
            *(self.toolkit.search_images(window_params) for window_params in scoped_params)
        )
        merged: list[StacItemResult] = []
        for results in batches:
            merged.extend(results)
        return merged

    async def _execute_inspect(self, args: Dict[str, Any]) -> Dict[str, Any]:
        asset_url = args.get("asset_url")
        bbox = args.get("bbox") or []
        band_name = args.get("band_name")
        if not asset_url or not bbox:
            return {"error": "asset_url or bbox missing"}
        try:
            return await self.toolkit.inspect_region(asset_url, bbox, band_name)
        except Exception as exc:
            logger.error("stac.agent.inspect_failed", error=str(exc))
            return {"error": str(exc)}

    def _parse_args(self, raw: Any) -> Dict[str, Any]:
        if not raw:
            return {}
        try:
            return parse_json_content(raw, expected="object").value
        except JsonParseError:
            logger.warning("stac.agent.bad_args", raw=raw)
            raise ValueError(
                "Los parametros de busqueda satelital no pudieron interpretarse. "
                "Por favor, reformule su consulta."
            )

    def _results_to_schema(
        self,
        items: List[StacItemResult],
        *,
        temporal_selection: Optional[TemporalSelection] = None,
        temporal_contract: Optional[TemporalComparisonContract] = None,
    ) -> StacResults:
        converted: List[StacItem] = []
        for res in items:
            thumb = None
            # Priorizamos nuestro thumbnail generado
            if "thumbnail" in res.assets:
                thumb = res.assets["thumbnail"]
            else:
                for key in ("rendered_preview", "overview"):
                    if key in res.assets:
                        thumb = res.assets[key]
                        break

            assets: List[StacAsset] = []
            for key, href in res.assets.items():
                assets.append(
                    StacAsset(
                        href=href,
                        title=key,
                        mime_type=None,
                        thumbnail=thumb if key in {"thumbnail", "rendered_preview"} else None,
                    )
                )

            converted.append(
                StacItem(
                    id=res.id,
                    datetime=res.datetime,
                    bbox=res.bbox,
                    collection=res.collection,
                    properties=res.properties,
                    cloud_cover=res.cloud_cover,
                    product_type=res.product_type,
                    product_label=res.product_label,
                    index_name=res.index_name,
                    index_stats=(
                        RemoteSensingStats(index_name=res.index_name, **res.index_stats)
                        if res.index_stats
                        else None
                    ),
                    quality=SceneQuality(**res.quality) if res.quality else None,
                    change_preview_href=res.change_preview_href,
                    assets=assets,
                )
            )

        return StacResults(
            items=converted,
            temporal_selection=temporal_selection,
            temporal_contract=temporal_contract,
        )

    @staticmethod
    def _should_plan_temporal_search(params: STACSearchInput, user_query: AgentInput) -> bool:
        query_text = getattr(user_query, "query", "") or ""
        rationale = getattr(params, "rationale", "") or ""
        combined_text = f"{query_text}\n{rationale}"
        if str(getattr(params, "temporal_strategy", "auto") or "auto").lower() != "auto":
            return params.limit > 1
        if detect_temporal_intent(combined_text):
            return params.limit > 1
        if (params.limit or 0) > 2:
            return True
        return False

    @staticmethod
    def _preferred_gap_days(
        params: STACSearchInput, *, force_recent_pair: bool
    ) -> int:
        if force_recent_pair:
            return PREFERRED_MIN_GAP_DAYS
        strategy = str(getattr(params, "temporal_strategy", "auto") or "auto").lower()
        explicit_gap = getattr(params, "target_gap_days", None)
        if isinstance(explicit_gap, int) and explicit_gap > 0:
            return explicit_gap
        if strategy == "recent_pair":
            return PREFERRED_MIN_GAP_DAYS
        if strategy == "monitoring_window":
            return 45
        if strategy == "seasonal_baseline":
            return 120
        if strategy in {"annual_baseline", "long_term_change"}:
            return 180
        return PREFERRED_MIN_GAP_DAYS

    @staticmethod
    def _selection_strategy_label(params: STACSearchInput, used_multi_window: bool) -> str:
        prefix = "multi_window" if used_multi_window else "single_window"
        strategy = str(getattr(params, "temporal_strategy", "auto") or "auto").lower()
        return f"{prefix}:{strategy}"
