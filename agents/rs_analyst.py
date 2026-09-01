from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger

from agents.base import BaseAgent
from libs.agro_references import get_threshold_context
from libs.temporal_selection import PREFERRED_MIN_GAP_DAYS, select_temporal_pair
from libs.schemas import (
    AgentInput,
    AgentRefs,
    ImageInsight,
    ImageInsights,
    LLMImageInterpretation,
    MeteoContext,
    RSAgentOutput,
    RemoteSensingChange,
    RemoteSensingFocus,
    RSAnalysisConfig,
    StacItem,
    StacResults,
    TimeSeriesPoint,
    TrendData,
)


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _fmt_num(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "sin dato"
    return f"{value:.{digits}f}"


def _quality_label(item: StacItem) -> str:
    quality = getattr(item, "quality", None)
    return getattr(quality, "label", None) or "desconocida"


def _stats_mean(item: StacItem) -> Optional[float]:
    stats = getattr(item, "index_stats", None)
    return getattr(stats, "mean", None) if stats else None


def _has_metric(item: StacItem) -> bool:
    stats = getattr(item, "index_stats", None)
    return bool(stats and stats.mean is not None and stats.valid_pixels > 0)


def _latest_observation(observations: Any) -> tuple[Optional[str], Optional[str]]:
    if not isinstance(observations, list) or not observations:
        return None, None

    def _obs_sort_key(obs: Any) -> str:
        for attr in ("date", "timestamp", "created_at"):
            val = getattr(obs, attr, None) or (obs.get(attr) if isinstance(obs, dict) else None)
            if val:
                return str(val)
        return ""

    latest = max(observations, key=_obs_sort_key, default=observations[0])
    if hasattr(latest, "parcel"):
        return latest.parcel, f"Observacion reciente en {latest.parcel}: {latest.note}"
    if isinstance(latest, dict):
        parcel = str(latest.get("parcel") or "parcela")
        note = str(latest.get("note") or "sin detalle")
        return parcel, f"Observacion reciente en {parcel}: {note}"
    return None, None


def _severity(delta: float, config: Optional[RSAnalysisConfig] = None) -> str:
    cfg = config or RSAnalysisConfig()
    magnitude = abs(delta)
    if magnitude >= cfg.severity_high:
        return "alta"
    if magnitude >= cfg.severity_medium:
        return "media"
    return "baja"


def _change_label(delta: float, metric: str, config: Optional[RSAnalysisConfig] = None) -> str:
    cfg = config or RSAnalysisConfig()
    if delta <= -cfg.severity_medium:
        return f"Descenso de {metric}"
    if delta >= cfg.severity_medium:
        return f"Crecimiento de {metric}"
    if abs(delta) > cfg.delta_trivial:
        return f"Variacion leve de {metric}"
    return f"{metric} estable"


def _is_radar_metric(item: StacItem) -> bool:
    product_type = (getattr(item, "product_type", None) or "").lower()
    metric = (getattr(item, "index_name", None) or "").upper()
    collection = (getattr(item, "collection", None) or "").lower()
    return product_type == "radar" or metric.startswith("S1_") or "sentinel-1" in collection


def _is_landcover(item: StacItem) -> bool:
    product_type = (getattr(item, "product_type", None) or "").lower()
    metric = (getattr(item, "index_name", None) or "").upper()
    collection = (getattr(item, "collection", None) or "").lower()
    return product_type == "landcover" or metric == "ESA_WORLDCOVER" or "worldcover" in collection


def _index_limitations(metric: Optional[str]) -> list[str]:
    normalized = (metric or "").upper()
    if normalized == "NDWI":
        return ["NDWI senala agua o humedad superficial visible; no confirma riego por si solo."]
    if normalized == "NDMI":
        return [
            "NDMI orienta sobre humedad de vegetacion/canopia; no diagnostica estres hidrico por si solo."
        ]
    return []


def _index_interpretation(metric: str, delta: float) -> str:
    if metric == "NDWI":
        if delta > 0.04:
            return "Senal compatible con mayor presencia relativa de agua o humedad superficial visible; contrastar encharcamiento, riego o lluvia reciente."
        if delta < -0.04:
            return "Senal compatible con menor presencia relativa de agua o humedad superficial visible; revisar disponibilidad hidrica y contexto meteorologico."
        return "La variacion de NDWI es baja; usarla como contexto de agua superficial, no como confirmacion de riego."
    if metric == "NDMI":
        if delta > 0.04:
            return "Senal compatible con mayor humedad relativa de la vegetacion/canopia; contrastar con riego, suelo y fenologia."
        if delta < -0.04:
            return "Senal compatible con menor humedad relativa de la vegetacion/canopia; conviene validar riego, suelo y demanda atmosferica."
        return "La variacion de NDMI es baja; usarla como contexto de humedad, no como diagnostico cerrado."
    if delta < -0.04:
        return "Senal compatible con menor vigor relativo; conviene validar riego, plaga, nascencia o manejo."
    if delta > 0.04:
        return "Senal compatible con mayor vigor relativo; revisar si responde a cultivo, fenologia o manejo."
    return (
        "La variacion media es baja; usar la imagen como referencia, no como diagnostico cerrado."
    )


def _landcover_limitations() -> list[str]:
    return [
        "WorldCover tiene resolucion de 10 m y puede mezclar clases en bordes de parcela.",
        "Es una clasificacion global de cobertura; no sustituye SIGPAC, catastro ni verificacion de campo.",
        "Puede estar desactualizado respecto al manejo actual de la explotacion.",
    ]


def _class_stats_summary(item: StacItem, limit: int = 3) -> str:
    stats = getattr(item, "index_stats", None)
    class_stats = list(getattr(stats, "class_stats", []) or []) if stats else []
    if not class_stats:
        return "sin distribucion de clases disponible"
    parts = [f"{entry.label} {_fmt_num(entry.percent, 1)}%" for entry in class_stats[:limit]]
    return ", ".join(parts)


def _confidence(
    previous: StacItem, current: StacItem, delta: float,
    config: Optional[RSAnalysisConfig] = None,
) -> tuple[float, list[str], bool]:
    cfg = config or RSAnalysisConfig()
    limitations: list[str] = []
    reliable = True
    score = cfg.confidence_base
    if _quality_label(previous) == "baja" or _quality_label(current) == "baja":
        limitations.append("Alguna escena tiene calidad baja; interpretar el cambio con cautela.")
        score -= cfg.quality_penalty
        reliable = False
    previous_dt = _parse_dt(previous.datetime)
    current_dt = _parse_dt(current.datetime)
    if previous_dt and current_dt:
        days = abs((current_dt - previous_dt).days)
        if days < cfg.min_temporal_gap_days:
            limitations.append(
                f"Las escenas estan separadas por solo {days} dias; el cambio puede ser ruido o condicion atmosferica."
            )
            score -= cfg.small_gap_penalty
            reliable = False
        if days > cfg.max_temporal_gap_days:
            limitations.append(
                f"Las escenas estan separadas por {days} dias; puede mezclar fenologia y manejo."
            )
            score -= cfg.large_gap_penalty
        doy_gap = abs(previous_dt.timetuple().tm_yday - current_dt.timetuple().tm_yday)
        doy_gap = min(doy_gap, 365 - doy_gap)
        if doy_gap > cfg.phenological_gap_days:
            limitations.append(
                "Las fechas no son fenologicamente equivalentes; comparar con el calendario del cultivo."
            )
            score -= cfg.phenological_gap_penalty
    if previous.collection and current.collection and previous.collection != current.collection:
        limitations.append(
            "Las escenas proceden de colecciones distintas; no se considera una comparacion homogenea."
        )
        score -= cfg.collection_mismatch_penalty
        reliable = False
    if previous.index_name and current.index_name and previous.index_name != current.index_name:
        limitations.append("Los indices comparados no coinciden.")
        score -= cfg.index_mismatch_penalty
        reliable = False
    if abs(delta) < cfg.delta_trivial:
        limitations.append(
            "La variacion media es pequena; no basta para afirmar una incidencia por si sola."
        )
    if _is_radar_metric(previous) or _is_radar_metric(current):
        limitations.append(
            "Sentinel-1 es sensible a humedad superficial, rugosidad, laboreo, orientacion y geometria de adquisicion."
        )
        limitations.append(
            "La senal radar debe tratarse como evidencia auxiliar y contrastarse con observaciones de campo."
        )
        score -= cfg.radar_penalty
    score = max(cfg.confidence_floor, min(cfg.confidence_ceiling, score))
    return score, limitations, reliable


def _crop_interpretation(metric: str, delta: float, crop_type: Optional[str], growth_stage: Optional[str]) -> Optional[str]:
    """Interpretacion contextualizada segun cultivo y fenologia."""
    if not crop_type:
        return None
    crop = crop_type.lower().strip()
    stage = (growth_stage or "").lower().strip()

    if crop in {"cafe", "café"}:
        if stage in {"cosecha", "cosech"}:
            if delta < -0.04:
                return f"Descenso de {metric} esperado en fase de cosecha; reduccion normal de biomasa verde."
            return f"Niveles de {metric} en rango esperado para fase de cosecha."
        if stage in {"floracion", "floración", "florecimiento"}:
            if delta < -0.04:
                return f"Descenso de {metric} durante floracion; revisar estres hidrico o nutricional que pueda afectar cuajado."
            return f"{metric} estable durante floracion; condicion favorable para cuajado."
        if stage in {"vegetativo", "crecimiento"}:
            if delta < -0.07:
                return f"Descenso significativo de {metric} en fase vegetativa; posible estres, plaga o limitacion nutricional."
            if delta > 0.07:
                return f"Aumento de {metric} en fase vegetativa; respuesta favorable a manejo o condiciones climaticas."

    if crop in {"maiz", "maíz", "maize"}:
        if stage in {"siembra", "nascencia"}:
            return f"Evaluacion de cobertura inicial; {metric} puede ser bajo por baja densidad de cobertura."
        if stage in {"vegetativo", "crecimiento", "llenado"}:
            if delta < -0.07:
                return f"Descenso de {metric} en fase productiva; riesgo de perdida de rendimiento si persiste."
            return f"{metric} en fase productiva dentro de parametros esperados."

    if crop in {"trigo", "cebada", "avena", "centeno"}:
        if stage in {"siembra", "nascencia", "Macollaje", "macollaje"}:
            return f"Evaluacion de cobertura inicial en cereal; {metric} puede ser bajo por baja densidad."
        if stage in {"macollaje", "hinchazon", "espigazon"}:
            if delta < -0.07:
                return f"Descenso de {metric} durante desarrollo; posible estres hidrico o helada."
            return f"{metric} en desarrollo dentro de parametros esperados."
        if stage in {"cosecha", "cosech", "madurez"}:
            if delta < -0.04:
                return f"Descenso de {metric} esperado en madurez; senal de senescencia natural."

    if crop in {"soja", "soya"}:
        if stage in {"siembra", "nascencia", "emergencia"}:
            return f"Cobertura inicial baja; {metric} puede ser bajo por germinacion reciente."
        if stage in {"vegetativo", "crecimiento", "floracion", "floración"}:
            if delta < -0.07:
                return f"Descenso de {metric} en fase productiva; revisar estres, plaga o nematodos."
            return f"{metric} estable en fase productiva."
        if stage in {"llenado", "maduracion", "maduración"}:
            if delta < -0.04:
                return f"Descenso de {metric} en llenado de granos; posible senescencia prematura."

    if crop in {"girasol", "girasoles"}:
        if stage in {"vegetativo", "crecimiento", "floracion", "floración"}:
            if delta < -0.07:
                return f"Descenso de {metric} en girasol; revisar estres hidrico o plaga."
            return f"{metric} estable en girasol."

    if crop in {"vid", "uva", "viñedo", "viñedo"}:
        if stage in {"floracion", "floración", "cuajado"}:
            if delta < -0.04:
                return f"Descenso de {metric} durante floracion/cuajado; riesgo de perdida de carga frutal."
        if stage in {"envero", "maduracion", "maduración"}:
            if delta < -0.04:
                return f"Descenso de {metric} en envero; posible senescencia foliar o estrés hídrico."

    if crop in {"pasto", "pradera", "forraje"}:
        if delta < -0.04:
            return f"Descenso de {metric} en pradera; revisar carga ganadera, sequia o degradacion."
        if delta > 0.04:
            return f"Mejora de {metric} en pradera; respuesta favorable a descanso o lluvia."

    if stage:
        if delta < -0.07:
            return f"Descenso significativo de {metric} en {crop} (fase {stage}); revisar causas."
        if delta > 0.07:
            return f"Aumento de {metric} en {crop} (fase {stage}); condicion favorable."

    return None


def _compute_trends(items: list[StacItem]) -> dict[str, TrendData]:
    """Calcula tendencia lineal (regresion) para cada grupo (collection, index_name)."""
    from collections import defaultdict
    groups: dict[tuple[str, str], list[tuple[datetime, float]]] = defaultdict(list)
    for item in items:
        if not _has_metric(item):
            continue
        dt = _parse_dt(item.datetime)
        mean = _stats_mean(item)
        if dt is not None and mean is not None:
            groups[(item.collection or "", item.index_name or "")].append((dt, mean))

    trends: dict[str, TrendData] = {}
    for (collection, index_name), points in groups.items():
        if len(points) < 3:
            continue
        points.sort(key=lambda p: p[0])
        n = len(points)
        x = [(p[0] - points[0][0]).total_seconds() / 86400.0 for p in points]
        y = [p[1] for p in points]
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        ss_xy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        ss_xx = sum((xi - x_mean) ** 2 for xi in x)
        ss_yy = sum((yi - y_mean) ** 2 for yi in y)
        if ss_xx < 1e-12 or ss_yy < 1e-12:
            continue
        slope = ss_xy / ss_xx
        r_squared = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_yy > 1e-12 else 0.0
        r_squared = min(1.0, max(0.0, r_squared))
        if abs(slope) < 1e-6:
            direction = "stable"
        elif slope > 0:
            direction = "ascending"
        else:
            direction = "descending"
        key = f"{collection}:{index_name}"
        if direction == "descending":
            interpretation = (
                f"Descenso consistente de {index_name} ({n} puntos, R²={r_squared:.2f})"
            )
        elif direction == "ascending":
            interpretation = (
                f"Ascenso consistente de {index_name} ({n} puntos, R²={r_squared:.2f})"
            )
        else:
            interpretation = f"{index_name} estable en el periodo ({n} puntos)"
        trends[key] = TrendData(
            metric=index_name,
            slope=slope,
            r_squared=r_squared,
            direction=direction,
            n_dates=n,
            date_range=f"{points[0][0].date()}/{points[-1][0].date()}",
            interpretation=interpretation,
        )
    return trends


class RSAnalystAgent(BaseAgent):
    name = "rs_analyst"
    output_model = RSAgentOutput

    async def _run(self, agent_input: AgentInput) -> RSAgentOutput:
        mission = str((agent_input.context or {}).get("mission") or "").strip()
        stac = agent_input.context.get("stac_results") if agent_input.context else None
        observations = agent_input.context.get("observations") if agent_input.context else None
        crop_type = (agent_input.context or {}).get("crop_type")
        growth_stage = (agent_input.context or {}).get("growth_stage")
        meteo_raw = (agent_input.context or {}).get("meteo")
        meteo = meteo_raw if isinstance(meteo_raw, MeteoContext) else None
        config_raw = (agent_input.context or {}).get("rs_config")
        config = config_raw if isinstance(config_raw, RSAnalysisConfig) else RSAnalysisConfig()
        user_memory = (agent_input.context or {}).get("user_memory", "")
        case_history = (agent_input.context or {}).get("case_history", [])

        if not isinstance(stac, StacResults):
            stac = StacResults(items=[])

        latest_parcel, latest_observation = _latest_observation(observations)
        ordered_items = sorted(stac.items, key=lambda item: item.datetime or "", reverse=True)

        trends = _compute_trends(ordered_items)

        insights = [
            self._scene_insight(item, latest_observation, crop_type, growth_stage)
            for item in ordered_items
        ]
        temporal_changes = self._temporal_changes(stac, ordered_items, config, crop_type, growth_stage)
        self._attach_trends_to_changes(temporal_changes, trends)
        self._apply_meteo_limitations(temporal_changes, meteo)
        focus_areas = self._focus_areas(temporal_changes, latest_parcel, latest_observation)

        await self._run_vision_analysis(ordered_items, stac, insights, latest_observation)

        overview = self._generate_overview(insights, temporal_changes, focus_areas, trends, meteo)
        if case_history and isinstance(case_history, list) and len(case_history) > 0:
            overview += f" Contexto: {len(case_history)} caso(s) previo(s) en memoria."

        time_series = sorted([
            TimeSeriesPoint(
                date=(item.datetime or "")[:10],
                mean=insight.stats.mean if insight.stats else None,
                valid_pixels=insight.stats.valid_pixels if insight.stats else 0,
                quality=insight.quality.label if insight.quality else None,
            )
            for item, insight in zip(ordered_items, insights)
            if insight.stats and insight.stats.mean is not None
        ], key=lambda p: p.date)

        data = ImageInsights(
            overview=overview,
            insights=insights,
            temporal_changes=temporal_changes,
            focus_areas=focus_areas,
            trends=trends,
            time_series=time_series,
        )
        return RSAgentOutput(
            agent=self.name,
            summary=overview,
            refs=AgentRefs(),
            data=data,
        )

    def _generate_overview(
        self,
        insights: list[ImageInsight],
        temporal_changes: list[RemoteSensingChange],
        focus_areas: list[RemoteSensingFocus],
        trends: dict[str, TrendData],
        meteo: Optional[MeteoContext],
    ) -> str:
        if not insights:
            return "Sin escenas satelitales disponibles para analizar."

        n_scenes = len(insights)
        n_with_metrics = sum(1 for i in insights if i.stats and i.stats.mean is not None)
        parts = [f"Analisis de {n_scenes} escena(s) satelital(es)"]
        if n_with_metrics > 0:
            parts.append(f" ({n_with_metrics} con metrica calculada)")

        single_scene_change = next(
            (c for c in temporal_changes if "escena unica" in (c.label or "").lower()), None
        )
        if single_scene_change:
            parts.append(f"; {single_scene_change.detail}")
        elif n_with_metrics < 2:
            parts.append("; evidencia satelital disponible, pero sin metrica temporal suficiente para concluir cambio")
        elif temporal_changes:
            high_sev = sum(1 for c in temporal_changes if c.severity == "alta")
            if high_sev:
                parts.append(f"; {high_sev} cambio(s) de severidad alta detectado(s)")
            else:
                parts.append(f"; {len(temporal_changes)} comparacion(es) temporal(es) realizada(s)")
        else:
            parts.append("; sin cambios robustos para concluir una incidencia por si sola")

        if trends:
            declining = sum(1 for t in trends.values() if t.direction == "descending" and t.r_squared > 0.5)
            if declining:
                parts.append(f"; {declining} metrica(s) en tendencia descendente consistente")

        if meteo and meteo.precipitation_irregularity_index is not None and meteo.precipitation_irregularity_index <= -1.0:
            parts.append("; condicion de sequia en el periodo")

        n_llm = sum(1 for i in insights if i.llm_interpretation)
        if n_llm > 0:
            parts.append(f"; {n_llm} imagen(es) analizada(s) con IA")

        return "".join(parts) + "."

    def _scene_insight(
        self,
        item: StacItem,
        latest_observation: Optional[str],
        crop_type: Optional[str] = None,
        growth_stage: Optional[str] = None,
    ) -> ImageInsight:
        product = item.product_label or item.index_name or item.collection or "producto satelital"
        quality = _quality_label(item)
        mean = _stats_mean(item)
        limitations: list[str] = []
        if item.quality and item.quality.reasons:
            limitations.extend(item.quality.reasons)
        limitations.extend(_index_limitations(getattr(item, "index_name", None)))

        if _is_landcover(item):
            limitations.extend(_landcover_limitations())
            summary = (
                f"{product}: Contexto de cobertura del suelo; clases principales: "
                f"{_class_stats_summary(item)}. Usarlo como contexto territorial, no como diagnostico temporal."
            )
            if latest_observation:
                summary = f"{summary} Contrastar con campo: {latest_observation}."
            return ImageInsight(
                item_id=item.id,
                summary=summary,
                confidence=0.62,
                product_label=product,
                stats=item.index_stats,
                quality=item.quality,
                limitations=limitations,
            )

        spatial_uniformity = None
        threshold = None

        if mean is None:
            summary = f"{product}: escena disponible con calidad {quality}; no hay estadistica cuantitativa del indice."
            confidence = 0.45 if quality == "baja" else 0.55
        else:
            mask_note = (
                " con mascara de calidad aplicada" if item.index_stats.quality_mask_applied else ""
            )
            summary = (
                f"{product}: media {item.index_name or 'indice'} {_fmt_num(mean)} "
                f"sobre {item.index_stats.valid_pixels} pixeles validos{mask_note}; calidad {quality}."
            )
            confidence = 0.74 if quality in {"alta", "media"} else 0.55

            if crop_type and growth_stage and item.index_name:
                threshold = get_threshold_context(crop_type, growth_stage, item.index_name, mean)

            if item.index_stats and item.index_stats.cv is not None:
                cv = item.index_stats.cv
                hotspots = item.index_stats.hotspots or {}
                low_pct = hotspots.get("low_pct", 0)
                high_pct = hotspots.get("high_pct", 0)
                if cv > 0.25:
                    spatial_uniformity = (
                        f"Campo heterogeneo (CV={cv:.2f}); {low_pct:.0f}% valores bajos, "
                        f"{high_pct:.0f}% valores altos."
                    )
                elif cv > 0.12:
                    spatial_uniformity = f"Variabilidad moderada (CV={cv:.2f})."
                else:
                    spatial_uniformity = f"Campo uniforme (CV={cv:.2f})."

        if latest_observation:
            summary = f"{summary} Contrastar con campo: {latest_observation}."

        return ImageInsight(
            item_id=item.id,
            summary=summary,
            confidence=confidence,
            product_label=product,
            stats=item.index_stats,
            quality=item.quality,
            limitations=limitations,
            threshold=threshold,
            spatial_uniformity=spatial_uniformity,
        )

    def _temporal_changes(
        self,
        stac: StacResults,
        ordered_desc: list[StacItem],
        config: RSAnalysisConfig,
        crop_type: Optional[str],
        growth_stage: Optional[str],
    ) -> list[RemoteSensingChange]:
        metric_items = [
            item
            for item in sorted(ordered_desc, key=lambda value: value.datetime or "")
            if item.index_name and _has_metric(item) and not _is_landcover(item)
        ]
        if len(metric_items) == 0:
            return []
        if len(metric_items) == 1:
            return self._single_scene_analysis(metric_items[0], config, crop_type, growth_stage)

        comparable_groups: dict[tuple[str, str], list[StacItem]] = {}
        for item in metric_items:
            key = (item.collection or "", item.index_name or "")
            comparable_groups.setdefault(key, []).append(item)

        changes: list[RemoteSensingChange] = []
        selection = getattr(stac, "temporal_selection", None)
        priority = {"NDVI": 0, "NDWI": 1, "NDMI": 2}
        groups = sorted(
            (items for items in comparable_groups.values() if len(items) >= 2),
            key=lambda items: priority.get((items[0].index_name or ""), 50),
        )

        for group_items in groups:
            pairs_to_compare: list[tuple[StacItem, StacItem]] = []
            pair = select_temporal_pair(
                group_items,
                preferred_min_gap_days=config.min_temporal_gap_days or PREFERRED_MIN_GAP_DAYS,
                selected_previous_id=getattr(selection, "previous_item_id", None),
                selected_current_id=getattr(selection, "current_item_id", None),
            )
            if pair is not None:
                pairs_to_compare.append((pair.previous, pair.current))

            if len(group_items) >= 3:
                best_delta = 0.0
                best_pair: tuple[StacItem, StacItem] | None = None
                primary_pair = pairs_to_compare[0] if pairs_to_compare else None
                for i in range(len(group_items) - 1):
                    candidate_previous = group_items[i]
                    candidate_current = group_items[i + 1]
                    if primary_pair == (candidate_previous, candidate_current):
                        continue
                    previous_dt = _parse_dt(candidate_previous.datetime)
                    current_dt = _parse_dt(candidate_current.datetime)
                    if previous_dt and current_dt:
                        gap_days = abs((current_dt - previous_dt).days)
                        if gap_days < config.min_temporal_gap_days:
                            continue
                    m1 = _stats_mean(candidate_previous)
                    m2 = _stats_mean(candidate_current)
                    if m1 is not None and m2 is not None:
                        d = abs(m2 - m1)
                        if d > best_delta:
                            best_delta = d
                            best_pair = (candidate_previous, candidate_current)
                if best_pair is not None:
                    pairs_to_compare.append(best_pair)

            for previous, current in pairs_to_compare:
                previous_mean = _stats_mean(previous)
                current_mean = _stats_mean(current)
                if previous_mean is None or current_mean is None:
                    continue

                delta = current_mean - previous_mean
                metric = previous.index_name or "indice"
                severity = _severity(delta, config)
                conf_val, limitations, reliable = _confidence(previous, current, delta, config)
                limitations.extend(_index_limitations(metric))
                label = _change_label(delta, metric, config)
                detail = (
                    f"{metric} medio pasa de {_fmt_num(previous_mean)} a {_fmt_num(current_mean)} "
                    f"(delta {_fmt_num(delta)}). "
                )
                is_radar = _is_radar_metric(previous) or _is_radar_metric(current)
                if is_radar:
                    detail += (
                        "Senal radar auxiliar: puede reflejar cambios de humedad superficial, "
                        "rugosidad, laboreo o estructura; no equivale por si sola a vigor del cultivo."
                    )
                else:
                    detail += _index_interpretation(metric, delta)

                crop_interp = _crop_interpretation(metric, delta, crop_type, growth_stage)
                if crop_interp:
                    detail = f"{detail} {crop_interp}"

                threshold = None
                if crop_type and growth_stage and metric:
                    threshold = get_threshold_context(crop_type, growth_stage, metric, current_mean)

                changes.append(
                    RemoteSensingChange(
                        from_item_id=previous.id,
                        to_item_id=current.id,
                        label=label,
                        detail=detail,
                        confidence=conf_val,
                        metric=metric,
                        collection=previous.collection,
                        group_key=f"{previous.collection or ''}:{metric}",
                        delta_mean=delta,
                        severity=severity,
                        reliable=reliable and abs(delta) >= config.delta_trivial,
                        limitations=limitations,
                        preview_href=current.change_preview_href,
                        threshold=threshold,
                    )
                )
        return changes

    def _single_scene_analysis(
        self,
        item: StacItem,
        config: RSAnalysisConfig,
        crop_type: Optional[str],
        growth_stage: Optional[str],
    ) -> list[RemoteSensingChange]:
        mean = _stats_mean(item)
        if mean is None:
            return []
        metric = item.index_name or "indice"
        quality = _quality_label(item)
        threshold = None
        if crop_type and growth_stage and metric:
            threshold = get_threshold_context(crop_type, growth_stage, metric, mean)
        threshold_text = ""
        if threshold and threshold.status:
            status_map = {
                "optimo": "en rango optimo",
                "bajo": "por debajo del rango de referencia",
                "alto": "por encima del rango de referencia",
                "critico_bajo": "en zona critica baja",
                "critico_alto": "en zona critica alta",
            }
            threshold_text = f"; valor {status_map.get(threshold.status, threshold.status)} para {crop_type} en fase {growth_stage}"
        limitations = _index_limitations(metric)
        limitations.append(
            "Escena unica: no hay comparacion temporal posible con los datos actuales."
        )
        detail = (
            f"{metric} medio actual: {_fmt_num(mean)}, calidad {quality}"
            f"{threshold_text}. "
            f"Para comparar evolucion temporal, se necesitan imagenes adicionales "
            f"del mismo periodo en otros anos."
        )
        product = item.product_label or metric
        changes = [RemoteSensingChange(
            from_item_id=item.id,
            to_item_id=item.id,
            label=f"{product} (escena unica)",
            detail=detail,
            confidence=0.55 if quality == "baja" else 0.70,
            metric=metric,
            collection=item.collection,
            group_key=f"{item.collection or ''}:{metric}",
            delta_mean=0.0,
            severity="baja",
            reliable=False,
            limitations=limitations,
            preview_href=item.change_preview_href,
            threshold=threshold,
        )]
        return changes

    def _attach_trends_to_changes(
        self,
        changes: list[RemoteSensingChange],
        trends: dict[str, TrendData],
    ) -> None:
        for change in changes:
            if not change.group_key:
                continue
            trend = trends.get(change.group_key)
            if trend is None:
                continue
            if trend.direction == "descending" and trend.r_squared > 0.5:
                change.trend_context = (
                    f"Tendencia descendente consistente detectada (R²={trend.r_squared:.2f}, "
                    f"pendiente={trend.slope:.6f}/dia) en {trend.n_dates} fechas."
                )
            elif trend.direction == "ascending" and trend.r_squared > 0.5:
                change.trend_context = (
                    f"Tendencia ascendente consistente detectada (R²={trend.r_squared:.2f}) "
                    f"en {trend.n_dates} fechas."
                )

    def _apply_meteo_limitations(
        self,
        changes: list[RemoteSensingChange],
        meteo: Optional[MeteoContext],
    ) -> None:
        if not meteo:
            return
        if meteo.precipitation_irregularity_index is not None and meteo.precipitation_irregularity_index <= -1.0:
            for change in changes:
                if change.delta_mean is not None and change.delta_mean < 0:
                    change.limitations.append(
                        "Sequia detectada en el periodo; el descenso de indice puede estar relacionado con falta de lluvia."
                    )
        if meteo.total_precip_mm is not None and meteo.period_start and meteo.period_end:
            try:
                _d1 = _parse_dt(meteo.period_start)
                _d2 = _parse_dt(meteo.period_end)
                if _d1 is None or _d2 is None:
                    return
                _n_days = max(1, (_d2 - _d1).days)
                _daily_rate = meteo.total_precip_mm / _n_days
                if _daily_rate > 15:
                    for change in changes:
                        change.limitations.append(
                            f"Exceso de lluvia en el periodo ({meteo.total_precip_mm:.0f}mm en "
                            f"{_n_days}d, {_daily_rate:.1f}mm/d); posible encharcamiento o "
                            f"nubosidad persistente."
                        )
            except (ValueError, TypeError):
                pass

    async def _run_vision_analysis(
        self,
        ordered_items: list[StacItem],
        stac: StacResults,
        insights: list[ImageInsight],
        latest_observation: Optional[str],
    ) -> None:
        if not self.external_enabled():
            return
        selection = getattr(stac, "temporal_selection", None)
        target_ids: set[str] = set()
        if selection:
            target_ids.add(getattr(selection, "current_item_id", ""))
            target_ids.add(getattr(selection, "previous_item_id", ""))
        if not target_ids and ordered_items:
            target_ids.add(ordered_items[0].id)

        analyzed = 0
        for item in ordered_items:
            if analyzed >= 2:
                break
            if item.id not in target_ids:
                continue
            thumbnail = self._extract_thumbnail(item)
            if not thumbnail:
                continue
            try:
                interp = await self._analyze_image_with_llm(item, thumbnail, latest_observation)
                for insight in insights:
                    if insight.item_id == item.id:
                        insight.llm_interpretation = interp
                        visual_note = self._visual_summary(interp)
                        if visual_note and visual_note not in insight.summary:
                            insight.summary = f"{insight.summary} Observacion visual IA: {visual_note}."
                        if interp and interp.supports_index_signal == "conflicts":
                            insight.limitations.append(
                                "La lectura visual no refuerza claramente la senal cuantitativa; conviene interpretar la escena con mas cautela."
                            )
                            insight.confidence = max(0.35, insight.confidence - 0.08)
                        elif interp and interp.supports_index_signal == "supports" and interp.confidence > 0.7:
                            insight.confidence = min(0.95, insight.confidence + 0.05)
                        break
                analyzed += 1
            except Exception as exc:
                logger.debug("rs_analyst.vision_error", item_id=item.id, error=str(exc))

    def _extract_thumbnail(self, item: StacItem) -> Optional[str]:
        scored: list[tuple[int, str]] = []
        for asset in item.assets:
            title = (asset.title or "").lower()
            mime_type = (asset.mime_type or "").lower()
            for candidate in [asset.thumbnail, asset.href]:
                if not candidate or not (
                    candidate.startswith("data:image")
                    or candidate.startswith("http://")
                    or candidate.startswith("https://")
                ):
                    continue
                lower_candidate = candidate.lower()
                if any(lower_candidate.endswith(ext) for ext in (".tif", ".tiff", ".jp2", ".nc")):
                    continue
                score = 0
                if asset.thumbnail and candidate == asset.thumbnail:
                    score += 100
                if title == "thumbnail":
                    score += 90
                elif "quicklook" in title:
                    score += 80
                elif "preview" in title or title == "rendered_preview":
                    score += 70
                elif "render" in title or "thumb" in title:
                    score += 60
                elif mime_type in {"image/png", "image/jpeg", "image/jpg", "image/webp"}:
                    score += 50
                elif mime_type.startswith("image/"):
                    score += 20
                if score > 0:
                    scored.append((score, candidate))
        if scored:
            scored.sort(key=lambda item: item[0], reverse=True)
            return scored[0][1]
        return None

    @staticmethod
    def _visual_summary(interp: Optional[LLMImageInterpretation]) -> str:
        if not interp:
            return ""
        parts: list[str] = []
        if interp.visible_patterns:
            parts.append(interp.visible_patterns[0])
        if interp.health_indicators:
            parts.append(f"salud/vigor: {interp.health_indicators[0]}")
        if interp.anomalies:
            anomalies = interp.anomalies[:2]
            parts.append(f"anomalias: {'; '.join(anomalies)}")
        if interp.supports_index_signal == "conflicts":
            parts.append("la lectura visual no refuerza el indice cuantitativo")
        elif interp.supports_index_signal == "supports":
            parts.append("la lectura visual refuerza el indice cuantitativo")
        return "; ".join(parts[:4])

    async def _analyze_image_with_llm(
        self,
        item: StacItem,
        thumbnail: str,
        latest_observation: Optional[str],
    ) -> Optional[LLMImageInterpretation]:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "visible_patterns": {"type": "array", "items": {"type": "string"}},
                "health_indicators": {"type": "array", "items": {"type": "string"}},
                "anomalies": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number"},
                "caveats": {"type": "array", "items": {"type": "string"}},
                "supports_index_signal": {
                    "type": "string",
                    "enum": ["supports", "conflicts", "unclear"],
                },
            },
            "required": [
                "visible_patterns",
                "health_indicators",
                "anomalies",
                "confidence",
                "caveats",
                "supports_index_signal",
            ],
        }
        system = (
            "Eres un analista de teledeteccion agricola experto. "
            "Analiza la imagen satelital junto con sus metadatos. "
            "Debes describir solo observaciones visuales plausibles y separarlas de cualquier inferencia causal. "
            "La salida es auxiliar: no diagnostica por si sola."
        )
        product = item.product_label or item.index_name or item.collection or "producto satelital"
        mean = _stats_mean(item)
        question = (
            f"Analiza esta imagen satelital. Producto: {product}. "
            f"Fecha: {item.datetime or 'sin fecha'}. Calidad: {_quality_label(item)}."
        )
        if mean is not None:
            question += f" Valor medio del indice: {_fmt_num(mean)}."
        if latest_observation:
            question += f" Observacion de campo reciente: {latest_observation}"

        user = (
            f"{question} "
            "Devuelve JSON con listas breves de patrones visibles, indicadores de salud/apariencia y anomalias. "
            "Incluye caveats visuales, una confianza entre 0 y 1, y si la lectura visual parece apoyar, "
            "contradecir o no aclarar la senal del indice."
        )
        payload = await self.call_llm_vision_json(
            system=system,
            images=[thumbnail],
            question=user,
            schema=schema,
            temperature=0.1,
        )
        if not payload:
            return None

        patterns = [item for item in payload.get("visible_patterns", []) if isinstance(item, str) and item.strip()]
        health = [item for item in payload.get("health_indicators", []) if isinstance(item, str) and item.strip()]
        anomalies = [item for item in payload.get("anomalies", []) if isinstance(item, str) and item.strip()]
        caveats = [item for item in payload.get("caveats", []) if isinstance(item, str) and item.strip()]
        supports_signal = str(payload.get("supports_index_signal", "unclear") or "unclear")
        raw_parts = {
            "supports_index_signal": supports_signal,
            "caveats": caveats,
            "payload": payload,
        }
        conf = payload.get("confidence", 0.65)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.65
        if caveats and len(anomalies) < 3:
            anomalies = anomalies + caveats[: 3 - len(anomalies)]
        return LLMImageInterpretation(
            item_id=item.id,
            visible_patterns=patterns[:5],
            health_indicators=health[:3],
            anomalies=anomalies[:3],
            caveats=caveats[:3],
            supports_index_signal=supports_signal if supports_signal in {"supports", "conflicts", "unclear"} else "unclear",
            confidence=min(0.9, max(0.3, conf)),
            raw_description=json.dumps(raw_parts, ensure_ascii=False)[:500],
        )

    def _focus_areas(
        self,
        changes: list[RemoteSensingChange],
        latest_parcel: Optional[str],
        latest_observation: Optional[str],
    ) -> list[RemoteSensingFocus]:
        focus_areas: list[RemoteSensingFocus] = []
        for change in changes[:2]:
            if change.reliable or change.severity in {"alta", "media"}:
                focus_areas.append(
                    RemoteSensingFocus(
                        title=f"Validar {change.label.lower()}",
                        detail=change.detail,
                        parcel=latest_parcel,
                        priority="alta" if change.severity == "alta" else "media",
                    )
                )
        if latest_observation:
            focus_areas.append(
                RemoteSensingFocus(
                    title="Contrastar observacion de campo con satelite",
                    detail=latest_observation,
                    parcel=latest_parcel,
                    priority="alta",
                )
            )
        return focus_areas[:3]
