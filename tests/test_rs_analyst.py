import pytest

from agents.rs_analyst import RSAnalystAgent, _compute_trends, _confidence, _severity, _crop_interpretation, _change_label, _latest_observation
from libs.schemas import (
    AgentInput,
    AgentRefs,
    FieldObservation,
    MeteoContext,
    RemoteSensingStats,
    RSAnalysisConfig,
    SceneQuality,
    StacAgentOutput,
    StacAsset,
    StacItem,
    StacResults,
    TrendData,
)


@pytest.mark.asyncio
async def test_rs_analyst_emits_structured_temporal_changes_and_focus():
    agent = RSAnalystAgent()
    result = await agent.run(
        AgentInput(
            query="Revisa evolución reciente",
            context={
                "stac_results": StacResults(
                    items=[
                        StacItem(
                            id="scene-new",
                            datetime="2026-04-01T10:00:00Z",
                            collection="sentinel-2-l2a",
                            product_label="NDVI recortado",
                            index_name="NDVI",
                            index_stats=RemoteSensingStats(
                                index_name="NDVI",
                                mean=0.42,
                                min=0.1,
                                max=0.7,
                                std=0.12,
                                valid_pixels=100,
                                quality_mask_applied=True,
                            ),
                            quality=SceneQuality(
                                label="alta", cloud_cover=5, reasons=["Cobertura de nubes baja."]
                            ),
                            change_preview_href="data:image/png;base64,diff",
                            assets=[
                                StacAsset(
                                    href="https://example.com/new.tif",
                                    thumbnail="https://example.com/new.png",
                                )
                            ],
                        ),
                        StacItem(
                            id="scene-old",
                            datetime="2026-02-10T10:00:00Z",
                            collection="sentinel-2-l2a",
                            product_label="NDVI recortado",
                            index_name="NDVI",
                            index_stats=RemoteSensingStats(
                                index_name="NDVI",
                                mean=0.58,
                                min=0.2,
                                max=0.8,
                                std=0.10,
                                valid_pixels=100,
                                quality_mask_applied=True,
                            ),
                            quality=SceneQuality(
                                label="alta", cloud_cover=3, reasons=["Cobertura de nubes baja."]
                            ),
                            assets=[
                                StacAsset(
                                    href="https://example.com/old.tif",
                                    thumbnail="https://example.com/old.png",
                                )
                            ],
                        ),
                    ]
                ),
                "observations": [
                    FieldObservation(
                        date="2026-04-02",
                        parcel="Parcela Norte",
                        campaign="2026",
                        note="Vigor irregular en borde oeste.",
                        severity="media",
                    )
                ],
            },
        )
    )

    assert result.data.temporal_changes
    assert result.data.temporal_changes[0].from_item_id == "scene-old"
    assert result.data.temporal_changes[0].to_item_id == "scene-new"
    assert result.data.temporal_changes[0].delta_mean == pytest.approx(-0.16)
    assert result.data.temporal_changes[0].metric == "NDVI"
    assert result.data.temporal_changes[0].reliable is True
    assert result.data.temporal_changes[0].preview_href == "data:image/png;base64,diff"
    assert result.data.insights[0].stats.quality_mask_applied is True
    assert result.data.focus_areas
    assert result.data.focus_areas[0].parcel == "Parcela Norte"
    assert "escena(s) satelital(es)" in result.data.overview


@pytest.mark.asyncio
async def test_rs_analyst_does_not_claim_change_without_quantitative_metrics():
    agent = RSAnalystAgent()
    result = await agent.run(
        AgentInput(
            query="Revisa evolucion reciente",
            context={
                "stac_results": StacResults(
                    items=[
                        StacItem(id="scene-new", datetime="2026-04-01T10:00:00Z"),
                        StacItem(id="scene-old", datetime="2026-02-10T10:00:00Z"),
                    ]
                )
            },
        )
    )

    assert result.data.temporal_changes == []
    assert "sin metrica temporal suficiente" in result.data.overview.lower()


@pytest.mark.asyncio
async def test_rs_analyst_does_not_compare_different_collections():
    agent = RSAnalystAgent()
    result = await agent.run(
        AgentInput(
            query="Compara vigor",
            context={
                "stac_results": StacResults(
                    items=[
                        StacItem(
                            id="s2",
                            datetime="2026-04-01T10:00:00Z",
                            collection="sentinel-2-l2a",
                            index_name="NDVI",
                            index_stats=RemoteSensingStats(
                                index_name="NDVI", mean=0.4, valid_pixels=100
                            ),
                        ),
                        StacItem(
                            id="landsat",
                            datetime="2026-02-10T10:00:00Z",
                            collection="landsat-c2-l2",
                            index_name="NDVI",
                            index_stats=RemoteSensingStats(
                                index_name="NDVI", mean=0.6, valid_pixels=100
                            ),
                        ),
                    ]
                )
            },
        )
    )

    assert result.data.temporal_changes == []


@pytest.mark.asyncio
async def test_rs_analyst_marks_very_short_interval_as_unreliable():
    agent = RSAnalystAgent()
    result = await agent.run(
        AgentInput(
            query="Compara vigor",
            context={
                "stac_results": StacResults(
                    items=[
                        StacItem(
                            id="new",
                            datetime="2026-04-05T10:00:00Z",
                            collection="sentinel-2-l2a",
                            index_name="NDVI",
                            index_stats=RemoteSensingStats(
                                index_name="NDVI", mean=0.3, valid_pixels=100
                            ),
                        ),
                        StacItem(
                            id="old",
                            datetime="2026-04-01T10:00:00Z",
                            collection="sentinel-2-l2a",
                            index_name="NDVI",
                            index_stats=RemoteSensingStats(
                                index_name="NDVI", mean=0.6, valid_pixels=100
                            ),
                        ),
                    ]
                )
            },
        )
    )

    assert result.data.temporal_changes
    assert result.data.temporal_changes[0].reliable is False
    assert any("solo 4 dias" in item for item in result.data.temporal_changes[0].limitations)


@pytest.mark.asyncio
async def test_rs_analyst_describes_sentinel1_change_as_auxiliary_radar_signal():
    agent = RSAnalystAgent()
    result = await agent.run(
        AgentInput(
            query="Compara Sentinel-1 VV",
            context={
                "stac_results": StacResults(
                    items=[
                        StacItem(
                            id="s1-new",
                            datetime="2026-04-20T06:00:00Z",
                            collection="sentinel-1-rtc",
                            product_type="radar",
                            product_label="Radar Sentinel-1 VV recortado",
                            index_name="S1_VV",
                            index_stats=RemoteSensingStats(
                                index_name="S1_VV", mean=-12.0, valid_pixels=100
                            ),
                        ),
                        StacItem(
                            id="s1-old",
                            datetime="2026-04-01T06:00:00Z",
                            collection="sentinel-1-rtc",
                            product_type="radar",
                            product_label="Radar Sentinel-1 VV recortado",
                            index_name="S1_VV",
                            index_stats=RemoteSensingStats(
                                index_name="S1_VV", mean=-9.0, valid_pixels=100
                            ),
                        ),
                    ]
                )
            },
        )
    )

    change = result.data.temporal_changes[0]
    assert change.metric == "S1_VV"
    assert "Senal radar auxiliar" in change.detail
    assert "menor vigor" not in change.detail
    assert any("humedad superficial" in item for item in change.limitations)


@pytest.mark.asyncio
async def test_rs_analyst_describes_worldcover_as_landcover_context_without_temporal_change():
    agent = RSAnalystAgent()
    result = await agent.run(
        AgentInput(
            query="Revisa WorldCover",
            context={
                "stac_results": StacResults(
                    items=[
                        StacItem(
                            id="wc-new",
                            datetime="2021-01-01T00:00:00Z",
                            collection="esa-worldcover",
                            product_type="landcover",
                            product_label="ESA WorldCover recortado",
                            index_name="ESA_WORLDCOVER",
                            index_stats=RemoteSensingStats(
                                index_name="ESA_WORLDCOVER",
                                valid_pixels=100,
                                class_stats=[
                                    {
                                        "code": 40,
                                        "label": "cultivo",
                                        "pixels": 70,
                                        "percent": 70.0,
                                    },
                                    {
                                        "code": 10,
                                        "label": "arbolado",
                                        "pixels": 30,
                                        "percent": 30.0,
                                    },
                                ],
                            ),
                        ),
                        StacItem(
                            id="wc-old",
                            datetime="2020-01-01T00:00:00Z",
                            collection="esa-worldcover",
                            product_type="landcover",
                            product_label="ESA WorldCover recortado",
                            index_name="ESA_WORLDCOVER",
                            index_stats=RemoteSensingStats(
                                index_name="ESA_WORLDCOVER",
                                valid_pixels=100,
                                class_stats=[
                                    {
                                        "code": 40,
                                        "label": "cultivo",
                                        "pixels": 60,
                                        "percent": 60.0,
                                    }
                                ],
                            ),
                        ),
                    ]
                )
            },
        )
    )

    assert result.data.temporal_changes == []
    assert "Contexto de cobertura" in result.data.insights[0].summary
    assert "cultivo 70.0%" in result.data.insights[0].summary
    assert any("SIGPAC" in item for item in result.data.insights[0].limitations)


@pytest.mark.asyncio
async def test_rs_analyst_emits_separate_temporal_changes_for_multiple_indices():
    agent = RSAnalystAgent()
    items = []
    for suffix, dt, ndvi, ndmi in [
        ("old", "2026-03-01T10:00:00Z", 0.62, 0.38),
        ("new", "2026-04-01T10:00:00Z", 0.46, 0.25),
    ]:
        items.extend(
            [
                StacItem(
                    id=f"scene-{suffix}::NDVI",
                    datetime=dt,
                    collection="sentinel-2-l2a",
                    product_type="spectral_index",
                    product_label="NDVI recortado",
                    index_name="NDVI",
                    index_stats=RemoteSensingStats(index_name="NDVI", mean=ndvi, valid_pixels=100),
                ),
                StacItem(
                    id=f"scene-{suffix}::NDMI",
                    datetime=dt,
                    collection="sentinel-2-l2a",
                    product_type="spectral_index",
                    product_label="NDMI recortado",
                    index_name="NDMI",
                    index_stats=RemoteSensingStats(index_name="NDMI", mean=ndmi, valid_pixels=100),
                ),
            ]
        )

    result = await agent.run(
        AgentInput(
            query="Compara vigor y humedad", context={"stac_results": StacResults(items=items)}
        )
    )

    assert [change.metric for change in result.data.temporal_changes] == ["NDVI", "NDMI"]
    assert any("humedad" in item.lower() for item in result.data.temporal_changes[1].limitations)


# ── Trend tests ──────────────────────────────────────────────────────────


def test_compute_trends_returns_descending_when_data_declines():
    items = []
    for i, (dt, mean) in enumerate(
        [
            ("2026-01-01T10:00:00Z", 0.60),
            ("2026-02-01T10:00:00Z", 0.55),
            ("2026-03-01T10:00:00Z", 0.50),
            ("2026-04-01T10:00:00Z", 0.45),
        ]
    ):
        items.append(
            StacItem(
                id=f"t{i}",
                datetime=dt,
                collection="sentinel-2-l2a",
                index_name="NDVI",
                index_stats=RemoteSensingStats(index_name="NDVI", mean=mean, valid_pixels=100),
            )
        )
    trends = _compute_trends(items)
    key = "sentinel-2-l2a:NDVI"
    assert key in trends
    assert trends[key].direction == "descending"
    assert trends[key].r_squared > 0.9
    assert trends[key].n_dates == 4


def test_compute_trends_returns_empty_when_fewer_than_three_dates():
    items = [
        StacItem(
            id="a",
            datetime="2026-01-01T10:00:00Z",
            collection="sentinel-2-l2a",
            index_name="NDVI",
            index_stats=RemoteSensingStats(index_name="NDVI", mean=0.5, valid_pixels=100),
        ),
        StacItem(
            id="b",
            datetime="2026-02-01T10:00:00Z",
            collection="sentinel-2-l2a",
            index_name="NDVI",
            index_stats=RemoteSensingStats(index_name="NDVI", mean=0.6, valid_pixels=100),
        ),
    ]
    trends = _compute_trends(items)
    assert trends == {}


# ── Multi-pair consecutive comparison ────────────────────────────────────


@pytest.mark.asyncio
async def test_rs_analyst_compares_consecutive_pairs_when_three_items():
    agent = RSAnalystAgent()
    items = [
        StacItem(
            id="s3",
            datetime="2026-05-01T10:00:00Z",
            collection="sentinel-2-l2a",
            index_name="NDVI",
            index_stats=RemoteSensingStats(index_name="NDVI", mean=0.30, valid_pixels=100),
        ),
        StacItem(
            id="s2",
            datetime="2026-04-01T10:00:00Z",
            collection="sentinel-2-l2a",
            index_name="NDVI",
            index_stats=RemoteSensingStats(index_name="NDVI", mean=0.50, valid_pixels=100),
        ),
        StacItem(
            id="s1",
            datetime="2026-03-01T10:00:00Z",
            collection="sentinel-2-l2a",
            index_name="NDVI",
            index_stats=RemoteSensingStats(index_name="NDVI", mean=0.60, valid_pixels=100),
        ),
    ]
    result = await agent.run(
        AgentInput(query="Compara vigor", context={"stac_results": StacResults(items=items)})
    )
    assert len(result.data.temporal_changes) >= 1
    pairs = [(c.from_item_id, c.to_item_id) for c in result.data.temporal_changes]
    assert ("s2", "s3") in pairs


@pytest.mark.asyncio
async def test_rs_analyst_respects_min_gap_when_adding_secondary_pairs():
    agent = RSAnalystAgent()
    items = [
        StacItem(
            id="s3",
            datetime="2026-05-20T10:00:00Z",
            collection="sentinel-2-l2a",
            index_name="NDVI",
            index_stats=RemoteSensingStats(index_name="NDVI", mean=0.30, valid_pixels=100),
        ),
        StacItem(
            id="s2",
            datetime="2026-05-19T10:00:00Z",
            collection="sentinel-2-l2a",
            index_name="NDVI",
            index_stats=RemoteSensingStats(index_name="NDVI", mean=0.60, valid_pixels=100),
        ),
        StacItem(
            id="s1",
            datetime="2026-05-01T10:00:00Z",
            collection="sentinel-2-l2a",
            index_name="NDVI",
            index_stats=RemoteSensingStats(index_name="NDVI", mean=0.55, valid_pixels=100),
        ),
    ]
    result = await agent.run(
        AgentInput(query="Compara vigor", context={"stac_results": StacResults(items=items)})
    )
    pairs = {(c.from_item_id, c.to_item_id) for c in result.data.temporal_changes}
    assert ("s2", "s3") not in pairs


# ── Crop-aware interpretation ────────────────────────────────────────────


def test_crop_interpretation_returns_cafe_harvest_when_cafe_cosecha():
    interp = _crop_interpretation("NDVI", -0.08, "cafe", "cosecha")
    assert interp is not None
    assert "cosecha" in interp.lower()


def test_crop_interpretation_returns_none_when_no_crop():
    assert _crop_interpretation("NDVI", -0.08, None, None) is None


def test_crop_interpretation_returns_none_for_unknown_crop_no_stage():
    assert _crop_interpretation("NDVI", -0.08, "tomate", None) is None


@pytest.mark.asyncio
async def test_rs_analyst_includes_crop_context_in_temporal_change_detail():
    agent = RSAnalystAgent()
    items = [
        StacItem(
            id="new",
            datetime="2026-04-01T10:00:00Z",
            collection="sentinel-2-l2a",
            index_name="NDVI",
            index_stats=RemoteSensingStats(index_name="NDVI", mean=0.30, valid_pixels=100),
        ),
        StacItem(
            id="old",
            datetime="2026-03-01T10:00:00Z",
            collection="sentinel-2-l2a",
            index_name="NDVI",
            index_stats=RemoteSensingStats(index_name="NDVI", mean=0.55, valid_pixels=100),
        ),
    ]
    result = await agent.run(
        AgentInput(
            query="Revisa cafe",
            context={
                "stac_results": StacResults(items=items),
                "crop_type": "cafe",
                "growth_stage": "cosecha",
            },
        )
    )
    change = result.data.temporal_changes[0]
    assert "cosecha" in change.detail.lower()


# ── Meteo context integration ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rs_analyst_applies_drought_limitation_when_meteo_dry():
    agent = RSAnalystAgent()
    items = [
        StacItem(
            id="new",
            datetime="2026-04-01T10:00:00Z",
            collection="sentinel-2-l2a",
            index_name="NDVI",
            index_stats=RemoteSensingStats(index_name="NDVI", mean=0.30, valid_pixels=100),
        ),
        StacItem(
            id="old",
            datetime="2026-03-01T10:00:00Z",
            collection="sentinel-2-l2a",
            index_name="NDVI",
            index_stats=RemoteSensingStats(index_name="NDVI", mean=0.55, valid_pixels=100),
        ),
    ]
    meteo = MeteoContext(precipitation_irregularity_index=-1.5, total_precip_mm=10.0)
    result = await agent.run(
        AgentInput(
            query="Revisa vigor",
            context={
                "stac_results": StacResults(items=items),
                "meteo": meteo,
            },
        )
    )
    change = result.data.temporal_changes[0]
    assert any("sequia" in lim.lower() for lim in change.limitations)


@pytest.mark.asyncio
async def test_rs_analyst_applies_drought_limitation_at_threshold_minus_one():
    agent = RSAnalystAgent()
    items = [
        StacItem(
            id="new",
            datetime="2026-04-01T10:00:00Z",
            collection="sentinel-2-l2a",
            index_name="NDVI",
            index_stats=RemoteSensingStats(index_name="NDVI", mean=0.30, valid_pixels=100),
        ),
        StacItem(
            id="old",
            datetime="2026-03-01T10:00:00Z",
            collection="sentinel-2-l2a",
            index_name="NDVI",
            index_stats=RemoteSensingStats(index_name="NDVI", mean=0.55, valid_pixels=100),
        ),
    ]
    meteo = MeteoContext(precipitation_irregularity_index=-1.0, total_precip_mm=0.0)
    result = await agent.run(
        AgentInput(query="Revisa vigor", context={"stac_results": StacResults(items=items), "meteo": meteo})
    )
    assert any("sequia" in lim.lower() for lim in result.data.temporal_changes[0].limitations)


@pytest.mark.asyncio
async def test_rs_analyst_applies_drought_limitation_to_all_negative_changes():
    agent = RSAnalystAgent()
    items = [
        StacItem(
            id="new-ndvi",
            datetime="2026-04-01T10:00:00Z",
            collection="sentinel-2-l2a",
            index_name="NDVI",
            index_stats=RemoteSensingStats(index_name="NDVI", mean=0.30, valid_pixels=100),
        ),
        StacItem(
            id="old-ndvi",
            datetime="2026-03-01T10:00:00Z",
            collection="sentinel-2-l2a",
            index_name="NDVI",
            index_stats=RemoteSensingStats(index_name="NDVI", mean=0.55, valid_pixels=100),
        ),
        StacItem(
            id="new-ndmi",
            datetime="2026-04-01T10:00:00Z",
            collection="sentinel-2-l2a",
            index_name="NDMI",
            index_stats=RemoteSensingStats(index_name="NDMI", mean=0.10, valid_pixels=100),
        ),
        StacItem(
            id="old-ndmi",
            datetime="2026-03-01T10:00:00Z",
            collection="sentinel-2-l2a",
            index_name="NDMI",
            index_stats=RemoteSensingStats(index_name="NDMI", mean=0.35, valid_pixels=100),
        ),
    ]
    meteo = MeteoContext(precipitation_irregularity_index=-1.5, total_precip_mm=10.0)
    result = await agent.run(
        AgentInput(query="Revisa vigor", context={"stac_results": StacResults(items=items), "meteo": meteo})
    )
    assert result.data.temporal_changes
    assert all(
        any("sequia" in lim.lower() for lim in change.limitations)
        for change in result.data.temporal_changes
        if (change.delta_mean or 0) < 0
    )


# ── Configurable thresholds ──────────────────────────────────────────────


def test_severity_uses_config_thresholds():
    config = RSAnalysisConfig(severity_high=0.30, severity_medium=0.15)
    assert _severity(0.35, config) == "alta"
    assert _severity(0.20, config) == "media"
    assert _severity(0.05, config) == "baja"


def test_confidence_respects_config_floor_and_ceiling():
    config = RSAnalysisConfig(confidence_floor=0.5, confidence_ceiling=0.8)
    prev = StacItem(
        id="a",
        datetime="2026-01-01T10:00:00Z",
        collection="sentinel-2-l2a",
        index_name="NDVI",
        index_stats=RemoteSensingStats(index_name="NDVI", mean=0.5, valid_pixels=100),
        quality=SceneQuality(label="baja"),
    )
    curr = StacItem(
        id="b",
        datetime="2026-04-01T10:00:00Z",
        collection="sentinel-2-l2a",
        index_name="NDVI",
        index_stats=RemoteSensingStats(index_name="NDVI", mean=0.3, valid_pixels=100),
        quality=SceneQuality(label="baja"),
    )
    conf, _, _ = _confidence(prev, curr, -0.2, config)
    assert conf >= config.confidence_floor
    assert conf <= config.confidence_ceiling


# ── Overview generation ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rs_analyst_overview_includes_trend_when_present():
    agent = RSAnalystAgent()
    items = [
        StacItem(
            id=f"t{i}",
            datetime=dt,
            collection="sentinel-2-l2a",
            index_name="NDVI",
            index_stats=RemoteSensingStats(index_name="NDVI", mean=mean, valid_pixels=100),
        )
        for i, (dt, mean) in enumerate(
            [
                ("2026-01-01T10:00:00Z", 0.60),
                ("2026-02-01T10:00:00Z", 0.50),
                ("2026-03-01T10:00:00Z", 0.40),
                ("2026-04-01T10:00:00Z", 0.30),
            ]
        )
    ]
    result = await agent.run(
        AgentInput(query="Analisis tendencia", context={"stac_results": StacResults(items=items)})
    )
    assert "tendencia" in result.data.overview.lower()


# ── Vision LLM integration ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rs_analyst_skips_vision_when_external_disabled():
    agent = RSAnalystAgent()
    items = [
        StacItem(
            id="scene1",
            datetime="2026-04-01T10:00:00Z",
            collection="sentinel-2-l2a",
            index_name="NDVI",
            index_stats=RemoteSensingStats(index_name="NDVI", mean=0.45, valid_pixels=100),
            assets=[
                StacAsset(
                    href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==",
                    title="thumbnail",
                )
            ],
        )
    ]
    result = await agent.run(
        AgentInput(query="Analisis visual", context={"stac_results": StacResults(items=items)})
    )
    assert result.data.insights[0].llm_interpretation is None


class VisionEnabledRSAnalyst(RSAnalystAgent):
    def external_enabled(self) -> bool:
        return True

    async def call_llm_vision_json(self, **kwargs) -> dict:
        return {
            "visible_patterns": ["vigor heterogeneo en el borde oeste"],
            "health_indicators": ["zonas de menor vigor aparente"],
            "anomalies": ["pequena mancha clara central"],
            "confidence": 0.78,
            "caveats": ["resolucion insuficiente para atribuir causa"],
            "supports_index_signal": "supports",
        }


class VisionConflictRSAnalyst(RSAnalystAgent):
    def external_enabled(self) -> bool:
        return True

    async def call_llm_vision_json(self, **kwargs) -> dict:
        return {
            "visible_patterns": ["cobertura visual heterogenea y poco concluyente"],
            "health_indicators": ["sin patron visual claro de apoyo"],
            "anomalies": [],
            "confidence": 0.82,
            "caveats": ["la vista previa no refuerza claramente el indice"],
            "supports_index_signal": "conflicts",
        }


@pytest.mark.asyncio
async def test_rs_analyst_adds_visual_observation_to_scene_summary():
    agent = VisionEnabledRSAnalyst()
    items = [
        StacItem(
            id="scene1",
            datetime="2026-04-01T10:00:00Z",
            collection="sentinel-2-l2a",
            index_name="NDVI",
            index_stats=RemoteSensingStats(index_name="NDVI", mean=0.45, valid_pixels=100),
            assets=[
                StacAsset(
                    href="https://example.test/thumbnail.png",
                    title="quicklook",
                    mime_type="image/png",
                )
            ],
        )
    ]
    result = await agent.run(
        AgentInput(query="Analisis visual", context={"stac_results": StacResults(items=items)})
    )
    insight = result.data.insights[0]
    assert insight.llm_interpretation is not None
    assert "observacion visual ia" in insight.summary.lower()
    assert "heterogeneo" in insight.summary.lower()


@pytest.mark.asyncio
async def test_rs_analyst_visual_conflict_adds_caution_and_does_not_raise_confidence():
    agent = VisionConflictRSAnalyst()
    items = [
        StacItem(
            id="scene1",
            datetime="2026-04-01T10:00:00Z",
            collection="sentinel-2-l2a",
            index_name="NDVI",
            index_stats=RemoteSensingStats(index_name="NDVI", mean=0.45, valid_pixels=100),
            assets=[StacAsset(href="https://example.test/preview.png", title="preview", mime_type="image/png")],
        )
    ]
    result = await agent.run(
        AgentInput(query="Analisis visual", context={"stac_results": StacResults(items=items)})
    )
    insight = result.data.insights[0]
    assert insight.llm_interpretation is not None
    assert insight.llm_interpretation.supports_index_signal == "conflicts"
    assert any("no refuerza claramente" in lim.lower() for lim in insight.limitations)
    assert insight.confidence < 0.74


# ── Trends exposed in ImageInsights ──────────────────────────────────────


@pytest.mark.asyncio
async def test_rs_analyst_exposes_trends_in_image_insights():
    agent = RSAnalystAgent()
    items = [
        StacItem(
            id=f"t{i}",
            datetime=dt,
            collection="sentinel-2-l2a",
            index_name="NDVI",
            index_stats=RemoteSensingStats(index_name="NDVI", mean=mean, valid_pixels=100),
        )
        for i, (dt, mean) in enumerate(
            [
                ("2026-01-01T10:00:00Z", 0.60),
                ("2026-02-01T10:00:00Z", 0.50),
                ("2026-03-01T10:00:00Z", 0.40),
                ("2026-04-01T10:00:00Z", 0.30),
            ]
        )
    ]
    result = await agent.run(
        AgentInput(query="Analisis tendencia", context={"stac_results": StacResults(items=items)})
    )
    assert "sentinel-2-l2a:NDVI" in result.data.trends
    assert result.data.trends["sentinel-2-l2a:NDVI"].direction == "descending"


# ── No crop interpretation on individual scenes ──────────────────────────


@pytest.mark.asyncio
async def test_rs_analyst_does_not_add_crop_interp_to_scene_insights():
    agent = RSAnalystAgent()
    result = await agent.run(
        AgentInput(
            query="Revisa cafe",
            context={
                "stac_results": StacResults(
                    items=[
                        StacItem(
                            id="s1",
                            datetime="2026-04-01T10:00:00Z",
                            collection="sentinel-2-l2a",
                            index_name="NDVI",
                            index_stats=RemoteSensingStats(index_name="NDVI", mean=0.5, valid_pixels=100),
                        )
                    ]
                ),
                "crop_type": "cafe",
                "growth_stage": "cosecha",
            },
        )
    )
    assert "cosecha" not in result.data.insights[0].summary.lower()


# ── Trends match by metric exactly, not substring ────────────────────────


@pytest.mark.asyncio
async def test_rs_analyst_trends_match_by_metric_not_substring():
    agent = RSAnalystAgent()
    items = [
        StacItem(
            id=f"s2-{i}",
            datetime=dt,
            collection="sentinel-2-l2a",
            index_name="NDVI",
            index_stats=RemoteSensingStats(index_name="NDVI", mean=mean, valid_pixels=100),
        )
        for i, (dt, mean) in enumerate(
            [
                ("2026-01-01T10:00:00Z", 0.60),
                ("2026-02-01T10:00:00Z", 0.50),
                ("2026-03-01T10:00:00Z", 0.40),
            ]
        )
    ]
    items += [
        StacItem(
            id=f"ls-{i}",
            datetime=dt,
            collection="landsat-c2-l2",
            index_name="NDVI",
            index_stats=RemoteSensingStats(index_name="NDVI", mean=mean, valid_pixels=100),
        )
        for i, (dt, mean) in enumerate(
            [
                ("2026-01-01T10:00:00Z", 0.55),
                ("2026-02-01T10:00:00Z", 0.45),
                ("2026-03-01T10:00:00Z", 0.35),
            ]
        )
    ]
    result = await agent.run(
        AgentInput(query="Analisis tendencia", context={"stac_results": StacResults(items=items)})
    )
    assert len(result.data.trends) == 2
    s2_trend = result.data.trends.get("sentinel-2-l2a:NDVI")
    ls_trend = result.data.trends.get("landsat-c2-l2:NDVI")
    assert s2_trend is not None
    assert ls_trend is not None
    assert s2_trend.n_dates == 3
    assert ls_trend.n_dates == 3


@pytest.mark.asyncio
async def test_rs_analyst_applies_trend_to_matching_collection_only():
    agent = RSAnalystAgent()
    items = [
        StacItem(
            id="s2-old",
            datetime="2026-01-01T10:00:00Z",
            collection="sentinel-2-l2a",
            index_name="NDVI",
            index_stats=RemoteSensingStats(index_name="NDVI", mean=0.70, valid_pixels=100),
        ),
        StacItem(
            id="s2-mid",
            datetime="2026-02-01T10:00:00Z",
            collection="sentinel-2-l2a",
            index_name="NDVI",
            index_stats=RemoteSensingStats(index_name="NDVI", mean=0.60, valid_pixels=100),
        ),
        StacItem(
            id="s2-new",
            datetime="2026-03-01T10:00:00Z",
            collection="sentinel-2-l2a",
            index_name="NDVI",
            index_stats=RemoteSensingStats(index_name="NDVI", mean=0.50, valid_pixels=100),
        ),
        StacItem(
            id="ls-old",
            datetime="2026-01-01T10:00:00Z",
            collection="landsat-c2-l2",
            index_name="NDVI",
            index_stats=RemoteSensingStats(index_name="NDVI", mean=0.30, valid_pixels=100),
        ),
        StacItem(
            id="ls-mid",
            datetime="2026-02-01T10:00:00Z",
            collection="landsat-c2-l2",
            index_name="NDVI",
            index_stats=RemoteSensingStats(index_name="NDVI", mean=0.40, valid_pixels=100),
        ),
        StacItem(
            id="ls-new",
            datetime="2026-03-01T10:00:00Z",
            collection="landsat-c2-l2",
            index_name="NDVI",
            index_stats=RemoteSensingStats(index_name="NDVI", mean=0.50, valid_pixels=100),
        ),
    ]
    result = await agent.run(
        AgentInput(query="Analisis tendencia", context={"stac_results": StacResults(items=items)})
    )
    changes_by_collection = {change.collection: change for change in result.data.temporal_changes}
    assert "descendente" in (changes_by_collection["sentinel-2-l2a"].trend_context or "").lower()
    assert "ascendente" in (changes_by_collection["landsat-c2-l2"].trend_context or "").lower()


def test_rs_analyst_prefers_previewish_assets_over_raster_sources():
    agent = RSAnalystAgent()
    item = StacItem(
        id="scene1",
        assets=[
            StacAsset(href="https://example.test/full.tif", title="data", mime_type="image/tiff"),
            StacAsset(href="https://example.test/quicklook.png", title="quicklook", mime_type="image/png"),
            StacAsset(href="https://example.test/thumb.jpg", title="thumbnail", mime_type="image/jpeg"),
        ],
    )
    assert agent._extract_thumbnail(item) == "https://example.test/thumb.jpg"


# ── Meteo threshold is period-relative ───────────────────────────────────


@pytest.mark.asyncio
async def test_rs_analyst_meteo_excess_rain_uses_daily_rate():
    agent = RSAnalystAgent()
    items = [
        StacItem(
            id="new",
            datetime="2026-04-10T10:00:00Z",
            collection="sentinel-2-l2a",
            index_name="NDVI",
            index_stats=RemoteSensingStats(index_name="NDVI", mean=0.30, valid_pixels=100),
        ),
        StacItem(
            id="old",
            datetime="2026-04-01T10:00:00Z",
            collection="sentinel-2-l2a",
            index_name="NDVI",
            index_stats=RemoteSensingStats(index_name="NDVI", mean=0.55, valid_pixels=100),
        ),
    ]
    meteo = MeteoContext(
        total_precip_mm=200.0,
        period_start="2026-04-01",
        period_end="2026-04-10",
    )
    result = await agent.run(
        AgentInput(
            query="Revisa lluvia",
            context={
                "stac_results": StacResults(items=items),
                "meteo": meteo,
            },
        )
    )
    change = result.data.temporal_changes[0]
    assert any("Exceso de lluvia" in lim for lim in change.limitations)
    assert "200mm en 9d" in next(lim for lim in change.limitations if "Exceso" in lim)


def test_change_label_uses_config_severity_medium():
    cfg = RSAnalysisConfig(severity_medium=0.05)
    assert _change_label(-0.06, "NDVI", cfg) == "Descenso de NDVI"
    assert _change_label(0.06, "NDVI", cfg) == "Crecimiento de NDVI"
    assert _change_label(0.045, "NDVI", cfg) == "Variacion leve de NDVI"
    assert _change_label(0.01, "NDVI", cfg) == "NDVI estable"


def test_change_label_default_config():
    assert _change_label(-0.08, "NDWI") == "Descenso de NDWI"
    assert _change_label(0.08, "NDWI") == "Crecimiento de NDWI"
    assert _change_label(0.05, "NDWI") == "Variacion leve de NDWI"
    assert _change_label(0.01, "NDWI") == "NDWI estable"


def test_change_label_does_not_say_mejora():
    cfg = RSAnalysisConfig(severity_medium=0.05)
    result = _change_label(0.10, "NDWI", cfg)
    assert "Mejora" not in result
    assert "Crecimiento" in result


def test_latest_observation_returns_most_recent_by_date():
    obs_old = FieldObservation(date="2026-01-15", parcel="Parcela A", note="obs vieja")
    obs_new = FieldObservation(date="2026-04-01", parcel="Parcela B", note="obs nueva")
    obs_mid = FieldObservation(date="2026-02-20", parcel="Parcela C", note="obs media")
    parcel, note = _latest_observation([obs_old, obs_new, obs_mid])
    assert parcel == "Parcela B"
    assert "obs nueva" in note


def test_latest_observation_returns_most_recent_from_dicts():
    obs_old = {"date": "2026-01-15", "parcel": "A", "note": "old"}
    obs_new = {"date": "2026-04-01", "parcel": "B", "note": "new"}
    parcel, note = _latest_observation([obs_old, obs_new])
    assert parcel == "B"
    assert "new" in note


def test_latest_observation_returns_none_for_empty():
    assert _latest_observation([]) == (None, None)
    assert _latest_observation(None) == (None, None)


def test_confidence_uses_config_penalties():
    cfg = RSAnalysisConfig(
        small_gap_penalty=0.25,
        large_gap_penalty=0.15,
        phenological_gap_penalty=0.15,
        index_mismatch_penalty=0.30,
    )
    previous = StacItem(
        id="prev",
        datetime="2026-03-01T10:00:00Z",
        collection="sentinel-2-l2a",
        index_name="NDVI",
        quality=SceneQuality(label="alta", cloud_cover=5, reasons=[]),
        index_stats=RemoteSensingStats(index_name="NDVI", mean=0.5, valid_pixels=100),
    )
    current = StacItem(
        id="curr",
        datetime="2026-03-05T10:00:00Z",
        collection="sentinel-2-l2a",
        index_name="NDVI",
        quality=SceneQuality(label="alta", cloud_cover=5, reasons=[]),
        index_stats=RemoteSensingStats(index_name="NDVI", mean=0.4, valid_pixels=100),
    )
    score, _, _ = _confidence(previous, current, -0.1, cfg)
    expected = cfg.confidence_base - cfg.small_gap_penalty
    assert abs(score - expected) < 0.01


@pytest.mark.asyncio
async def test_rs_analyst_single_scene_analysis():
    agent = RSAnalystAgent()
    result = await agent.run(
        AgentInput(
            query="Analiza NDVI de Mora de Rubielos",
            context={
                "stac_results": StacResults(
                    items=[
                        StacItem(
                            id="single-scene",
                            datetime="2026-07-03T10:00:00Z",
                            collection="sentinel-2-l2a",
                            index_name="NDVI",
                            quality=SceneQuality(label="alta", cloud_cover=3, reasons=[]),
                            index_stats=RemoteSensingStats(
                                index_name="NDVI",
                                mean=0.253,
                                valid_pixels=201127,
                                quality_mask_applied=True,
                            ),
                        ),
                    ],
                ),
            },
        )
    )
    assert result.status == "ok"
    assert len(result.data.temporal_changes) == 1
    change = result.data.temporal_changes[0]
    assert "escena unica" in change.label.lower()
    assert "0.253" in change.detail
    assert change.reliable is False
    assert any("escena unica" in lim.lower() for lim in change.limitations)


@pytest.mark.asyncio
async def test_rs_analyst_single_scene_overview():
    agent = RSAnalystAgent()
    result = await agent.run(
        AgentInput(
            query="Analiza NDVI",
            context={
                "stac_results": StacResults(
                    items=[
                        StacItem(
                            id="single",
                            datetime="2026-07-03T10:00:00Z",
                            collection="sentinel-2-l2a",
                            index_name="NDVI",
                            quality=SceneQuality(label="alta", cloud_cover=5, reasons=[]),
                            index_stats=RemoteSensingStats(
                                index_name="NDVI",
                                mean=0.35,
                                valid_pixels=50000,
                            ),
                        ),
                    ],
                ),
            },
        )
    )
    assert "escena unica" in result.data.overview.lower() or "1 escena" in result.data.overview.lower()
    assert "0.35" in result.data.overview
