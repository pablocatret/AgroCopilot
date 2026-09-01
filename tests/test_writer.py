import pytest

from agents.writer import WriterAgent
from libs.schemas import (
    AgentInput,
    AgentRef,
    ImageInsight,
    ImageInsights,
    LLMImageInterpretation,
    Reference,
    RemoteSensingChange,
    RemoteSensingStats,
    SceneQuality,
    StacAsset,
    StacItem,
    StacResults,
    WebResearch,
)


@pytest.mark.asyncio
async def test_writer_conversation_mode_returns_markdown_message(monkeypatch):
    monkeypatch.setattr("backend.deps.settings.DISABLE_EXTERNALS", True)
    writer = WriterAgent()

    output = await writer.run(
        AgentInput(query="Que miro antes de regar?", response_mode="conversation")
    )

    assert output.data.message_md
    assert output.data.report_md == output.data.message_md


@pytest.mark.asyncio
async def test_writer_fast_path_search_discloses_search_and_sources(monkeypatch):
    monkeypatch.setattr("backend.deps.settings.DISABLE_EXTERNALS", False)
    writer = WriterAgent()

    async def fake_search(query: str):
        return WebResearch(
            references=[
                Reference(
                    title="Precio medio del aceite",
                    url="https://example.org/precio",
                    snippet="Precio medio actualizado.",
                )
            ]
        ), [
            AgentRef(
                ref_id="writer-web-1",
                title="Precio medio del aceite",
                source="web",
                url="https://example.org/precio",
                snippet="Precio medio actualizado.",
            )
        ]

    async def fake_text(*, system: str, user: str, temperature: float = 0.2):
        assert "Busqueda puntual del writer" in user
        return "He usado una busqueda puntual y he incorporado la fuente Precio medio del aceite."

    monkeypatch.setattr(writer, "_run_fast_path_search", fake_search)
    monkeypatch.setattr(writer, "call_llm_text", fake_text)

    output = await writer.run(
        AgentInput(
            query="Precio actual del aceite de oliva",
            response_mode="conversation",
            context={"_plan": {"steps": ["writer"], "policy": {"writer_search_allowed": True}}},
        )
    )

    assert output.data.response_path == "single_agent_fast_path"
    assert output.data.search_used is True
    assert output.data.fast_path.enabled is True
    assert output.data.fast_path.search_allowed is True
    assert output.data.fast_path.search_used is True
    assert output.data.references[0].source == "web"
    assert output.data.research is not None


@pytest.mark.asyncio
async def test_writer_fast_path_marks_escalation_for_legal_or_attachment_queries(monkeypatch):
    monkeypatch.setattr("backend.deps.settings.DISABLE_EXTERNALS", True)
    writer = WriterAgent()

    output = await writer.run(
        AgentInput(
            query="Necesito revisar la normativa vigente del expediente PAC",
            response_mode="conversation",
            context={"_plan": {"steps": ["writer"], "policy": {"writer_search_allowed": True}}},
        )
    )

    assert output.data.response_path == "single_agent_fast_path"
    assert output.data.escalation_required is True
    assert output.data.escalation_reason
    assert output.data.fast_path.escalation_required is True


@pytest.mark.asyncio
async def test_writer_does_not_search_when_policy_disallows_fast_path_search(monkeypatch):
    monkeypatch.setattr("backend.deps.settings.DISABLE_EXTERNALS", False)
    writer = WriterAgent()

    async def fake_text(*, system: str, user: str, temperature: float = 0.2):
        assert "Busqueda puntual del writer:\nNo usada" in user
        return "Respuesta sin busqueda."

    monkeypatch.setattr(writer, "call_llm_text", fake_text)

    output = await writer.run(
        AgentInput(
            query="Precio actual del aceite de oliva",
            response_mode="conversation",
            context={"_plan": {"steps": ["writer"], "policy": {"writer_search_allowed": False}}},
        )
    )

    assert output.data.search_used is False
    assert output.data.research is None
    assert output.data.fast_path.search_allowed is False


@pytest.mark.asyncio
async def test_writer_assembles_conversation_evidence_bundle_for_fast_path_search(monkeypatch):
    monkeypatch.setattr("backend.deps.settings.DISABLE_EXTERNALS", False)
    writer = WriterAgent()

    async def fake_search(query: str):
        return WebResearch(
            references=[
                Reference(
                    title="Precio medio del aceite",
                    url="https://example.org/precio",
                    snippet="Precio medio actualizado.",
                )
            ]
        ), [
            AgentRef(
                ref_id="writer-web-1",
                title="Precio medio del aceite",
                source="web",
                url="https://example.org/precio",
                snippet="Precio medio actualizado.",
            )
        ]

    monkeypatch.setattr(writer, "_run_fast_path_search", fake_search)

    evidence = await writer._assemble_conversation_evidence(
        AgentInput(
            query="Precio actual del aceite de oliva",
            response_mode="conversation",
            context={"_plan": {"steps": ["writer"], "policy": {"writer_search_allowed": True}}},
        ),
        {"_plan": {"steps": ["writer"], "policy": {"writer_search_allowed": True}}},
        refs_blocks=[],
        agent_summary="Sin resumen disponible.",
        execution_summary="Sin incidencias",
        memory_summary="Sin memoria",
        context_window="Sin contexto",
    )

    assert evidence.response_path == "single_agent_fast_path"
    assert evidence.search_used is True
    assert evidence.research is not None
    assert evidence.web_refs[0].source == "web"
    prompt = writer._conversation_prompt_context(
        AgentInput(
            query="Precio actual del aceite de oliva",
            response_mode="conversation",
            context={
                "_conversation_history": [{"role": "user", "query": "Consulta previa"}],
                "_memory_reuse": {"remote_sensing": {"status": "miss", "reason": "sin artefactos"}},
            },
        ),
        evidence,
    )
    assert "Busqueda puntual del writer" in prompt
    assert "Historial conversacional" in prompt
    assert "Reutilizacion estructurada de memoria" in prompt
    assert "Contexto RS para razonar" in prompt


def test_writer_builds_rs_reasoning_context_without_forcing_template():
    rs_context = WriterAgent._build_rs_reasoning_context(
        {
            "rs_analyst": type(
                "RSOut",
                (),
                {
                    "data": ImageInsights(
                        overview="Analisis de 2 escenas satelitales; 1 cambio de severidad alta detectado.",
                        insights=[
                            ImageInsight(
                                item_id="scene-new",
                                summary="Menor vigor en borde oeste.",
                                confidence=0.74,
                            )
                        ],
                        temporal_changes=[
                            RemoteSensingChange(
                                from_item_id="scene-old",
                                to_item_id="scene-new",
                                label="Descenso de NDVI",
                                detail="NDVI medio pasa de 0.580 a 0.420.",
                                confidence=0.74,
                                metric="NDVI",
                                delta_mean=-0.16,
                                severity="alta",
                                reliable=True,
                                limitations=["Las escenas no bastan para confirmar causa sin campo."],
                                trend_context="Tendencia descendente consistente detectada.",
                            )
                        ],
                    )
                },
            )(),
        }
    )

    assert "Hallazgo principal" in rs_context
    assert "Contexto temporal" in rs_context
    assert "Regla de uso visible" not in rs_context
    assert "no robusta" not in rs_context


def test_writer_rs_reasoning_context_mentions_visual_conflict_as_caution():
    rs_context = WriterAgent._build_rs_reasoning_context(
        {
            "rs_analyst": type(
                "RSOut",
                (),
                {
                    "data": ImageInsights(
                        insights=[
                            ImageInsight(
                                item_id="scene-new",
                                summary="Escena con observacion visual auxiliar.",
                                confidence=0.66,
                                llm_interpretation=LLMImageInterpretation(
                                    item_id="scene-new",
                                    visible_patterns=["cobertura heterogenea"],
                                    health_indicators=["sin apoyo visual claro"],
                                    caveats=["la vista previa no refuerza claramente el indice"],
                                    supports_index_signal="conflicts",
                                    confidence=0.82,
                                ),
                            )
                        ]
                    )
                },
            )(),
        }
    )

    assert "Observacion visual auxiliar" in rs_context
    assert "no refuerza claramente" in rs_context


def test_writer_builds_temporal_comparison_from_stac_and_rs():
    stac = StacResults(
        items=[
            StacItem(
                id="scene-old",
                datetime="2026-02-10T10:00:00Z",
                product_label="NDVI recortado",
                index_name="NDVI",
                index_stats=RemoteSensingStats(index_name="NDVI", mean=0.58, valid_pixels=100),
                quality=SceneQuality(label="alta", cloud_cover=3),
                assets=[
                    StacAsset(
                        href="https://example.com/old.tif", thumbnail="https://example.com/old.png"
                    )
                ],
            ),
            StacItem(
                id="scene-new",
                datetime="2026-04-01T10:00:00Z",
                product_label="NDVI recortado",
                index_name="NDVI",
                index_stats=RemoteSensingStats(index_name="NDVI", mean=0.42, valid_pixels=100),
                quality=SceneQuality(label="alta", cloud_cover=5),
                assets=[
                    StacAsset(
                        href="https://example.com/new.tif", thumbnail="https://example.com/new.png"
                    )
                ],
            ),
        ]
    )
    remote_sensing = ImageInsights(
        insights=[
            ImageInsight(
                item_id="scene-old", summary="Vigor estable en referencia.", confidence=0.6
            ),
            ImageInsight(
                item_id="scene-new", summary="Menor vigor en borde oeste.", confidence=0.7
            ),
        ],
        temporal_changes=[
            RemoteSensingChange(
                from_item_id="scene-old",
                to_item_id="scene-new",
                label="Descenso de NDVI",
                detail="NDVI medio pasa de 0.580 a 0.420 (delta -0.160).",
                confidence=0.74,
                metric="NDVI",
                delta_mean=-0.16,
                severity="alta",
                reliable=True,
            )
        ],
    )

    comparison = WriterAgent._fallback_temporal_comparison(stac, remote_sensing)

    assert comparison is not None
    assert comparison.available is True
    assert comparison.previous is not None
    assert comparison.current is not None
    assert comparison.previous.item_id == "scene-old"
    assert comparison.current.item_id == "scene-new"
    assert comparison.current.preview_href == "https://example.com/new.png"
    assert comparison.label == "Cambio satelital medido"
    assert comparison.metric == "NDVI"
    assert comparison.delta_mean == -0.16
    assert comparison.previous.stats is not None
    assert any("Menor vigor" in item for item in comparison.key_changes)


def test_writer_does_not_build_temporal_comparison_for_worldcover_context():
    stac = StacResults(
        items=[
            StacItem(
                id="wc-2020",
                datetime="2020-01-01T00:00:00Z",
                collection="esa-worldcover",
                product_type="landcover",
                product_label="ESA WorldCover recortado",
                index_name="ESA_WORLDCOVER",
                index_stats=RemoteSensingStats(index_name="ESA_WORLDCOVER", valid_pixels=100),
            ),
            StacItem(
                id="wc-2021",
                datetime="2021-01-01T00:00:00Z",
                collection="esa-worldcover",
                product_type="landcover",
                product_label="ESA WorldCover recortado",
                index_name="ESA_WORLDCOVER",
                index_stats=RemoteSensingStats(index_name="ESA_WORLDCOVER", valid_pixels=100),
            ),
        ]
    )

    comparison = WriterAgent._fallback_temporal_comparison(stac, ImageInsights())

    assert comparison is None


def test_writer_prioritizes_ndvi_temporal_comparison_when_multiple_indices_exist():
    stac = StacResults(
        items=[
            StacItem(
                id="old::NDMI",
                datetime="2026-03-01T10:00:00Z",
                collection="sentinel-2-l2a",
                product_label="NDMI recortado",
                index_name="NDMI",
                index_stats=RemoteSensingStats(index_name="NDMI", mean=0.4, valid_pixels=100),
            ),
            StacItem(
                id="new::NDMI",
                datetime="2026-04-01T10:00:00Z",
                collection="sentinel-2-l2a",
                product_label="NDMI recortado",
                index_name="NDMI",
                index_stats=RemoteSensingStats(index_name="NDMI", mean=0.2, valid_pixels=100),
            ),
            StacItem(
                id="old::NDVI",
                datetime="2026-03-01T10:00:00Z",
                collection="sentinel-2-l2a",
                product_label="NDVI recortado",
                index_name="NDVI",
                index_stats=RemoteSensingStats(index_name="NDVI", mean=0.6, valid_pixels=100),
            ),
            StacItem(
                id="new::NDVI",
                datetime="2026-04-01T10:00:00Z",
                collection="sentinel-2-l2a",
                product_label="NDVI recortado",
                index_name="NDVI",
                index_stats=RemoteSensingStats(index_name="NDVI", mean=0.4, valid_pixels=100),
            ),
        ]
    )
    remote_sensing = ImageInsights(
        temporal_changes=[
            RemoteSensingChange(
                from_item_id="old::NDMI",
                to_item_id="new::NDMI",
                label="Descenso de NDMI",
                detail="NDMI bajo.",
                metric="NDMI",
                delta_mean=-0.2,
                severity="alta",
                reliable=True,
            ),
            RemoteSensingChange(
                from_item_id="old::NDVI",
                to_item_id="new::NDVI",
                label="Descenso de NDVI",
                detail="NDVI bajo.",
                metric="NDVI",
                delta_mean=-0.2,
                severity="alta",
                reliable=True,
            ),
        ]
    )

    comparison = WriterAgent._fallback_temporal_comparison(stac, remote_sensing)

    assert comparison is not None
    assert comparison.metric == "NDVI"
    assert comparison.previous.item_id == "old::NDVI"
    assert comparison.current.item_id == "new::NDVI"


def test_writer_prunes_generic_noise_and_document_duplicates():
    actions, missing, docs = WriterAgent._prune_list_noise(
        next_actions=[
            "Obtener el certificado de titularidad.",
            "Obtener el certificado de titularidad.",
            "Revisar mas documentacion.",
        ],
        missing_information=[
            "Certificado de titularidad.",
            "Firma ilegible.",
            "Cualquier otro documento relevante.",
        ],
        documents_needed=[
            "Certificado de titularidad.",
            "Cualquier otro documento relevante.",
            "Justificante de superficie declarada.",
        ],
    )
    assert actions == ["Obtener el certificado de titularidad."]
    assert missing == ["Firma ilegible."]
    assert docs == ["Certificado de titularidad.", "Justificante de superficie declarada."]


def test_writer_filters_internal_technical_details_from_visible_items():
    visible = WriterAgent._clean_visible_items(
        [
            "Confirmar en campo la zona oeste para validar el descenso de vigor.",
            "retry failed in case_manager",
            "case_manager devolvio error tecnico que impide integrar automaticamente observaciones historicas temporales.",
            "TypeError: object is not iterable",
            "stac disabled by configuration",
        ]
    )

    assert visible == ["Confirmar en campo la zona oeste para validar el descenso de vigor."]


def test_writer_filters_internal_details_from_all_visible_lists():
    source = [
        "Mantener la comprobacion de riego para confirmar la causa observada.",
        "case_manager retry failed",
    ]

    assert WriterAgent._clean_visible_items(source) == [
        "Mantener la comprobacion de riego para confirmar la causa observada."
    ]


def test_writer_filters_internal_technical_details_from_visible_markdown():
    text = WriterAgent._clean_visible_text(
        "# Lectura\nMantener seguimiento para confirmar el riesgo.\nTypeError en stac disabled."
    )

    assert "Mantener seguimiento" in text
    assert "TypeError" not in text
    assert "stac disabled" not in text


def test_writer_translates_execution_failures_to_professional_limitations():
    limitation = WriterAgent._professional_execution_limitation(
        "stac",
        "hard_error",
        {"instances": [{"message": "TypeError: stac disabled"}]},
    )

    assert limitation is not None
    assert "teledeteccion" in limitation
    assert "stac" not in limitation.lower()
    assert "typeerror" not in limitation.lower()


def test_writer_keeps_up_to_ten_evidence_linked_actions():
    source_actions = [
        f"Comprobar punto {idx} para resolver la incertidumbre {idx} observada en el caso."
        for idx in range(12)
    ]

    actions, _, _ = WriterAgent._prune_list_noise(
        next_actions=source_actions,
        missing_information=[],
        documents_needed=[],
    )

    assert len(actions) == 10
    assert all(WriterAgent._has_action_purpose(item) for item in actions)


def test_writer_open_tasks_are_professional_actions_not_raw_titles():
    action = WriterAgent._action_from_open_task("Subir informe de riego")

    assert action == "Subir informe de riego."



