import pytest

from agents.organizer import ORGANIZER_REPLAN_SCHEMA, _agent_catalog
from agents.writer import WRITER_RESPONSE_SCHEMA_BASE
from libs.context_engineering import (
    summarize_attachments,
    summarize_execution_report,
    summarize_memory_context,
    summarize_refs,
    truncate_text,
)
from libs.prompts import compose_system_prompt, render_prompt
from libs.schemas import AgentRef, AttachmentMeta


def test_truncate_text_compacts_large_values():
    text = "a" * 200 + "b" * 200
    result = truncate_text(text, 80)
    assert len(result) <= 80
    assert "..." in result or "…" in result


def test_context_helpers_return_compact_human_readable_blocks():
    attachments = [
        AttachmentMeta(
            attachment_id="a1",
            filename="informe.pdf",
            content_type="application/pdf",
            size_bytes=123,
            summary="Resumen de prueba",
        )
    ]
    refs = [AgentRef(ref_id="r1", title="Reglamento UE", source="legal", snippet="Texto relevante")]
    execution = {
        "legal": {"final_level": "soft_error", "instances": [{"message": "timeout parcial"}]}
    }

    assert "informe.pdf" in summarize_attachments(attachments)
    assert "soft_error" in summarize_execution_report(execution)
    assert "Contexto persistente" not in summarize_memory_context("perfil demo")
    assert "Reglamento UE" in summarize_refs(refs)


def test_compose_system_prompt_adds_shared_contract():
    prompt = compose_system_prompt(
        agent_name="writer", body="Cuerpo base", output_contract="Solo JSON"
    )
    assert "multi-agent agricultural copiloting system" in prompt
    assert "Solo JSON" in prompt


def test_render_prompt_raises_on_missing_required_variable():
    with pytest.raises(Exception):
        render_prompt("writer_system.txt")


def test_current_agent_user_prompts_render():
    names = [
        "stac_user.txt",
        "document_user.txt",
        "spreadsheet_user.txt",
        "vision_user.txt",
        "case_manager_user.txt",
    ]
    for name in names:
        kwargs = {
            "query": "demo",
            "decision_mode": "case",
            "memory_summary": "perfil demo",
            "attachments_summary": "sin adjuntos",
            "case_history_summary": "sin historial",
            "observations_summary": "sin observaciones",
            "temporal_focus": "sin foco temporal",
            "monitoring_summary": "sin seguimiento activo",
        }
        if name == "stac_user.txt":
            kwargs["catalog_context"] = "catalogo"
        if name in {"document_user.txt", "spreadsheet_user.txt", "vision_user.txt"}:
            kwargs["extracted_context"] = "contexto extraido"
        if name == "case_manager_user.txt":
            kwargs.update(
                {
                    "legal_summary": "sin legal",
                    "readiness_summary": "sin readiness",
                    "rs_summary": "sin rs",
                    "rs_context": "sin contexto rs",
                    "execution_summary": "sin incidencias",
                    "evidence_ledger": "[]",
                    "deterministic_draft": "{}",
                    "case_history": "sin historial",
                }
            )
        assert render_prompt(name, **kwargs)


def test_writer_and_organizer_prompts_render_with_current_contract():
    writer_user = render_prompt(
        "writer_user.txt",
        language="es",
        query="Necesito decidir que hacer con una parcela con observaciones recientes",
        decision_mode="case",
        agent_summary="stac y rs_analyst aportaron senales utiles",
        execution_summary="Sin incidencias",
        memory_summary="Perfil de olivar ecologico",
        length_mode="STANDARD",
        rs_context="Hallazgo principal: descenso de NDVI con cautela moderada.",
        context_window="Bloque compacto de contexto",
        references_block="Sin referencias",
    )
    organizer_plan = render_prompt(
        "organizer_plan.txt",
        agent_catalog="- legal: normativa\n- stac: satelite",
        stac_hint="STAC IS enabled",
        sentinel_example='{"steps":["stac","rs_analyst","writer"]}',
    )
    organizer_replan = render_prompt(
        "organizer_replan.txt",
        agent_catalog="- legal: normativa\n- stac: satelite",
        candidates="legal, stac, writer",
    )

    assert "<query>" in writer_user
    assert "stac y rs_analyst" in writer_user
    assert "<evidence_context>" in writer_user
    assert "<remote_sensing_context>" in writer_user
    assert "<references>" in writer_user
    assert "writer_mode" in organizer_plan
    assert "Information retrieval" in organizer_plan
    assert "Interpretation and case resolution" in organizer_plan
    assert "extra_steps" in organizer_replan


def test_hardened_agent_prompts_expose_contracts():
    writer_system = render_prompt("writer_system.txt", length_mode="STANDARD")
    legal_system = render_prompt("legal_system.txt")

    assert "final writer" in writer_system
    assert "MUST NOT" in writer_system
    assert "internal agents" in writer_system
    assert "output_format" in writer_system
    assert "confidence_taxonomy" in legal_system


def test_writer_system_prompt_does_not_expose_internal_response_paths_or_thresholds():
    writer_system = render_prompt("writer_system.txt", length_mode="STANDARD")

    assert "single_agent_fast_path" not in writer_system
    assert "R² > 0.5" not in writer_system
    assert "<= -1.0" not in writer_system
    assert "precipitation_irregularity_index" not in writer_system


def test_writer_rs_context_no_longer_appends_visible_usage_rule():
    from agents.writer import WriterAgent

    rs_context = WriterAgent._build_rs_reasoning_context({})

    assert "Regla de uso visible" not in rs_context
    assert rs_context == "Sin evidencia de teledeteccion relevante."


def test_organizer_catalog_exposes_agent_tool_capabilities():
    catalog = _agent_catalog(["legal", "stac", "document_analyst", "vision_ocr", "writer"])
    assert "Tools:" in catalog
    assert "web_search" in catalog
    assert "search_satellite_images" in catalog
    assert "local PDF" in catalog
    assert "local PNG" in catalog
    assert "single-agent fast path" in catalog


def test_structured_output_schemas_cover_current_product_contract():
    writer_props = WRITER_RESPONSE_SCHEMA_BASE["properties"]
    assert "executive_summary" in writer_props
    assert "next_actions" in writer_props
    assert "missing_information" in writer_props
    assert "documents_needed" in writer_props
    assert "evidence_summary" in writer_props

    replan_props = ORGANIZER_REPLAN_SCHEMA["properties"]
    assert "extra_steps" in replan_props
    assert "writer_mode" in replan_props
    assert "stop" in replan_props
