import pytest

from agents.organizer import OrganizerAgent
from libs.schemas import AgentInput, AttachmentMeta


@pytest.mark.asyncio
async def test_plan_steps():
    plan = await OrganizerAgent().plan(AgentInput(query="evaluar finca"))
    assert "writer" in plan.steps


def test_fallback_routes_phytosanitary_and_disease_queries_to_specialists():
    organizer = OrganizerAgent()
    steps = organizer._fallback_steps(
        "Tengo una enfermedad foliar y quiero saber si puedo usar un fitosanitario"
    )
    assert "legal" in steps
    assert "case_manager" in steps
    assert steps[-1] == "writer"


def test_fallback_skips_remote_sensing_agents_when_memory_reuse_is_hit():
    organizer = OrganizerAgent()
    steps = organizer._fallback_steps(
        "Monitoriza la parcela norte",
        monitoring_hint=True,
        context={"_memory_reuse": {"remote_sensing": {"status": "hit"}}},
    )

    assert "stac" not in steps
    assert "rs_analyst" not in steps
    assert steps == ["writer"]


@pytest.mark.asyncio
async def test_simple_question_can_use_writer_fast_path_without_specialists():
    organizer = OrganizerAgent()
    plan = await organizer.plan(
        AgentInput(query="¿Cuanto cuesta hoy el azufre mojable?")
    )

    assert plan.steps == ["writer"]
    assert plan.policy.writer_search_allowed is True
    assert plan.policy.fast_path.enabled is True
    assert plan.policy.fast_path.allow_search is True
    assert plan.diagnostics.fallback_reason == "single_agent_fast_path"


@pytest.mark.asyncio
async def test_replan_uses_heuristic_fallback_when_llm_returns_no_steps_for_legal_gap(monkeypatch):
    organizer = OrganizerAgent()

    async def fake_chat_json(*args, **kwargs):
        return {"extra_steps": [], "writer_mode": "STANDARD", "rationale": "", "stop": False}

    monkeypatch.setattr(organizer, "_chat_json", fake_chat_json)

    plan = await organizer.replan(
        AgentInput(query="Necesito revisar el reglamento vigente del expediente PAC"),
        {
            "legal": {
                "summary": "fuente no disponible",
                "execution": {"final_level": "soft_error"},
            }
        },
    )

    assert plan.steps == ["legal", "writer"]
    assert plan.diagnostics.planner_source == "heuristic"
    assert plan.diagnostics.fallback_reason == "heuristic_replan_gap"


@pytest.mark.asyncio
async def test_replan_uses_attachment_extractor_fallback_when_document_processing_failed(monkeypatch):
    organizer = OrganizerAgent()

    async def fake_chat_json(*args, **kwargs):
        return {"extra_steps": [], "writer_mode": "STANDARD", "rationale": "", "stop": False}

    monkeypatch.setattr(organizer, "_chat_json", fake_chat_json)

    plan = await organizer.replan(
        AgentInput(
            query="Revisa el expediente adjunto",
            attachments=[
                AttachmentMeta(
                    attachment_id="a1",
                    filename="expediente.pdf",
                    content_type="application/pdf",
                    size_bytes=123,
                )
            ],
        ),
        {
            "document_analyst": {
                "summary": "sin datos utiles",
                "execution": {"final_level": "insufficient_data"},
            }
        },
    )

    assert plan.steps == ["document_analyst", "writer"]


@pytest.mark.asyncio
async def test_detect_ambiguity_skipped_when_externals_disabled():
    organizer = OrganizerAgent()
    plan = await organizer.plan(AgentInput(query="ayuda"))
    assert plan.clarification is None


@pytest.mark.asyncio
async def test_detect_ambiguity_skipped_for_queries_with_domain_keywords():
    organizer = OrganizerAgent()
    plan = await organizer.plan(AgentInput(query="análisis satelital de mi parcela"))
    assert plan.clarification is None


@pytest.mark.asyncio
async def test_detect_ambiguity_skipped_when_attachments_present():
    organizer = OrganizerAgent()
    plan = await organizer.plan(
        AgentInput(
            query="ayuda",
            attachments=[
                AttachmentMeta(
                    attachment_id="a1",
                    filename="doc.pdf",
                    content_type="application/pdf",
                    size_bytes=100,
                )
            ],
        )
    )
    assert plan.clarification is None


@pytest.mark.asyncio
async def test_detect_ambiguity_skipped_when_context_has_observations():
    organizer = OrganizerAgent()
    plan = await organizer.plan(
        AgentInput(query="ayuda", context={"observations": [{"date": "2025-01-01", "parcel": "P1", "note": "test", "severity": "baja"}]})
    )
    assert plan.clarification is None


@pytest.mark.asyncio
async def test_fallback_clarification_for_parcela_query():
    organizer = OrganizerAgent()
    clarification = organizer._fallback_clarification("mi parcela")
    assert "parcela" in clarification.question.lower()
    assert len(clarification.options) == 3
    keys = {opt.key for opt in clarification.options}
    assert keys == {"satellite", "legal", "general"}


@pytest.mark.asyncio
async def test_fallback_clarification_for_olivo_query():
    organizer = OrganizerAgent()
    clarification = organizer._fallback_clarification("olivar")
    assert "olivar" in clarification.question.lower()


@pytest.mark.asyncio
async def test_fallback_clarification_for_generic_query():
    organizer = OrganizerAgent()
    clarification = organizer._fallback_clarification("necesito ayuda urgente")
    assert "consulta" in clarification.question.lower()
    assert all(opt.enriched_query for opt in clarification.options)


@pytest.mark.asyncio
async def test_clarification_plan_has_empty_steps():
    organizer = OrganizerAgent()
    clarification = organizer._fallback_clarification("test")
    from libs.schemas import AgentPlan
    plan = AgentPlan(
        steps=[],
        runs={},
        dependencies={},
        clarification=clarification,
    )
    assert plan.steps == []
    assert plan.clarification is not None
    assert len(plan.clarification.options) == 3


def test_detect_ambiguity_returns_none_for_long_queries():
    organizer = OrganizerAgent()
    result = organizer._detect_ambiguity(
        "Necesito un análisis completo de mi parcela de olivos con datos satelitales y consejos de riego para la próxima campaña",
        [],
        {},
    )
    assert result is None


def test_detect_ambiguity_returns_none_for_queries_with_context():
    organizer = OrganizerAgent()
    result = organizer._detect_ambiguity(
        "ayuda",
        [],
        {"case_history": [{"query": "test"}]},
    )
    assert result is None


def test_detect_ambiguity_returns_reason_for_single_word():
    organizer = OrganizerAgent()
    result = organizer._detect_ambiguity("olivos", [], {})
    assert result is not None


def test_clarification_schema_has_required_fields():
    from agents.organizer import CLARIFICATION_SCHEMA
    assert "question" in CLARIFICATION_SCHEMA["properties"]
    assert "options" in CLARIFICATION_SCHEMA["properties"]
    assert "question" in CLARIFICATION_SCHEMA["required"]
    assert "options" in CLARIFICATION_SCHEMA["required"]


@pytest.mark.asyncio
async def test_clarification_option_schema():
    from libs.schemas import ClarificationOption, ClarificationRequest
    opt = ClarificationOption(
        key="satellite",
        label="Satellite Analysis",
        description="NDVI and crop vigor",
        enriched_query="Analyze satellite imagery of my farm",
    )
    assert opt.key == "satellite"
    assert opt.enriched_query == "Analyze satellite imagery of my farm"

    req = ClarificationRequest(
        question="What analysis do you need?",
        options=[opt],
        rationale="test",
    )
    assert req.question == "What analysis do you need?"
    assert len(req.options) == 1


@pytest.mark.asyncio
async def test_full_clarification_flow_via_plan(monkeypatch):
    organizer = OrganizerAgent()

    async def fake_chat_json(*args, **kwargs):
        return {
            "question": "¿Qué análisis necesitas?",
            "options": [
                {"key": "satellite", "label": "Satélite", "description": "NDVI", "enriched_query": "Analiza mi parcela con satélite"},
                {"key": "legal", "label": "Legal", "description": "Normativa", "enriched_query": "Revisa la normativa de mi parcela"},
            ],
            "rationale": "test",
        }

    monkeypatch.setattr(organizer, "_chat_json", fake_chat_json)
    monkeypatch.setattr("agents.organizer.settings.DISABLE_EXTERNALS", False)
    monkeypatch.setattr(organizer, "client", True)

    plan = await organizer.plan(AgentInput(query="ayuda"))

    assert plan.clarification is not None
    assert plan.clarification.question == "¿Qué análisis necesitas?"
    assert len(plan.clarification.options) == 2
    assert plan.steps == []
    assert plan.clarification.options[0].enriched_query == "Analiza mi parcela con satélite"


@pytest.mark.asyncio
async def test_full_clarification_flow_fallback_when_llm_fails(monkeypatch):
    organizer = OrganizerAgent()

    async def fake_chat_json(*args, **kwargs):
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(organizer, "_chat_json", fake_chat_json)
    monkeypatch.setattr("agents.organizer.settings.DISABLE_EXTERNALS", False)
    monkeypatch.setattr(organizer, "client", True)

    plan = await organizer.plan(AgentInput(query="ayuda"))

    assert plan.clarification is not None
    assert "consulta" in plan.clarification.question.lower()
    assert len(plan.clarification.options) == 3


@pytest.mark.asyncio
async def test_clarification_skipped_for_contextual_queries(monkeypatch):
    organizer = OrganizerAgent()

    async def fake_chat_json(*args, **kwargs):
        return {"question": "test", "options": [{"key": "a", "label": "A", "enriched_query": "x"}, {"key": "b", "label": "B", "enriched_query": "y"}]}

    monkeypatch.setattr(organizer, "_chat_json", fake_chat_json)
    monkeypatch.setattr("agents.organizer.settings.DISABLE_EXTERNALS", False)
    monkeypatch.setattr(organizer, "client", True)

    plan = await organizer.plan(
        AgentInput(
            query="ayuda",
            context={"observations": [{"date": "2025-01-01", "parcel": "P1", "note": "test", "severity": "baja"}]},
        )
    )

    assert plan.clarification is None
    assert len(plan.steps) > 0


def test_fallback_enriched_queries_are_self_contained():
    organizer = OrganizerAgent()
    clarification = organizer._fallback_clarification("ayuda")
    for opt in clarification.options:
        assert "ayuda" not in opt.enriched_query
        assert len(opt.enriched_query) > 20


def test_fallback_enriched_queries_contain_label_context():
    organizer = OrganizerAgent()
    clarification = organizer._fallback_clarification("olivar")
    satellite_opt = next(o for o in clarification.options if o.key == "satellite")
    assert "olivar" in satellite_opt.enriched_query
