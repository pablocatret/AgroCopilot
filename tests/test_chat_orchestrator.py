import asyncio
import time

import pytest

from agents.writer import WriterAgent
from backend.memory_store import UserMemoryStore
from libs.schemas import (
    AgentInput,
    AgentPlan,
    FieldObservation,
    FinalAnswer,
    ImageInsights,
    LegalFindings,
    LegalAgentOutput,
    MeteoContext,
    StacItem,
    StacResults,
    WebResearch,
    WriterAgentOutput,
    StacAgentOutput,
    RSAgentOutput,
    RemoteSensingMemoryArtifact,
)
from backend.services.chat_orchestrator import ChatOrchestratorService


class FakeBroker:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def publish(self, conversation_id: str, payload: dict) -> None:
        self.events.append((conversation_id, payload))


class FakeOrganizer:
    async def plan(self, _: AgentInput) -> AgentPlan:
        return AgentPlan(steps=["legal", "writer"], runs={"legal": 1, "writer": 1})


class FakeOrganizerWriterOnly(FakeOrganizer):
    async def plan(self, _: AgentInput) -> AgentPlan:
        return AgentPlan(
            steps=["writer"],
            runs={"writer": 1},
            policy={"writer_search_allowed": True, "fast_path": {"enabled": True, "allow_search": True}},
        )


class FakeOrganizerRetry(FakeOrganizer):
    async def plan(self, _: AgentInput) -> AgentPlan:
        return AgentPlan(
            steps=["legal", "writer"],
            runs={"legal": 1, "writer": 1},
            policy={"allow_retries": True, "max_rounds": 1, "retry_candidates": ["legal"]},
            diagnostics={"planner_source": "llm", "fallback_reason": None, "rationale": "retry"},
        )


class FakeLegal:
    async def run(self, _: AgentInput) -> LegalAgentOutput:
        return LegalAgentOutput(agent="legal", summary="ok", data=LegalFindings())


class FakeLegalError:
    async def run(self, _: AgentInput) -> LegalAgentOutput:
        return LegalAgentOutput(
            agent="legal", status="error", summary="fuente no disponible", data=LegalFindings()
        )


class FlakyLegal:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, _: AgentInput) -> LegalAgentOutput:
        self.calls += 1
        if self.calls == 1:
            return LegalAgentOutput(
                agent="legal", status="error", summary="timeout parcial", data=LegalFindings()
            )
        return LegalAgentOutput(agent="legal", summary="ok", data=LegalFindings())


class NullAgent:
    async def run(self, *_, **__):
        return None


class CountingNullAgent(NullAgent):
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, *args, **kwargs):
        self.calls += 1
        return await super().run(*args, **kwargs)


class FakeWriter:
    async def run(self, _: AgentInput) -> WriterAgentOutput:
        answer = FinalAnswer(
            executive_summary="ok",
            legal=LegalFindings(),
            remote_sensing=ImageInsights(insights=[]),
            research=WebResearch(findings=[]),
            stac=StacResults(items=[]),
            report_md="# Demo\nTexto",
        )
        return WriterAgentOutput(agent="writer", summary="ok", data=answer)


class FakeWriterCapturing(FakeWriter):
    def __init__(self) -> None:
        self.last_input: AgentInput | None = None

    async def run(self, agent_input: AgentInput) -> WriterAgentOutput:
        self.last_input = agent_input
        return await super().run(agent_input)


class FakeNamedWriter(FakeWriterCapturing):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.calls = 0

    async def run(self, agent_input: AgentInput) -> WriterAgentOutput:
        self.calls += 1
        output = await super().run(agent_input)
        output.agent = self.name
        return output


class FakeOrganizerUnknownStep(FakeOrganizer):
    async def plan(self, _: AgentInput) -> AgentPlan:
        return AgentPlan(steps=["legal", "invented_agent", "writer"])


class FakeOrganizerCampaign(FakeOrganizer):
    async def plan(self, _: AgentInput) -> AgentPlan:
        return AgentPlan(steps=["legal", "case_manager", "writer"])


class FakeOrganizerAgronomy(FakeOrganizer):
    async def plan(self, _: AgentInput) -> AgentPlan:
        return AgentPlan(steps=["document_analyst", "case_manager", "writer"])


class FakeOrganizerStacRS(FakeOrganizer):
    async def plan(self, _: AgentInput) -> AgentPlan:
        return AgentPlan(steps=["stac", "rs_analyst", "writer"])


class FakeOrganizerParallel(FakeOrganizer):
    async def plan(self, _: AgentInput) -> AgentPlan:
        return AgentPlan(
            steps=["legal", "document_analyst", "writer"],
            dependencies={"writer": ["legal", "document_analyst"]},
        )


class FakeOrganizerWithNoOpReplan(FakeOrganizer):
    async def plan(self, _: AgentInput) -> AgentPlan:
        return AgentPlan(
            steps=["legal", "writer"],
            runs={"legal": 1, "writer": 1},
            allow_replan=True,
        )

    async def replan(self, _: AgentInput, __: dict) -> AgentPlan:
        return AgentPlan(
            steps=[],
            diagnostics={"planner_source": "llm", "fallback_reason": "llm_replan_empty", "rationale": "sin cambios"},
        )


class FakeOrganizerWithAppliedReplan(FakeOrganizer):
    async def plan(self, _: AgentInput) -> AgentPlan:
        return AgentPlan(
            steps=["legal", "writer"],
            runs={"legal": 1, "writer": 1},
            allow_replan=True,
        )

    async def replan(self, _: AgentInput, __: dict) -> AgentPlan:
        return AgentPlan(
            steps=["document_analyst", "writer"],
            dependencies={"writer": ["document_analyst"]},
            diagnostics={"planner_source": "llm", "fallback_reason": None, "rationale": "hace falta leer documento"},
        )


class SlowAgent:
    def __init__(self, name: str, delay: float, marks: list[tuple[str, float]]) -> None:
        self.name = name
        self.delay = delay
        self.marks = marks

    async def run(self, _: AgentInput) -> LegalAgentOutput:
        self.marks.append((f"{self.name}:start", time.perf_counter()))
        await asyncio.sleep(self.delay)
        self.marks.append((f"{self.name}:end", time.perf_counter()))
        return LegalAgentOutput(agent=self.name, summary="ok", data=LegalFindings())


@pytest.mark.asyncio
async def test_chat_orchestrator_executes_plan_and_returns_payload():
    broker = FakeBroker()
    service = ChatOrchestratorService(
        agents={
            "organizer": FakeOrganizer(),
            "legal": FakeLegal(),
            "stac": NullAgent(),
            "rs_analyst": NullAgent(),
            "writer": FakeWriter(),
            "document_analyst": NullAgent(),
            "spreadsheet_analyst": NullAgent(),
            "vision_ocr": NullAgent(),
        },
        broker=broker,
    )

    payload = await service.execute(query="demo", conversation_id="conv-1")

    assert payload["conversation_id"] == "conv-1"
    assert payload["plan"]["steps"] == ["legal", "writer"]
    assert payload["answer"]["report_md"] == "# Demo\nTexto"
    assert payload["plan"]["policy"]["allow_retries"] is False
    assert any(event["type"] == "plan" for _, event in broker.events)
    assert any(
        event["type"] == "status" and event["stage"] == "completed" for _, event in broker.events
    )


@pytest.mark.asyncio
async def test_chat_orchestrator_retries_agents_selected_by_policy():
    broker = FakeBroker()
    legal = FlakyLegal()
    service = ChatOrchestratorService(
        agents={
            "organizer": FakeOrganizerRetry(),
            "legal": legal,
            "writer": FakeWriter(),
        },
        broker=broker,
    )

    payload = await service.execute(query="demo", conversation_id="conv-retry")

    assert payload["answer"]["report_md"] == "# Demo\nTexto"
    assert legal.calls == 2
    retry_events = [event for _, event in broker.events if event.get("status") == "retrying"]
    assert retry_events


@pytest.mark.asyncio
async def test_chat_orchestrator_runs_independent_agents_in_parallel():
    broker = FakeBroker()
    marks: list[tuple[str, float]] = []
    service = ChatOrchestratorService(
        agents={
            "organizer": FakeOrganizerParallel(),
            "legal": SlowAgent("legal", 0.08, marks),
            "document_analyst": SlowAgent("document_analyst", 0.08, marks),
            "writer": FakeWriter(),
        },
        broker=broker,
    )

    started = time.perf_counter()
    payload = await service.execute(query="demo", conversation_id="conv-parallel")
    elapsed = time.perf_counter() - started

    assert payload["plan"]["dependencies"]["writer"] == ["legal", "document_analyst"]
    assert elapsed < 0.18
    starts = {name: ts for name, ts in marks if name.endswith(":start")}
    assert abs(starts["legal:start"] - starts["document_analyst:start"]) < 0.04


@pytest.mark.asyncio
async def test_chat_orchestrator_emits_replan_result_even_when_no_extra_steps():
    broker = FakeBroker()
    service = ChatOrchestratorService(
        agents={
            "organizer": FakeOrganizerWithNoOpReplan(),
            "legal": FakeLegal(),
            "writer": FakeWriter(),
        },
        broker=broker,
    )

    payload = await service.execute(query="demo", conversation_id="conv-replan-noop")

    assert payload["plan"]["replan"]["attempted"] is True
    assert payload["plan"]["replan"]["applied"] is False
    replan_events = [event for _, event in broker.events if event.get("type") == "replan"]
    assert replan_events
    assert replan_events[0]["applied"] is False


@pytest.mark.asyncio
async def test_chat_orchestrator_applies_replanned_steps_and_reports_them():
    broker = FakeBroker()
    document_agent = FakeWriterCapturing()
    service = ChatOrchestratorService(
        agents={
            "organizer": FakeOrganizerWithAppliedReplan(),
            "legal": FakeLegal(),
            "document_analyst": document_agent,
            "writer": FakeWriter(),
        },
        broker=broker,
    )

    payload = await service.execute(query="demo", conversation_id="conv-replan-applied")

    assert payload["plan"]["replan"]["attempted"] is True
    assert payload["plan"]["replan"]["applied"] is True
    assert payload["plan"]["replan"]["extra_steps"] == ["document_analyst"]
    assert "document_analyst" in payload["plan"]["steps"]
    assert document_agent.last_input is not None


@pytest.mark.asyncio
async def test_chat_orchestrator_always_uses_direct_writer():
    direct_writer = FakeNamedWriter("direct_writer")
    service = ChatOrchestratorService(
        agents={
            "organizer": FakeOrganizer(),
            "legal": FakeLegal(),
            "direct_writer": direct_writer,
        },
        broker=FakeBroker(),
    )

    conversation_payload = await service.execute(query="demo", conversation_id="conv-direct")
    legacy_payload = await service.execute(
        query="demo", conversation_id="conv-report", response_mode="report"
    )

    assert conversation_payload["plan"]["steps"] == ["legal", "writer"]
    assert conversation_payload["plan"]["writer_agent"] == "direct_writer"
    assert legacy_payload["plan"]["writer_agent"] == "direct_writer"
    assert direct_writer.calls == 2
    assert direct_writer.last_input is not None
    assert direct_writer.last_input.response_mode == "conversation"
    assert direct_writer.last_input.context["_plan"]["writer_agent"] == "direct_writer"


@pytest.mark.asyncio
async def test_chat_orchestrator_raises_for_unknown_plan_steps():
    broker = FakeBroker()
    service = ChatOrchestratorService(
        agents={
            "organizer": FakeOrganizerUnknownStep(),
            "legal": FakeLegal(),
            "stac": NullAgent(),
            "rs_analyst": NullAgent(),
            "writer": FakeWriter(),
            "document_analyst": NullAgent(),
            "spreadsheet_analyst": NullAgent(),
            "vision_ocr": NullAgent(),
        },
        broker=broker,
    )

    with pytest.raises(RuntimeError, match="Plan inv"):
        await service.execute(query="demo", conversation_id="conv-unknown")

    assert any(event["type"] == "error" for _, event in broker.events)
    assert not any(
        event["type"] == "status" and event.get("stage") == "completed"
        for _, event in broker.events
    )


@pytest.mark.asyncio
async def test_chat_orchestrator_surfaces_partial_failures_in_limitations():
    broker = FakeBroker()
    service = ChatOrchestratorService(
        agents={
            "organizer": FakeOrganizer(),
            "legal": FakeLegalError(),
            "stac": NullAgent(),
            "rs_analyst": NullAgent(),
            "writer": FakeWriter(),
            "document_analyst": NullAgent(),
            "spreadsheet_analyst": NullAgent(),
            "vision_ocr": NullAgent(),
        },
        broker=broker,
    )

    payload = await service.execute(query="demo", conversation_id="conv-partial")

    limitations = payload["answer"]["limitations"]
    assert any("evidencia documental o normativa es incompleta" in item for item in limitations)
    assert not any("legal#1" in item or "EjecuciÃ³n parcial" in item for item in limitations)


@pytest.mark.asyncio
async def test_chat_orchestrator_passes_execution_report_to_writer():
    broker = FakeBroker()
    writer = FakeWriterCapturing()
    service = ChatOrchestratorService(
        agents={
            "organizer": FakeOrganizer(),
            "legal": FakeLegalError(),
            "stac": NullAgent(),
            "rs_analyst": NullAgent(),
            "writer": writer,
            "document_analyst": NullAgent(),
            "spreadsheet_analyst": NullAgent(),
            "vision_ocr": NullAgent(),
        },
        broker=broker,
    )

    await service.execute(query="demo", conversation_id="conv-exec-report")

    assert writer.last_input is not None
    execution = writer.last_input.context["_execution"]
    assert execution["legal"]["final_level"] == "soft_error"
    assert execution["legal"]["instances"][0]["message"] == "legal#1: fuente no disponible"


@pytest.mark.asyncio
async def test_chat_orchestrator_loads_memory_and_persists_decision_log(tmp_path, monkeypatch):
    broker = FakeBroker()
    writer = FakeWriterCapturing()
    memory = UserMemoryStore(base_dir=tmp_path / "memory")
    user_id = "finca-demo"
    memory.ensure_user_files(user_id)
    (tmp_path / "memory" / user_id / "default" / "profile.md").write_text(
        "# Perfil\n\n- Cultivo principal: olivar\n", encoding="utf-8"
    )
    (tmp_path / "memory" / user_id / "default" / "farm_context.md").write_text(
        "# Contexto de explotaciÃ³n\n\n- Zona: secano\n", encoding="utf-8"
    )
    monkeypatch.setattr("backend.services.chat_orchestrator.memory_store", memory)

    service = ChatOrchestratorService(
        agents={
            "organizer": FakeOrganizer(),
            "legal": FakeLegal(),
            "stac": NullAgent(),
            "rs_analyst": NullAgent(),
            "writer": writer,
            "document_analyst": NullAgent(),
            "spreadsheet_analyst": NullAgent(),
            "vision_ocr": NullAgent(),
        },
        broker=broker,
    )

    payload = await service.execute(
        query="Â¿QuÃ© harÃ­a esta semana?",
        conversation_id="conv-memory",
        user_id=user_id,
        decision_mode="case",
        memory_enabled=True,
    )

    assert writer.last_input is not None
    memory_meta = writer.last_input.context["_memory"]
    assert memory_meta["enabled"] is True
    assert memory_meta["user_id"] == user_id
    assert "Cultivo principal" in memory_meta["context"]
    assert payload["answer"]["memory"]["enabled"] is True
    assert payload["answer"]["memory"]["memory_id"] == "default"
    decision_log = (tmp_path / "memory" / user_id / "default" / "decision_log.md").read_text(encoding="utf-8")
    assert "case" not in decision_log


@pytest.mark.asyncio
async def test_chat_orchestrator_passes_conversation_history_to_writer():
    broker = FakeBroker()
    writer = FakeWriterCapturing()
    service = ChatOrchestratorService(
        agents={
            "organizer": FakeOrganizer(),
            "legal": FakeLegal(),
            "writer": writer,
        },
        broker=broker,
    )

    await service.execute(query="Primera consulta", conversation_id="conv-history")
    await service.execute(query="Y sobre lo anterior, que haria ahora?", conversation_id="conv-history")

    assert writer.last_input is not None
    history = writer.last_input.context.get("_conversation_history")
    assert isinstance(history, list)
    assert any(item.get("query") == "Primera consulta" for item in history if isinstance(item, dict))


@pytest.mark.asyncio
async def test_chat_orchestrator_preserves_writer_fast_path_with_technical_context(monkeypatch):
    monkeypatch.setattr("backend.deps.settings.DISABLE_EXTERNALS", True)
    service = ChatOrchestratorService(
        agents={
            "organizer": FakeOrganizerWriterOnly(),
            "writer": WriterAgent(),
        },
        broker=FakeBroker(),
    )

    payload = await service.execute(query="Precio actual del aceite de oliva", conversation_id="conv-fast-path")

    assert payload["answer"]["response_path"] == "single_agent_fast_path"


@pytest.mark.asyncio
async def test_chat_orchestrator_reuses_remote_sensing_memory_without_rerunning_agents(
    tmp_path, monkeypatch
):
    broker = FakeBroker()
    writer = FakeWriterCapturing()
    stac = CountingNullAgent()
    rs = CountingNullAgent()
    memory = UserMemoryStore(base_dir=tmp_path / "memory")
    user_id = "finca-demo"
    memory.append_observation(
        user_id,
        FieldObservation(
            date="2026-07-01",
            parcel="Parcela Norte",
            campaign="2026",
            note="Seguimiento de vigor en la misma parcela.",
            severity="media",
        ),
    )
    memory.save_remote_sensing_artifact(
        user_id,
        RemoteSensingMemoryArtifact(
            generated_at="2099-07-01T10:00:00Z",
            query="Analiza la parcela norte con satelite",
            query_intent="monitoring",
            evidence_level="analyzed_temporal",
            decision_mode="case",
            parcel="Parcela Norte",
            latest_scene_date="2099-06-30",
            stac_item_ids=["scene-1", "scene-2"],
            scene_count=2,
            summary="Analisis reciente de 2 escenas satelitales.",
            change_highlights=["Descenso de vigor en borde oeste."],
        ),
    )
    monkeypatch.setattr("backend.services.chat_orchestrator.memory_store", memory)

    service = ChatOrchestratorService(
        agents={
            "organizer": FakeOrganizerStacRS(),
            "stac": stac,
            "rs_analyst": rs,
            "writer": writer,
            "legal": NullAgent(),
            "document_analyst": NullAgent(),
            "case_manager": NullAgent(),
            "spreadsheet_analyst": NullAgent(),
            "vision_ocr": NullAgent(),
        },
        broker=broker,
    )

    payload = await service.execute(
        query="Monitoriza la parcela norte",
        conversation_id="conv-rs-memory-hit",
        user_id=user_id,
        memory_enabled=True,
    )

    assert payload["plan"]["steps"] == ["writer"]
    assert stac.calls == 0
    assert rs.calls == 0
    assert writer.last_input is not None
    reuse = writer.last_input.context.get("_memory_reuse")
    assert reuse["remote_sensing"]["status"] == "hit"


@pytest.mark.asyncio
async def test_chat_orchestrator_does_not_reuse_retrieval_only_remote_sensing_memory(
    tmp_path, monkeypatch
):
    broker = FakeBroker()
    writer = FakeWriterCapturing()
    stac = CountingNullAgent()
    rs = CountingNullAgent()
    memory = UserMemoryStore(base_dir=tmp_path / "memory")
    user_id = "finca-demo"
    memory.save_remote_sensing_artifact(
        user_id,
        RemoteSensingMemoryArtifact(
            generated_at="2099-07-01T10:00:00Z",
            latest_scene_date="2099-06-30",
            query="Analiza la parcela norte con satelite",
            query_intent="monitoring",
            evidence_level="retrieval_only",
            decision_mode="case",
            parcel="Parcela Norte",
            summary="2 escenas STAC recuperadas.",
        ),
    )
    monkeypatch.setattr("backend.services.chat_orchestrator.memory_store", memory)

    service = ChatOrchestratorService(
        agents={
            "organizer": FakeOrganizerStacRS(),
            "stac": stac,
            "rs_analyst": rs,
            "writer": writer,
            "legal": NullAgent(),
            "document_analyst": NullAgent(),
            "case_manager": NullAgent(),
            "spreadsheet_analyst": NullAgent(),
            "vision_ocr": NullAgent(),
        },
        broker=broker,
    )

    await service.execute(
        query="Monitoriza la parcela norte",
        conversation_id="conv-rs-memory-retrieval-only",
        user_id=user_id,
        memory_enabled=True,
    )

    assert stac.calls == 1
    assert rs.calls == 1


@pytest.mark.asyncio
async def test_chat_orchestrator_passes_history_and_observations_to_case_manager(
    tmp_path, monkeypatch
):
    broker = FakeBroker()
    writer = FakeWriterCapturing()
    tracker = FakeWriterCapturing()
    memory = UserMemoryStore(base_dir=tmp_path / "memory")
    user_id = "finca-demo"
    memory.append_case_history(
        user_id,
        title="Caso previo",
        decision_mode="case",
        summary="Seguimiento previo.",
        next_actions=["Revisar borde norte"],
        blocked_by=["Falta nueva visita"],
    )
    memory.append_observation(
        user_id,
        FieldObservation(
            date="2026-04-02",
            parcel="Parcela Norte",
            campaign="2026",
            note="Vigor irregular en borde oeste.",
            severity="media",
        ),
    )
    monkeypatch.setattr("backend.services.chat_orchestrator.memory_store", memory)

    service = ChatOrchestratorService(
        agents={
            "organizer": FakeOrganizerCampaign(),
            "legal": FakeLegal(),
            "case_manager": tracker,
            "stac": NullAgent(),
            "rs_analyst": NullAgent(),
            "writer": writer,
            "document_analyst": NullAgent(),
            "spreadsheet_analyst": NullAgent(),
            "vision_ocr": NullAgent(),
        },
        broker=broker,
    )

    await service.execute(
        query="demo", conversation_id="conv-campaign", user_id=user_id, memory_enabled=True
    )

    assert tracker.last_input is not None
    assert "case_context" in tracker.last_input.context
    assert "case_history" not in tracker.last_input.context
    assert "observations" not in tracker.last_input.context


@pytest.mark.asyncio
async def test_chat_orchestrator_passes_case_context_to_document_agent(tmp_path, monkeypatch):
    broker = FakeBroker()
    document_agent = FakeWriterCapturing()
    writer = FakeWriterCapturing()
    memory = UserMemoryStore(base_dir=tmp_path / "memory")
    user_id = "finca-demo"
    memory.append_case_history(
        user_id,
        title="Caso previo",
        decision_mode="case",
        summary="Se observÃ³ menor vigor en una zona concreta.",
        next_actions=["Revisar parcela norte"],
        blocked_by=[],
    )
    memory.append_observation(
        user_id,
        FieldObservation(
            date="2026-04-02",
            parcel="Parcela Norte",
            campaign="2026",
            note="ApariciÃ³n de clorosis leve en hojas jÃ³venes.",
            severity="media",
        ),
    )
    monkeypatch.setattr("backend.services.chat_orchestrator.memory_store", memory)

    service = ChatOrchestratorService(
        agents={
            "organizer": FakeOrganizerAgronomy(),
            "legal": NullAgent(),
            "case_manager": NullAgent(),
            "stac": NullAgent(),
            "rs_analyst": NullAgent(),
            "writer": writer,
            "document_analyst": document_agent,
            "spreadsheet_analyst": NullAgent(),
            "vision_ocr": NullAgent(),
        },
        broker=broker,
    )

    await service.execute(
        query="Necesito ideas para biodiversidad funcional",
        conversation_id="conv-agro",
        user_id=user_id,
        memory_enabled=True,
    )

    assert document_agent.last_input is not None
    assert "case_context" in document_agent.last_input.context
    assert "case_history" not in document_agent.last_input.context
    assert "observations" not in document_agent.last_input.context


class FakeStacWithBbox:
    async def run(self, _: AgentInput) -> StacAgentOutput:
        return StacAgentOutput(
            agent="stac",
            summary="ok",
            data=StacResults(
                items=[
                    StacItem(
                        id="item-1",
                        bbox=[-3.75, 40.35, -3.65, 40.45],
                        datetime="2025-05-15T09:15:00Z",
                    ),
                    StacItem(
                        id="item-2",
                        bbox=[-3.75, 40.35, -3.65, 40.45],
                        datetime="2025-06-01T11:45:00Z",
                    ),
                ],
            ),
        )


class RSCapturing:
    def __init__(self) -> None:
        self.last_input: AgentInput | None = None

    async def run(self, agent_input: AgentInput) -> RSAgentOutput:
        self.last_input = agent_input
        return RSAgentOutput(agent="rs_analyst", summary="ok", data=ImageInsights())


@pytest.mark.asyncio
async def test_fetch_meteo_for_stac_passes_bbox_and_dates(monkeypatch):
    fake_meteo = MeteoContext(total_precip_mm=42.0, precipitation_irregularity_index=-0.8)
    calls: list[dict] = []

    async def fake_fetch_meteo_context_async(bbox, start_date, end_date):
        calls.append({"bbox": bbox, "start": start_date, "end": end_date})
        return fake_meteo

    monkeypatch.setattr(
        "backend.services.chat_orchestrator.fetch_meteo_context_async",
        fake_fetch_meteo_context_async,
    )

    rs_agent = RSCapturing()
    broker = FakeBroker()
    service = ChatOrchestratorService(
        agents={
            "organizer": FakeOrganizerStacRS(),
            "stac": FakeStacWithBbox(),
            "rs_analyst": rs_agent,
            "writer": FakeWriter(),
            "legal": NullAgent(),
            "document_analyst": NullAgent(),
            "case_manager": NullAgent(),
            "spreadsheet_analyst": NullAgent(),
            "vision_ocr": NullAgent(),
        },
        broker=broker,
    )

    await service.execute(query="analisis de vigor", conversation_id="conv-meteo")

    assert len(calls) == 1
    assert calls[0]["bbox"] == [-3.75, 40.35, -3.65, 40.45]
    assert calls[0]["start"] == "2025-05-15"
    assert calls[0]["end"] == "2025-06-01"
    assert rs_agent.last_input is not None
    assert rs_agent.last_input.context.get("meteo") is fake_meteo


@pytest.mark.asyncio
async def test_fetch_meteo_for_stac_returns_none_without_bbox(monkeypatch):
    calls: list[dict] = []

    async def fake_fetch_meteo_context_async(bbox, start_date, end_date):
        calls.append({"bbox": bbox})
        return MeteoContext(total_precip_mm=10.0)

    monkeypatch.setattr(
        "backend.services.chat_orchestrator.fetch_meteo_context_async",
        fake_fetch_meteo_context_async,
    )

    rs_agent = RSCapturing()
    broker = FakeBroker()
    service = ChatOrchestratorService(
        agents={
            "organizer": FakeOrganizerStacRS(),
            "stac": FakeStacWithBbox(),
            "rs_analyst": rs_agent,
            "writer": FakeWriter(),
            "legal": NullAgent(),
            "document_analyst": NullAgent(),
            "case_manager": NullAgent(),
            "spreadsheet_analyst": NullAgent(),
            "vision_ocr": NullAgent(),
        },
        broker=broker,
    )

    # Override stac to return items WITHOUT bbox
    class FakeStacNoBbox:
        async def run(self, _: AgentInput) -> StacAgentOutput:
            return StacAgentOutput(
                agent="stac",
                summary="ok",
                data=StacResults(
                    items=[StacItem(id="item-nobbx", datetime="2025-05-15")],
                ),
            )

    service.agents["stac"] = FakeStacNoBbox()
    await service.execute(query="analisis de vigor", conversation_id="conv-nobbx")

    assert len(calls) == 0
    assert rs_agent.last_input is not None
    assert rs_agent.last_input.context.get("meteo") is None


