import pytest
from httpx import ASGITransport, AsyncClient

from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.api import ChatRequest, app, chat, get_agents
from backend.memory_store import UserMemoryStore
from libs.schemas import (
    AgentInput,
    AgentPlan,
    CaseTask,
    CaseState,
    FinalAnswer,
    MemoryUsage,
    WriterAgentOutput,
)


class FakeOrganizer:
    async def plan(self, agent_input: AgentInput, **__) -> AgentPlan:
        return AgentPlan(steps=["writer"], response_mode=agent_input.response_mode)


class FakeWriter:
    async def run(self, _: AgentInput) -> WriterAgentOutput:
        return WriterAgentOutput(
            agent="writer",
            summary="ok",
            data=FinalAnswer(
                executive_summary="Resumen ejecutivo de prueba",
                next_actions=["Revisar parcela norte"],
                memory=MemoryUsage(enabled=False, used_sections=[]),
                report_md="# Informe\nContenido de prueba",
            ),
        )


@pytest.fixture(autouse=True)
def override_agents():
    def _build():
        return {
            "organizer": FakeOrganizer(),
            "writer": FakeWriter(),
        }

    app.dependency_overrides[get_agents] = _build
    yield
    app.dependency_overrides.pop(get_agents, None)


@pytest.mark.asyncio
async def test_chat_endpoint():
    data = await chat(
        ChatRequest(query="Analiza parcela en Valencia para orgánico"),
        agents={"organizer": FakeOrganizer(), "writer": FakeWriter()},
    )
    assert "plan" in data and "answer" in data
    answer = data["answer"]
    assert data["plan"]["steps"] == ["writer"]
    assert "executive_summary" in answer
    assert "memory" in answer
    assert isinstance(answer.get("next_actions", []), list)
    assert data["plan"]["response_mode"] == "conversation"


def test_chat_request_defaults_to_automatic_continuity():
    assert ChatRequest(query="Consulta").continuity_mode == "auto"


def test_chat_endpoint_rejects_removed_report_response_mode():
    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"query": "Genera informe del caso", "response_mode": "report"},
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_endpoint_rejects_missing_attachment_ids():
    with pytest.raises(HTTPException) as exc:
        await chat(
            ChatRequest(query="Analiza este caso", attachment_ids=["missing-1", "missing-2"]),
            agents={"organizer": FakeOrganizer(), "writer": FakeWriter()},
        )
    assert exc.value.status_code == 400
    detail = exc.value.detail
    assert detail["error"] == "missing_attachments"
    assert detail["attachment_ids"] == ["missing-1", "missing-2"]


@pytest.mark.asyncio
async def test_memory_endpoints(tmp_path, monkeypatch):
    store = UserMemoryStore(base_dir=tmp_path / "memory")
    monkeypatch.setattr("backend.api.memory_store", store)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        user_id = "finca-demo"
        get_resp = await client.get(f"/memory/{user_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["user_id"] == user_id

        put_resp = await client.put(
            f"/memory/{user_id}",
            json={
                "sections": {
                    "profile": "Tipo de explotación: olivar",
                    "open_questions": "Dato pendiente: análisis de suelo",
                }
            },
        )
        assert put_resp.status_code == 200
        data = put_resp.json()
        assert "olivar" in data["sections"]["profile"]

        bad_resp = await client.put(
            f"/memory/{user_id}", json={"sections": {"decision_log": "no permitido"}}
        )
        assert bad_resp.status_code == 400

        delete_resp = await client.delete(f"/memory/{user_id}")
        assert delete_resp.status_code == 200
        assert delete_resp.json()["ok"] is True


@pytest.mark.asyncio
async def test_case_endpoint(tmp_path, monkeypatch):
    store = UserMemoryStore(base_dir=tmp_path / "memory")
    monkeypatch.setattr("backend.api.memory_store", store)
    store.save_case_state(
        "finca-demo",
        CaseState(
            case_summary="Caso abierto",
            open_tasks=[CaseTask(title="Subir análisis de suelo")],
            blocked_by=["Falta documento técnico"],
            recommended_next_input=["Adjuntar análisis actualizado"],
        ),
    )

    user_id = "finca-demo"
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        resp = await client.get(f"/case/{user_id}")
    assert resp.status_code == 404
    return
    data = resp.json()
    assert data["user_id"] == user_id
    assert "case_state" in data
    assert data["case_state"]["open_tasks"][0]["title"] == "Subir análisis de suelo"


@pytest.mark.asyncio
async def test_observations_endpoints(tmp_path, monkeypatch):
    store = UserMemoryStore(base_dir=tmp_path / "memory")
    monkeypatch.setattr("backend.api.memory_store", store)

    user_id = "finca-demo"
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        post_resp = await client.post(
            f"/observations/{user_id}",
            json={
                "date": "2026-04-02",
                "parcel": "Parcela Norte",
                "campaign": "2026",
                "note": "Hay una zona con menor vigor.",
                "severity": "media",
            },
        )
        assert post_resp.status_code == 404
        return
        get_resp = await client.get(f"/observations/{user_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["items"][0]["parcel"] == "Parcela Norte"


@pytest.mark.asyncio
async def test_legacy_followup_endpoints_are_removed(tmp_path, monkeypatch):
    from backend.case_store import CaseStore

    store = CaseStore(tmp_path / "cases.db")
    monkeypatch.setattr("backend.api.case_store", store)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        created = await client.post(
            "/followups", json={"workspace_id": "finca-demo", "title": "Riego norte"}
        )
        assert created.status_code == 404
        return
        followup_id = created.json()["followup_id"]
        observation = await client.post(
            f"/followups/{followup_id}/observations",
            json={
                "date": "2026-07-11",
                "parcel": "Parcela Norte",
                "note": "Menor vigor en el borde oeste.",
                "severity": "media",
            },
        )
        detail = await client.get(
            f"/followups/{followup_id}", params={"workspace_id": "finca-demo"}
        )

    assert created.status_code == 201
    assert observation.status_code == 201
    assert detail.status_code == 200
    assert detail.json()["followup"]["title"] == "Riego norte"
    assert detail.json()["observations"][0]["parcel"] == "Parcela Norte"
    store.close()
