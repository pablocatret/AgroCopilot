import pytest

from backend.api import ChatRequest, chat
from libs.schemas import (
    AgentPlan,
    AgentInput,
    FinalAnswer,
    ImageInsights,
    LegalFindings,
    LegalAgentOutput,
    StacResults,
    WebResearch,
    WriterAgentOutput,
)


class FakeOrganizer:
    async def plan(self, _: AgentInput, **__) -> AgentPlan:
        return AgentPlan(steps=["legal", "writer"])


class FakeLegal:
    async def run(self, _: AgentInput) -> LegalAgentOutput:
        return LegalAgentOutput(agent="legal", summary="ok", data=LegalFindings())


class NullAgent:
    async def run(self, *_, **__):
        return None


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


@pytest.mark.asyncio
async def test_chat_endpoint_with_fakes():
    data = await chat(
        ChatRequest(query="demo"),
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
    )
    assert data["plan"]["steps"] == ["legal", "writer"]
    assert "report_md" in data["answer"]
