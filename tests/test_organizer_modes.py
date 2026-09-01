from agents.organizer import OrganizerAgent
from libs.schemas import AgentInput, FieldObservation
import pytest


def test_case_route_uses_core_agents_for_documents_and_monitoring():
    organizer = OrganizerAgent()
    steps = organizer._steps_for_decision_mode(
        "case",
        attachments=[
            {"filename": "expediente.pdf", "content_type": "application/pdf"},
            {"filename": "foto.jpg", "content_type": "image/jpeg"},
        ],
        query="Revisa la evolucion reciente de la parcela norte con satelite",
        context={
            "observations": [
                FieldObservation(
                    date="2026-04-01",
                    parcel="Parcela Norte",
                    note="Perdida de vigor en varias lineas.",
                    severity="media",
                )
            ]
        },
    )
    assert "document_analyst" in steps
    assert "vision_ocr" in steps
    assert "stac" in steps
    assert "rs_analyst" in steps
    assert "case_manager" in steps
    assert steps[-1] == "writer"
    assert set(steps).issubset(
        {
            "document_analyst",
            "vision_ocr",
            "stac",
            "rs_analyst",
            "case_manager",
            "writer",
        }
    )


def test_fallback_routes_html_attachment_to_document_analyst():
    steps = OrganizerAgent()._fallback_steps(
        "Resume el documento adjunto",
        attachments=[
            {"filename": "attachment_1.html", "content_type": "text/html"}
        ],
    )
    assert steps == ["document_analyst", "case_manager", "writer"]


@pytest.mark.asyncio
async def test_organizer_uses_conversation_response_contract():
    organizer = OrganizerAgent()
    plan = await organizer.plan(
        AgentInput(query="Analiza el caso y propone los siguientes pasos")
    )
    assert plan.response_mode == "conversation"
