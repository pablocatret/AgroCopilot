# backend/api.py
from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
import uuid

from agents.case_manager import CaseManagerAgent
from agents.legal import LegalAgent
from agents.document_analyst import DocumentAnalystAgent
from agents.free import FreeAgent
from agents.organizer import OrganizerAgent
from agents.rs_analyst import RSAnalystAgent
from agents.spreadsheet_analyst import SpreadsheetAnalystAgent
from agents.stac_search import StacSearchAgent
from agents.vision_ocr import VisionOcrAgent
from agents.writer import DirectResponseWriterAgent
from backend.cost_store import cost_store
from backend.deps import VERSION, configure_logging, settings
from backend.events import broker, record_delivery
from libs.schemas import (
    AgentPlan,
    ClarificationRequest,
    FinalAnswer,
    DecisionMode,
    MemoryListItem,
    MemoryMeta,
    ResponseMode,
)
from backend.services.chat_orchestrator import ChatOrchestratorService
from backend.memory_store import EDITABLE_SECTIONS, memory_store
from backend.case_store import CASE_STATUSES, case_store
from backend.conversation_store import conversation_store
from backend.storage import attachments_store
from libs.costs.pricing import get_pricing_catalog


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    logger.bind(component="api").info("API started")
    yield


app = FastAPI(title="AgroCopilot API", version=VERSION, lifespan=lifespan)

allowed_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str
    conversation_id: str | None = None
    user_id: str | None = None
    language: str | None = None
    decision_mode: DecisionMode = "case"
    response_mode: ResponseMode = "conversation"
    memory_enabled: bool = False
    continuity_mode: Literal["auto", "off", "explicit"] = "auto"
    attachment_ids: list[str] | None = None
    case_id: str | None = None


class ChatResponse(BaseModel):
    plan: AgentPlan
    answer: FinalAnswer
    conversation_id: str | None = None
    clarification: ClarificationRequest | None = None


class MemoryResponse(BaseModel):
    user_id: str
    sections: dict[str, str]
    used_sections: list[str]


class MemoryUpdateRequest(BaseModel):
    sections: dict[str, str]


class MemoryListResponse(BaseModel):
    user_id: str
    items: list[MemoryListItem]


class MemoryCreateRequest(BaseModel):
    name: str = "Mi memoria"


class MemorySwitchRequest(BaseModel):
    memory_id: str


class MemoryRenameRequest(BaseModel):
    name: str


class CaseCreateRequest(BaseModel):
    workspace_id: str = "local"
    title: str
    objective: str = ""


class CaseUpdateRequest(BaseModel):
    workspace_id: str = "local"
    title: str | None = None
    objective: str | None = None


class CaseStatusRequest(BaseModel):
    workspace_id: str = "local"
    status: Literal["active", "on_hold", "closed", "archived", "deleted"]


class AssertionCreateRequest(BaseModel):
    workspace_id: str = "local"
    key: str
    value: str
    scope: Literal["case", "global"] = "case"
    provenance: str = "user_statement"
    status: Literal["proposed", "confirmed"] = "confirmed"
    assertion_type: str = "fact"
    display_text: str = ""
    confidence: float | None = None


class AssertionStatusRequest(BaseModel):
    workspace_id: str = "local"
    status: Literal["proposed", "confirmed", "superseded", "retracted", "expired"]


class AssertionCorrectionRequest(BaseModel):
    workspace_id: str = "local"
    value: str
    display_text: str | None = None


class CaseTaskCreateRequest(BaseModel):
    workspace_id: str = "local"
    title: str
    rationale: str = ""
    priority: Literal["high", "medium", "low"] = "medium"
    status: Literal["proposed", "open", "blocked", "done", "cancelled"] = "open"


class CaseTaskStatusRequest(BaseModel):
    workspace_id: str = "local"
    status: Literal["proposed", "open", "blocked", "done", "cancelled"]


class CaseObservationRequest(BaseModel):
    workspace_id: str = "local"
    date: str
    parcel: str
    campaign: str | None = None
    note: str
    severity: Literal["baja", "media", "alta"] = "media"


class WorkspaceContextRequest(BaseModel):
    name: str = ""
    zone: str = ""
    crops: str = ""
    infrastructure: str = ""
    constraints: str = ""
    preferences: str = ""


@lru_cache(maxsize=1)
def _cached_agents():
    return {
        "organizer": OrganizerAgent(),
        "legal": LegalAgent(),
        "case_manager": CaseManagerAgent(),
        "stac": StacSearchAgent(),
        "rs_analyst": RSAnalystAgent(),
        "free": FreeAgent(),
        "direct_writer": DirectResponseWriterAgent(),
        "document_analyst": DocumentAnalystAgent(),
        "spreadsheet_analyst": SpreadsheetAnalystAgent(),
        "vision_ocr": VisionOcrAgent(),
    }


def get_agents():
    return _cached_agents()


@app.post("/attachments")
async def upload_attachments(files: list[UploadFile] = File(...)):
    saved = await attachments_store.save_files(files)
    return {"attachments": [meta.model_dump() for meta in saved]}


@app.get("/health")
async def health():
    return {"status": "ok", "version": VERSION}


@app.get("/version")
async def version():
    return {"version": VERSION}


@app.get("/costs/summary")
async def costs_summary(days: int = 7):
    return cost_store.summary(days=days)


@app.get("/costs/conversation/{conversation_id}")
async def costs_conversation(conversation_id: str):
    return cost_store.summarize_conversation(conversation_id)


@app.get("/costs/pricing")
async def costs_pricing():
    return get_pricing_catalog()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, agents=Depends(get_agents)):
    requested_ids = req.attachment_ids or []
    attachments = attachments_store.list(requested_ids)
    missing = sorted(set(requested_ids) - {a.attachment_id for a in attachments})
    if missing:
        logger.bind(component="api").warning(
            "Missing attachments requested",
            attachment_ids=missing,
        )
        raise HTTPException(
            status_code=400,
            detail={
                "error": "missing_attachments",
                "message": "Faltan adjuntos solicitados para ejecutar el caso.",
                "attachment_ids": missing,
            },
        )
    orchestrator = ChatOrchestratorService(agents=agents, broker=broker)
    try:
        payload = await orchestrator.execute(
            query=req.query,
            language=req.language or "es",
            conversation_id=req.conversation_id,
            user_id=req.user_id,
            decision_mode=req.decision_mode,
            response_mode=req.response_mode,
            memory_enabled=req.memory_enabled,
            continuity_mode=req.continuity_mode,
            attachments=attachments,
            case_id=req.case_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return payload


@app.get("/cases")
async def list_cases(workspace_id: str = "local", status: str | None = None, limit: int = 50):
    if status and status not in CASE_STATUSES:
        raise HTTPException(status_code=400, detail="Invalid case status")
    return {"workspace_id": workspace_id, "items": case_store.list_cases(workspace_id=workspace_id, status=status, limit=limit)}


@app.post("/cases")
async def create_case(req: CaseCreateRequest):
    return case_store.create_case(workspace_id=req.workspace_id, title=req.title, objective=req.objective)


@app.get("/workspace-context/{workspace_id}")
async def get_workspace_context(workspace_id: str):
    return case_store.get_workspace_context(workspace_id)


@app.put("/workspace-context/{workspace_id}")
async def save_workspace_context(workspace_id: str, req: WorkspaceContextRequest):
    return case_store.save_workspace_context(workspace_id, req.model_dump())


@app.get("/cases/{case_id}")
async def get_case(case_id: str, workspace_id: str = "local"):
    try:
        return case_store.get_case(case_id, workspace_id=workspace_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Case not found")


@app.patch("/cases/{case_id}")
async def update_case(case_id: str, req: CaseUpdateRequest):
    try:
        return case_store.update_case(case_id, workspace_id=req.workspace_id, title=req.title, objective=req.objective)
    except KeyError:
        raise HTTPException(status_code=404, detail="Case not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/cases/{case_id}/status")
async def set_case_status(case_id: str, req: CaseStatusRequest):
    try:
        return case_store.set_case_status(case_id, workspace_id=req.workspace_id, status=req.status)
    except KeyError:
        raise HTTPException(status_code=404, detail="Case not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/cases/{case_id}/events")
async def get_case_events(case_id: str, workspace_id: str = "local", limit: int = 100):
    try:
        case_store.get_case(case_id, workspace_id=workspace_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Case not found")
    return {"case_id": case_id, "items": case_store.list_events(case_id, limit=limit)}


@app.get("/cases/{case_id}/assertions")
async def get_case_assertions(case_id: str, workspace_id: str = "local", status: str | None = None):
    statuses = [status] if status else None
    return {"case_id": case_id, "items": case_store.list_assertions(workspace_id=workspace_id, case_id=case_id, statuses=statuses)}


@app.post("/cases/{case_id}/assertions")
async def create_case_assertion(case_id: str, req: AssertionCreateRequest):
    try:
        case_store.get_case(case_id, workspace_id=req.workspace_id)
        return case_store.create_assertion(
            workspace_id=req.workspace_id, case_id=case_id, key=req.key, value=req.value,
            scope="case", provenance=req.provenance, status=req.status,
            assertion_type=req.assertion_type, display_text=req.display_text, confidence=req.confidence,
            actor_type="user",
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/assertions")
async def list_global_assertions(workspace_id: str = "local", status: str | None = None):
    statuses = [status] if status else None
    return {"workspace_id": workspace_id, "items": case_store.list_assertions(workspace_id=workspace_id, scope="global", statuses=statuses)}


@app.post("/assertions")
async def create_global_assertion(req: AssertionCreateRequest):
    if req.scope != "global":
        raise HTTPException(status_code=400, detail="Use the case assertion endpoint for case-scoped data")
    try:
        return case_store.create_assertion(
            workspace_id=req.workspace_id, case_id=None, key=req.key, value=req.value,
            scope="global", provenance=req.provenance, status=req.status,
            assertion_type=req.assertion_type, display_text=req.display_text, confidence=req.confidence,
            actor_type="user",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/assertions/{assertion_id}/status")
async def set_assertion_status(assertion_id: str, req: AssertionStatusRequest):
    try:
        return case_store.set_assertion_status(assertion_id, workspace_id=req.workspace_id, status=req.status)
    except KeyError:
        raise HTTPException(status_code=404, detail="Assertion not found")


@app.post("/assertions/{assertion_id}/correct")
async def correct_assertion(assertion_id: str, req: AssertionCorrectionRequest):
    try:
        return case_store.correct_assertion(assertion_id, workspace_id=req.workspace_id, value=req.value, display_text=req.display_text)
    except KeyError:
        raise HTTPException(status_code=404, detail="Assertion not found")


@app.post("/cases/{case_id}/tasks")
async def create_case_task(case_id: str, req: CaseTaskCreateRequest):
    try:
        return case_store.create_task(workspace_id=req.workspace_id, case_id=case_id, title=req.title, rationale=req.rationale, priority=req.priority, status=req.status, actor_type="user")
    except KeyError:
        raise HTTPException(status_code=404, detail="Case not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/cases/{case_id}/observations", status_code=201)
async def create_case_observation(case_id: str, req: CaseObservationRequest):
    try:
        return case_store.create_observation(
            case_id=case_id,
            workspace_id=req.workspace_id,
            date=req.date,
            parcel=req.parcel,
            campaign=req.campaign,
            note=req.note,
            severity=req.severity,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Case not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/case-tasks/{task_id}/status")
async def set_case_task_status(task_id: str, req: CaseTaskStatusRequest):
    try:
        return case_store.update_task(task_id, workspace_id=req.workspace_id, status=req.status)
    except KeyError:
        raise HTTPException(status_code=404, detail="Task not found")


@app.get("/memory/{user_id}", response_model=MemoryResponse)
async def get_memory(user_id: str):
    memory = memory_store.load(user_id)
    return {
        "user_id": memory.user_id,
        "sections": memory_store.editable_snapshot(user_id),
        "used_sections": memory.used_sections,
    }


@app.put("/memory/{user_id}", response_model=MemoryResponse)
async def update_memory(user_id: str, req: MemoryUpdateRequest):
    invalid = sorted(set(req.sections) - set(EDITABLE_SECTIONS))
    if invalid:
        raise HTTPException(
            status_code=400, detail=f"Secciones no permitidas: {', '.join(invalid)}"
        )
    memory = memory_store.replace_sections(user_id, req.sections)
    return {
        "user_id": memory.user_id,
        "sections": memory_store.editable_snapshot(user_id),
        "used_sections": memory.used_sections,
    }


@app.delete("/memory/{user_id}")
async def delete_memory(user_id: str):
    memory_store.delete_user(user_id)
    return {"ok": True, "user_id": user_id}


@app.get("/memory/{user_id}/list", response_model=MemoryListResponse)
async def list_memories(user_id: str):
    items = memory_store.list_memories(user_id)
    return {"user_id": user_id, "items": items}


@app.post("/memory/{user_id}/create", response_model=MemoryMeta)
async def create_memory(user_id: str, req: MemoryCreateRequest):
    try:
        meta = memory_store.create_memory(user_id, req.name)
        return meta
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.put("/memory/{user_id}/current", response_model=MemoryMeta)
async def switch_memory(user_id: str, req: MemorySwitchRequest):
    try:
        meta = memory_store.set_current_memory(user_id, req.memory_id)
        return meta
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.put("/memory/{user_id}/{memory_id}/rename", response_model=MemoryMeta)
async def rename_memory(user_id: str, memory_id: str, req: MemoryRenameRequest):
    try:
        meta = memory_store.rename_memory(user_id, memory_id, req.name)
        return meta
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.delete("/memory/{user_id}/{memory_id}")
async def delete_single_memory(user_id: str, memory_id: str):
    try:
        ok = memory_store.delete_memory(user_id, memory_id)
        return {"ok": ok, "user_id": user_id, "memory_id": memory_id}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/events/{conversation_id}")
async def events(conversation_id: str):
    async def event_generator():
        stream_id = str(uuid.uuid4())
        async for data in broker.stream(conversation_id):
            record_delivery(conversation_id, data, stream_id)
            yield {"data": data}

    return EventSourceResponse(
        event_generator(),
        headers={"Cache-Control": "no-cache"},
        media_type="text/event-stream",
    )


@app.get("/conversations")
async def list_conversations(user_id: str | None = None, limit: int = 50):
    items = conversation_store.list_conversations(limit=limit)
    if user_id:
        items = [c for c in items if c.get("user_id") == user_id]
    return {"conversations": items}


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    conv = conversation_store.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str):
    conv = conversation_store.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = conversation_store.get_messages(conversation_id)
    return {"conversation_id": conversation_id, "messages": messages}


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    ok = conversation_store.delete_conversation(conversation_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"ok": True, "conversation_id": conversation_id}
