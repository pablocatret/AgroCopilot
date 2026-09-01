from __future__ import annotations

from typing import Any, Dict, List

from loguru import logger

from agents.base import BaseAgent
from backend.deps import settings
from libs.prompts import compose_system_prompt, render_prompt
from libs.schemas import (
    AgentInput,
    AgentRef,
    AgentRefs,
    FreeAgentData,
    FreeAgentOutput,
    Reference,
)


FREE_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "findings": {"type": "string", "minLength": 1},
        "confidence": {"type": "string", "enum": ["alta", "media", "baja"]},
        "limitations": {
            "type": "array",
            "items": {"type": "string"},
            "default": [],
        },
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "url": {"type": "string"},
                    "snippet": {"type": "string", "default": ""},
                },
                "required": ["title", "url"],
            },
            "default": [],
        },
    },
    "required": ["findings", "confidence", "limitations", "sources"],
}


class FreeAgent(BaseAgent):
    name = "free"
    output_model = FreeAgentOutput
    requires_llm = True
    _provider_key = "LLM_PROVIDER_FREE"

    def __init__(self) -> None:
        super().__init__()
        self.model = settings.resolve_openai_model("OPENAI_MODEL_FREE")

    def _build_tools(self) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        from libs.search_tool import WebSearchTool

        search_tool = WebSearchTool(
            description=(
                "Busca informacion en internet. Usar para obtener datos actualizados, "
                "precios, tecnicas, normativa vigente, variedades, plagas, enfermedades, "
                "o cualquier informacion factual que requiera fuentes externas."
            ),
            max_fetch=5,
            do_fetch=True,
        )
        tools = [search_tool.tool_spec()]
        tool_map: Dict[str, Any] = {"web_search": search_tool.run}
        return tools, tool_map

    async def _run(self, agent_input: AgentInput) -> FreeAgentOutput:
        ctx = agent_input.context or {}
        task_description = str(ctx.get("task_description") or "").strip()
        if not task_description:
            task_description = agent_input.query

        memory_summary = ""
        user_memory = ctx.get("user_memory")
        if user_memory:
            memory_summary = str(user_memory).strip()[:600]

        conversation_history = ctx.get("_conversation_history")
        history_block = ""
        if conversation_history and isinstance(conversation_history, list):
            recent = conversation_history[-6:]
            lines = []
            for msg in recent:
                role = str(msg.get("role", "")).strip()
                content = str(msg.get("content", "")).strip()
                if role and content:
                    lines.append(f"- {role}: {content[:200]}")
            if lines:
                history_block = "\n".join(lines)

        user_prompt = render_prompt(
            "free_user.txt",
            task_description=task_description,
            query=agent_input.query,
            memory_summary=memory_summary or "Sin memoria previa.",
            conversation_history=history_block or "Sin historial previo.",
        )

        tools, tool_map = self._build_tools()

        if self.external_enabled():
            try:
                parsed = await self.call_llm_json_with_tools(
                    system=compose_system_prompt(
                        agent_name="free",
                        body=render_prompt("free_system.txt"),
                        output_contract=(
                            "Devuelve exclusivamente JSON valido con findings, confidence, limitations y sources. "
                            "No expliques el proceso ni menciones agentes internos."
                        ),
                    ),
                    user=user_prompt,
                    schema=FREE_RESPONSE_SCHEMA,
                    tools=tools,
                    tool_map=tool_map,
                    temperature=0.3,
                    max_iterations=5,
                )
            except Exception as exc:
                logger.bind(agent="free").warning("LLM call failed: {}", exc)
                parsed = {}
        else:
            parsed = {}

        findings = str(parsed.get("findings") or "").strip()
        confidence = str(parsed.get("confidence") or "media").strip()
        if confidence not in {"alta", "media", "baja"}:
            confidence = "media"
        limitations = [
            str(item).strip()
            for item in (parsed.get("limitations") or [])
            if isinstance(item, str) and item.strip()
        ]

        raw_sources = parsed.get("sources") or []
        refs: List[AgentRef] = []
        sources: List[Reference] = []
        for idx, src in enumerate(raw_sources, start=1):
            if not isinstance(src, dict):
                continue
            title = str(src.get("title") or "").strip()
            url = str(src.get("url") or "").strip()
            snippet = str(src.get("snippet") or "").strip()
            if not url:
                continue
            refs.append(
                AgentRef(
                    ref_id=f"free-web-{idx}",
                    title=title or f"Fuente {idx}",
                    source="web",
                    url=url,
                    snippet=snippet,
                )
            )
            sources.append(Reference(title=title or f"Fuente {idx}", url=url, snippet=snippet))

        if not findings:
            findings = (
                "No se pudieron obtener hallazgos para la tarea asignada. "
                "Consulte con el organizer para reintentar con otra estrategia."
            )
            confidence = "baja"

        summary = findings.split("\n\n")[0][:300].strip()
        if not summary:
            summary = "Investigacion general completada."

        return FreeAgentOutput(
            agent=self.name,
            summary=summary,
            refs=AgentRefs(items=refs),
            data=FreeAgentData(
                findings=findings,
                sources=sources,
                confidence=confidence,
                limitations=limitations,
            ),
        )
