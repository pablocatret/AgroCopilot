from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Tuple, Type

from openai import AsyncOpenAI
from loguru import logger

from agents.base import BaseAgent, _build_client, openai_strict_json_schema, validate_openai_strict_json_schema
from agents.case_manager import CaseManagerAgent
from agents.document_analyst import DocumentAnalystAgent
from agents.free import FreeAgent
from agents.legal import LegalAgent
from agents.rs_analyst import RSAnalystAgent
from agents.spreadsheet_analyst import SpreadsheetAnalystAgent
from agents.stac_search import StacSearchAgent
from agents.vision_ocr import VisionOcrAgent
from agents.writer import WriterAgent
from backend.deps import require_openai_key, settings
from libs.costs.tracker import cost_context, record_openai_chat_usage
from libs.robust_json import JsonParseError, extract_llm_content, parse_json_content
from libs.openai_compat import chat_temperature_kwargs, completion_token_kwargs
from libs.context_engineering import (
    summarize_attachments,
    summarize_case_history,
    summarize_memory_context,
    summarize_memory_reuse,
    summarize_monitoring_signal,
    summarize_observations,
)
from libs.prompts import compose_system_prompt, render_prompt
from libs.schemas import (
    AgentInput,
    AgentPlan,
    ClarificationOption,
    ClarificationRequest,
    EffectivePlanPolicy,
    MissionEntry,
    PlanDiagnostics,
    PlanPolicy,
    WriterFastPathPolicy,
)


def _normalize_spanish(text: str) -> str:
    """Remove accents: 'certificación' -> 'certificacion' for regex matching."""
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    return normalized.encode("ascii", "ignore").decode()


@dataclass(frozen=True)
class AgentSpec:
    name: str
    cls: Type[BaseAgent]
    description: str
    enabled: Callable[[], bool] = lambda: True


def _stac_enabled() -> bool:
    return settings.ENABLE_STAC


def _rs_enabled() -> bool:
    return settings.ENABLE_STAC and settings.ENABLE_RS_ANALYST


AGENT_SPECS: Tuple[AgentSpec, ...] = (
    AgentSpec(
        name="legal",
        cls=LegalAgent,
        description=(
            "Recuperacion legal: obtiene normativa agricola y fuentes oficiales para alimentar "
            "decisiones documentales o de cumplimiento."
        ),
    ),
    AgentSpec(
        name="case_manager",
        cls=CaseManagerAgent,
        description=(
            "Solucion y seguimiento: usa memoria, observaciones y evidencias para mantener el caso "
            "abierto, priorizar bloqueos y definir el siguiente dato util."
        ),
    ),
    AgentSpec(
        name="stac",
        cls=StacSearchAgent,
        description=(
            "Recuperacion satelital: busca escenas STAC, indices y previsualizaciones cuando la "
            "consulta requiere evidencia remota."
        ),
        enabled=_stac_enabled,
    ),
    AgentSpec(
        name="rs_analyst",
        cls=RSAnalystAgent,
        description=(
            "Solucion satelital: interpreta escenas e indices ya recuperados para detectar cambios, "
            "zonas de revision y limites de confianza."
        ),
        enabled=_rs_enabled,
    ),
    AgentSpec(
        name="document_analyst",
        cls=DocumentAnalystAgent,
        description="Recuperacion documental: extrae texto y datos utiles de PDF/DOC/DOCX/TXT.",
    ),
    AgentSpec(
        name="spreadsheet_analyst",
        cls=SpreadsheetAnalystAgent,
        description="Recuperacion tabular: perfila CSV/XLS/XLSX y extrae columnas, faltantes y muestras.",
    ),
    AgentSpec(
        name="vision_ocr",
        cls=VisionOcrAgent,
        description="Recuperacion visual: extrae texto y senales utiles de imagenes o escaneos.",
    ),
    AgentSpec(
        name="free",
        cls=FreeAgent,
        description=(
            "Investigacion general: busca informacion en internet y razona sobre tareas "
            "asignadas por el organizer que no encajan en agentes especializados."
        ),
    ),
    AgentSpec(
        name="writer",
        cls=WriterAgent,
        description=(
            "Respuesta final: convierte informacion y razonamiento disponible en respuesta directa "
            "y accionable para el usuario."
        ),
    ),
)
ORGANIZER_PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "steps": {
            "type": "array",
            "items": {"type": "string"},
            "default": [],
        },
        "writer_mode": {
            "type": "string",
            "enum": ["BRIEFING", "STANDARD", "DEEP_DIVE"],
        },
        "dependencies": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "legal": {"type": "array", "items": {"type": "string"}, "default": []},
                "case_manager": {"type": "array", "items": {"type": "string"}, "default": []},
                "stac": {"type": "array", "items": {"type": "string"}, "default": []},
                "rs_analyst": {"type": "array", "items": {"type": "string"}, "default": []},
                "document_analyst": {"type": "array", "items": {"type": "string"}, "default": []},
                "spreadsheet_analyst": {"type": "array", "items": {"type": "string"}, "default": []},
                "vision_ocr": {"type": "array", "items": {"type": "string"}, "default": []},
                "free": {"type": "array", "items": {"type": "string"}, "default": []},
                "writer": {"type": "array", "items": {"type": "string"}, "default": []},
            },
            "required": [
                "legal",
                "case_manager",
                "stac",
                "rs_analyst",
                "document_analyst",
                "spreadsheet_analyst",
                "vision_ocr",
                "free",
                "writer",
            ],
        },
        "allow_replan": {"type": "boolean"},
        "rationale": {"type": "string"},
        "missions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "agent": {"type": "string"},
                    "instruction": {"type": "string"},
                },
                "required": ["agent", "instruction"],
                "additionalProperties": False,
            },
            "default": [],
        },
        "policy": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "allow_retries": {"type": "boolean"},
                "max_rounds": {"type": "integer"},
                "retry_candidates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                },
                "writer_search_allowed": {"type": "boolean"},
                "fast_path": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "allow_search": {"type": "boolean"},
                        "disclose_search_use": {"type": "boolean"},
                        "disclose_sources": {"type": "boolean"},
                        "escalate_when_specialized": {"type": "boolean"},
                    },
                    "required": [
                        "enabled",
                        "allow_search",
                        "disclose_search_use",
                        "disclose_sources",
                        "escalate_when_specialized",
                    ],
                },
            },
            "required": [
                "allow_retries",
                "max_rounds",
                "retry_candidates",
                "writer_search_allowed",
                "fast_path",
            ],
        },
    },
    "required": ["steps", "writer_mode", "dependencies", "allow_replan", "rationale", "policy"],
}

ORGANIZER_REPLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "extra_steps": {
            "type": "array",
            "items": {"type": "string"},
            "default": [],
        },
        "writer_mode": {
            "type": "string",
            "enum": ["BRIEFING", "STANDARD", "DEEP_DIVE"],
        },
        "rationale": {"type": "string"},
        "missions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "agent": {"type": "string"},
                    "instruction": {"type": "string"},
                },
                "required": ["agent", "instruction"],
                "additionalProperties": False,
            },
            "default": [],
        },
        "stop": {"type": "boolean"},
    },
    "required": ["extra_steps", "writer_mode", "rationale", "stop"],
}

CLARIFICATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "question": {"type": "string"},
        "options": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "label": {"type": "string"},
                    "description": {"type": "string", "default": ""},
                    "enriched_query": {"type": "string"},
                },
                "required": ["key", "label", "enriched_query"],
            },
            "minItems": 2,
            "maxItems": 4,
        },
        "rationale": {"type": "string", "default": ""},
    },
    "required": ["question", "options"],
}


# Palabras clave para fallback determinista
KW_LEGAL = re.compile(
    r"\b(organico|ecologico|reglament\w*|normativ\w*|ley|globalg\.?a\.?p|ifa|certificacion|etiquetado|mrl|residuos|exportacion|pasaporte|trazabilidad|fitosanitari\w*|pesticida[s]?|herbicida[s]?|fungicida[s]?|insecticida[s]?)\b",
    re.I,
)
KW_STAC = re.compile(
    r"\b(sentinel|landsat|stac|satelit|nube[s]?|cloud|bbox|coordenad|teledeteccion|imagen)\b",
    re.I,
)
KW_MONITORING = re.compile(
    r"\b(seguimiento|evolucion|comparar|antes|ahora|serie temporal|temporal|parcela|campana|revisar de nuevo|cambio[s]?)\b",
    re.I,
)
KW_SIMPLE_SINGLE_AGENT = re.compile(
    r"^(que|qué|como|cómo|cuando|cuándo|cuanto|cuánto|cual|cuál|donde|dónde|puedo|debo|merece|conviene)\b",
    re.I,
)


def agent_registry(include_disabled: bool = False) -> Dict[str, AgentSpec]:
    specs = AGENT_SPECS if include_disabled else [spec for spec in AGENT_SPECS if spec.enabled()]
    return {spec.name: spec for spec in specs}


def _available_agents() -> List[str]:
    return [spec.name for spec in AGENT_SPECS if spec.enabled()]


def _agent_tools_hint(name: str) -> str:
    tools_by_agent = {
        "legal": "Legalize RAG; conditional web_search for official sources/currency",
        "case_manager": "consolidated context and memory; no external tools",
        "stac": "geocode_place, search_satellite_images, inspect_region",
        "rs_analyst": "deterministic StacResults analysis; no external tools",
        "document_analyst": "local PDF/DOC/DOCX/TXT extraction + LLM enrichment",
        "spreadsheet_analyst": "local CSV/XLS/XLSX profiling + LLM enrichment",
        "vision_ocr": "local PNG/JPG/TIF OCR + LLM enrichment",
        "free": "web_search for general research; LLM reasoning on assigned tasks",
        "writer": "single-agent fast path with targeted web search and final synthesis; no specialized tools",
    }
    return tools_by_agent.get(name, "no tools declared")


def _agent_catalog(candidates: List[str]) -> str:
    registry = agent_registry()
    lines = []
    for name in candidates:
        spec = registry.get(name)
        if spec:
            lines.append(f"- {name}: {spec.description} Tools: {_agent_tools_hint(name)}.")
    return "\n".join(lines) if lines else "No agents available."


def _enforce_rules(steps: List[str]) -> List[str]:
    """Normaliza orden y consistencia: sin duplicados, respeta flags y writer al final."""
    candidates = set(_available_agents())
    steps = [step for step in steps if step in candidates]
    seen = set()
    uniq = []
    for step in steps:
        if step not in seen:
            seen.add(step)
            uniq.append(step)
    if "rs_analyst" in uniq and "stac" not in uniq:
        uniq = [step for step in uniq if step != "rs_analyst"]
    # Solo se añade case_manager si fue solicitado explícitamente en steps
    has_case_manager = "case_manager" in candidates and "case_manager" in steps
    uniq = [step for step in uniq if step not in {"writer", "case_manager"}]
    if has_case_manager:
        uniq.append("case_manager")
    uniq = [step for step in uniq if step != "writer"] + ["writer"]
    return uniq


def _default_dependencies(steps: List[str]) -> Dict[str, List[str]]:
    step_set = set(steps)
    deps: Dict[str, List[str]] = {}
    if "rs_analyst" in step_set and "stac" in step_set:
        deps["rs_analyst"] = ["stac"]
    if "case_manager" in step_set:
        deps["case_manager"] = [
            dep
            for dep in [
                "legal",
                "document_analyst",
                "spreadsheet_analyst",
                "vision_ocr",
                "stac",
                "rs_analyst",
            ]
            if dep in step_set
        ]
    if "writer" in step_set:
        deps["writer"] = [step for step in steps if step != "writer"]
    return deps


def _remote_sensing_reuse_status(context: Dict[str, Any] | None) -> str:
    context = context or {}
    raw = context.get("_memory_reuse")
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump(exclude_none=True)
    if not isinstance(raw, dict):
        return "miss"
    rs = raw.get("remote_sensing")
    if hasattr(rs, "model_dump"):
        rs = rs.model_dump(exclude_none=True)
    if not isinstance(rs, dict):
        return "miss"
    return str(rs.get("status") or "miss").strip().lower()


class OrganizerAgent:
    """
    Planificador con LLM + fallback determinista.
    - plan(): plan inicial minimo y suficiente.
    - replan(): pasos adicionales en funcion de observaciones/salidas previas.
    """

    def __init__(self) -> None:
        self._provider = settings.resolve_provider("LLM_PROVIDER_ORGANIZER")
        self.client = _build_client(self._provider) if settings.OPENAI_API_KEY else None
        self.model = settings.resolve_openai_model("OPENAI_MODEL_ORGANIZER")
        self.evaluation_temperature: float | None = None
        self.evaluation_max_tokens: int | None = None

    def _system_plan(self, candidates: List[str]) -> str:
        stac_hint = "STAC IS enabled" if settings.ENABLE_STAC else "STAC is NOT enabled"
        sentinel_example = (
            '{"steps": ["stac","rs_analyst","writer"], "writer_mode": "STANDARD", "dependencies": {"rs_analyst": ["stac"], "writer": ["stac","rs_analyst"]}, "allow_replan": false, "rationale": "La consulta requiere recuperar escenas y luego interpretarlas.", "policy": {"allow_retries": false, "max_rounds": 0, "retry_candidates": [], "writer_search_allowed": false, "fast_path": {"enabled": false, "allow_search": false, "disclose_search_use": true, "disclose_sources": true, "escalate_when_specialized": true}}}'
            if settings.ENABLE_STAC
            else '{"steps": ["writer"], "writer_mode": "BRIEFING", "dependencies": {"writer": []}, "allow_replan": false, "rationale": "STAC no esta habilitado y solo procede respuesta directa.", "policy": {"allow_retries": false, "max_rounds": 0, "retry_candidates": [], "writer_search_allowed": true, "fast_path": {"enabled": true, "allow_search": true, "disclose_search_use": true, "disclose_sources": true, "escalate_when_specialized": true}}}'
        )
        return compose_system_prompt(
            agent_name="organizer",
            body=render_prompt(
                "organizer_plan.txt",
                candidates=", ".join(candidates),
                stac_hint=stac_hint,
                sentinel_example=sentinel_example,
                agent_catalog=_agent_catalog(candidates),
            ),
            output_contract=(
                "Devuelve exclusivamente JSON valido con steps, writer_mode, dependencies, allow_replan, rationale y policy. "
                "No anadas explicacion fuera del objeto."
            ),
        )

    def _system_replan(self, candidates: List[str]) -> str:
        return compose_system_prompt(
            agent_name="organizer",
            body=render_prompt(
                "organizer_replan.txt",
                candidates=", ".join(candidates),
                agent_catalog=_agent_catalog(candidates),
            ),
            output_contract=(
                "Devuelve exclusivamente JSON valido con extra_steps, writer_mode, rationale y stop. "
                "No anadas explicacion fuera del objeto."
            ),
        )

    def _fallback_steps(
        self,
        q: str,
        *,
        attachments: List[dict] | None = None,
        monitoring_hint: bool = False,
        context: Dict[str, Any] | None = None,
    ) -> List[str]:
        steps: List[str] = []
        q_normalized = _normalize_spanish(q)
        reuse_status = _remote_sensing_reuse_status(context)
        if KW_LEGAL.search(q_normalized):
            steps.append("legal")
        if monitoring_hint and settings.ENABLE_STAC and reuse_status != "hit":
            steps.append("stac")
            if settings.ENABLE_RS_ANALYST:
                steps.append("rs_analyst")
        attachments = attachments or []
        for att in attachments:
            ctype = (att.get("content_type") or "").lower()
            fname = (att.get("filename") or "").lower()
            if "sheet" in ctype or fname.endswith((".csv", ".xlsx", ".xls")):
                steps.append("spreadsheet_analyst")
            elif "pdf" in ctype or "html" in ctype or fname.endswith(
                (".pdf", ".doc", ".docx", ".txt", ".html", ".htm")
            ):
                steps.append("document_analyst")
            elif "image" in ctype or fname.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                steps.append("vision_ocr")
        if settings.ENABLE_STAC and KW_STAC.search(q_normalized) and reuse_status != "hit":
            steps.append("stac")
            if settings.ENABLE_RS_ANALYST:
                steps.append("rs_analyst")
        # Incluir case_manager solo si se identificaron otros agentes
        if len(steps) > 0 and "case_manager" in _available_agents():
            steps.append("case_manager")
        if not steps:
            steps.append("writer")
        else:
            steps.append("writer")
        return _enforce_rules(steps)

    def _steps_for_decision_mode(
        self,
        decision_mode: str,
        attachments: List[dict] | None = None,
        *,
        query: str = "",
        context: Dict[str, Any] | None = None,
    ) -> List[str]:
        attachments = attachments or []
        context = context or {}
        observations = context.get("observations") if isinstance(context.get("observations"), list) else []
        monitoring_hint = bool(observations) or bool(KW_MONITORING.search(query))
        if decision_mode == "case":
            return self._fallback_steps(
                query,
                attachments=attachments,
                monitoring_hint=monitoring_hint,
                context=context,
            )
        return []

    def _fallback_replan_steps(
        self,
        user_query: AgentInput,
        observations: Dict[str, Any],
    ) -> List[str]:
        query = user_query.query.strip()
        attachments = (
            [a.model_dump() for a in user_query.attachments]
            if getattr(user_query, "attachments", None)
            else []
        )
        candidates = set(_available_agents())

        def execution_level(agent: str) -> str:
            payload = observations.get(agent)
            if not isinstance(payload, dict):
                return ""
            execution = payload.get("execution")
            if not isinstance(execution, dict):
                return ""
            return str(execution.get("final_level", "")).strip().lower()

        steps: list[str] = []
        if (
            KW_LEGAL.search(query)
            and "legal" in candidates
            and execution_level("legal") in {"soft_error", "hard_error", "insufficient_data"}
        ):
            steps.append("legal")

        monitoring_like = bool(KW_MONITORING.search(query) or KW_STAC.search(query))
        if monitoring_like and "stac" in candidates:
            stac_level = execution_level("stac")
            rs_level = execution_level("rs_analyst")
            if stac_level in {"soft_error", "hard_error", "insufficient_data"}:
                steps.append("stac")
            if (
                settings.ENABLE_RS_ANALYST
                and "rs_analyst" in candidates
                and rs_level in {"soft_error", "hard_error", "insufficient_data"}
            ):
                steps.extend(["stac", "rs_analyst"])

        if attachments:
            for att in attachments:
                ctype = (att.get("content_type") or "").lower()
                fname = (att.get("filename") or "").lower()
                if (
                    ("sheet" in ctype or fname.endswith((".csv", ".xlsx", ".xls")))
                    and "spreadsheet_analyst" in candidates
                    and execution_level("spreadsheet_analyst")
                    in {"soft_error", "hard_error", "insufficient_data"}
                ):
                    steps.append("spreadsheet_analyst")
                elif (
                    ("pdf" in ctype or fname.endswith((".pdf", ".doc", ".docx", ".txt")))
                    and "document_analyst" in candidates
                    and execution_level("document_analyst")
                    in {"soft_error", "hard_error", "insufficient_data"}
                ):
                    steps.append("document_analyst")
                elif (
                    ("image" in ctype or fname.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")))
                    and "vision_ocr" in candidates
                    and execution_level("vision_ocr")
                    in {"soft_error", "hard_error", "insufficient_data"}
                ):
                    steps.append("vision_ocr")

        return _enforce_rules(steps) if steps else []

    def _normalize_writer_mode(self, value: Any) -> Literal["BRIEFING", "STANDARD", "DEEP_DIVE"]:
        if not isinstance(value, str):
            return "STANDARD"
        normalized = value.strip().upper()
        if normalized in {"BRIEFING", "STANDARD", "DEEP_DIVE"}:
            return normalized
        return "STANDARD"

    def _fast_path_policy(self, *, enabled: bool, allow_search: bool) -> WriterFastPathPolicy:
        return WriterFastPathPolicy(
            enabled=enabled,
            allow_search=allow_search,
            disclose_search_use=True,
            disclose_sources=True,
            escalate_when_specialized=True,
        )

    def _coerce_policy(self, value: Any) -> EffectivePlanPolicy:
        if not isinstance(value, dict):
            return EffectivePlanPolicy()
        retry_candidates = [
            str(item).strip().lower()
            for item in (value.get("retry_candidates") or [])
            if str(item).strip()
        ]
        try:
            max_rounds = max(0, min(int(value.get("max_rounds", 0)), 3))
        except (TypeError, ValueError):
            max_rounds = 0
        allow_retries = bool(value.get("allow_retries")) and max_rounds > 0
        writer_search_allowed = bool(value.get("writer_search_allowed"))
        raw_fast_path = value.get("fast_path") if isinstance(value.get("fast_path"), dict) else {}
        fast_path = WriterFastPathPolicy(
            enabled=bool(raw_fast_path.get("enabled")),
            allow_search=bool(
                raw_fast_path.get("allow_search", writer_search_allowed)
            ),
            disclose_search_use=bool(raw_fast_path.get("disclose_search_use", True)),
            disclose_sources=bool(raw_fast_path.get("disclose_sources", True)),
            escalate_when_specialized=bool(
                raw_fast_path.get("escalate_when_specialized", True)
            ),
        )
        if writer_search_allowed and not fast_path.enabled:
            fast_path.enabled = True
        if fast_path.allow_search and not writer_search_allowed:
            writer_search_allowed = True
        return EffectivePlanPolicy(
            allow_retries=allow_retries,
            max_rounds=max_rounds,
            retry_candidates=retry_candidates,
            writer_search_allowed=writer_search_allowed,
            fast_path=fast_path,
        )

    async def _chat_json(
        self, system: str, user: str, *, schema: Dict[str, Any], schema_name: str
    ) -> Dict[str, Any]:
        if settings.DISABLE_EXTERNALS or not self.client:
            return {}
        strict_schema = openai_strict_json_schema(schema)
        schema_errors = validate_openai_strict_json_schema(strict_schema)
        if schema_errors:
            raise ValueError(
                f"Schema incompatible con Structured Outputs: {'; '.join(schema_errors)}"
            )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        limits = completion_token_kwargs(
            self.model,
            self._provider,
            self.evaluation_max_tokens or 2048,
        )
        with cost_context(agent="organizer", operation=f"organizer.{schema_name}"):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema_name,
                            "strict": True,
                            "schema": strict_schema,
                        },
                    },
                    **chat_temperature_kwargs(
                        self.model,
                        0.0 if self.evaluation_temperature is None else self.evaluation_temperature,
                    ),
                    **limits,
                )
            except Exception as structured_exc:
                logger.warning(
                    "organizer.structured_output_failed",
                    model=self.model,
                    schema=schema_name,
                    error=str(structured_exc),
                )
                fallback_messages = [
                    {"role": "system", "content": system + "\n\nDevuelve únicamente JSON válido, sin Markdown ni texto adicional."},
                    {"role": "user", "content": user + "\n\nEsquema JSON requerido:\n" + json.dumps(strict_schema, ensure_ascii=False)},
                ]
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=fallback_messages,
                    **chat_temperature_kwargs(
                        self.model,
                        0.0 if self.evaluation_temperature is None else self.evaluation_temperature,
                    ),
                    **limits,
                )
            if getattr(resp, "usage", None) is not None:
                record_openai_chat_usage(
                    self.model, resp.usage, operation=f"organizer.{schema_name}",
                    provider=self._provider,
                )
        content, _, _ = extract_llm_content(resp)
        try:
            return parse_json_content(content, expected="object").value
        except JsonParseError as exc:
            raise RuntimeError(f"Respuesta de planificación ilegible: {exc}") from exc

    def _is_conversational_or_simple(self, q: str, attachments: List[dict]) -> bool:
        if attachments:
            return False
        clean_q = q.strip().lower()
        for punc in ["?", "!", ",", ".", ";", ":"]:
            clean_q = clean_q.replace(punc, "")
        clean_q = clean_q.strip()
        conversational_terms = {
            "hola", "buenos dias", "buenas tardes", "buenas noches", "que tal", "como estas",
            "gracias", "muchas gracias", "de nada", "adios", "hasta luego", "saludos",
            "hola copiloto", "hola asistente", "hola de nuevo"
        }
        if clean_q in conversational_terms:
            return True
        words = clean_q.split()
        if len(words) <= 2 and any(w in conversational_terms for w in words):
            return True
        return False

    def _is_single_agent_fast_path_candidate(
        self,
        q: str,
        *,
        attachments: List[dict],
        context: Dict[str, Any] | None = None,
    ) -> bool:
        if attachments:
            return False
        context = context or {}
        if context.get("observations") or context.get("case_history"):
            return False
        q_normalized = _normalize_spanish(q)
        if KW_LEGAL.search(q_normalized) or KW_STAC.search(q_normalized) or KW_MONITORING.search(q_normalized):
            return False
        clean_q = q.strip()
        if not clean_q:
            return False
        words = re.findall(r"\w+", clean_q.lower())
        if len(words) < 3 or len(words) > 18:
            return False
        return clean_q.endswith("?") or bool(KW_SIMPLE_SINGLE_AGENT.search(clean_q))

    _AMBIGUITY_KEYWORDS = re.compile(
        r"\b(satélite|sentinel|ndvi|legal|pac|certificación|riego|fertiliz|plaga|cosecha"
        r"|olivar|viña|documento|pdf|informe|comparar|seguimiento|satellite|irrigation|pest"
        r"|harvest|olive|vineyard|canopy|vigour|vigor|biomass|moisture)\b",
        re.I,
    )

    def _detect_ambiguity(
        self,
        query: str,
        attachments: List[dict],
        context: Dict[str, Any],
    ) -> str | None:
        q = query.strip()
        words = re.findall(r"\w+", q.lower())
        if len(words) >= 8:
            return None
        if attachments:
            return None
        if context.get("case_history") or context.get("observations"):
            return None
        q_normalized = _normalize_spanish(q)
        if self._AMBIGUITY_KEYWORDS.search(q_normalized):
            return None
        if len(words) <= 1:
            return "Query demasiado corta para determinar la intención del usuario"
        return "Query sin dominio específico detectado; necesito saber qué análisis necesita"

    def _fallback_clarification(self, query: str) -> ClarificationRequest:
        q_lower = query.strip().lower()
        if any(w in q_lower for w in ("parcela", "campo", "terreno", "finca")):
            label = "la parcela"
        elif any(w in q_lower for w in ("olivo", "olivar", "aceite")):
            label = "el olivar"
        elif any(w in q_lower for w in ("viña", "viñedo", "uva", "vino")):
            label = "la viña"
        else:
            label = "la consulta"
        return ClarificationRequest(
            question=f"¿Qué tipo de análisis necesitas para {label}?",
            options=[
                ClarificationOption(
                    key="satellite",
                    label="Análisis satelital",
                    description="Búsqueda de escenas Sentinel-2, evolución NDVI, vigor del cultivo",
                    enriched_query=f"Realiza un análisis satelital completo de {label}: búsqueda de escenas recientes, evolución NDVI y assessment de vigor del cultivo",
                ),
                ClarificationOption(
                    key="legal",
                    label="Revisión normativa",
                    description="PAC, certificaciones orgánicas, cumplimiento normativo",
                    enriched_query=f"Revisa la normativa y requisitos aplicables a {label}: PAC, certificaciones, cumplimiento legal vigente",
                ),
                ClarificationOption(
                    key="general",
                    label="Consejo agronómico",
                    description="Riego, fertilización, manejo integrado de plagas",
                    enriched_query=f"Proporciona consejo agronómico para {label}: plan de riego, fertilización y manejo integrado de plagas",
                ),
            ],
            rationale="Fallback: query ambigua sin contexto suficiente para determinar dominio",
        )

    async def _request_clarification(
        self,
        query: str,
        reason: str,
    ) -> ClarificationRequest | None:
        if settings.DISABLE_EXTERNALS or not self.client:
            return self._fallback_clarification(query)
        try:
            obj = await self._chat_json(
                compose_system_prompt(
                    agent_name="organizer",
                    body=render_prompt(
                        "organizer_clarify.txt",
                        query=query,
                        ambiguity_reason=reason,
                    ),
                    output_contract=(
                        "Return a JSON object with 'question' (string) and "
                        "'options' (array of {key, label, description, enriched_query})."
                    ),
                ),
                user=query,
                schema=CLARIFICATION_SCHEMA,
                schema_name="organizer_clarification",
            )
            if obj and obj.get("question") and obj.get("options"):
                opts = []
                for o in obj["options"]:
                    if isinstance(o, dict) and o.get("key") and o.get("label") and o.get("enriched_query"):
                        opts.append(ClarificationOption(**o))
                if len(opts) >= 2:
                    return ClarificationRequest(
                        question=obj["question"],
                        options=opts,
                        rationale=obj.get("rationale", reason),
                    )
        except Exception:
            pass
        return self._fallback_clarification(query)

    async def plan(
        self,
        user_query: AgentInput,
    ) -> AgentPlan:
        q = user_query.query.strip()
        candidates = _available_agents()
        attachments = (
            [a.model_dump() for a in user_query.attachments]
            if getattr(user_query, "attachments", None)
            else []
        )
        
        # 1. Vía rápida determinista para saludos y mensajes conversacionales simples
        if self._is_conversational_or_simple(q, attachments):
            steps = _enforce_rules(["writer"])
            return AgentPlan(
                steps=steps,
                runs={step: 1 for step in steps},
                dependencies=_default_dependencies(steps),
                allow_replan=False,
                writer_mode="BRIEFING",
                response_mode=user_query.response_mode,
                policy=EffectivePlanPolicy(
                    writer_search_allowed=False,
                    fast_path=self._fast_path_policy(enabled=True, allow_search=False),
                ),
                diagnostics=PlanDiagnostics(
                    planner_source="simple_conversation",
                    rationale="Consulta conversacional simple sin necesidad de recuperar evidencia adicional.",
                ),
            )
        if self._is_single_agent_fast_path_candidate(
            q,
            attachments=attachments,
            context=user_query.context,
        ):
            steps = _enforce_rules(["writer"])
            return AgentPlan(
                steps=steps,
                runs={step: 1 for step in steps},
                dependencies=_default_dependencies(steps),
                allow_replan=False,
                writer_mode="BRIEFING",
                response_mode=user_query.response_mode,
                policy=EffectivePlanPolicy(
                    writer_search_allowed=True,
                    fast_path=self._fast_path_policy(enabled=True, allow_search=True),
                ),
                diagnostics=PlanDiagnostics(
                    planner_source="heuristic",
                    fallback_reason="single_agent_fast_path",
                    rationale=(
                        "Consulta breve sin adjuntos ni senales de necesidad especializada; "
                        "se habilita writer como fast path con busqueda acotada si procede."
                    ),
                ),
            )

        # 1b. Detección de ambigüedad: si la query es vaga, pedir clarificación
        # Solo cuando el LLM está disponible para generar opciones inteligentes
        if not settings.DISABLE_EXTERNALS and self.client:
            ambiguity_reason = self._detect_ambiguity(q, attachments, user_query.context)
            if ambiguity_reason:
                clarification = await self._request_clarification(q, ambiguity_reason)
                if clarification:
                    return AgentPlan(
                        steps=[],
                        runs={},
                        dependencies={},
                        allow_replan=False,
                        response_mode=user_query.response_mode,
                        clarification=clarification,
                        diagnostics=PlanDiagnostics(
                            planner_source="heuristic",
                            rationale=f"Ambigüedad detectada: {ambiguity_reason}",
                        ),
                    )

        # 2. Planificación dinámica por LLM (si está disponible y habilitada)
        obj = {}
        planner_source = "heuristic"
        fallback_reason: str | None = None
        planner_error_detail = ""
        if not settings.DISABLE_EXTERNALS and self.client:
            try:
                obj = await self._chat_json(
                    self._system_plan(candidates),
                    render_prompt(
                        "organizer_user.txt",
                        query=q,
                        decision_mode=user_query.decision_mode,
                        attachments_summary=summarize_attachments(user_query.attachments),
                        memory_summary=summarize_memory_context(
                            str(user_query.context.get("user_memory", "") or "")
                        ),
                        case_history_summary=summarize_case_history(
                            user_query.context.get("case_history", [])
                        ),
                        observations_summary=summarize_observations(
                            user_query.context.get("observations", [])
                        ),
                        monitoring_summary=summarize_monitoring_signal(
                            q,
                            user_query.context.get("observations", []),
                            user_query.context.get("case_history", []),
                            language=user_query.language,
                        ),
                        memory_reuse_summary=summarize_memory_reuse(
                            user_query.context.get("_memory_reuse")
                        ),
                    ),
                    schema=ORGANIZER_PLAN_SCHEMA,
                    schema_name="organizer_plan",
                )
                if obj:
                    planner_source = "llm"
            except Exception as exc:
                obj = {}
                fallback_reason = "llm_plan_error"
                planner_error_detail = f"{type(exc).__name__}: {exc}"[:500]
                logger.exception(
                    "organizer.plan_failed",
                    model=self.model,
                    error=planner_error_detail,
                )

        # 3. Fallback en caso de fallo del LLM o planificación deshabilitada/vacía
        steps = obj.get("steps") or []
        if not isinstance(steps, list) or not steps:
            if fallback_reason is None:
                fallback_reason = (
                    "llm_plan_empty" if planner_source == "llm" else "llm_planning_disabled"
                )
            # Intentar obtener pasos para el modo de decisión actual
            steps = self._steps_for_decision_mode(
                user_query.decision_mode,
                attachments,
                query=q,
                context=user_query.context,
            )
            # Si no se define nada, recurrir al fallback de reglas general
            if not steps:
                steps = self._fallback_steps(
                    q,
                    attachments=attachments,
                    monitoring_hint=bool(user_query.context.get("observations"))
                    and bool(KW_MONITORING.search(q) or user_query.decision_mode == "case"),
                    context=user_query.context,
                )

        steps = [str(step).strip().lower() for step in steps]
        steps = _enforce_rules(steps)
        writer_mode = self._normalize_writer_mode(obj.get("writer_mode"))
        policy = self._coerce_policy(obj.get("policy"))
        if steps == ["writer"]:
            policy.fast_path.enabled = True
            policy.fast_path.allow_search = bool(policy.writer_search_allowed)
            policy.writer_search_allowed = bool(policy.fast_path.allow_search)
        else:
            policy.fast_path.enabled = False
            policy.fast_path.allow_search = False
        raw_dependencies = obj.get("dependencies") if isinstance(obj.get("dependencies"), dict) else {}
        dependencies: Dict[str, List[str]] = {}
        for step, deps in raw_dependencies.items():
            if step in steps and isinstance(deps, list):
                dependencies[step] = [str(dep).strip().lower() for dep in deps if str(dep).strip().lower() in steps]
        defaults = _default_dependencies(steps)
        for step, deps in defaults.items():
            dependencies.setdefault(step, deps)
        raw_missions = obj.get("missions") if isinstance(obj.get("missions"), list) else []
        missions = [
            MissionEntry(agent=str(m.get("agent", "")).strip().lower(), instruction=str(m.get("instruction", "")))
            for m in raw_missions
            if isinstance(m, dict) and str(m.get("agent", "")).strip().lower() in steps
            and str(m.get("agent", "")).strip().lower() != "writer"
        ]
        return AgentPlan(
            steps=steps,
            missions=missions,
            runs={step: 1 for step in steps},
            dependencies=dependencies,
            allow_replan=bool(obj.get("allow_replan")),
            writer_mode=writer_mode,
            response_mode=user_query.response_mode,
            policy=policy,
            diagnostics=PlanDiagnostics(
                planner_source=planner_source,
                fallback_reason=fallback_reason,
                rationale=str(obj.get("rationale") or planner_error_detail).strip(),
            ),
        )

    async def replan(
        self,
        user_query: AgentInput,
        observations: Dict[str, Any],
    ) -> AgentPlan:
        """
        observations: resumen de salidas (p.ej. {"legal":{"status":"insuficiente"}, "web":{"citations":1}, "rs":{"confidence_avg":0.4}})
        Devuelve pasos ADICIONALES (no incluye los ya ejecutados) y termina en 'writer' si hay algo que decir.
        """
        q = user_query.query.strip()
        candidates = _available_agents()

        def compact(obs: Dict[str, Any]) -> str:
            try:
                s = json.dumps(obs, ensure_ascii=False)
                return s[:1000] + ("..." if len(s) > 1000 else "")
            except Exception:
                return str(obs)[:1000]

        obj = await self._chat_json(
            self._system_replan(candidates),
            f"Consulta original: {q}\nObservaciones previas (JSON): {compact(observations)}",
            schema=ORGANIZER_REPLAN_SCHEMA,
            schema_name="organizer_replan",
        )

        stop = bool(obj.get("stop")) if isinstance(obj, dict) else False
        extra = obj.get("extra_steps") or []
        if not isinstance(extra, list):
            extra = []
        extra = [str(step).strip().lower() for step in extra]

        extra = [step for step in extra if step in candidates and step != "writer"]
        fallback_reason: str | None = None
        rationale = str(obj.get("rationale") or "").strip()
        planner_source = "llm" if obj else "heuristic"
        if not extra and not stop:
            extra = self._fallback_replan_steps(user_query, observations)
            if extra:
                fallback_reason = "heuristic_replan_gap"
                planner_source = "heuristic"
                if not rationale:
                    rationale = (
                        "El replan LLM no anadio pasos, pero la ejecucion previa deja una brecha "
                        "accionable que justifica recuperar evidencia adicional."
                    )
        elif not obj:
            fallback_reason = "llm_replan_empty"
        if extra:
            extra.append("writer")
        extra = _enforce_rules(extra)
        writer_mode = self._normalize_writer_mode(obj.get("writer_mode"))
        raw_missions = obj.get("missions") if isinstance(obj.get("missions"), list) else []
        extra_missions = [
            MissionEntry(agent=str(m.get("agent", "")).strip().lower(), instruction=str(m.get("instruction", "")))
            for m in raw_missions
            if isinstance(m, dict) and str(m.get("agent", "")).strip().lower() in extra
            and str(m.get("agent", "")).strip().lower() != "writer"
        ]
        return AgentPlan(
            steps=extra,
            missions=extra_missions,
            runs={step: 1 for step in extra},
            dependencies=_default_dependencies(extra),
            allow_replan=False,
            writer_mode=writer_mode,
            response_mode=user_query.response_mode,
            diagnostics=PlanDiagnostics(
                planner_source=planner_source,
                fallback_reason=fallback_reason,
                rationale=rationale,
            ),
        )
