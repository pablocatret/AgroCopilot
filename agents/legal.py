# agents/legal.py
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import List

from agents.base import BaseAgent
from backend.deps import settings
from libs.context_engineering import summarize_attachments, summarize_memory_context, truncate_text
from libs.prompts import compose_system_prompt, render_prompt
from libs.rag.legalize import normalize_text
from libs.rag.retriever import build_legal_dossier, retrieve_legal_dossier
from libs.schemas import (
    AgentInput,
    AgentRef,
    AgentRefs,
    Citation,
    LegalAgentOutput,
    LegalFinding,
    LegalFindings,
    Reference,
)
from libs.search_tool import WebSearchTool


CHECKLIST = [
    "Certificacion organica UE (Reglamento (UE) 2018/848)",
    "Certificacion GlobalG.A.P. IFA v6 (frutas y hortalizas)",
]

ORG_KEYWORDS = ("2018/848", "reglamento (ue) 2018/848", "organic", "organico", "ecologico")
GG_KEYWORDS = ("globalg.a.p", "globalgap", "ifa v6", "ifa v7")
CURRENT_STATUSES = {"vigente", "current", "in_force"}
CURRENTNESS_TOKENS = (
    "vigente",
    "actual",
    "actualizado",
    "actualizada",
    "ultima",
    "última",
    "ultimas",
    "últimas",
    "reciente",
    "recientes",
    "hoy",
    "ahora",
    "este ano",
    "este año",
    "normativa vigente",
    "requisitos actuales",
    "reglamento actual",
    "boe",
    "diario oficial",
    "consultar",
    "buscar",
    "web",
    "internet",
)
YEAR_REFERENCE_RE = re.compile(r"\b(19|20)\d{2}\b")


def _matches_any(doc: dict, keywords: tuple[str, ...]) -> bool:
    haystack = normalize_text(
        f"{doc.get('title', '')} {doc.get('snippet', '')} {doc.get('article', '')}"
    )
    return any(normalize_text(keyword) in haystack for keyword in keywords)


def _infer_requirement_status(
    query: str, evidence_docs: List[dict], keywords: tuple[str, ...]
) -> str:
    query_lower = normalize_text(query)
    normalized_keywords = tuple(normalize_text(token) for token in keywords)
    if any(
        token in query_lower for token in ("no tengo", "sin ", "carezco de", "todavia no")
    ) and any(token in query_lower for token in normalized_keywords):
        return "no_cumple"
    if any(
        token in query_lower
        for token in ("tengo", "dispongo de", "cuento con", "ya tengo", "certificado")
    ) and any(token in query_lower for token in normalized_keywords):
        return "cumple"
    if evidence_docs:
        return "insuficiente"
    return "insuficiente"


def _doc_url(doc: dict) -> str:
    return (
        doc.get("official_source") or doc.get("fuente") or doc.get("url") or doc.get("path") or ""
    )


def _doc_status(doc: dict) -> str:
    return str(doc.get("source_status") or doc.get("estado") or "").lower()


def _doc_title(doc: dict) -> str:
    return str(doc.get("titulo") or doc.get("title") or "Normativa")


def _doc_limitations(doc: dict) -> list[str]:
    if not doc:
        return []
    limitations: list[str] = []
    status = _doc_status(doc)
    if status and status not in CURRENT_STATUSES:
        limitations.append(
            f"La fuente figura con estado '{status}'; verificar antes de usarla como base principal."
        )
    if not _doc_url(doc):
        limitations.append("La evidencia no incluye fuente oficial verificable.")
    if doc.get("corpus") == "legalize":
        limitations.append(
            "Legalize es una reproduccion automatizada: contrastar el texto con la fuente oficial."
        )
    return limitations


def _citations_from_docs(docs: List[dict]) -> List[Citation]:
    citations: List[Citation] = []
    for doc in docs:
        citations.append(
            Citation(
                title=_doc_title(doc),
                url=_doc_url(doc),
                source="legal",
            )
        )
    return citations


def _finding_from_doc(doc: dict) -> LegalFinding:
    article = doc.get("article") or ""
    requirement = f"{_doc_title(doc)} - {article}" if article else _doc_title(doc)
    return LegalFinding(
        requirement=requirement[:220],
        status="insuficiente",
        evidence=_citations_from_docs([doc]),
        jurisdiction=doc.get("jurisdiccion") or doc.get("jurisdiction") or doc.get("pais"),
        source_status=doc.get("source_status") or doc.get("estado"),
        updated_at=doc.get("updated_at") or doc.get("ultima_actualizacion"),
        official_source=_doc_url(doc),
        article=article or None,
        limitations=_doc_limitations(doc),
    )


def _certification_finding(
    query: str,
    requirement: str,
    docs: list[dict],
    fallback_docs: list[dict],
    keywords: tuple[str, ...],
) -> LegalFinding:
    source_doc = (docs or fallback_docs or [{}])[0]
    return LegalFinding(
        requirement=requirement,
        status=_infer_requirement_status(query, docs, keywords),
        evidence=_citations_from_docs(docs or fallback_docs),
        jurisdiction=source_doc.get("jurisdiccion")
        or source_doc.get("jurisdiction")
        or source_doc.get("pais"),
        source_status=source_doc.get("source_status") or source_doc.get("estado"),
        updated_at=source_doc.get("updated_at") or source_doc.get("ultima_actualizacion"),
        official_source=_doc_url(source_doc),
        article=source_doc.get("article"),
        limitations=_doc_limitations(source_doc),
    )


def _build_checklist(query: str, docs: List[dict]) -> List[LegalFinding]:
    organic_docs = [doc for doc in docs if _matches_any(doc, ORG_KEYWORDS)]
    globalgap_docs = [doc for doc in docs if _matches_any(doc, GG_KEYWORDS)]
    fallback_docs = docs[:2]
    query_norm = normalize_text(query)
    needs_certification_fallback = bool(
        organic_docs
        or globalgap_docs
        or any(
            token in query_norm
            for token in (
                "organico",
                "ecologico",
                "globalg.a.p",
                "globalgap",
                "certificado",
                "certificacion",
            )
        )
    )

    if needs_certification_fallback:
        return [
            _certification_finding(
                query,
                CHECKLIST[0],
                organic_docs,
                fallback_docs,
                ("organico", "ecologico", "organic", "2018/848"),
            ),
            _certification_finding(
                query,
                CHECKLIST[1],
                globalgap_docs,
                fallback_docs,
                ("globalg.a.p", "globalgap", "ifa"),
            ),
        ]

    findings: list[LegalFinding] = []
    seen: set[str] = set()
    for doc in docs:
        finding = _finding_from_doc(doc)
        key = f"{finding.requirement}:{finding.article}"
        if key in seen:
            continue
        seen.add(key)
        findings.append(finding)
        if len(findings) >= 5:
            break
    return findings


def _agent_refs_from_docs(docs: List[dict]) -> List[AgentRef]:
    refs: List[AgentRef] = []
    for doc in docs:
        title = _doc_title(doc)
        url = _doc_url(doc)
        refs.append(
            AgentRef(
                ref_id=f"legal-{abs(hash(url or doc.get('path') or title))}",
                title=title,
                source="legal",
                url=url,
                snippet=doc.get("snippet") or "",
                metadata={
                    "corpus": doc.get("corpus"),
                    "repo": doc.get("repo"),
                    "jurisdiction": doc.get("jurisdiccion")
                    or doc.get("jurisdiction")
                    or doc.get("pais"),
                    "source_status": doc.get("source_status") or doc.get("estado"),
                    "updated_at": doc.get("updated_at") or doc.get("ultima_actualizacion"),
                    "official_source": url,
                    "article": doc.get("article"),
                    "identifier": doc.get("identificador"),
                    "rank": doc.get("rango"),
                },
            )
        )
    return refs


def _build_context_for_llm(
    query: str,
    authoritative_docs: List[dict],
    supporting_docs: List[dict],
    max_docs: int = 5,
) -> str:
    lines = [
        f"Consulta del usuario: {query.strip()}".strip(),
        "",
        "Fuentes normativas autoritativas:",
    ]
    for i, doc in enumerate(authoritative_docs[:max_docs], start=1):
        title = _doc_title(doc)
        url = _doc_url(doc)
        snippet = (doc.get("snippet") or "").strip()
        snippet = snippet[:900] + ("..." if len(snippet) > 900 else "")
        meta = {
            "jurisdiccion": doc.get("jurisdiccion") or doc.get("jurisdiction") or doc.get("pais"),
            "estado": doc.get("source_status") or doc.get("estado"),
            "ultima_actualizacion": doc.get("updated_at") or doc.get("ultima_actualizacion"),
            "articulo": doc.get("article"),
            "fuente_oficial": url,
        }
        lines.append(f"[{i}] {title} - {json.dumps(meta, ensure_ascii=False)}\n{snippet}")
    if supporting_docs:
        lines.extend(["", "Fuentes de apoyo o contraste:"])
        for i, doc in enumerate(supporting_docs[:max_docs], start=1):
            title = _doc_title(doc)
            url = _doc_url(doc)
            snippet = (doc.get("snippet") or "").strip()
            snippet = snippet[:600] + ("..." if len(snippet) > 600 else "")
            lines.append(f"[S{i}] {title} - {url}\n{snippet}")
    return "\n".join(lines)


def _aggregate_metadata(docs: list[dict], findings: list[LegalFinding]) -> dict:
    first = docs[0] if docs else {}
    limitations: list[str] = []
    for finding in findings:
        for item in finding.limitations:
            if item and item not in limitations:
                limitations.append(item)
    if not docs:
        limitations.append("No se han recuperado normas del corpus local para esta consulta.")
    return {
        "jurisdiction": first.get("jurisdiccion") or first.get("jurisdiction") or first.get("pais"),
        "source_status": first.get("source_status") or first.get("estado"),
        "updated_at": first.get("updated_at") or first.get("ultima_actualizacion"),
        "official_source": _doc_url(first),
        "article": first.get("article"),
        "limitations": limitations[:6],
    }


def _query_requests_currentness(query: str) -> bool:
    query_norm = normalize_text(query)
    if any(token in query_norm for token in CURRENTNESS_TOKENS):
        return True
    current_year = datetime.now(timezone.utc).year
    requested_years = [
        int(match.group(0))
        for match in YEAR_REFERENCE_RE.finditer(query_norm)
    ]
    return any(year >= current_year for year in requested_years)


def _needs_external_verification(query: str, docs: list[dict]) -> bool:
    if not docs:
        return True
    if _query_requests_currentness(query):
        return True
    primary_docs = docs[:3]
    if any(_doc_status(doc) and _doc_status(doc) not in CURRENT_STATUSES for doc in primary_docs):
        return True
    if any(not _doc_url(doc) for doc in primary_docs):
        return True
    weak_snippets = sum(1 for doc in primary_docs if len((doc.get("snippet") or "").strip()) < 80)
    if weak_snippets >= 2:
        return True
    return False


LEGAL_ANSWER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {"type": "string", "minLength": 1},
        "references": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "url": {"type": "string"},
                    "snippet": {"type": "string"},
                },
                "required": ["title", "url", "snippet"],
            },
            "default": [],
        },
    },
    "required": ["answer", "references"],
}


class LegalAgent(BaseAgent):
    name = "legal"
    output_model = LegalAgentOutput
    requires_llm = True
    _provider_key = "LLM_PROVIDER_LEGAL"

    def __init__(self) -> None:
        super().__init__()
        self.model = settings.resolve_openai_model(
            "OPENAI_MODEL_LEGAL",
            "OPENAI_MODEL_LEGAL_WRITER",
        )

    async def _answer_with_research(
        self,
        query: str,
        authoritative_docs: List[dict],
        supporting_docs: List[dict],
        *,
        use_web: bool,
    ) -> tuple[str, List[Reference]]:
        if not authoritative_docs and not supporting_docs and (
            settings.DISABLE_EXTERNALS or not settings.OPENAI_API_KEY
        ):
            return (
                "No se han encontrado evidencias suficientes en el corpus legal local. "
                "Actualice o ingiera el corpus Legalize y contraste con la fuente oficial aplicable."
            ), []

        if settings.DISABLE_EXTERNALS or not settings.OPENAI_API_KEY:
            return "Resumen legal no disponible (LLM deshabilitado).", []

        system = compose_system_prompt(
            agent_name="legal",
            body=render_prompt("legal_system.txt"),
            output_contract=(
                "Devuelve exclusivamente JSON valido con answer y references. "
                "No afirmes cumplimiento legal definitivo. Prioriza fuentes oficiales y marca cautelas."
            ),
        )
        context = _build_context_for_llm(query, authoritative_docs, supporting_docs)
        user = (
            f"{render_prompt('legal_user.txt', context=context, query=query)}\n\n"
            "Devuelve SOLO JSON siguiendo este esquema:\n"
            f"{json.dumps(LEGAL_ANSWER_SCHEMA, ensure_ascii=False)}"
        )
        tool = WebSearchTool(
            description="Busca normativa vigente y fuentes oficiales cuando el corpus local sea insuficiente o necesite verificacion.",
        )
        data = await self.call_llm_json_with_tools(
            system=system,
            user=user,
            schema=LEGAL_ANSWER_SCHEMA,
            tools=[tool.tool_spec()] if use_web else [],
            tool_map={tool.name: tool.run} if use_web else {},
            temperature=0.2,
        )
        references = [
            Reference(
                title=ref.get("title", "Fuente legal"),
                url=ref.get("url", ""),
                snippet=ref.get("snippet", ""),
            )
            for ref in data.get("references") or []
        ]
        return data.get("answer", "").strip() or "Resumen legal no disponible.", references

    async def _run(self, user_query: AgentInput) -> LegalAgentOutput:
        docs, local_dossier = await retrieve_legal_dossier(user_query.query, k=8)
        findings = _build_checklist(user_query.query, docs)
        aggregate = _aggregate_metadata(docs, findings)
        use_web = bool(settings.SEARCH_API_KEY) and _needs_external_verification(
            user_query.query, docs
        )
        authoritative_urls = {
            ref.url for ref in local_dossier.authoritative_references if isinstance(ref.url, str)
        }
        authoritative_docs = [
            doc for doc in docs if authoritative_urls and _doc_url(doc) in authoritative_urls
        ][:5]
        supporting_docs = [
            doc for doc in docs if not authoritative_urls or _doc_url(doc) not in authoritative_urls
        ][:5]
        mission = str((user_query.context or {}).get("mission") or "").strip()
        memory_summary = summarize_memory_context(
            str(user_query.context.get("user_memory", "") or ""), max_chars=600
        )
        attachments_summary = summarize_attachments(
            user_query.attachments, max_items=4, max_summary_chars=120
        )

        enriched_query = (
            f"{truncate_text(user_query.query, 700)}\n\n"
            f"Contexto persistente del perfil:\n{memory_summary}\n\n"
            f"Adjuntos relevantes:\n{attachments_summary}"
        )
        if mission:
            enriched_query = f"Misión: {mission}\n\n{enriched_query}"
        answer_text, web_refs = await self._answer_with_research(
            enriched_query,
            authoritative_docs,
            supporting_docs,
            use_web=use_web,
        )
        local_refs = _agent_refs_from_docs(docs[:5])
        dossier = build_legal_dossier(
            docs,
            verification_mode="hybrid" if web_refs else local_dossier.verification_mode,
        )
        if web_refs:
            dossier.supporting_references.extend(web_refs)
        combined_refs = local_refs + [
            AgentRef(
                ref_id=f"legal-web-{abs(hash(ref.url or ref.title))}",
                title=ref.title,
                source="legal",
                url=ref.url,
                snippet=ref.snippet,
                metadata={"verification": "web"},
            )
            for ref in web_refs
        ]

        findings_model = LegalFindings(
            checklist=findings,
            answer=answer_text,
            citations=[citation for finding in findings for citation in finding.evidence],
            references=web_refs,
            dossier=dossier,
            jurisdiction=aggregate["jurisdiction"],
            source_status=aggregate["source_status"],
            updated_at=aggregate["updated_at"],
            official_source=aggregate["official_source"],
            article=aggregate["article"],
            limitations=aggregate["limitations"],
        )
        return LegalAgentOutput(
            agent=self.name,
            summary=(
                "Analisis legal completado con evidencias normativas."
                if docs
                else "Sin evidencias legales disponibles."
            ),
            refs=AgentRefs(items=combined_refs),
            data=findings_model,
        )
