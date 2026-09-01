from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Dict, List
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

from openai import AsyncOpenAI
from rank_bm25 import BM25Okapi

from backend.deps import settings
from agents.base import _build_client
from libs.costs.tracker import record_openai_embedding_usage
from libs.rag.legalize import normalize_text
from libs.schemas import LegalDossier, Reference
from libs.rag.vector_store import get_vector_store


def _build_rag_client():
    provider = settings.resolve_provider()
    return _build_client(provider) if settings.OPENAI_API_KEY else None


client = _build_rag_client()

_CORPUS_CACHE_TTL_S = 30.0
_CORPUS_CACHE: dict[str, dict[str, Any]] = {}


_STOPWORDS = {
    "a",
    "al",
    "and",
    "as",
    "con",
    "de",
    "del",
    "e",
    "el",
    "en",
    "for",
    "la",
    "las",
    "los",
    "of",
    "on",
    "para",
    "por",
    "se",
    "the",
    "to",
    "un",
    "una",
    "y",
}


def _tokenize(text: str) -> List[str]:
    normalized = normalize_text(text)
    return [
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if len(token) > 1 and token not in _STOPWORDS
    ]


def _doc_key(payload: Dict[str, Any]) -> str:
    primary = (
        payload.get("path")
        or payload.get("source_url")
        or payload.get("url")
        or payload.get("title")
        or ""
    )
    article = payload.get("article") or ""
    repo = payload.get("repo") or payload.get("corpus") or ""
    return f"{repo}:{primary}:{article}#{payload.get('chunk_id', 0)}"


def _legal_metadata_boost(query: str, payload: Dict[str, Any]) -> float:
    if payload.get("corpus") != "legalize":
        return 0.0
    query_norm = normalize_text(query)
    boost = 0.2
    status = normalize_text(str(payload.get("estado") or payload.get("source_status") or ""))
    if status == "vigente":
        boost += 0.45
    elif "derogada" in status:
        boost -= 0.8
    if payload.get("fuente") or payload.get("official_source") or payload.get("source_url"):
        boost += 0.15
    identifier = normalize_text(
        str(payload.get("identificador") or payload.get("identifier") or payload.get("celex") or "")
    )
    path = normalize_text(str(payload.get("path") or ""))
    if identifier and identifier in query_norm:
        boost += 5.0
    if path and any(part and part in query_norm for part in path.replace(".", "/").split("/")):
        boost += 1.0
    jurisdiction = normalize_text(
        str(payload.get("jurisdiccion") or payload.get("jurisdiction") or "")
    )
    if jurisdiction:
        if jurisdiction in {"es", "espana"} and any(
            token in query_norm for token in ("espana", "boe", "nacional", "estatal")
        ):
            boost += 0.25
        if jurisdiction in {"eu", "ue"} and any(
            token in query_norm for token in ("ue", "union europea", "reglamento europeo", "pac")
        ):
            boost += 0.25
        if jurisdiction in query_norm:
            boost += 0.2
    return boost


def _dedupe_ranked_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for item in items:
        key = _doc_key(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _corpus_cache_key() -> str:
    backend = (settings.VECTOR_BACKEND or "sqlite").lower()
    if backend == "qdrant":
        return f"qdrant:{settings.QDRANT_PATH}:{settings.QDRANT_COLLECTION}"
    return f"{backend}:{settings.DATABASE_URL}:{settings.QDRANT_COLLECTION}"


def _load_corpus_index(limit: int = 50_000) -> dict[str, Any]:
    cache_key = _corpus_cache_key()
    cached = _CORPUS_CACHE.get(cache_key)
    now = time.monotonic()
    if cached and now - cached["loaded_at"] <= _CORPUS_CACHE_TTL_S:
        return cached

    store = get_vector_store()
    try:
        payloads = store.scroll(settings.QDRANT_COLLECTION, limit=limit)
    finally:
        if hasattr(store, "close"):
            try:
                store.close()
            except Exception:
                pass
    docs = [
        " ".join(
            str(payload.get(key) or "")
            for key in (
                "identificador",
                "identifier",
                "title",
                "titulo",
                "article",
                "subjects",
                "text",
            )
        )
        for payload in payloads
    ]
    kept = [(index, doc) for index, doc in enumerate(docs) if doc and doc.strip()]
    if kept:
        indexes, docs_nonempty = zip(*kept)
        tokenized = [_tokenize(doc) for doc in docs_nonempty]
        bm25 = BM25Okapi(tokenized) if tokenized else None
        corpus_index = list(indexes)
    else:
        tokenized = []
        bm25 = None
        corpus_index = []
    snapshot = {
        "loaded_at": now,
        "payloads": list(payloads),
        "tokenized": tokenized,
        "corpus_index": corpus_index,
        "bm25": bm25,
    }
    _CORPUS_CACHE[cache_key] = snapshot
    return snapshot


def clear_retriever_cache() -> None:
    _CORPUS_CACHE.clear()


def _bm25_rank(query: str, corpus_snapshot: dict[str, Any], top_k: int = 8) -> List[Dict[str, Any]]:
    payloads = corpus_snapshot.get("payloads") or []
    bm25 = corpus_snapshot.get("bm25")
    indexes = corpus_snapshot.get("corpus_index") or []
    if not payloads or bm25 is None or not indexes:
        return []
    scores = bm25.get_scores(_tokenize(query))
    ranked: List[Dict[str, Any]] = []
    for rank_pos, raw_score in enumerate(scores):
        original_index = indexes[rank_pos]
        payload = dict(payloads[original_index])
        payload["_bm25"] = float(raw_score)
        payload["_legal_boost"] = _legal_metadata_boost(query, payload)
        payload["_rank_score"] = payload["_bm25"] + payload["_legal_boost"]
        ranked.append(payload)
    ranked.sort(key=lambda item: item["_rank_score"], reverse=True)
    ranked = ranked[:top_k]
    return _dedupe_ranked_items(ranked)


def _reciprocal_rank_fusion(
    bm25_list: List[Dict[str, Any]], vec_list: List[Dict[str, Any]], k: int = 6
) -> List[Dict[str, Any]]:
    def rankify(items: List[Dict[str, Any]]) -> Dict[str, int]:
        return {_doc_key(item): index for index, item in enumerate(items)}

    deduped_bm25 = _dedupe_ranked_items(bm25_list)
    deduped_vec = _dedupe_ranked_items(vec_list)
    bm25_rank = rankify(deduped_bm25)
    vec_rank = rankify(deduped_vec)
    fused: List[Dict[str, Any]] = []

    for key in set(bm25_rank) | set(vec_rank):
        score = 0.0
        if key in bm25_rank:
            score += 1.0 / (k + bm25_rank[key] + 1)
        if key in vec_rank:
            score += 1.0 / (k + vec_rank[key] + 1)
        payload = next((item for item in deduped_bm25 if _doc_key(item) == key), None)
        if payload is None:
            payload = next((item for item in deduped_vec if _doc_key(item) == key), None)
        record = dict(payload) if payload else {"text": "", "path": "", "title": ""}
        record["_rrf"] = score + (_legal_metadata_boost("", record) * 0.01)
        fused.append(record)

    fused.sort(key=lambda item: item["_rrf"], reverse=True)
    return fused[:k]


async def _vector_rank(query: str, top_k: int = 8) -> List[Dict[str, Any]]:
    if settings.DISABLE_EXTERNALS or not client:
        return []
    try:
        response = await asyncio.wait_for(
            client.with_options(timeout=15.0, max_retries=0).embeddings.create(
                model=settings.OPENAI_EMBEDDING_MODEL, input=[query]
            ),
            timeout=20,
        )
    except Exception as exc:
        logger.warning("embedding_request_error: %s", exc)
        return []
    if getattr(response, "usage", None) is not None:
        record_openai_embedding_usage(
            settings.OPENAI_EMBEDDING_MODEL,
            response.usage,
            operation="rag.retrieve.embedding",
        )
    embedding = response.data[0].embedding
    store = get_vector_store()
    try:
        results = store.search(settings.QDRANT_COLLECTION, embedding, limit=top_k)
        return _dedupe_ranked_items(results)
    finally:
        if hasattr(store, "close"):
            try:
                store.close()
            except Exception as exc:
                logger.debug("store_close_error: %s", exc)


def _format_retrieved(items: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    for item in items[:k]:
        text = item.get("text", "") or ""
        formatted.append(
            {
                "title": item.get("title") or "Documento",
                "url": item.get("source_url") or item.get("path") or "",
                "snippet": text[:300] + ("â€¦" if len(text) > 300 else ""),
                "jurisdiction": item.get("jurisdiction", ""),
                "version": item.get("version", ""),
                "rank_score": item.get("_rank_score"),
                "bm25_score": item.get("_bm25"),
                "vector_score": item.get("_score"),
                "rrf_score": item.get("_rrf"),
                "path": item.get("path", ""),
                "corpus": item.get("corpus", ""),
                "repo": item.get("repo", ""),
                "titulo": item.get("titulo") or item.get("title") or "",
                "identificador": item.get("identificador", ""),
                "pais": item.get("pais", ""),
                "jurisdiccion": item.get("jurisdiccion") or item.get("jurisdiction", ""),
                "rango": item.get("rango", ""),
                "estado": item.get("estado") or item.get("source_status", ""),
                "source_status": item.get("source_status") or item.get("estado", ""),
                "fecha_publicacion": item.get("fecha_publicacion", ""),
                "publication_date": item.get("publication_date", ""),
                "published_at": item.get("published_at") or item.get("fecha_publicacion", ""),
                "ultima_actualizacion": item.get("ultima_actualizacion")
                or item.get("updated_at")
                or item.get("last_updated", ""),
                "last_updated": item.get("last_updated", ""),
                "updated_at": item.get("updated_at")
                or item.get("ultima_actualizacion")
                or item.get("last_updated", ""),
                "fuente": item.get("fuente")
                or item.get("official_source")
                or item.get("source_url", ""),
                "official_source": item.get("official_source")
                or item.get("fuente")
                or item.get("source_url", ""),
                "article": item.get("article", ""),
            }
        )
    return formatted


def _is_authoritative_legal_doc(item: Dict[str, Any]) -> bool:
    status = normalize_text(str(item.get("source_status") or item.get("estado") or ""))
    official_source = str(
        item.get("official_source") or item.get("fuente") or item.get("source_url") or ""
    ).strip()
    if not official_source:
        return False
    if status and status not in {"vigente", "current", "in_force"}:
        return False
    parsed = urlparse(official_source)
    host = normalize_text(parsed.netloc or "")
    official_markers = (
        ".gob.",
        ".gov.",
        ".europa.eu",
        ".boe.es",
        ".eur-lex.europa.eu",
        ".legifrance.gouv.fr",
    )
    if any(marker in host for marker in official_markers):
        return True
    if host.startswith("boe.es") or host.startswith("eur-lex.europa.eu"):
        return True
    return False


def _to_reference(item: Dict[str, Any]) -> Reference:
    return Reference(
        title=str(item.get("title") or item.get("titulo") or "Documento"),
        url=str(item.get("official_source") or item.get("fuente") or item.get("url") or ""),
        snippet=str(item.get("snippet") or ""),
    )


def build_legal_dossier(items: List[Dict[str, Any]], *, verification_mode: str = "local") -> LegalDossier:
    authoritative: List[Reference] = []
    supporting: List[Reference] = []
    seen_auth: set[tuple[str, str]] = set()
    seen_support: set[tuple[str, str]] = set()
    for item in items:
        ref = _to_reference(item)
        key = (ref.title, ref.url)
        if _is_authoritative_legal_doc(item):
            if key in seen_auth:
                continue
            seen_auth.add(key)
            authoritative.append(ref)
        else:
            if key in seen_support:
                continue
            seen_support.add(key)
            supporting.append(ref)
    mode = verification_mode if verification_mode in {"local", "web", "hybrid"} else "local"
    return LegalDossier(
        authoritative_references=authoritative,
        supporting_references=supporting,
        verification_mode=mode,
    )


async def retrieve_legal_dossier(query: str, k: int = 6) -> tuple[List[Dict[str, Any]], LegalDossier]:
    docs = await retrieve(query, k=k)
    return docs, build_legal_dossier(docs, verification_mode="local")


async def retrieve(query: str, k: int = 6) -> List[Dict[str, Any]]:
    strategy = (settings.LEGAL_RAG_STRATEGY or "hybrid").lower()
    top_k = max(k, 8)

    if strategy == "vector":
        return _format_retrieved(await _vector_rank(query, top_k=top_k), k)

    corpus_snapshot = _load_corpus_index()
    if not (corpus_snapshot.get("payloads") or []):
        return []

    bm25_top: List[Dict[str, Any]] = []
    if strategy in {"bm25", "hybrid"}:
        bm25_top.extend(_bm25_rank(query, corpus_snapshot, top_k=top_k))
        bm25_top = _dedupe_ranked_items(bm25_top)

    if strategy == "bm25":
        return _format_retrieved(bm25_top, k)

    vec_top: List[Dict[str, Any]] = []
    if strategy == "hybrid":
        vec_top.extend(await _vector_rank(query, top_k=top_k))
        vec_top = _dedupe_ranked_items(vec_top)

    fused = _reciprocal_rank_fusion(bm25_top, vec_top, k=k)
    return _format_retrieved(fused, k)

