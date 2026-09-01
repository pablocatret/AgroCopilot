from __future__ import annotations
import os
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

from unstructured.partition.auto import partition
from openai import AsyncOpenAI
from qdrant_client.http.models import PointStruct
import tiktoken

import fitz  # PyMuPDF
import pytesseract

from backend.deps import settings
from agents.base import _build_client
from libs.costs.tracker import record_openai_embedding_usage
from libs.rag.legalize import build_chunks, parse_markdown_document
from libs.rag.vector_store import get_vector_store


def _build_rag_client():
    provider = settings.resolve_provider()
    return _build_client(provider) if settings.OPENAI_API_KEY else None


client = _build_rag_client()

# --- Parámetros de troceado / batch ---
CHUNK_MAX_CHARS = 900
CHUNK_OVERLAP_CHARS = 120
MAX_TOKENS_PER_REQUEST = 240_000
MAX_TOKENS_PER_ITEM = 2_048
MAX_ITEMS_PER_EMBEDDING_REQUEST = 2_048
ENC = None

MIN_TEXT_AFTER_PARSE = 400  # si menos que esto, probamos OCR
MIN_CHUNK_CHARS = 60  # descarta cachitos demasiado cortos


def _pdf_ocr_text(path: str, dpi: int = 220, langs: str = "spa+eng") -> str:
    """OCR de cada página con PyMuPDF + Tesseract. Devuelve texto."""
    doc = fitz.open(path)
    parts: List[str] = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)  # rasteriza
        text = pytesseract.image_to_string(pix.tobytes("ppm"), lang=langs)
        text = (text or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts).strip()


def _read_file(path: str) -> str:
    # 1) Intento normal con unstructured
    try:
        elements = partition(filename=path, languages=["es", "en"])
        texts = [el.text for el in elements if getattr(el, "text", None)]
        text = ("\n\n".join(texts)).strip()
    except Exception as exc:
        logger.warning("ingest_partition_error: %s", exc)
        text = ""

    # 2) Fallback OCR for PDF with little text
    if (not text or len(text) < MIN_TEXT_AFTER_PARSE) and path.lower().endswith(".pdf"):
        try:
            ocr_text = _pdf_ocr_text(path)
            if len(ocr_text) > len(text):
                text = ocr_text
        except Exception as exc:
            logger.warning("ingest_ocr_fallback_error: %s", exc)

    return text


def _chunk(
    text: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP_CHARS
) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + max_chars, n)
        chunk = text[i:j].strip()
        if len(chunk) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)
        if j == n:
            break
        i = j - overlap
    return chunks


def _tok_len(s: str) -> int:
    encoder = _encoding()
    return len(encoder.encode(s))


def _truncate_tokens(s: str, max_tokens: int = MAX_TOKENS_PER_ITEM) -> str:
    encoder = _encoding()
    ids = encoder.encode(s)
    if len(ids) <= max_tokens:
        return s
    ids = ids[:max_tokens]
    return encoder.decode(ids)


def _encoding():
    global ENC
    if ENC is not None:
        return ENC
    try:
        ENC = tiktoken.get_encoding("cl100k_base")
    except Exception:
        class _FallbackEncoding:
            @staticmethod
            def encode(value: str) -> list[int]:
                return list(str(value or "").encode("utf-8"))

            @staticmethod
            def decode(values: list[int]) -> str:
                return bytes(values).decode("utf-8", errors="ignore")

        ENC = _FallbackEncoding()
    return ENC


async def _embed_batched(texts: List[str]) -> List[List[float]]:
    if settings.DISABLE_EXTERNALS or not client:
        return []
    texts = [_truncate_tokens(t) for t in texts if t and t.strip()]
    if not texts:
        return []

    batches: List[List[str]] = []
    current_batch: List[str] = []
    current_tokens = 0

    for t in texts:
        tl = _tok_len(t)
        if tl > MAX_TOKENS_PER_REQUEST:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            batches.append([t])
            continue
        if (
            current_batch
            and (
                current_tokens + tl > MAX_TOKENS_PER_REQUEST
                or len(current_batch) >= MAX_ITEMS_PER_EMBEDDING_REQUEST
            )
        ):
            batches.append(current_batch)
            current_batch = [t]
            current_tokens = tl
        else:
            current_batch.append(t)
            current_tokens += tl
    if current_batch:
        batches.append(current_batch)

    vectors: List[List[float]] = []
    for batch in batches:
        resp = await client.embeddings.create(model=settings.OPENAI_EMBEDDING_MODEL, input=batch)
        if getattr(resp, "usage", None) is not None:
            record_openai_embedding_usage(
                settings.OPENAI_EMBEDDING_MODEL,
                resp.usage,
                operation="rag.ingest.embeddings",
                metadata={"items": len(batch)},
            )
        vectors.extend([d.embedding for d in resp.data])
    return vectors


def _infer_metadata(path: str) -> Dict[str, Any]:
    base = os.path.basename(path)
    title = os.path.splitext(base)[0]
    jurisdiction = "UE" if any(k in base.lower() for k in ("ue", "eu", "europa")) else "Global"
    return {
        "title": title,
        "jurisdiction": jurisdiction,
        "source_url": "",
        "version": "",
        "path": path,
    }


async def ingest_paths(paths: List[str]) -> int:
    if settings.DISABLE_EXTERNALS or not client:
        return 0
    texts_meta: List[Dict[str, Any]] = []
    for p in paths:
        if not os.path.isfile(p):
            continue
        if Path(p).suffix.lower() == ".md":
            try:
                document = parse_markdown_document(
                    Path(p), repo="local_markdown", repo_root=Path(p).parent
                )
                legal_chunks = build_chunks(document, include_all=True)
            except Exception:
                legal_chunks = []
            if legal_chunks:
                for chunk in legal_chunks:
                    texts_meta.append({"text": chunk.text, "meta": dict(chunk.payload)})
                continue
        raw = _read_file(p)
        if not raw:
            continue
        meta = _infer_metadata(p)
        for idx, ch in enumerate(_chunk(raw)):
            texts_meta.append({"text": ch, "meta": {**meta, "chunk_id": idx}})

    if not texts_meta:
        return 0

    vectors = await _embed_batched([tm["text"] for tm in texts_meta])
    if not vectors:
        return 0
    dim = len(vectors[0])

    store = get_vector_store()
    store.ensure_collection(settings.QDRANT_COLLECTION, dim)

    points = []
    for tm, vec in zip(texts_meta, vectors):
        pid = uuid.uuid4().hex
        payload = {
            "text": tm["text"],
            **tm["meta"],
            "ingested_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        points.append(PointStruct(id=pid, vector=vec, payload=payload))

    store.upsert(settings.QDRANT_COLLECTION, points)
    if hasattr(store, "close"):
        try:
            store.close()
        except Exception as exc:
            logger.debug("store_close_error: %s", exc)
    return len(points)
