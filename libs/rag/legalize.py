from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


REPO_URLS = {
    "es": "https://github.com/legalize-dev/legalize-es.git",
    "eu": "https://github.com/legalize-dev/legalize-eu.git",
    "legalize-es": "https://github.com/legalize-dev/legalize-es.git",
    "legalize-eu": "https://github.com/legalize-dev/legalize-eu.git",
}

REPO_NAMES = {
    "es": "legalize-es",
    "eu": "legalize-eu",
    "legalize-es": "legalize-es",
    "legalize-eu": "legalize-eu",
}

AGRO_KEYWORDS = (
    "agricultura",
    "agricola",
    "agrario",
    "explotacion agraria",
    "politica agricola comun",
    "pac",
    "feaga",
    "feader",
    "ecologico",
    "organico",
    "produccion ecologica",
    "fitosanitario",
    "sanidad vegetal",
    "fertilizante",
    "nitrato",
    "regadio",
    "riego",
    "agua",
    "trazabilidad",
    "etiquetado",
    "ayuda",
    "subvencion",
    "condicionalidad",
    "bienestar animal",
    "producto vegetal",
    "producto agricola",
    "desarrollo rural",
)

DEMO_KEYWORDS = (
    "2018/848",
    "produccion ecologica",
    "ecologico",
    "organico",
    "politica agricola comun",
    "pac",
    "condicionalidad",
    "ayuda",
    "subvencion",
    "fitosanitario",
    "sanidad vegetal",
    "fertilizante",
    "nitrato",
    "regadio",
    "riego",
    "agua",
    "trazabilidad",
    "etiquetado",
)

PROFILE_KEYWORDS = {
    "agro": AGRO_KEYWORDS,
    "demo": DEMO_KEYWORDS,
}

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
SECTION_RE = re.compile(
    r"(?im)^(?:#{1,6}\s*)?"
    r"("
    r"art[ií]culo\s+(?:\d+[^\n]*|[úu]nico[^\n]*)|"
    r"article\s+\d+[^\n]*|"
    r"anexo\s+[^\n]*|"
    r"annex\s+[^\n]*|"
    r"t[ií]tulo\s+[^\n]*|"
    r"cap[ií]tulo\s+[^\n]*"
    r")\s*$"
)


@dataclass(frozen=True)
class LegalizeDocument:
    repo: str
    path: Path
    relative_path: str
    metadata: dict[str, Any]
    body: str


@dataclass(frozen=True)
class LegalizeChunk:
    point_id: str
    text: str
    payload: dict[str, Any]


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value or "")
    ascii_text = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return ascii_text.lower()


def parse_markdown_document(
    path: Path, *, repo: str, repo_root: Path | None = None
) -> LegalizeDocument:
    raw = path.read_text(encoding="utf-8", errors="replace")
    match = FRONTMATTER_RE.match(raw)
    metadata: dict[str, Any] = {}
    body = raw
    if match:
        loaded = yaml.safe_load(match.group(1)) or {}
        metadata = loaded if isinstance(loaded, dict) else {}
        body = raw[match.end() :]

    root = repo_root or path.parent
    try:
        relative_path = path.relative_to(root).as_posix()
    except ValueError:
        relative_path = path.as_posix()

    return LegalizeDocument(
        repo=repo,
        path=path,
        relative_path=relative_path,
        metadata=metadata,
        body=body.strip(),
    )


def is_agro_relevant(
    document: LegalizeDocument, *, profile: str = "agro", include_all: bool = False
) -> bool:
    if include_all:
        return True
    keywords = PROFILE_KEYWORDS.get(profile)
    if not keywords:
        return True
    meta_text = " ".join(str(value) for value in document.metadata.values() if value is not None)
    haystack = normalize_text(f"{meta_text}\n{document.relative_path}\n{document.body[:8000]}")
    return any(keyword in haystack for keyword in keywords)


def split_legal_sections(
    text: str, *, max_chars: int = 1800, overlap: int = 180
) -> list[tuple[str | None, str]]:
    text = (text or "").strip()
    if not text:
        return []

    matches = list(SECTION_RE.finditer(text))
    sections: list[tuple[str | None, str]] = []
    if matches:
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            title = re.sub(r"^#{1,6}\s*", "", match.group(0)).strip()
            chunk = text[start:end].strip()
            if chunk:
                sections.extend(
                    (title, part)
                    for part in chunk_text(chunk, max_chars=max_chars, overlap=overlap)
                )
        return sections

    return [(None, part) for part in chunk_text(text, max_chars=max_chars, overlap=overlap)]


def chunk_text(text: str, *, max_chars: int = 1800, overlap: int = 180) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            break_at = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
            if break_at > start + max_chars // 2:
                end = break_at + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def build_chunks(
    document: LegalizeDocument, *, profile: str = "agro", include_all: bool = False
) -> list[LegalizeChunk]:
    if not is_agro_relevant(document, profile=profile, include_all=include_all):
        return []

    metadata = document.metadata
    title = str(metadata.get("titulo") or metadata.get("title") or document.path.stem)
    identifier = str(metadata.get("identificador") or metadata.get("id") or document.path.stem)
    country = str(metadata.get("pais") or metadata.get("country") or "")
    jurisdiction = str(
        metadata.get("jurisdiccion") or metadata.get("jurisdiction") or country or ""
    )
    status = str(metadata.get("estado") or metadata.get("status") or "").lower()
    updated_at = str(
        metadata.get("ultima_actualizacion")
        or metadata.get("updated_at")
        or metadata.get("last_updated")
        or ""
    )
    published_at = str(
        metadata.get("fecha_publicacion")
        or metadata.get("published_at")
        or metadata.get("publication_date")
        or ""
    )
    source_url = str(
        metadata.get("fuente") or metadata.get("source") or metadata.get("source_url") or ""
    )
    rank = str(metadata.get("rango") or metadata.get("rank") or metadata.get("type") or "")

    chunks: list[LegalizeChunk] = []
    for index, (article, text) in enumerate(split_legal_sections(document.body)):
        stable_key = (
            f"{document.repo}:{document.relative_path}:{article or ''}:{index}:{updated_at}"
        )
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))
        digest = hashlib.sha256(stable_key.encode("utf-8")).hexdigest()
        payload = {
            "text": text,
            "title": title,
            "url": source_url,
            "source_url": source_url,
            "path": document.relative_path,
            "corpus": "legalize",
            "repo": document.repo,
            "chunk_id": index,
            "point_id": point_id,
            "stable_hash": digest,
            "titulo": title,
            "identificador": identifier,
            "pais": country,
            "jurisdiccion": jurisdiction,
            "jurisdiction": jurisdiction,
            "rango": rank,
            "estado": status,
            "source_status": status,
            "fecha_publicacion": published_at,
            "publication_date": str(metadata.get("publication_date") or ""),
            "published_at": published_at,
            "ultima_actualizacion": updated_at,
            "last_updated": str(metadata.get("last_updated") or ""),
            "updated_at": updated_at,
            "fuente": source_url,
            "official_source": source_url,
            "article": article,
        }
        chunks.append(LegalizeChunk(point_id=point_id, text=text, payload=payload))
    return chunks


def iter_markdown_files(repo_dir: Path) -> Iterable[Path]:
    ignored_parts = {".git", ".github", "__pycache__"}
    for path in repo_dir.rglob("*.md"):
        if ignored_parts.intersection(path.parts):
            continue
        if path.name.lower() in {"readme.md", "license.md"}:
            continue
        yield path
