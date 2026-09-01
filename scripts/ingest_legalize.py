from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

from qdrant_client.http.models import PointStruct

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.deps import settings  # noqa: E402
from libs.rag.legalize import (  # noqa: E402
    REPO_NAMES,
    REPO_URLS,
    build_chunks,
    iter_markdown_files,
    parse_markdown_document,
)
from libs.rag.retriever import clear_retriever_cache  # noqa: E402
from libs.rag.vector_store import get_vector_store  # noqa: E402


DEFAULT_VECTOR_DIM = 1536
EMBEDDING_PRICE_PER_MILLION = 0.02
BATCH_EMBEDDING_PRICE_PER_MILLION = 0.01


def _run_git(args: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)


def _repo_aliases(raw: str) -> list[str]:
    aliases = [item.strip() for item in raw.split(",") if item.strip()]
    return aliases or ["es", "eu"]


def _parse_date(raw: object | None) -> date | None:
    if not raw:
        return None
    try:
        return date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


def _document_date(document) -> date | None:
    metadata = document.metadata
    return _parse_date(metadata.get("ultima_actualizacion")) or _parse_date(
        metadata.get("fecha_publicacion")
    )


def _document_matches_filters(document, *, status_filter: str | None, since: date | None) -> bool:
    if status_filter:
        status_aliases = {
            "vigente": "in_force",
            "en_vigor": "in_force",
            "in-force": "in_force",
        }
        expected = status_aliases.get(status_filter.lower(), status_filter.lower())
        status = str(
            document.metadata.get("estado") or document.metadata.get("status") or ""
        ).lower()
        status = status_aliases.get(status, status)
        if status != expected:
            return False
    if since:
        doc_date = _document_date(document)
        if doc_date is None or doc_date < since:
            return False
    return True


def _document_identifier(document) -> str:
    metadata = document.metadata
    return str(metadata.get("identificador") or metadata.get("identifier") or document.path.stem)


def _estimate_tokens(texts: Sequence[str]) -> int:
    # Offline approximation for Spanish legal prose. Good enough for cost control before billing.
    return sum(max(1, round(len(text) / 3.8)) for text in texts if text)


def _print_cost_estimate(
    *, total_chunks: int, total_tokens: int, existing_chunks: int, billable_chunks: int
) -> None:
    billable_tokens = (
        0 if total_chunks == 0 else round(total_tokens * (billable_chunks / total_chunks))
    )
    standard = billable_tokens / 1_000_000 * EMBEDDING_PRICE_PER_MILLION
    batch = billable_tokens / 1_000_000 * BATCH_EMBEDDING_PRICE_PER_MILLION
    print(
        "[legalize] coste_estimado_embeddings: "
        f"chunks_totales={total_chunks}, chunks_existentes={existing_chunks}, "
        f"chunks_facturables={billable_chunks}, tokens_estimados={billable_tokens}, "
        f"estandar_usd=${standard:.4f}, batch_usd=${batch:.4f}"
    )


def _ensure_repo(
    alias: str, base_dir: Path, *, update: bool = True, local_only: bool = False
) -> tuple[str, Path, str | None]:
    if alias not in REPO_URLS:
        return alias, base_dir / alias, f"Repositorio Legalize no soportado: {alias}"
    repo_name = REPO_NAMES[alias]
    repo_dir = base_dir / repo_name
    try:
        if local_only:
            if repo_dir.exists():
                return repo_name, repo_dir, None
            return repo_name, repo_dir, f"Repositorio local no encontrado: {repo_dir}"
        if (repo_dir / ".git").exists():
            if update:
                _run_git(["-C", str(repo_dir), "pull", "--ff-only"])
        else:
            repo_dir.parent.mkdir(parents=True, exist_ok=True)
            _run_git(["clone", "--depth", "1", REPO_URLS[alias], str(repo_dir)])
        return repo_name, repo_dir, None
    except subprocess.CalledProcessError as exc:
        return repo_name, repo_dir, str(exc)


def _collect_chunks(
    repo_name: str,
    repo_dir: Path,
    *,
    profile: str,
    include_all: bool,
    limit_files: int | None = None,
    max_docs: int | None = None,
    max_chunks: int | None = None,
    since: date | None = None,
    status_filter: str | None = None,
    identifiers: set[str] | None = None,
) -> tuple[int, int, list]:
    read_count = 0
    matched_count = 0
    chunks = []
    files: Iterable[Path] = iter_markdown_files(repo_dir)
    for path in files:
        if limit_files is not None and read_count >= limit_files:
            break
        if identifiers is not None and path.stem.lower() not in identifiers:
            continue
        read_count += 1
        try:
            document = parse_markdown_document(path, repo=repo_name, repo_root=repo_dir)
            if identifiers is not None and _document_identifier(document).lower() not in identifiers:
                continue
            if not _document_matches_filters(document, status_filter=status_filter, since=since):
                continue
            doc_chunks = build_chunks(document, profile=profile, include_all=include_all)
        except Exception as exc:
            print(f"[warn] No se pudo procesar {path}: {exc}")
            continue
        if not doc_chunks:
            continue
        if max_docs is not None and matched_count >= max_docs:
            break
        if max_chunks is not None:
            remaining = max_chunks - len(chunks)
            if remaining <= 0:
                break
            doc_chunks = doc_chunks[:remaining]
        matched_count += 1
        chunks.extend(doc_chunks)
        if max_chunks is not None and len(chunks) >= max_chunks:
            break
    return read_count, matched_count, chunks


def _existing_point_ids(collection: str) -> set[str]:
    store = get_vector_store()
    try:
        payloads = store.scroll(collection, limit=1_000_000)
        return {str(item.get("point_id")) for item in payloads if item.get("point_id")}
    except Exception as exc:
        print(f"[warn] No se pudieron leer chunks existentes para idempotencia: {exc}")
        return set()
    finally:
        if hasattr(store, "close"):
            store.close()


async def _vectors_for_chunks(texts: list[str], *, embeddings: bool) -> list[list[float]]:
    if embeddings:
        from libs.rag.ingest import _embed_batched

        vectors = await _embed_batched(texts)
        if vectors:
            return vectors
        print(
            "[warn] No se generaron embeddings; se usaran vectores cero para habilitar BM25/offline."
        )
    return [[0.0] * DEFAULT_VECTOR_DIM for _ in texts]


async def ingest_legalize(args: argparse.Namespace) -> int:
    base_dir = Path(args.data_dir or settings.LEGALIZE_DATA_DIR).resolve()
    aliases = _repo_aliases(args.repos or settings.LEGALIZE_DEFAULT_REPOS)
    since = _parse_date(args.since)
    identifiers = (
        {item.strip().lower() for item in args.identifiers.split(",") if item.strip()}
        if args.identifiers
        else None
    )
    errors: list[str] = []
    total_read = 0
    total_matched = 0
    all_chunks = []

    for alias in aliases:
        repo_name, repo_dir, error = _ensure_repo(
            alias, base_dir, update=not args.no_update, local_only=args.local_only
        )
        if error:
            errors.append(f"{repo_name}: {error}")
            continue
        remaining_docs = None if args.max_docs is None else max(0, args.max_docs - total_matched)
        remaining_chunks = (
            None if args.max_chunks is None else max(0, args.max_chunks - len(all_chunks))
        )
        if remaining_docs == 0 or remaining_chunks == 0:
            break
        read_count, matched_count, chunks = _collect_chunks(
            repo_name,
            repo_dir,
            profile=args.profile,
            include_all=args.include_all,
            limit_files=args.limit_files,
            max_docs=remaining_docs,
            max_chunks=remaining_chunks,
            since=since,
            status_filter=args.status,
            identifiers=identifiers,
        )
        total_read += read_count
        total_matched += matched_count
        all_chunks.extend(chunks)
        print(
            f"[legalize] {repo_name}: normas_leidas={read_count}, filtradas={matched_count}, chunks={len(chunks)}"
        )

    if not all_chunks:
        print("[legalize] Sin chunks para insertar.")
        for error in errors:
            print(f"[error] {error}")
        return 1 if errors else 0

    estimated_tokens = _estimate_tokens([chunk.text for chunk in all_chunks])
    existing_ids = set() if args.force else _existing_point_ids(settings.QDRANT_COLLECTION)
    chunks_to_insert = [
        chunk for chunk in all_chunks if args.force or chunk.point_id not in existing_ids
    ]
    existing_count = len(all_chunks) - len(chunks_to_insert)
    _print_cost_estimate(
        total_chunks=len(all_chunks),
        total_tokens=estimated_tokens,
        existing_chunks=existing_count,
        billable_chunks=len(chunks_to_insert),
    )

    if args.dry_run_cost:
        print(
            "[legalize] dry-run: no se han generado embeddings ni insertado chunks. "
            f"normas_leidas={total_read}, normas_filtradas={total_matched}, chunks={len(all_chunks)}"
        )
        for error in errors:
            print(f"[error] {error}")
        return 0 if not errors else 2

    if not chunks_to_insert:
        print("[legalize] No hay chunks nuevos para insertar. Usa --force para reingestar.")
        return 0 if not errors else 2

    vectors = await _vectors_for_chunks(
        [chunk.text for chunk in chunks_to_insert], embeddings=args.embeddings
    )
    if len(vectors) != len(chunks_to_insert):
        raise RuntimeError("El numero de embeddings no coincide con los chunks Legalize.")

    store = get_vector_store()
    try:
        store.ensure_collection(settings.QDRANT_COLLECTION, len(vectors[0]))
        points = [
            PointStruct(id=chunk.point_id, vector=vector, payload=chunk.payload)
            for chunk, vector in zip(chunks_to_insert, vectors)
        ]
        store.upsert(settings.QDRANT_COLLECTION, points)
    finally:
        if hasattr(store, "close"):
            store.close()
    clear_retriever_cache()

    print(
        "[legalize] completado: "
        f"normas_leidas={total_read}, normas_filtradas={total_matched}, "
        f"chunks_totales={len(all_chunks)}, chunks_existentes={existing_count}, "
        f"chunks_insertados={len(chunks_to_insert)}, coleccion={settings.QDRANT_COLLECTION}"
    )
    for error in errors:
        print(f"[error] {error}")
    return 0 if not errors else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingiere corpus Legalize Espana/UE en el RAG legal."
    )
    parser.add_argument("--repos", default=None, help="CSV de repos: es,eu,legalize-es,legalize-eu")
    parser.add_argument("--data-dir", default=None, help="Directorio base para clones Legalize.")
    parser.add_argument(
        "--profile",
        default=settings.LEGALIZE_INGEST_PROFILE,
        help="Perfil de filtrado: demo, agro u otro.",
    )
    parser.add_argument(
        "--all",
        dest="include_all",
        action="store_true",
        help="Ingiere todo el corpus sin filtro de perfil.",
    )
    parser.add_argument(
        "--dry-run-cost",
        action="store_true",
        help="Estima chunks/tokens/coste sin generar embeddings ni insertar.",
    )
    parser.add_argument(
        "--max-docs", type=int, default=None, help="Maximo de normas filtradas a ingerir."
    )
    parser.add_argument("--max-chunks", type=int, default=None, help="Maximo de chunks a ingerir.")
    parser.add_argument(
        "--since", default=None, help="Filtra normas actualizadas/publicadas desde YYYY-MM-DD."
    )
    parser.add_argument(
        "--status", default=None, help="Filtra por estado Legalize, por ejemplo vigente."
    )
    parser.add_argument(
        "--identifiers",
        default=None,
        help="CSV de identificadores exactos a ingerir, por ejemplo 32018R0848,BOE-A-2018-14803.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Reingesta chunks aunque ya existan en la coleccion."
    )
    parser.add_argument(
        "--embeddings", dest="embeddings", action="store_true", help="Genera embeddings OpenAI."
    )
    parser.add_argument(
        "--no-embeddings",
        dest="embeddings",
        action="store_false",
        help="Ingesta BM25/offline con vectores cero.",
    )
    parser.add_argument(
        "--no-update", action="store_true", help="No hace git pull si el repo ya existe."
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Usa repos locales existentes y no clona ni actualiza.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Limite de ficheros por repo para smoke tests.",
    )
    parser.set_defaults(embeddings=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(asyncio.run(ingest_legalize(args)))


if __name__ == "__main__":
    main()
