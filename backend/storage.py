from __future__ import annotations

import atexit
import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import List

from fastapi import HTTPException, UploadFile

from backend.deps import settings
from libs.attachments import extract_artifact_from_document
from libs.schemas import AttachmentMeta

MAX_ATTACHMENT_FILES = 6
MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024
READ_CHUNK_SIZE = 1024 * 1024
ALLOWED_ATTACHMENT_EXTENSIONS = {
    ".csv",
    ".doc",
    ".docx",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".tif",
    ".tiff",
    ".txt",
    ".xls",
    ".xlsx",
}
ALLOWED_ATTACHMENT_MIME_PREFIXES = ("image/", "text/")
ALLOWED_ATTACHMENT_MIME_TYPES = {
    "application/csv",
    "application/msword",
    "application/octet-stream",
    "application/pdf",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/csv",
}
EXTRACTABLE_ATTACHMENT_EXTENSIONS = {
    ".doc",
    ".docx",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".tif",
    ".tiff",
    ".txt",
}


class AttachmentStore:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path(settings.ATTACHMENTS_DIR)
        self.db_path = self.base_dir / "attachments.db"
        self._lock = threading.Lock()
        self._closed = False
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS attachments (
                    attachment_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    storage_path TEXT,
                    extracted_text TEXT,
                    summary TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            self.conn.commit()
        self._prune_missing_files()

    def _row_to_meta(self, row: sqlite3.Row) -> AttachmentMeta:
        metadata_raw = row["metadata_json"] or "{}"
        try:
            metadata = json.loads(metadata_raw)
        except json.JSONDecodeError:
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        return AttachmentMeta(
            attachment_id=row["attachment_id"],
            filename=row["filename"],
            content_type=row["content_type"],
            size_bytes=int(row["size_bytes"]),
            storage_path=row["storage_path"],
            extracted_text=row["extracted_text"],
            summary=row["summary"],
            metadata=metadata,
        )

    def _upsert_meta(self, meta: AttachmentMeta) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO attachments (
                    attachment_id, filename, content_type, size_bytes,
                    storage_path, extracted_text, summary, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(attachment_id) DO UPDATE SET
                    filename = excluded.filename,
                    content_type = excluded.content_type,
                    size_bytes = excluded.size_bytes,
                    storage_path = excluded.storage_path,
                    extracted_text = excluded.extracted_text,
                    summary = excluded.summary,
                    metadata_json = excluded.metadata_json
                """,
                (
                    meta.attachment_id,
                    meta.filename,
                    meta.content_type,
                    meta.size_bytes,
                    meta.storage_path,
                    meta.extracted_text,
                    meta.summary,
                    json.dumps(meta.metadata, ensure_ascii=False),
                ),
            )
            self.conn.commit()

    def _hydrate_extraction(self, meta: AttachmentMeta) -> AttachmentMeta:
        if not meta.storage_path:
            return meta
        path = Path(meta.storage_path)
        if not path.exists():
            return meta
        suffix = path.suffix.lower()
        extraction_meta = meta.metadata.get("extraction")
        if (
            suffix not in EXTRACTABLE_ATTACHMENT_EXTENSIONS
            or (meta.extracted_text and extraction_meta)
        ):
            return meta
        artifact = extract_artifact_from_document(str(path))
        extraction = {
            "kind": artifact.kind,
            "extractor": artifact.extractor,
            "confidence": artifact.confidence,
            "warnings": artifact.warnings,
            **artifact.metadata,
        }
        hydrated = meta.model_copy(
            update={
                "extracted_text": artifact.text or meta.extracted_text,
                "summary": artifact.summary or meta.summary,
                "metadata": {**meta.metadata, "extraction": extraction},
            }
        )
        self._upsert_meta(hydrated)
        return hydrated

    def _delete_attachment(self, attachment_id: str) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM attachments WHERE attachment_id = ?", (attachment_id,))
            self.conn.commit()

    def _prune_missing_files(self) -> None:
        stale_ids: list[str] = []
        with self._lock:
            rows = self.conn.execute(
                "SELECT attachment_id, storage_path FROM attachments"
            ).fetchall()
        for row in rows:
            storage_path = row["storage_path"]
            if storage_path and Path(storage_path).exists():
                continue
            stale_ids.append(row["attachment_id"])
        if not stale_ids:
            return
        with self._lock:
            self.conn.executemany(
                "DELETE FROM attachments WHERE attachment_id = ?",
                [(attachment_id,) for attachment_id in stale_ids],
            )
            self.conn.commit()

    def _validate_file_contract(self, upload: UploadFile) -> str:
        suffix = Path(upload.filename or "").suffix.lower()
        content_type = (upload.content_type or "application/octet-stream").lower()
        mime_allowed = content_type in ALLOWED_ATTACHMENT_MIME_TYPES or content_type.startswith(
            ALLOWED_ATTACHMENT_MIME_PREFIXES
        )
        if suffix not in ALLOWED_ATTACHMENT_EXTENSIONS or not mime_allowed:
            raise HTTPException(
                status_code=415,
                detail={
                    "error": "unsupported_attachment_type",
                    "message": "Tipo de adjunto no permitido.",
                    "filename": upload.filename,
                    "content_type": upload.content_type,
                },
            )
        return suffix

    async def _read_limited(self, upload: UploadFile) -> bytes:
        chunks: list[bytes] = []
        total = 0
        while True:
            try:
                chunk = await upload.read(READ_CHUNK_SIZE)
            except TypeError:
                chunk = await upload.read()
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_ATTACHMENT_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail={
                        "error": "attachment_too_large",
                        "message": "Attachment exceeds maximum allowed size.",
                        "max_bytes": MAX_ATTACHMENT_BYTES,
                        "filename": upload.filename,
                    },
                )
            chunks.append(chunk)
        return b"".join(chunks)

    async def save_files(self, files: List[UploadFile]) -> List[AttachmentMeta]:
        if len(files) > MAX_ATTACHMENT_FILES:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "too_many_attachments",
                    "message": "Se han enviado demasiados adjuntos.",
                    "max_files": MAX_ATTACHMENT_FILES,
                },
            )
        saved: List[AttachmentMeta] = []
        self.base_dir.mkdir(parents=True, exist_ok=True)
        for upload in files:
            attachment_id = uuid.uuid4().hex
            suffix = self._validate_file_contract(upload)
            path = self.base_dir / f"{attachment_id}{suffix}"
            content = await self._read_limited(upload)
            path.write_bytes(content)
            meta = AttachmentMeta(
                attachment_id=attachment_id,
                filename=upload.filename or path.name,
                content_type=upload.content_type or "application/octet-stream",
                size_bytes=len(content),
                storage_path=str(path),
            )
            self._upsert_meta(meta)
            saved.append(meta)
        return saved

    def get(self, attachment_id: str) -> AttachmentMeta | None:
        with self._lock:
            row = self.conn.execute(
                """
                SELECT attachment_id, filename, content_type, size_bytes,
                       storage_path, extracted_text, summary, metadata_json
                FROM attachments
                WHERE attachment_id = ?
                """,
                (attachment_id,),
            ).fetchone()
        if row is None:
            return None
        meta = self._row_to_meta(row)
        if meta.storage_path and Path(meta.storage_path).exists():
            return self._hydrate_extraction(meta)
        self._delete_attachment(attachment_id)
        return None

    def list(self, attachment_ids: List[str]) -> List[AttachmentMeta]:
        metas: List[AttachmentMeta] = []
        for attachment_id in attachment_ids:
            meta = self.get(attachment_id)
            if meta is not None:
                metas.append(meta)
        return metas

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self.conn.close()
            self._closed = True


attachments_store = AttachmentStore()
atexit.register(attachments_store.close)
