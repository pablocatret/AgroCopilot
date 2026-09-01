from pathlib import Path

import pytest
from fastapi import HTTPException

from backend.storage import MAX_ATTACHMENT_BYTES, MAX_ATTACHMENT_FILES, AttachmentStore


class FakeUploadFile:
    def __init__(self, name: str, content: bytes, content_type: str) -> None:
        self.filename = name
        self.content_type = content_type
        self._content = content
        self._offset = 0

    async def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            chunk = self._content[self._offset :]
            self._offset = len(self._content)
            return chunk
        chunk = self._content[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


def build_upload_file(name: str, content: bytes, content_type: str) -> FakeUploadFile:
    return FakeUploadFile(name, content, content_type)


@pytest.mark.asyncio
async def test_attachment_store_persists_metadata_to_disk(tmp_path: Path):
    store = AttachmentStore(base_dir=tmp_path)
    upload = build_upload_file("demo.txt", b"hola mundo", "text/plain")

    saved = await store.save_files([upload])
    store.close()

    reloaded = AttachmentStore(base_dir=tmp_path)
    listed = reloaded.list([saved[0].attachment_id])

    assert len(listed) == 1
    assert listed[0].filename == "demo.txt"
    assert Path(listed[0].storage_path).exists()
    assert listed[0].summary
    assert listed[0].metadata["extraction"]["extractor"] == "plain_text"
    reloaded.close()


@pytest.mark.asyncio
async def test_attachment_store_prunes_missing_files_on_reload(tmp_path: Path):
    store = AttachmentStore(base_dir=tmp_path)
    upload = build_upload_file("ghost.txt", b"contenido", "text/plain")

    saved = await store.save_files([upload])
    stored_path = Path(saved[0].storage_path)
    stored_path.unlink()
    store.close()

    reloaded = AttachmentStore(base_dir=tmp_path)
    listed = reloaded.list([saved[0].attachment_id])

    assert listed == []


@pytest.mark.asyncio
async def test_attachment_store_get_removes_stale_entry(tmp_path: Path):
    store = AttachmentStore(base_dir=tmp_path)
    upload = build_upload_file("stale.txt", b"hola", "text/plain")

    saved = await store.save_files([upload])
    attachment_id = saved[0].attachment_id
    Path(saved[0].storage_path).unlink()

    assert store.get(attachment_id) is None
    assert store.list([attachment_id]) == []
    store.close()


@pytest.mark.asyncio
async def test_attachment_store_rejects_too_many_files(tmp_path: Path):
    store = AttachmentStore(base_dir=tmp_path)
    uploads = [
        build_upload_file(f"demo-{idx}.txt", b"hola", "text/plain")
        for idx in range(MAX_ATTACHMENT_FILES + 1)
    ]

    with pytest.raises(HTTPException) as exc:
        await store.save_files(uploads)

    assert exc.value.status_code == 413
    assert exc.value.detail["error"] == "too_many_attachments"
    store.close()


@pytest.mark.asyncio
async def test_attachment_store_rejects_oversized_file(tmp_path: Path):
    store = AttachmentStore(base_dir=tmp_path)
    upload = build_upload_file("large.txt", b"x" * (MAX_ATTACHMENT_BYTES + 1), "text/plain")

    with pytest.raises(HTTPException) as exc:
        await store.save_files([upload])

    assert exc.value.status_code == 413
    assert exc.value.detail["error"] == "attachment_too_large"
    store.close()


@pytest.mark.asyncio
async def test_attachment_store_rejects_unsupported_type(tmp_path: Path):
    store = AttachmentStore(base_dir=tmp_path)
    upload = build_upload_file("script.exe", b"not allowed", "application/x-msdownload")

    with pytest.raises(HTTPException) as exc:
        await store.save_files([upload])

    assert exc.value.status_code == 415
    assert exc.value.detail["error"] == "unsupported_attachment_type"
    store.close()
