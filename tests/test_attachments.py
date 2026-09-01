from pathlib import Path
import zipfile

from PIL import Image

import pytest

from agents.document_analyst import DocumentAnalystAgent
from agents.spreadsheet_analyst import SpreadsheetAnalystAgent
from agents.vision_ocr import VisionOcrAgent
from libs.attachments import extract_artifact_from_document
from libs.schemas import AgentInput, AttachmentMeta


def build_docx(path: Path, text: str) -> None:
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>{text}</w:t></w:r></w:p>
  </w:body>
</w:document>
"""
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", document_xml)


@pytest.mark.asyncio
async def test_document_agent_with_text(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(DocumentAnalystAgent, "external_enabled", lambda self: False)
    doc_path = tmp_path / "note.txt"
    doc_path.write_text("Informe agronómico básico.", encoding="utf-8")
    attachment = AttachmentMeta(
        attachment_id="doc-1",
        filename="note.txt",
        content_type="text/plain",
        size_bytes=doc_path.stat().st_size,
        storage_path=str(doc_path),
    )
    output = await DocumentAnalystAgent().run(AgentInput(query="demo", attachments=[attachment]))
    assert output.data["documents"]


@pytest.mark.asyncio
async def test_spreadsheet_agent_with_csv(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(SpreadsheetAnalystAgent, "external_enabled", lambda self: False)
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("cultivo,kg\nmaiz,10\ntrigo,20\n", encoding="utf-8")
    attachment = AttachmentMeta(
        attachment_id="sheet-1",
        filename="data.csv",
        content_type="text/csv",
        size_bytes=csv_path.stat().st_size,
        storage_path=str(csv_path),
    )
    output = await SpreadsheetAnalystAgent().run(AgentInput(query="demo", attachments=[attachment]))
    assert output.data["tables"][0]["row_count"] == 2


@pytest.mark.asyncio
async def test_vision_agent_with_image(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(VisionOcrAgent, "external_enabled", lambda self: False)
    img_path = tmp_path / "photo.png"
    Image.new("RGB", (32, 32), color="white").save(img_path)
    attachment = AttachmentMeta(
        attachment_id="img-1",
        filename="photo.png",
        content_type="image/png",
        size_bytes=img_path.stat().st_size,
        storage_path=str(img_path),
    )
    output = await VisionOcrAgent().run(AgentInput(query="demo", attachments=[attachment]))
    assert output.data["images"]


def test_extract_artifact_from_docx_reads_document_xml(tmp_path: Path):
    docx_path = tmp_path / "memo.docx"
    build_docx(docx_path, "Informe de cultivo de prueba")

    artifact = extract_artifact_from_document(str(docx_path))

    assert artifact.extractor == "docx_xml"
    assert "cultivo" in artifact.text.lower()
    assert artifact.confidence == pytest.approx(0.92)


def test_extract_artifact_from_doc_marks_unsupported_extraction(tmp_path: Path):
    doc_path = tmp_path / "legacy.doc"
    doc_path.write_bytes(b"legacy-binary-placeholder")

    artifact = extract_artifact_from_document(str(doc_path))

    assert artifact.extractor == "doc_unsupported"
    assert artifact.metadata["supported"] is False
    assert artifact.warnings


@pytest.mark.asyncio
async def test_document_agent_surfaces_llm_enrichment_failure_as_limitation(tmp_path: Path, monkeypatch):
    doc_path = tmp_path / "note.txt"
    doc_path.write_text("Resumen tecnico de parcela.", encoding="utf-8")
    attachment = AttachmentMeta(
        attachment_id="doc-2",
        filename="note.txt",
        content_type="text/plain",
        size_bytes=doc_path.stat().st_size,
        storage_path=str(doc_path),
    )
    async def fail_llm(*args, **kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr(DocumentAnalystAgent, "call_llm_json", fail_llm)
    monkeypatch.setattr(DocumentAnalystAgent, "external_enabled", lambda self: True)

    output = await DocumentAnalystAgent().run(AgentInput(query="demo", attachments=[attachment]))

    assert output.data["documents"]
    limitations = output.data["documents"][0]["limitations"]
    assert "Could not enrich document with LLM analysis." in limitations


@pytest.mark.asyncio
async def test_document_agent_accepts_html_attachment(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(DocumentAnalystAgent, "external_enabled", lambda self: False)
    doc_path = tmp_path / "guide.html"
    doc_path.write_text("<h1>Materia orgánica</h1><p>Guía de suelo.</p>", encoding="utf-8")
    attachment = AttachmentMeta(
        attachment_id="doc-html",
        filename="guide.html",
        content_type="text/html",
        size_bytes=doc_path.stat().st_size,
        storage_path=str(doc_path),
    )
    output = await DocumentAnalystAgent().run(AgentInput(query="demo", attachments=[attachment]))
    assert output.data["documents"]


@pytest.mark.asyncio
async def test_vision_agent_surfaces_llm_enrichment_failure_as_limitation(tmp_path: Path, monkeypatch):
    img_path = tmp_path / "photo2.png"
    Image.new("RGB", (32, 32), color="white").save(img_path)
    attachment = AttachmentMeta(
        attachment_id="img-2",
        filename="photo2.png",
        content_type="image/png",
        size_bytes=img_path.stat().st_size,
        storage_path=str(img_path),
    )
    async def fail_llm(*args, **kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr(VisionOcrAgent, "call_llm_vision_json", fail_llm)
    monkeypatch.setattr(VisionOcrAgent, "external_enabled", lambda self: True)

    output = await VisionOcrAgent().run(AgentInput(query="demo", attachments=[attachment]))

    assert output.data["images"]
    limitations = output.data["images"][0]["limitations"]
    assert "Could not enrich image with LLM analysis." in limitations
