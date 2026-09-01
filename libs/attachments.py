from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tempfile
from typing import Dict, List
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd
from pypdf import PdfReader
import fitz  # PyMuPDF

from libs.ocr import get_ocr_backend


@dataclass
class ExtractionArtifact:
    kind: str
    text: str
    summary: str
    extractor: str
    confidence: float | None = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class TabularSummary:
    columns: List[str]
    row_count: int
    missing: Dict[str, int]
    numeric_summary: Dict[str, Dict[str, float]]
    sample: List[Dict[str, object]]


def _summarize_text(text: str, limit: int = 400) -> str:
    clean = " ".join((text or "").split()).strip()
    if not clean:
        return ""
    return clean[:limit] + ("..." if len(clean) > limit else "")


def _docx_text(path: str) -> ExtractionArtifact:
    warnings: List[str] = []
    paragraphs: List[str] = []
    try:
        with zipfile.ZipFile(path) as zf:
            xml_bytes = zf.read("word/document.xml")
        root = ET.fromstring(xml_bytes)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        for paragraph in root.findall(".//w:p", ns):
            runs = [
                node.text.strip()
                for node in paragraph.findall(".//w:t", ns)
                if isinstance(node.text, str) and node.text.strip()
            ]
            if runs:
                paragraphs.append("".join(runs))
    except KeyError:
        warnings.append("DOCX without readable word/document.xml.")
    except Exception:
        warnings.append("Could not parse DOCX.")
    text = "\n".join(paragraphs).strip()
    return ExtractionArtifact(
        kind="document",
        text=text,
        summary=_summarize_text(text) or "DOCX document without readable text.",
        extractor="docx_xml",
        confidence=0.92 if text else 0.0,
        warnings=warnings,
        metadata={"format": "docx", "paragraphs": len(paragraphs)},
    )


def extract_artifact_from_pdf(path: str) -> ExtractionArtifact:
    warnings: List[str] = []
    text = ""
    page_count = 0
    try:
        reader = PdfReader(path)
        page_count = len(reader.pages)
        text_parts = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(text_parts).strip()
    except Exception:
        warnings.append("PDF text extraction failed; attempting OCR.")
    if text:
        return ExtractionArtifact(
            kind="document",
            text=text,
            summary=_summarize_text(text),
            extractor="pypdf",
            confidence=0.96,
            warnings=warnings,
            metadata={"format": "pdf", "pages": page_count},
        )
    try:
        doc = fitz.open(path)
        page_count = len(doc)
        ocr = get_ocr_backend()
        parts: List[str] = []
        confidences: List[float] = []
        with tempfile.TemporaryDirectory(prefix="ocr-pdf-") as tmp_dir:
            tmp_root = Path(tmp_dir)
            for index, page in enumerate(doc, start=1):
                pix = page.get_pixmap(dpi=220)
                tmp_path = tmp_root / f"page-{index}.png"
                pix.save(tmp_path)
                result = ocr.extract(str(tmp_path))
                if result.text:
                    parts.append(result.text)
                if result.confidence is not None:
                    confidences.append(result.confidence)
                for item in result.warnings or []:
                    if item not in warnings:
                        warnings.append(item)
        text = "\n".join([p for p in parts if p]).strip()
        confidence = round(sum(confidences) / len(confidences), 3) if confidences else None
        return ExtractionArtifact(
            kind="document",
            text=text,
            summary=_summarize_text(text) or "Scanned PDF without sufficient OCR text.",
            extractor="ocr_pdf",
            confidence=confidence,
            warnings=warnings,
            metadata={"format": "pdf", "pages": page_count},
        )
    except Exception:
        warnings.append("Could not extract text from PDF.")
        return ExtractionArtifact(
            kind="document",
            text="",
            summary="PDF without extractable text.",
            extractor="pdf_failed",
            confidence=None,
            warnings=warnings,
            metadata={"format": "pdf", "pages": page_count},
        )


def extract_artifact_from_image(path: str) -> ExtractionArtifact:
    ocr = get_ocr_backend()
    result = ocr.extract(path)
    return ExtractionArtifact(
        kind="image",
        text=result.text,
        summary=_summarize_text(result.text) or "Image without sufficient OCR text.",
        extractor="ocr_image",
        confidence=result.confidence,
        warnings=list(result.warnings or []),
        metadata=dict(result.metadata or {}),
    )


def extract_artifact_from_document(path: str) -> ExtractionArtifact:
    lower = path.lower()
    if lower.endswith(".pdf"):
        return extract_artifact_from_pdf(path)
    if lower.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
        return extract_artifact_from_image(path)
    if lower.endswith(".doc"):
        return ExtractionArtifact(
            kind="document",
            text="",
            summary="Documento .doc sin extractor compatible.",
            extractor="doc_unsupported",
            confidence=None,
            warnings=[
                "El formato .doc esta admitido como adjunto, pero no tiene extraccion estructurada disponible.",
                "Conviene convertirlo a DOCX o PDF legible antes del analisis.",
            ],
            metadata={"format": "doc", "supported": False},
        )
    if lower.endswith(".docx"):
        return _docx_text(path)
    try:
        text = Path(path).read_text(encoding="utf-8")
        return ExtractionArtifact(
            kind="document",
            text=text.strip(),
            summary=_summarize_text(text),
            extractor="plain_text",
            confidence=0.98 if text.strip() else 0.0,
            metadata={"format": Path(path).suffix.lower().lstrip(".")},
        )
    except Exception:
        return ExtractionArtifact(
            kind="document",
            text="",
            summary="Document without extractable text.",
            extractor="read_failed",
            confidence=None,
            warnings=["Could not read file as text or DOCX."],
            metadata={"format": Path(path).suffix.lower().lstrip(".")},
        )


def extract_text_from_pdf(path: str) -> str:
    return extract_artifact_from_pdf(path).text


def extract_text_from_image(path: str) -> str:
    return extract_artifact_from_image(path).text


def extract_text_from_document(path: str) -> str:
    return extract_artifact_from_document(path).text


def summarize_table(path: str, max_rows: int = 5) -> TabularSummary:
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    columns = list(df.columns)
    row_count = int(df.shape[0])
    missing = {col: int(df[col].isna().sum()) for col in columns}
    numeric_summary: Dict[str, Dict[str, float]] = {}
    for col in df.select_dtypes(include=["number"]).columns:
        desc = df[col].describe()
        numeric_summary[col] = {
            "mean": float(desc.get("mean", 0.0)),
            "min": float(desc.get("min", 0.0)),
            "max": float(desc.get("max", 0.0)),
        }
    sample = df.head(max_rows).fillna("").to_dict(orient="records")
    return TabularSummary(
        columns=columns,
        row_count=row_count,
        missing=missing,
        numeric_summary=numeric_summary,
        sample=sample,
    )
