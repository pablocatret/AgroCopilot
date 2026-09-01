from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image, ImageOps
import pytesseract
from pytesseract import Output

from backend.deps import settings


@dataclass(frozen=True)
class OCRResult:
    text: str
    confidence: float | None = None
    warnings: list[str] | None = None
    metadata: dict[str, object] | None = None


class OCRBackend(Protocol):
    def extract(self, image_path: str) -> OCRResult: ...

    def extract_text(self, image_path: str) -> str: ...


class TesseractOCR:
    @staticmethod
    def _prepare_image(image_path: str) -> Image.Image:
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        img = ImageOps.grayscale(img)
        img = ImageOps.autocontrast(img)
        return img

    @staticmethod
    def _mean_confidence(data: dict[str, list[object]]) -> float | None:
        values: list[float] = []
        for raw in data.get("conf") or []:
            try:
                value = float(str(raw).strip())
            except (TypeError, ValueError):
                continue
            if value >= 0:
                values.append(value)
        if not values:
            return None
        return round(sum(values) / len(values) / 100.0, 3)

    def extract(self, image_path: str) -> OCRResult:
        warnings: list[str] = []
        try:
            img = self._prepare_image(image_path)
            text = (pytesseract.image_to_string(img) or "").strip()
            data = pytesseract.image_to_data(img, output_type=Output.DICT)
            confidence = self._mean_confidence(data)
            if not text:
                warnings.append("OCR without readable text.")
            if confidence is not None and confidence < 0.45:
                warnings.append("Low OCR confidence; validate original.")
            return OCRResult(
                text=text,
                confidence=confidence,
                warnings=warnings,
                metadata={"engine": "tesseract"},
            )
        except Exception:
            return OCRResult(
                text="",
                confidence=None,
                warnings=["OCR not available for this file."],
                metadata={"engine": "tesseract"},
            )

    def extract_text(self, image_path: str) -> str:
        return self.extract(image_path).text


class NoOCR:
    def extract(self, image_path: str) -> OCRResult:
        return OCRResult(
            text="",
            confidence=None,
            warnings=["OCR disabled by configuration."],
            metadata={"engine": "none"},
        )

    def extract_text(self, image_path: str) -> str:
        return ""


def get_ocr_backend() -> OCRBackend:
    backend = (settings.OCR_BACKEND or "tesseract").lower()
    if backend == "none":
        return NoOCR()
    return TesseractOCR()
