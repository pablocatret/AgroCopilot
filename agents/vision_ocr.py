from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import List

from agents.base import BaseAgent
from backend.deps import settings
from libs.attachments import extract_artifact_from_image
from libs.context_engineering import summarize_memory_context
from libs.prompts import compose_system_prompt, render_prompt
from libs.schemas import AgentInput, AgentRef, AgentRefs, VisionAgentOutput

VISION_ANALYSIS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "images": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "attachment_id": {"type": "string"},
                    "filename": {"type": "string"},
                    "used_for": {"type": "string"},
                    "confidence": {"type": ["number", "null"]},
                    "key_signals": {"type": "array", "items": {"type": "string"}, "default": []},
                    "limitations": {"type": "array", "items": {"type": "string"}, "default": []},
                },
                "required": [
                    "attachment_id",
                    "filename",
                    "used_for",
                    "confidence",
                    "key_signals",
                    "limitations",
                ],
            },
            "default": [],
        }
    },
    "required": ["images"],
}


def _image_data_url(path: str, content_type: str) -> str | None:
    """Encode a local image for the multimodal chat API.

    OCR alone cannot expose symptoms such as foliar lesions to the model. The
    evaluation runner supplies local fixture paths, so the vision agent must
    carry the image bytes as well as its OCR text.
    """
    source = Path(path)
    if not source.is_file():
        return None
    mime = content_type if content_type.startswith("image/") else mimetypes.guess_type(source.name)[0]
    if not mime or not mime.startswith("image/"):
        return None
    encoded = base64.b64encode(source.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


class VisionOcrAgent(BaseAgent):
    name = "vision_ocr"
    output_model = VisionAgentOutput
    _provider_key = "LLM_PROVIDER_VISION_OCR"

    def __init__(self) -> None:
        super().__init__()
        self.model = settings.resolve_openai_model(
            "OPENAI_MODEL_VISION_OCR",
            "OPENAI_MODEL_VISION",
        )

    async def _run(self, agent_input: AgentInput) -> VisionAgentOutput:
        images = []
        image_data_urls: List[str] = []
        refs: List[AgentRef] = []
        for attachment in agent_input.attachments:
            fname = attachment.filename.lower()
            if not fname.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                continue
            if not attachment.storage_path:
                continue
            extraction_meta = (
                attachment.metadata.get("extraction", {})
                if isinstance(attachment.metadata.get("extraction"), dict)
                else {}
            )
            confidence = extraction_meta.get("confidence")
            limitations = [
                str(item).strip()
                for item in (extraction_meta.get("warnings") or [])
                if str(item).strip()
            ]
            text = (attachment.extracted_text or attachment.summary or "").strip()
            if not text:
                artifact = extract_artifact_from_image(attachment.storage_path)
                text = artifact.text
                confidence = artifact.confidence
                limitations = [item for item in artifact.warnings if item]
            excerpt = (text[:300] + "...") if len(text) > 300 else text
            images.append(
                {
                    "attachment_id": attachment.attachment_id,
                    "filename": attachment.filename,
                    "ocr_text": text,
                    "excerpt": excerpt,
                    "used_for": "Aun sin clasificar",
                    "confidence": confidence,
                    "key_signals": [],
                    "limitations": limitations,
                }
            )
            data_url = _image_data_url(attachment.storage_path, attachment.content_type)
            if data_url:
                image_data_urls.append(data_url)
            refs.append(
                AgentRef(
                    ref_id=f"img-{attachment.attachment_id}",
                    title=attachment.filename,
                    source="vision",
                    url=f"attachment:{attachment.attachment_id}",
                    snippet=excerpt,
                )
            )

        if images and self.external_enabled():
            mission = str((agent_input.context or {}).get("mission") or "").strip()
            user = (
                f"{render_prompt('vision_user.txt', query=agent_input.query, decision_mode=agent_input.decision_mode, memory_summary=summarize_memory_context(str(agent_input.context.get('user_memory', '') or '')), extracted_context=json.dumps(images, ensure_ascii=False))}\n\n"
                + (f"Misión: {mission}\n\n" if mission else "")
                + "Devuelve SOLO JSON conforme a este esquema:\n"
                f"{json.dumps(VISION_ANALYSIS_SCHEMA, ensure_ascii=False)}"
            )
            try:
                system = compose_system_prompt(
                    agent_name="vision_ocr",
                    body=render_prompt("vision_system.txt"),
                    output_contract="Devuelve exclusivamente JSON valido siguiendo el esquema proporcionado.",
                )
                if image_data_urls:
                    data = await self.call_llm_vision_json(
                        system=system,
                        images=image_data_urls,
                        question=user,
                        schema=VISION_ANALYSIS_SCHEMA,
                        temperature=0.1,
                    )
                else:
                    data = await self.call_llm_json(
                        system=system,
                        user=user,
                        schema=VISION_ANALYSIS_SCHEMA,
                        temperature=0.1,
                    )
                enriched = {
                    item["attachment_id"]: item
                    for item in (data.get("images") or [])
                    if item.get("attachment_id")
                }
                for item in images:
                    extra = enriched.get(item["attachment_id"])
                    if not extra:
                        continue
                    item["used_for"] = extra.get("used_for", item["used_for"])
                    item["confidence"] = extra.get("confidence", item["confidence"])
                    item["key_signals"] = extra.get("key_signals") or []
                    item["limitations"] = extra.get("limitations") or item["limitations"]
            except Exception:
                for item in images:
                    limitations = list(item.get("limitations") or [])
                    message = "Could not enrich image with LLM analysis."
                    if message not in limitations:
                        limitations.append(message)
                    item["limitations"] = limitations

        summary = "No images for OCR." if not images else "OCR completed."
        return VisionAgentOutput(
            agent=self.name,
            summary=summary,
            refs=AgentRefs(items=refs),
            data={"images": images},
        )
