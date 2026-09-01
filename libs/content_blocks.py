from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from libs.schemas import AgentRef, ContentBlock, FinalAnswer, ResolvedCitation

_BLOCK_RE = re.compile(
    r"^BLOCK\s+type=(\w+)(?:\s+ref_id=(\S+))?(?:\s+title=(.+?))?\s*\n(.*?)\n/BLOCK$",
    re.MULTILINE | re.DOTALL,
)
_CITATION_RE = re.compile(r"\[(\d+)\]")


def parse_block_markers(markdown: str) -> tuple[str, List[ContentBlock]]:
    """Extract BLOCK type=X ... /BLOCK markers from markdown.

    Returns cleaned markdown and list of ContentBlock objects.
    """
    blocks: List[ContentBlock] = []
    if not markdown:
        return markdown, blocks

    def _replace(match: re.Match) -> str:
        block_type = match.group(1)
        ref_id = match.group(2) or f"parsed-{len(blocks)}"
        title = match.group(3) or ""
        raw_data = match.group(4) or "{}"
        try:
            data = json.loads(raw_data)
        except (json.JSONDecodeError, TypeError):
            data = {"raw": raw_data}
        blocks.append(ContentBlock(
            block_type=block_type,
            ref_id=ref_id,
            title=title,
            data=data,
        ))
        return f"{{ref:{ref_id}}}"

    cleaned = _BLOCK_RE.sub(_replace, markdown)
    return cleaned, blocks


def resolve_citations(
    markdown: str,
    references: List[AgentRef],
) -> List[ResolvedCitation]:
    """Find [N] citation markers and map them to AgentRef entries."""
    if not markdown or not references:
        return []

    resolved: List[ResolvedCitation] = []
    for match in _CITATION_RE.finditer(markdown):
        idx = int(match.group(1))
        # Out-of-range citations are silently ignored — the raw [N] text
        # remains in the markdown and renders as plain text.
        if 1 <= idx <= len(references):
            ref = references[idx - 1]
            resolved.append(ResolvedCitation(
                index=idx,
                ref_id=ref.ref_id,
                start_char=match.start(),
                end_char=match.end(),
            ))
    return resolved


def auto_generate_blocks(final: FinalAnswer) -> List[ContentBlock]:
    """Generate ContentBlocks from structured data when not already present."""
    blocks: List[ContentBlock] = []
    existing_ids = {b.ref_id for b in final.content_blocks}

    _gen_time_series(blocks, existing_ids, final)
    _gen_threshold_callouts(blocks, existing_ids, final)
    _gen_scene_table(blocks, existing_ids, final)
    _gen_focus_callouts(blocks, existing_ids, final)
    _gen_satellite_images(blocks, existing_ids, final)
    _gen_legal_sources(blocks, existing_ids, final)
    _gen_web_sources(blocks, existing_ids, final)

    return blocks


def _gen_time_series(
    blocks: List[ContentBlock],
    existing: set[str],
    final: FinalAnswer,
) -> None:
    rs = final.remote_sensing
    if not rs or not rs.time_series or len(rs.time_series) < 2:
        return
    if "auto-ts" in existing:
        return

    points = [
        {"date": p.date, "value": p.mean}
        for p in rs.time_series
        if p.mean is not None
    ]
    if len(points) < 2:
        return

    metric_name = "Índice"
    for change in rs.temporal_changes or []:
        if change.metric:
            metric_name = change.metric.upper()
            break
    series = [{"label": metric_name, "points": points}]

    thresholds_data = []
    for change in rs.temporal_changes or []:
        if change.threshold and change.metric:
            thresholds_data.append({
                "label": f"Ref {change.metric}",
                "range": change.threshold.reference_range,
                "status": change.threshold.status,
            })

    data: Dict[str, Any] = {"series": series}
    if thresholds_data:
        data["thresholds"] = thresholds_data

    blocks.append(ContentBlock(
        block_type="chart",
        ref_id="auto-ts",
        title="Evolucion temporal del indice",
        data=data,
    ))


def _gen_threshold_callouts(
    blocks: List[ContentBlock],
    existing: set[str],
    final: FinalAnswer,
) -> None:
    rs = final.remote_sensing
    if not rs:
        return

    for i, change in enumerate(rs.temporal_changes or []):
        ref_id = f"auto-threshold-{i}"
        if ref_id in existing or not change.threshold:
            continue
        variant = "alert" if change.threshold.status == "below" else "info"
        if change.threshold.status == "above":
            variant = "warning"
        blocks.append(ContentBlock(
            block_type="callout",
            ref_id=ref_id,
            title=f"Umbral {change.metric or 'indice'}",
            data={
                "variant": variant,
                "message": change.threshold.message,
            },
        ))


def _gen_scene_table(
    blocks: List[ContentBlock],
    existing: set[str],
    final: FinalAnswer,
) -> None:
    rs = final.remote_sensing
    if not rs or not rs.insights:
        return
    if "auto-scenes" in existing:
        return

    headers = ["Escena", "Indice", "Media", "Calidad", "Confianza"]
    rows = []
    for ins in rs.insights:
        mean_val = "-"
        if ins.stats and ins.stats.mean is not None:
            mean_val = f"{ins.stats.mean:.3f}"
        quality_val = ins.quality.label if ins.quality else "-"
        rows.append([
            ins.item_id[:16],
            ins.product_label or "-",
            mean_val,
            quality_val,
            f"{ins.confidence:.0%}",
        ])

    blocks.append(ContentBlock(
        block_type="table",
        ref_id="auto-scenes",
        title="Resumen de escenas",
        data={"headers": headers, "rows": rows},
    ))


def _gen_focus_callouts(
    blocks: List[ContentBlock],
    existing: set[str],
    final: FinalAnswer,
) -> None:
    rs = final.remote_sensing
    if not rs:
        return

    for i, focus in enumerate(rs.focus_areas or []):
        ref_id = f"auto-focus-{i}"
        if ref_id in existing:
            continue
        if focus.priority != "alta":
            continue
        blocks.append(ContentBlock(
            block_type="callout",
            ref_id=ref_id,
            title=focus.title,
            data={
                "variant": "alert",
                "message": focus.detail,
            },
        ))


def _gen_satellite_images(
    blocks: List[ContentBlock],
    existing: set[str],
    final: FinalAnswer,
) -> None:
    if not final.stac or not final.stac.items:
        return

    count = 0
    for item in final.stac.items:
        if count >= 4:
            break
        for asset in item.assets:
            thumb = getattr(asset, "thumbnail", None)
            if not thumb:
                continue
            ref_id = f"auto-img-{item.id}"
            if ref_id in existing:
                break
            label = item.product_label or item.index_name or item.collection or "Satelite"
            dt_str = (item.datetime or "")[:10]
            blocks.append(ContentBlock(
                block_type="image",
                ref_id=ref_id,
                title=f"{label} - {dt_str}",
                data={
                    "src": thumb,
                    "alt": label,
                    "caption": f"{label} ({dt_str}) - Calidad: {getattr(item.quality, 'label', '-')}",
                },
            ))
            count += 1
            break


def _gen_legal_sources(
    blocks: List[ContentBlock],
    existing: set[str],
    final: FinalAnswer,
) -> None:
    if not final.legal or not final.legal.references:
        return
    if "auto-legal" in existing:
        return

    refs = final.legal.references[:8]
    lines = []
    for r in refs:
        if r.url:
            lines.append(f"- [{r.title}]({r.url})")
        else:
            lines.append(f"- {r.title}")

    blocks.append(ContentBlock(
        block_type="callout",
        ref_id="auto-legal",
        title="Referencias legales",
        data={
            "variant": "info",
            "message": "\n".join(lines),
        },
    ))


def _gen_web_sources(
    blocks: List[ContentBlock],
    existing: set[str],
    final: FinalAnswer,
) -> None:
    web_refs = [r for r in final.references if r.source == "web"]
    if len(web_refs) < 2:
        return
    if "auto-web-refs" in existing:
        return

    headers = ["Fuente", "Enlace"]
    rows = [[r.title, r.url or r.ref_id] for r in web_refs[:10]]

    blocks.append(ContentBlock(
        block_type="table",
        ref_id="auto-web-refs",
        title="Fuentes web consultadas",
        data={"headers": headers, "rows": rows},
    ))
