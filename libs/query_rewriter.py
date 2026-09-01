from __future__ import annotations

from typing import Any

from loguru import logger

from agents.base import _build_client
from backend.deps import settings
from libs.context_engineering import summarize_conversation_history
from libs.costs.tracker import cost_context, record_openai_chat_usage
from libs.openai_compat import chat_temperature_kwargs


_REWRITE_SYSTEM = """\
Eres un reescritor de consultas. Dado el historial de una conversación \
entre un usuario y un asistente, reescribe la última pregunta del usuario \
para que sea autocontenida y clara.

Reglas:
- Resuelve pronombres y referencias ambiguas ("y en Madrid", "y eso", \
"¿también?", "¿y allí?")
- Mantén el significado exacto del usuario
- SOLO devuelve el query reescrito, sin explicaciones ni comillas
- Si la pregunta ya es autocontenida, devuélvela sin cambios"""


async def rewrite_query(raw_query: str, history: list[dict[str, Any]]) -> str:
    """Reescribe queries ambiguas usando el historial conversacional.

    Devuelve el query original si no hay historial suficiente o si falla
    la llamada LLM.
    """
    if not history or len(history) < 2:
        return raw_query

    formatted = summarize_conversation_history(history, max_turns=4)
    if not formatted or formatted.startswith("Sin historial"):
        return raw_query

    try:
        client = _build_client("openai")
        model = settings.resolve_openai_model("OPENAI_MODEL_QUERY_REWRITER")
        user_prompt = (
            f"Historial reciente:\n{formatted}\n\n"
            f"Pregunta actual del usuario: {raw_query}\n\n"
            f"Query reescrito:"
        )
        with cost_context(agent="query_rewriter", operation="query_rewrite"):
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _REWRITE_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=150,
            )
            if getattr(resp, "usage", None) is not None:
                record_openai_chat_usage(
                    model,
                    resp.usage,
                    operation="query_rewrite",
                    provider="openai",
                )
        rewritten = (resp.choices[0].message.content or raw_query).strip()
        if rewritten and rewritten.lower() != raw_query.lower():
            logger.info(
                "Query rewriting: '{}' -> '{}'", raw_query[:80], rewritten[:80]
            )
            return rewritten
        return raw_query
    except Exception:
        logger.warning("Query rewriting failed, using raw query")
        return raw_query
