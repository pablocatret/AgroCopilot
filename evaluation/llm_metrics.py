"""Juez LLM multi-métricas: genera un vector completo de mÃ©tricas en cada llamada.

Cada llamada produce 10 dimensiones de calidad + 2 meta + 3 coverage + 3 cualitativas.
Diseñado para maximizar la informaciÃ³n obtenida por token gastado.
"""
from __future__ import annotations

import json
from typing import Any

from evaluation.llm_support import call_llm_json, LLMCallTracker
from evaluation.schemas import (
    CaseSpec,
    JudgeDimensionScore,
    JudgeMultiMetrics,
    NormalizedOutput,
)


# â”€â”€ Schema de salida del juez â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

JUDGE_DIMENSIONS = [
    "factual_correctness",
    "domain_accuracy",
    "responsible_action_quality",
    "actionability",
    "decision_support_quality",
    "evidence_utilization",
    "transparent_confidence",
    "case_personalization",
    "practical_value",
    "overall_quality",
]

MULTI_METRIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "scores": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                dim: {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "score": {"type": "integer", "minimum": 1, "maximum": 5},
                        "rationale": {"type": "string"},
                    },
                    "required": ["score", "rationale"],
                }
                for dim in JUDGE_DIMENSIONS
            },
            "required": JUDGE_DIMENSIONS,
        },
        "meta": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "judge_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "perceived_difficulty": {"type": "integer", "minimum": 1, "maximum": 3},
            },
            "required": ["judge_confidence", "perceived_difficulty"],
        },
        "gold_coverage": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "concepts_coverage": {
                    "type": "number", "minimum": 0, "maximum": 1,
                    "description": "FracciÃ³n de must_contain_concepts que la respuesta transmite",
                },
                "actions_coverage": {
                    "type": "number", "minimum": 0, "maximum": 1,
                    "description": "FracciÃ³n de must_actions que la respuesta recomienda",
                },
                "facts_coverage": {
                    "type": "number", "minimum": 0, "maximum": 1,
                    "description": "FracciÃ³n de must_mention_facts que la respuesta menciona",
                },
                "clarification_detected": {
                    "type": "boolean",
                    "description": "True si la respuesta pide informaciÃ³n adicional al usuario",
                },
                "forbidden_claims_violated": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Claims prohibidos que la respuesta AFIRMA activamente (no solo menciona de pasada o descarta)",
                },
            },
            "required": ["concepts_coverage", "actions_coverage", "facts_coverage", "clarification_detected", "forbidden_claims_violated"],
        },
        "qualitative": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "strengths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 5,
                },
                "weaknesses": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 0,
                    "maxItems": 5,
                },
                "missing_elements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 0,
                    "maxItems": 5,
                },
            },
            "required": ["strengths", "weaknesses", "missing_elements"],
        },
    },
    "required": ["scores", "meta", "gold_coverage", "qualitative"],
}


# â”€â”€ Prompt del juez â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

JUDGE_SYSTEM_PROMPT = """\
Eres AgroBench Judge, un evaluador especializado en sistemas de apoyo a decisiones agrÃ­colas.

CONTEXTO CRÃTICO:
Evaluas un COPILO AGRÃCOLA que debe AYUDAR a agricultores a tomar decisiones.

REGLA FUNDAMENTAL:
Un sistema que NUNCA recomienda nada es TAN PELIGROSO como uno que recomienda MAL.

REGLAS ANTI-CAUTELA:
1. LA INACCIÃ“N NO ES PRUDENCIA â€” Un sistema que nunca recomienda nada falla en su funciÃ³n.
2. LA CAUTELA DEBE SER PROPORCIONAL AL RIESGO â€” Bajo riesgo = mÃ¡s directo; Alto riesgo = mÃ¡s cauteloso PERO DAR ORIENTACIÃ“N.
3. DISTINGUIR "NO SÃ‰ NADA" DE "CON LO QUE SÃ‰ PUEDO ORIENTAR" â€” Una cosa es no tener toda la informaciÃ³n, otra es no poder ayudar en nada.
4. LA UTILIDAD ES UNA DIMENSIÃ“N DE CALIDAD â€” Un sistema perfectamente seguro pero inÃºtil NO es bueno.
5. PENALIZAR PASIVIDAD DISFRAZADA â€” "Depende de muchos factores" sin orientaciÃ³n es PASIVIDAD, no prudencia.

RÃšBRICA 1-5:
  1 = Deficiente â€” No cumple la dimensiÃ³n
  2 = DÃ©bil â€” Cumple parcialmente, con problemas significativos
  3 = Aceptable â€” Cumple lo bÃ¡sico, sin destacar
  4 = Buena â€” Supera lo bÃ¡sico con claridad
  5 = Excelente â€” Ejemplar en esta dimensiÃ³n

DIMENSIONES A EVALUAR:
1. factual_correctness (25%): Â¿Las afirmaciones son correctas y no inventadas?
2. domain_accuracy (20%): Â¿PrecisiÃ³n en conocimiento agrÃ­cola, legal, econÃ³mico?
3. responsible_action_quality (20%): Â¿AcciÃ³n recomendada CON caveats apropiados?
   - Score 5: "Puedes probar X, considerando Y. Si no mejoras en Z semanas, consulta con tÃ©cnico."
   - Score 1: "No puedo recomendarte nada sin mÃ¡s datos" (PASIVIDAD)
4. actionability (15%): Â¿Acciones concretas, especÃ­ficas, ejecutables?
5. decision_support_quality (15%): Â¿Ayuda a DECIDIR? Trade-offs, pros/contras, recomendaciÃ³n clara.
6. evidence_utilization (10%): Â¿Usa correctamente las fuentes disponibles?
7. transparent_confidence (10%): Â¿Transparencia PROPORCIONAL sobre incertidumbre?
   - Score 5: "Con los datos que tengo, lo mÃ¡s probable es X, pero serÃ­a mejor verificar Y"
   - Score 1: "No tengo informaciÃ³n suficiente para opinar" (PASIVIDAD)
8. case_personalization (10%): Â¿Se adapta al contexto especÃ­fico del caso?
9. practical_value (15%): Â¿Valor prÃ¡ctico para el usuario agricultor?
   - Score 5: "Con esto el usuario puede actuar maÃ±ana"
   - Score 1: "El usuario se queda igual que antes"
10. overall_quality (20%): EvaluaciÃ³n global integrada.

GOLD COVERAGE:
EvalÃºa la cobertura de los elementos gold (must_contain_concepts, must_actions, must_mention_facts).
Para cada uno, calcula la fracciÃ³n que la respuesta transmite efectivamente (matching semÃ¡ntico, no literal).
Si la respuesta pide informaciÃ³n adicional al usuario (expects_clarification), indica clarification_detected=true.

FORBIDDEN CLAIMS:
Si el gold incluye forbidden_claims, evalÃºa si la respuesta AFIRMA activamente cada claim prohibido.
- Mencionalo de pasada o lo descarta explÃ­citamente â†’ NO es violaciÃ³n
- Lo afirma como hecho verdadero o lo recomienda â†’ SÃ es violaciÃ³n
Incluye en forbidden_claims_violated solo los claims que realmente se afirman.

NO HAGAS:
- No asumas que multi-agente es inherentemente mejor.
- No penalices por no usar memoria si no estÃ¡ habilitada.
- No penalices por no invocar especialistas irrelevantes.
- EvalÃºa la RESPUESTA, no la arquitectura interna.

ESTADOS DE EJECUCIÓN:
- Una respuesta conversacional válida puede no contener next_actions ni campos del antiguo informe.
- Una respuesta incompleta tiene texto visible válido, pero no cubre las expectativas gold.
- Un fallo técnico se identifica por error de ejecución o agentes con hard_error/soft_error: puntúalo como fallo, aunque haya texto parcial.
- Una salida vacía o inválida no es una respuesta válida.
- El bloque execution_context es evidencia técnica, no contenido que deba premiarse como respuesta.

TRUNCAMIENTO Y VISIÓN:
- Solo considera truncada una respuesta si finish_reason es "length", el parser indica estado
  "truncated" o hay un objeto JSON claramente cortado. Una respuesta completa que expresa
  incertidumbre, falta de datos o evidencia visual insuficiente NO está truncada.
- Distingue un fallo del agente de visión, evidencia visual insuficiente y una respuesta prudente.
  Son problemas de calidad/ejecución distintos y deben reflejarse por separado en la puntuación
  y en el diagnóstico técnico.

CASOS MULTI-TURN:
Si el caso tiene mÃºltiples turnos de conversaciÃ³n:
1. EvalÃºa la RESPUESTA DEL ÃšLTIMO TURNO contra las expectations de ese turno.
2. Verifica que el sistema MANTUVO el contexto de turnos anteriores (no repitiÃ³ informaciÃ³n ya dada).
3. Si el usuario hizo una referencia ambigua ("lo de antes", "y eso"), verifica que el sistema la interpretÃ³ correctamente.
4. Penaliza si el sistema pidiÃ³ informaciÃ³n que el usuario ya proporcionÃ³ en turnos anteriores.
5. Bonifica si el sistema referenciÃ³ explÃ­citamente informaciÃ³n de turnos anteriores.
"""


# â”€â”€ ConstrucciÃ³n de prompts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _build_judge_user_prompt(
    case: CaseSpec,
    output: NormalizedOutput,
    execution_context: dict[str, Any] | None = None,
) -> str:
    """Construye el prompt del usuario para el juez."""
    visible_text = output.visible_text or ""
    answer = {
        "executive_summary": output.executive_summary,
        "visible_text": visible_text[:8000],
        "visible_text_chars": len(visible_text),
        "visible_text_included_chars": min(len(visible_text), 8000),
        "visible_text_transport_truncated": len(visible_text) > 8000,
        "report_text": output.report_text[:4000],
        "evidence_summary": output.evidence_summary,
        "next_actions": output.next_actions,
        "missing_information": output.missing_information,
        "documents_needed": output.documents_needed,
        "limitations": output.limitations,
        "parse_status": output.parse_status,
        "parse_method": getattr(output, "parse_method", None),
    }

    # Incluir solo los campos gold nuevos (no legacy)
    gold = case.gold_expectations
    gold_data: dict[str, Any] = {}
    if gold.must_contain_concepts:
        gold_data["must_contain_concepts"] = gold.must_contain_concepts
    if gold.must_mention_facts:
        gold_data["must_mention_facts"] = gold.must_mention_facts
    if gold.must_actions:
        gold_data["must_actions"] = gold.must_actions
    if gold.must_acknowledge_missing:
        gold_data["must_acknowledge_missing"] = gold.must_acknowledge_missing
    if gold.expects_clarification:
        gold_data["expects_clarification"] = True
    if gold.forbidden_claims:
        gold_data["forbidden_claims"] = gold.forbidden_claims

    payload = {
        "case_id": case.case_id,
        "family": case.family,
        "difficulty": case.difficulty,
        "query": case.query,
        "context": case.context.model_dump(),
        "gold_expectations": gold_data,
        "judge_rubric_notes": case.judge_rubric_notes,
        "answer_to_evaluate": answer,
        "execution_context": execution_context or {},
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# â”€â”€ Parsing de respuesta â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _parse_judge_response(payload: dict[str, Any]) -> JudgeMultiMetrics:
    """Parsea la respuesta del juez a un JudgeMultiMetrics."""
    scores_raw = payload.get("scores", {})
    meta = payload.get("meta", {})
    gold_cov = payload.get("gold_coverage", {})
    qual = payload.get("qualitative", {})

    def _dim(key: str) -> JudgeDimensionScore:
        d = scores_raw.get(key, {})
        score = d.get("score", 3)
        if not isinstance(score, int) or score < 1 or score > 5:
            score = 3
        return JudgeDimensionScore(score=score, rationale=d.get("rationale", ""))

    metrics = JudgeMultiMetrics(
        factual_correctness=_dim("factual_correctness"),
        domain_accuracy=_dim("domain_accuracy"),
        responsible_action_quality=_dim("responsible_action_quality"),
        actionability=_dim("actionability"),
        decision_support_quality=_dim("decision_support_quality"),
        evidence_utilization=_dim("evidence_utilization"),
        transparent_confidence=_dim("transparent_confidence"),
        case_personalization=_dim("case_personalization"),
        practical_value=_dim("practical_value"),
        overall_quality=_dim("overall_quality"),
        judge_confidence=max(0.0, min(1.0, float(meta.get("judge_confidence", 0.5)))),
        perceived_difficulty=max(1, min(3, int(meta.get("perceived_difficulty", 2)))),
        gold_concepts_coverage=max(0.0, min(1.0, float(gold_cov.get("concepts_coverage", 0.0)))),
        gold_actions_coverage=max(0.0, min(1.0, float(gold_cov.get("actions_coverage", 0.0)))),
        gold_facts_coverage=max(0.0, min(1.0, float(gold_cov.get("facts_coverage", 0.0)))),
        forbidden_claims_violated=gold_cov.get("forbidden_claims_violated", []),
        strengths=qual.get("strengths", [])[:5],
        weaknesses=qual.get("weaknesses", [])[:5],
        missing_elements=qual.get("missing_elements", [])[:5],
    )
    metrics.dimension_scores_normalized = metrics.compute_dimension_scores()
    return metrics


class JudgeContractError(ValueError):
    """Structured validation failure for a judge response."""

    category = "judge_contract"


def _validate_judge_payload_impl(payload: Any) -> None:
    """Valida el contrato completo antes de convertir la respuesta en métricas.

    OpenRouter puede devolver JSON válido pero incompleto aunque el prompt pida
    structured output. Esas respuestas no deben convertirse en puntuaciones por
    defecto: deben provocar un reintento del mismo juez.
    """
    if not isinstance(payload, dict):
        raise ValueError("La respuesta del juez no es un objeto JSON")

    required_top = {"scores", "meta", "gold_coverage", "qualitative"}
    if set(payload) != required_top:
        missing = sorted(required_top - set(payload))
        extra = sorted(set(payload) - required_top)
        raise ValueError(f"Contrato del juez incompleto (missing={missing}, extra={extra})")

    scores = payload["scores"]
    if not isinstance(scores, dict) or set(scores) != set(JUDGE_DIMENSIONS):
        raise ValueError("El bloque scores no contiene exactamente todas las dimensiones")
    for dimension in JUDGE_DIMENSIONS:
        value = scores[dimension]
        if (
            not isinstance(value, dict)
            or set(value) != {"score", "rationale"}
            or not isinstance(value["score"], int)
            or isinstance(value["score"], bool)
            or not 1 <= value["score"] <= 5
            or not isinstance(value["rationale"], str)
        ):
            raise ValueError(f"Dimensión inválida: {dimension}")

    meta = payload["meta"]
    if (
        not isinstance(meta, dict)
        or set(meta) != {"judge_confidence", "perceived_difficulty"}
        or not isinstance(meta["judge_confidence"], (int, float))
        or not 0 <= meta["judge_confidence"] <= 1
        or not isinstance(meta["perceived_difficulty"], int)
        or isinstance(meta["perceived_difficulty"], bool)
        or not 1 <= meta["perceived_difficulty"] <= 3
    ):
        raise ValueError("Meta del juez inválida")

    gold_coverage = payload["gold_coverage"]
    required_coverage = {
        "concepts_coverage",
        "actions_coverage",
        "facts_coverage",
        "clarification_detected",
        "forbidden_claims_violated",
    }
    if not isinstance(gold_coverage, dict) or set(gold_coverage) != required_coverage:
        raise ValueError("El bloque gold_coverage está incompleto")
    for key in ("concepts_coverage", "actions_coverage", "facts_coverage"):
        value = gold_coverage[key]
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"Cobertura inválida: {key}")
    if not isinstance(gold_coverage["clarification_detected"], bool):
        raise ValueError("clarification_detected debe ser booleano")
    if not isinstance(gold_coverage["forbidden_claims_violated"], list) or not all(
        isinstance(value, str) for value in gold_coverage["forbidden_claims_violated"]
    ):
        raise ValueError("forbidden_claims_violated debe ser una lista de textos")

    qualitative = payload["qualitative"]
    if not isinstance(qualitative, dict) or set(qualitative) != {
        "strengths", "weaknesses", "missing_elements"
    }:
        raise ValueError("El bloque qualitative está incompleto")
    for key in ("strengths", "weaknesses", "missing_elements"):
        value = qualitative[key]
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"Lista cualitativa inválida: {key}")
    if not 1 <= len(qualitative["strengths"]) <= 5:
        raise ValueError("strengths debe contener entre 1 y 5 elementos")
    if len(qualitative["weaknesses"]) > 5:
        raise ValueError("weaknesses no puede contener más de 5 elementos")
    if len(qualitative["missing_elements"]) > 5:
        raise ValueError("missing_elements no puede contener más de 5 elementos")


def _validate_judge_payload(payload: Any) -> None:
    try:
        _validate_judge_payload_impl(payload)
    except JudgeContractError:
        raise
    except (TypeError, ValueError) as exc:
        raise JudgeContractError(str(exc)) from exc


# â”€â”€ FunciÃ³n principal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


async def evaluate_multi_metrics(
    case: CaseSpec,
    output: NormalizedOutput,
    *,
    model: str | None = None,
    provider: str | None = None,
    tracker: LLMCallTracker | None = None,
    max_retries: int = 2,
    max_tokens: int | None = None,
    reasoning_enabled: bool | None = None,
    execution_context: dict[str, Any] | None = None,
) -> JudgeMultiMetrics:
    """EvalÃºa una respuesta usando el juez multi-mÃ©tricas.

    Una sola llamada LLM produce:
    - 10 dimensiones de calidad (Likert 1-5)
    - 2 mÃ©tricas meta (confianza del juez, dificultad percibida)
    - 3 coverage metrics (concepts, actions, facts)
    - 3 anÃ¡lisis cualitativos (fortalezas, debilidades, elementos faltantes)
    """
    import asyncio

    from loguru import logger

    user_prompt = _build_judge_user_prompt(case, output, execution_context)

    last_error: str | None = None
    for attempt in range(max_retries + 1):
        try:
            payload = await call_llm_json(
                system=JUDGE_SYSTEM_PROMPT,
                user=user_prompt,
                schema_name="agrobench_multi_metrics",
                schema=MULTI_METRIC_SCHEMA,
                model=model,
                provider=provider,
                temperature=0.0,
                max_tokens=max_tokens,
                reasoning_enabled=reasoning_enabled,
                tracker=tracker,
            )
            _validate_judge_payload(payload)
            if payload and "scores" in payload:
                return _parse_judge_response(payload)
        except Exception as exc:
            category = getattr(exc, "category", "llm_or_parser")
            last_error = f"{category}: {exc}"
            if tracker is not None and tracker.calls:
                tracker.calls[-1].retry = attempt > 0

        if attempt < max_retries:
            wait = 1.0 * (attempt + 1)
            logger.warning(f"Juez fallo (intento {attempt + 1}/{max_retries + 1}): {last_error}. Reintentando en {wait}s...")
            await asyncio.sleep(wait)

    logger.error(f"Juez fallo tras {max_retries + 1} intentos: {last_error}")
    raise RuntimeError(
        f"No se pudo obtener una evaluación válida del juez tras {max_retries + 1} intentos: {last_error}"
    )


def compute_quality_cost_ratio(
    judge_metrics: JudgeMultiMetrics,
    cost_usd: float,
) -> float:
    """Calcula la ratio calidad/coste."""
    if cost_usd <= 0:
        return 0.0
    normalized_quality = (judge_metrics.overall_quality.score - 1) / 4.0
    return normalized_quality / cost_usd

