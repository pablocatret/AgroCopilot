"""Modelos de datos para evaluación de modelos en AgroCopilot.

Diseñado para comparar LLMs: cada modelo ejecuta el sistema multi-agente
y se evalúa calidad, routing de agentes, coste y latencia.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "2.0"

# ── Casos de evaluación ──────────────────────────────────────────────


class GoldExpectations(BaseModel):
    """Oro estructurado: qué se espera de la respuesta.

    Campos semánticos (evaluados por juez LLM):
      - must_contain_concepts: conceptos que la respuesta debe transmitir
      - must_mention_facts: datos técnicos verificables
      - must_actions: acciones concretas que debe recomendar
      - must_acknowledge_missing: información que debe reconocer como faltante

    Campos determinísticos (evaluados por regex/token overlap):
      - forbidden_claims: afirmaciones prohibidas específicas
      - forbidden_overclaim: patrones de sobreafirmación

    Comportamiento:
      - expects_clarification: True si el sistema debe pedir info adicional
    """

    # — Matching semántico (juez LLM evalúa) —
    must_contain_concepts: list[str] = Field(
        default_factory=list,
        description="Conceptos que la respuesta debe transmitir (matching semántico)",
    )
    must_mention_facts: list[str] = Field(
        default_factory=list,
        description="Datos técnicos verificables que la respuesta debe mencionar",
    )
    must_actions: list[str] = Field(
        default_factory=list,
        description="Acciones concretas que la respuesta debe recomendar",
    )
    must_acknowledge_missing: list[str] = Field(
        default_factory=list,
        description="Información que la respuesta debe reconocer como faltante",
    )

    # — Matching estricto (métricas deterministas) —
    forbidden_claims: list[str] = Field(
        default_factory=list,
        description="Afirmaciones prohibidas específicas (overlap ≥0.6 = violación)",
    )
    forbidden_overclaim: list[str] = Field(
        default_factory=list,
        description="Patrones de sobreafirmación (regex match = violación)",
    )

    # — Comportamiento esperado —
    expects_clarification: bool = Field(
        default=False,
        description="True si el sistema debe pedir información adicional en vez de responder",
    )

    # — Legacy compat —
    must_mention: list[str] = Field(
        default_factory=list,
        description="LEGACY: se mapea a must_contain_concepts en from_legacy()",
    )
    legacy_raw: dict[str, Any] = Field(
        default_factory=dict,
        description="Gold expectations originales sin procesar (solo para legacy)",
    )

    @classmethod
    def from_legacy(cls, data: dict) -> "GoldExpectations":
        """Convierte formato legacy al nuevo formato."""
        must_concepts = list(data.get("required_evidence", []))
        for spec in data.get("required_specialists", []):
            must_concepts.append(f"consultar {spec}" if not spec.startswith("consultar") else spec)

        must_ack = list(data.get("required_missing_information", []))
        must_ack.extend(data.get("required_clarifications", []))

        return cls(
            must_contain_concepts=must_concepts,
            must_mention_facts=list(data.get("required_contextual_facts", [])),
            must_actions=list(data.get("required_next_actions", [])),
            must_acknowledge_missing=must_ack,
            forbidden_claims=list(data.get("forbidden_claims", [])),
            forbidden_overclaim=[],
            expects_clarification=False,
            must_mention=must_concepts,
            legacy_raw=data,
        )


class CaseContext(BaseModel):
    """Contexto adicional del caso."""

    user_role: str = "agricultor"
    crop_type: str = ""
    region: str = ""
    previous_context: str = ""


class ConversationTurn(BaseModel):
    """Un turno en una conversación multi-turn."""

    turn: int
    query: str
    gold_expectations: GoldExpectations = Field(default_factory=GoldExpectations)
    context_override: dict[str, Any] = Field(default_factory=dict)
    expected_behavior: str = ""


class CaseSpec(BaseModel):
    """Especificación de un caso de evaluación."""

    case_id: str
    family: Literal["diagnosis", "compliance", "decision", "general", "attachment_analysis"]
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    query: str
    context: CaseContext = Field(default_factory=CaseContext)
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    gold_expectations: GoldExpectations = Field(default_factory=GoldExpectations)
    judge_rubric_notes: str = ""

    # Multi-turn support
    is_multiturn: bool = Field(
        default=False,
        description="True si el caso tiene múltiples turnos de conversación",
    )
    turns: list[ConversationTurn] = Field(
        default_factory=list,
        description="Turnos de conversación para casos multi-turn",
    )

    # Routing assertions
    expected_route: list[str] = Field(
        default_factory=list,
        description="Agentes que se esperan en el routing (ej: ['legal', 'writer'])",
    )
    optional_route: list[str] = Field(
        default_factory=list,
        description="Agentes aceptables pero no obligatorios para que el caso sea vÃ¡lido",
    )
    routing_assertion: str = Field(
        default="",
        description="Aserción sobre routing que debe verificarse",
    )

    # Legacy fields (ignored but accepted for backward compat)
    decision_mode: str | None = None
    user_id: str | None = None

    @classmethod
    def model_validate(cls, obj: Any, **kwargs) -> "CaseSpec":
        """Override para soportar formato legacy de gold_expectations."""
        if isinstance(obj, dict):
            obj = obj.copy()
            gold = obj.get("gold_expectations", {})
            if gold and "required_evidence" in gold:
                obj["gold_expectations"] = GoldExpectations.from_legacy(gold)
        return super().model_validate(obj, **kwargs)


# ── Salida normalizada del sistema ───────────────────────────────────


class NormalizedOutput(BaseModel):
    """Salida normalizada del sistema evaluado."""

    schema_version: str = SCHEMA_VERSION
    executive_summary: str = ""
    report_text: str = ""
    message_md: str = ""
    evidence_summary: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    documents_needed: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    references: list[dict[str, Any]] = Field(default_factory=list)
    structured_fields_present: list[str] = Field(default_factory=list)
    parse_status: Literal["ok", "partial", "failed"] = "ok"

    @property
    def visible_text(self) -> str:
        """Texto principal visible para el usuario."""
        return self.message_md or self.report_text or self.executive_summary

    @property
    def evaluable_text(self) -> str:
        """All user-facing and structured content available to evaluators."""
        parts: list[str] = [
            self.executive_summary,
            self.report_text,
            self.message_md,
            *self.evidence_summary,
            *self.next_actions,
            *self.missing_information,
            *self.documents_needed,
            *self.limitations,
        ]
        return "\n".join(str(part) for part in parts if str(part).strip())


# ── Métricas de ejecución ────────────────────────────────────────────


@dataclass
class ExecutionMetrics:
    """Métricas deterministas de una ejecución."""

    schema_version: str = SCHEMA_VERSION
    success: bool = False
    latency_ms: float = 0.0
    system_latency_ms: float = 0.0
    judge_latency_ms: float = 0.0
    task_wall_latency_ms: float = 0.0
    queue_latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    system_cost_usd: float = 0.0
    vision_cost_usd: float = 0.0
    judge_cost_usd: float = 0.0
    model_calls: int = 0
    token_prompt_total: int = 0
    token_completion_total: int = 0
    answer_completeness: float = 0.0
    forbidden_claim_rate: float = 0.0
    overclaim_count: float = 0.0
    actionability: float = 0.0
    actionability_structured: float = 0.0
    actionability_visible: float = 0.0
    clarification_detected: bool = False
    agents_invoked: list[str] = field(default_factory=list)
    agents_planned: list[str] = field(default_factory=list)
    agents_failed: list[str] = field(default_factory=list)
    agents_extra: list[str] = field(default_factory=list)
    agents_ok: int = 0
    agents_error: int = 0
    route_observed: list[str] = field(default_factory=list)
    routing_score: float = 1.0
    routing_assertion_pass: bool = True
    routing_precision: float = 1.0
    routing_recall: float = 1.0
    routing_order_score: float = 1.0
    execution_status: str = "failed"
    failure_reason: str = ""
    required_agents_missing: list[str] = field(default_factory=list)
    visual_evidence_status: str = "not_required"
    visual_evidence_used: bool = False


# ── Métricas del juez LLM ───────────────────────────────────────────


class JudgeDimensionScore(BaseModel):
    """Score de una dimensión individual del juez."""

    score: int = Field(ge=1, le=5, description="Escala Likert 1-5")
    rationale: str = ""


class JudgeMultiMetrics(BaseModel):
    """Vector completo de métricas producido por una llamada al juez.

    Incluye 10 dimensiones de calidad + 2 meta + 3 cualitativas + 1 semántica.
    """

    schema_version: str = SCHEMA_VERSION

    # Dimensiones de calidad (1-5)
    factual_correctness: JudgeDimensionScore = Field(
        default_factory=lambda: JudgeDimensionScore(score=3)
    )
    domain_accuracy: JudgeDimensionScore = Field(
        default_factory=lambda: JudgeDimensionScore(score=3)
    )
    responsible_action_quality: JudgeDimensionScore = Field(
        default_factory=lambda: JudgeDimensionScore(score=3),
        description="Calidad de la acción recomendada (balance prudencia-utilidad)",
    )
    actionability: JudgeDimensionScore = Field(
        default_factory=lambda: JudgeDimensionScore(score=3)
    )
    decision_support_quality: JudgeDimensionScore = Field(
        default_factory=lambda: JudgeDimensionScore(score=3)
    )
    evidence_utilization: JudgeDimensionScore = Field(
        default_factory=lambda: JudgeDimensionScore(score=3)
    )
    transparent_confidence: JudgeDimensionScore = Field(
        default_factory=lambda: JudgeDimensionScore(score=3),
        description="Transparencia PROPORCIONAL sobre incertidumbre",
    )
    case_personalization: JudgeDimensionScore = Field(
        default_factory=lambda: JudgeDimensionScore(score=3)
    )
    practical_value: JudgeDimensionScore = Field(
        default_factory=lambda: JudgeDimensionScore(score=3),
        description="Valor práctico para el usuario agricultor",
    )
    overall_quality: JudgeDimensionScore = Field(
        default_factory=lambda: JudgeDimensionScore(score=3)
    )

    # Métricas meta
    judge_confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    perceived_difficulty: int = Field(ge=1, le=3, default=2)

    # Matching semántico (evaluación del juez contra gold)
    gold_concepts_coverage: float = Field(
        ge=0.0, le=1.0, default=0.0,
        description="Fracción de must_contain_concepts que la respuesta transmite",
    )
    gold_actions_coverage: float = Field(
        ge=0.0, le=1.0, default=0.0,
        description="Fracción de must_actions que la respuesta recomienda",
    )
    gold_facts_coverage: float = Field(
        ge=0.0, le=1.0, default=0.0,
        description="Fracción de must_mention_facts que la respuesta menciona",
    )

    # Claims prohibidos (evaluación semántica por juez)
    forbidden_claims_violated: list[str] = Field(
        default_factory=list,
        description="Claims prohibidos que la respuesta AFIRMA activamente (no solo menciona de pasada)",
    )

    # Análisis cualitativo
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    missing_elements: list[str] = Field(default_factory=list)

    # Derivadas (calculadas post-hoc)
    dimension_scores_normalized: dict[str, float] = Field(default_factory=dict)

    def compute_dimension_scores(self) -> dict[str, float]:
        """Normaliza los scores 1-5 a [0,1] para facilitar análisis."""
        dims = {
            "factual_correctness": self.factual_correctness.score,
            "domain_accuracy": self.domain_accuracy.score,
            "responsible_action_quality": self.responsible_action_quality.score,
            "actionability": self.actionability.score,
            "decision_support_quality": self.decision_support_quality.score,
            "evidence_utilization": self.evidence_utilization.score,
            "transparent_confidence": self.transparent_confidence.score,
            "case_personalization": self.case_personalization.score,
            "practical_value": self.practical_value.score,
            "overall_quality": self.overall_quality.score,
        }
        return {k: (v - 1) / 4.0 for k, v in dims.items()}

    def to_flat_dict(self) -> dict[str, Any]:
        """Convierte a diccionario plano para agregación."""
        result: dict[str, Any] = {}
        for dim_name in [
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
        ]:
            dim = getattr(self, dim_name)
            result[f"judge_{dim_name}"] = dim.score
            result[f"judge_{dim_name}_rationale"] = dim.rationale
        result["judge_confidence"] = self.judge_confidence
        result["judge_perceived_difficulty"] = self.perceived_difficulty
        result["gold_concepts_coverage"] = self.gold_concepts_coverage
        result["gold_actions_coverage"] = self.gold_actions_coverage
        result["gold_facts_coverage"] = self.gold_facts_coverage
        result["forbidden_claims_violated"] = self.forbidden_claims_violated
        result["judge_strengths"] = self.strengths
        result["judge_weaknesses"] = self.weaknesses
        result["judge_missing_elements"] = self.missing_elements
        return result


# ── Artefacto de ejecución ───────────────────────────────────────────


@dataclass
class RunArtifact:
    """Resultado completo de una ejecución."""

    run_id: str
    case_id: str
    model: str
    input_query: str
    normalized_output: NormalizedOutput
    metrics: ExecutionMetrics
    judge_results: dict[str, JudgeMultiMetrics | None] = field(default_factory=dict)
    judge_metrics: JudgeMultiMetrics | None = None  # LEGACY: primer juez
    agent_routing: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    timestamp_iso: str = ""
    family: str = ""
    difficulty: str = ""
    run_idx: int = 0
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        # Migrar judge_metrics legacy a judge_results
        if self.judge_metrics is not None and not self.judge_results:
            self.judge_results["default"] = self.judge_metrics
        elif self.judge_results and self.judge_metrics is None:
            first = next(iter(self.judge_results.values()), None)
            if first is not None:
                self.judge_metrics = first
