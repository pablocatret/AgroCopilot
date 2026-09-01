"""Tests for evaluation/metrics.py — deterministic metric functions."""
import pytest

from evaluation.metrics import (
    compute_actionability,
    compute_actionability_structured,
    compute_actionability_visible,
    compute_execution_metrics,
    compute_forbidden_claim_rate,
    compute_overclaim_rate,
    detect_clarification,
)
from evaluation.schemas import CaseSpec, GoldExpectations, NormalizedOutput


def _make_case(**gold_kwargs) -> CaseSpec:
    return CaseSpec(
        case_id="test-case",
        family="diagnosis",
        query="test query",
        gold_expectations=GoldExpectations(**gold_kwargs),
    )


def _make_output(**kwargs) -> NormalizedOutput:
    defaults = {"executive_summary": "resumen", "message_md": "respuesta", "parse_status": "ok"}
    defaults.update(kwargs)
    return NormalizedOutput(**defaults)


# ── forbidden_claim_rate ─────────────────────────────────────────────


class TestForbiddenClaimRate:
    def test_no_forbidden_returns_0(self):
        case = _make_case()
        output = _make_output()
        assert compute_forbidden_claim_rate(case, output) == 0.0

    def test_violation_detected(self):
        case = _make_case(forbidden_claims=["virus del mosaico"])
        output = _make_output(message_md="Es virus del mosaico")
        assert compute_forbidden_claim_rate(case, output) == 1.0

    def test_no_violation(self):
        case = _make_case(forbidden_claims=["virus del mosaico"])
        output = _make_output(message_md="Es tizion tardia")
        assert compute_forbidden_claim_rate(case, output) == 0.0

    def test_explicit_negation_is_not_a_violation(self):
        case = _make_case(forbidden_claims=["virus del mosaico"])
        output = _make_output(message_md="No hay evidencia de virus del mosaico")
        assert compute_forbidden_claim_rate(case, output) == 0.0


# ── overclaim ────────────────────────────────────────────────────────


class TestOverclaimRate:
    def test_no_overclaims(self):
        output = _make_output(message_md="Podria ser util")
        assert compute_overclaim_rate(output) == 0.0

    def test_garantiza_detected(self):
        output = _make_output(message_md="Esto garantiza la cura")
        assert compute_overclaim_rate(output) >= 1.0

    def test_sin_riesgo_detected(self):
        output = _make_output(message_md="Es sin riesgo")
        assert compute_overclaim_rate(output) >= 1.0


# ── actionability ────────────────────────────────────────────────────


class TestActionability:
    def test_no_actions_returns_0(self):
        output = _make_output(next_actions=[])
        assert compute_actionability(output) == 0.0

    def test_specific_action(self):
        output = _make_output(
            next_actions=["Comprobar humedad del suelo con tensiometro"]
        )
        assert compute_actionability(output) == 1.0

    def test_vague_action(self):
        output = _make_output(next_actions=["Ver que pasa"])
        assert compute_actionability(output) == 0.0

    def test_visible_conversation_action_is_scored_when_structured_field_is_absent(self):
        output = _make_output(
            message_md="Antes de actuar, comprueba la humedad del suelo y verifica el drenaje."
        )
        assert compute_actionability_structured(output) == 0.0
        assert compute_actionability_visible(output) > 0.0
        assert compute_actionability(output) > 0.0


# ── clarification detection ──────────────────────────────────────────


class TestClarificationDetection:
    def test_no_clarification(self):
        output = _make_output(message_md="Aplica fungicida cada 15 dias")
        assert detect_clarification(output) is False

    def test_clarification_detected(self):
        output = _make_output(message_md="Necesitaria que me indicaras que variedad tienes")
        assert detect_clarification(output) is True

    def test_clarification_with_missing_info(self):
        output = _make_output(message_md="Faltan datos sobre la superficie cultivada")
        assert detect_clarification(output) is True

    def test_clarification_with_question(self):
        output = _make_output(message_md="Podrias indicar que tipo de cultivo tienes?")
        assert detect_clarification(output) is True


# ── compute_execution_metrics ────────────────────────────────────────


class TestComputeExecutionMetrics:
    def test_populates_all_fields(self):
        case = _make_case(
            forbidden_claims=["virus"],
        )
        output = _make_output(
            message_md="Respuesta completa",
            next_actions=["Medir humedad del suelo"],
        )
        metrics = compute_execution_metrics(
            case, output, latency_ms=100.0, cost_usd=0.005, model_calls=2,
            agents_invoked=["organizer", "legal"], agents_ok=2, agents_error=0,
        )
        assert metrics.success is True
        assert metrics.latency_ms == 100.0
        assert metrics.estimated_cost_usd == 0.005
        assert metrics.model_calls == 2
        assert metrics.forbidden_claim_rate == 0.0
        assert isinstance(metrics.overclaim_count, float)
        assert 0.0 <= metrics.actionability <= 1.0
        assert metrics.agents_invoked == ["organizer", "legal"]
        assert metrics.agents_ok == 2
        assert metrics.agents_error == 0

    def test_failed_output(self):
        case = _make_case()
        output = NormalizedOutput(parse_status="failed")
        metrics = compute_execution_metrics(case, output)
        assert metrics.success is False

    def test_clarification_flag(self):
        case = _make_case()
        output = _make_output(message_md="Podrias indicarme que variedad tienes?")
        metrics = compute_execution_metrics(case, output)
        assert metrics.clarification_detected is True
