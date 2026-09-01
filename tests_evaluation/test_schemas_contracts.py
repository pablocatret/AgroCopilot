"""Tests for evaluation/schemas.py — data model contracts."""
import pytest

from evaluation.schemas import (
    CaseSpec,
    ExecutionMetrics,
    GoldExpectations,
    JudgeDimensionScore,
    JudgeMultiMetrics,
    NormalizedOutput,
)


# ── GoldExpectations ─────────────────────────────────────────────────


class TestGoldExpectations:
    def test_default_fields(self):
        g = GoldExpectations()
        assert g.must_contain_concepts == []
        assert g.must_mention_facts == []
        assert g.must_actions == []
        assert g.forbidden_claims == []
        assert g.expects_clarification is False
        assert g.legacy_raw == {}

    def test_from_legacy_maps_fields(self):
        legacy = {
            "required_evidence": ["phosphorus"],
            "required_next_actions": ["medir"],
            "required_missing_information": ["variedad"],
            "forbidden_claims": ["virus"],
            "required_specialists": ["fitopatologo"],
            "required_clarifications": ["antiguedad"],
            "required_documents": ["certificado"],
            "required_contextual_facts": ["zona semiarida"],
            "required_tradeoff_dimensions": ["coste"],
            "expected_route": "simple",
        }
        g = GoldExpectations.from_legacy(legacy)
        assert "phosphorus" in g.must_contain_concepts
        assert "consultar fitopatologo" in g.must_contain_concepts
        assert g.must_actions == ["medir"]
        assert "variedad" in g.must_acknowledge_missing
        assert "antiguedad" in g.must_acknowledge_missing
        assert g.forbidden_claims == ["virus"]
        assert g.must_mention_facts == ["zona semiarida"]
        assert g.legacy_raw == legacy

    def test_from_legacy_empty(self):
        g = GoldExpectations.from_legacy({})
        assert g.must_contain_concepts == []
        assert g.legacy_raw == {}

    def test_new_format_direct(self):
        g = GoldExpectations(
            must_contain_concepts=["concepto A", "concepto B"],
            must_mention_facts=["dato tecnico"],
            must_actions=["accion concreta"],
            must_acknowledge_missing=["info faltante"],
            forbidden_claims=["afirmacion prohibida"],
            forbidden_overclaim=["pattern 100%"],
            expects_clarification=True,
        )
        assert len(g.must_contain_concepts) == 2
        assert g.expects_clarification is True
        assert g.forbidden_overclaim == ["pattern 100%"]


# ── CaseSpec ─────────────────────────────────────────────────────────


class TestCaseSpec:
    def test_basic_creation(self):
        case = CaseSpec(
            case_id="c1",
            family="diagnosis",
            query="test",
        )
        assert case.case_id == "c1"
        assert case.difficulty == "medium"

    def test_legacy_gold_conversion(self):
        raw = {
            "case_id": "c2",
            "family": "compliance",
            "query": "test",
            "gold_expectations": {
                "required_evidence": ["reglamento"],
                "forbidden_claims": ["FDA"],
            },
        }
        case = CaseSpec.model_validate(raw)
        assert "reglamento" in case.gold_expectations.must_contain_concepts
        assert case.gold_expectations.forbidden_claims == ["FDA"]
        assert "required_evidence" in case.gold_expectations.legacy_raw

    def test_accepts_attachment_analysis_family(self):
        case = CaseSpec(
            case_id="c3",
            family="attachment_analysis",
            query="test",
        )
        assert case.family == "attachment_analysis"


# ── NormalizedOutput ────────────────────────────────────────────────


class TestNormalizedOutput:
    def test_visible_text_priority(self):
        o1 = NormalizedOutput(message_md="msg", report_text="rpt", executive_summary="sum")
        assert o1.visible_text == "msg"

        o2 = NormalizedOutput(report_text="rpt", executive_summary="sum")
        assert o2.visible_text == "rpt"

        o3 = NormalizedOutput(executive_summary="sum")
        assert o3.visible_text == "sum"


# ── ExecutionMetrics ────────────────────────────────────────────────


class TestExecutionMetrics:
    def test_has_all_fields(self):
        m = ExecutionMetrics(
            success=True,
            latency_ms=100.0,
            estimated_cost_usd=0.01,
            model_calls=3,
            forbidden_claim_rate=0.1,
            overclaim_count=2.0,
            actionability=0.9,
            clarification_detected=True,
            agents_invoked=["organizer", "legal"],
            agents_ok=2,
            agents_error=0,
        )
        assert m.success is True
        assert m.forbidden_claim_rate == 0.1
        assert m.overclaim_count == 2.0
        assert m.actionability == 0.9
        assert m.clarification_detected is True
        assert m.agents_invoked == ["organizer", "legal"]
        assert m.agents_ok == 2
        assert m.agents_error == 0

    def test_defaults(self):
        m = ExecutionMetrics()
        assert m.success is False
        assert m.forbidden_claim_rate == 0.0
        assert m.actionability == 0.0
        assert m.clarification_detected is False
        assert m.agents_invoked == []


# ── JudgeDimensionScore ─────────────────────────────────────────────


class TestJudgeDimensionScore:
    def test_valid_range(self):
        s = JudgeDimensionScore(score=1)
        assert s.score == 1
        s2 = JudgeDimensionScore(score=5)
        assert s2.score == 5

    def test_rejects_out_of_range(self):
        with pytest.raises(Exception):
            JudgeDimensionScore(score=0)
        with pytest.raises(Exception):
            JudgeDimensionScore(score=6)


# ── JudgeMultiMetrics ───────────────────────────────────────────────


class TestJudgeMultiMetrics:
    def test_compute_dimension_scores(self):
        jm = JudgeMultiMetrics(
            factual_correctness=JudgeDimensionScore(score=5),
            overall_quality=JudgeDimensionScore(score=1),
        )
        scores = jm.compute_dimension_scores()
        assert scores["factual_correctness"] == 1.0  # (5-1)/4
        assert scores["overall_quality"] == 0.0  # (1-1)/4

    def test_to_flat_dict(self):
        jm = JudgeMultiMetrics(
            overall_quality=JudgeDimensionScore(score=4, rationale="good"),
            gold_concepts_coverage=0.8,
            gold_actions_coverage=0.6,
            gold_facts_coverage=0.9,
        )
        d = jm.to_flat_dict()
        assert d["judge_overall_quality"] == 4
        assert d["judge_overall_quality_rationale"] == "good"
        assert d["gold_concepts_coverage"] == 0.8
        assert d["gold_actions_coverage"] == 0.6
        assert d["gold_facts_coverage"] == 0.9
        assert "judge_confidence" in d
