"""Tests para sistema multi-judge de evaluación."""
from __future__ import annotations

import json

import pytest

from evaluation.config import EvalConfig, JudgeConfig, ModelConfig
from evaluation.model_packs import DEFAULT_JUDGES, get_pack_config, list_packs
from evaluation.schemas import (
    ExecutionMetrics,
    JudgeDimensionScore,
    JudgeMultiMetrics,
    NormalizedOutput,
    RunArtifact,
)


class TestJudgeConfig:
    def test_creation(self):
        j = JudgeConfig(name="mimo", model="xiaomi/mimo-v2.5", provider="openrouter")
        assert j.name == "mimo"
        assert j.model == "xiaomi/mimo-v2.5"
        assert j.provider == "openrouter"

    def test_default_provider(self):
        j = JudgeConfig(name="test", model="test-model")
        assert j.provider == "openrouter"


class TestEvalConfigJudges:
    def test_from_json_with_judges(self, tmp_path):
        data = {
            "models": {"m": {"model_id": "gpt-5-mini", "provider": "openai"}},
            "judges": [
                {"name": "j1", "model": "xiaomi/mimo-v2.5", "provider": "openrouter"},
                {"name": "j2", "model": "z-ai/glm-5.2", "provider": "openrouter"},
            ],
        }
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data))
        config = EvalConfig.from_json(path)
        assert len(config.judges) == 2
        assert config.judges[0].name == "j1"
        assert config.judges[1].name == "j2"

    def test_from_json_legacy_migrates(self, tmp_path):
        data = {
            "models": {},
            "judge_model": "xiaomi/mimo-v2.5",
            "judge_provider": "openrouter",
        }
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data))
        config = EvalConfig.from_json(path)
        assert len(config.judges) == 1
        assert config.judges[0].model == "xiaomi/mimo-v2.5"

    def test_to_json_judges(self, tmp_path):
        config = EvalConfig()
        config.judges = [
            JudgeConfig(name="j1", model="m1", provider="openrouter"),
            JudgeConfig(name="j2", model="m2", provider="openai"),
        ]
        path = tmp_path / "out.json"
        config.to_json(path)
        data = json.loads(path.read_text())
        assert len(data["judges"]) == 2
        assert data["judges"][0]["name"] == "j1"
        assert data["judges"][1]["provider"] == "openai"

    def test_get_judge_names(self):
        config = EvalConfig()
        config.judges = [
            JudgeConfig(name="alpha", model="m1"),
            JudgeConfig(name="beta", model="m2"),
        ]
        assert config.get_judge_names() == ["alpha", "beta"]

    def test_backward_compat_no_judges(self, tmp_path):
        data = {"models": {}}
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data))
        config = EvalConfig.from_json(path)
        assert config.judges == []


class TestDefaultJudges:
    def test_has_three_judges(self):
        assert len(DEFAULT_JUDGES) == 3

    def test_names_are_unique(self):
        names = [j.name for j in DEFAULT_JUDGES]
        assert len(names) == len(set(names))

    def test_all_openrouter(self):
        for j in DEFAULT_JUDGES:
            assert j.provider == "openrouter"


class TestRunArtifactMultiJudge:
    def _make_judge_metrics(self, score: int = 3) -> JudgeMultiMetrics:
        dims = {
            f"dim_{i}": JudgeDimensionScore(dimension=f"dim_{i}", score=score)
            for i in range(10)
        }
        return JudgeMultiMetrics(
            factual_correctness=dims["dim_0"],
            domain_accuracy=dims["dim_1"],
            responsible_action_quality=dims["dim_2"],
            actionability=dims["dim_3"],
            decision_support_quality=dims["dim_4"],
            evidence_utilization=dims["dim_5"],
            transparent_confidence=dims["dim_6"],
            case_personalization=dims["dim_7"],
            practical_value=dims["dim_8"],
            overall_quality=dims["dim_9"],
            judge_confidence=0.9,
            perceived_difficulty=2,
            gold_concepts_coverage=0.8,
            gold_actions_coverage=0.7,
            gold_facts_coverage=0.6,
            strengths=["test"],
            weaknesses=["test"],
            missing_elements=["test"],
        )

    def test_judge_results_dict(self):
        jm1 = self._make_judge_metrics(3)
        jm2 = self._make_judge_metrics(4)
        artifact = RunArtifact(
            run_id="test",
            case_id="case1",
            model="gpt-5-mini",
            input_query="test query",
            normalized_output=NormalizedOutput(parse_status="ok"),
            metrics=ExecutionMetrics(),
            judge_results={"mimo-v2.5": jm1, "glm-5.2": jm2},
        )
        assert len(artifact.judge_results) == 2
        assert artifact.judge_results["mimo-v2.5"].overall_quality.score == 3
        assert artifact.judge_results["glm-5.2"].overall_quality.score == 4

    def test_legacy_judge_metrics_fallback(self):
        jm = self._make_judge_metrics(3)
        artifact = RunArtifact(
            run_id="test",
            case_id="case1",
            model="gpt-5-mini",
            input_query="test",
            normalized_output=NormalizedOutput(parse_status="ok"),
            metrics=ExecutionMetrics(),
            judge_metrics=jm,
        )
        assert artifact.judge_results.get("default") is jm

    def test_post_init_migrates_to_judge_results(self):
        jm = self._make_judge_metrics(4)
        artifact = RunArtifact(
            run_id="test",
            case_id="case1",
            model="gpt-5-mini",
            input_query="test",
            normalized_output=NormalizedOutput(parse_status="ok"),
            metrics=ExecutionMetrics(),
            judge_metrics=jm,
        )
        assert "default" in artifact.judge_results
        assert artifact.judge_results["default"] is jm


class TestPackConfigMultiJudge:
    def test_all_packs_have_judges(self):
        for pack_name in list_packs():
            config = get_pack_config(pack_name)
            assert len(config.judges) == 3, f"{pack_name} should have 3 judges"

    def test_openai_buena_uses_gpt56luna(self):
        config = get_pack_config("openai_buena")
        m = list(config.models.values())[0]
        assert m.model_id == "gpt-5.6-luna"

    def test_judge_names_match_default(self):
        config = get_pack_config("china_barata")
        expected = [j.name for j in DEFAULT_JUDGES]
        actual = [j.name for j in config.judges]
        assert actual == expected


class TestJudgeFailureHandling:
    def test_openrouter_uses_native_structured_outputs_when_supported(self, monkeypatch):
        import asyncio
        from types import SimpleNamespace
        from evaluation import llm_support as module

        captured = {}

        class FakeCompletions:
            async def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
                    usage=None,
                )

        client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        monkeypatch.setattr(module, "llm_enabled", lambda: True)
        monkeypatch.setattr(module, "provider_enabled", lambda provider: True)
        monkeypatch.setattr(module, "_build_client", lambda provider: client)
        monkeypatch.setattr(
            module,
            "_get_openrouter_capabilities",
            lambda model: asyncio.sleep(0, result={"response_format"}),
        )

        result = asyncio.run(
            module.call_llm_json(
                system="system",
                user="user",
                schema_name="probe",
                schema={
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
                model="z-ai/glm-5.2",
                provider="openrouter",
            )
        )

        assert result == {"ok": True}
        assert captured["response_format"]["type"] == "json_schema"
        assert captured["response_format"]["json_schema"]["strict"] is True
        assert captured["extra_body"] == {"provider": {"require_parameters": True}}

    def test_retries_when_structured_output_is_incomplete(self, monkeypatch):
        import asyncio
        from evaluation import llm_metrics as module
        from evaluation.schemas import CaseSpec, CaseContext, GoldExpectations

        valid_payload = {
            "scores": {
                dimension: {"score": 4, "rationale": "ok"}
                for dimension in module.JUDGE_DIMENSIONS
            },
            "meta": {"judge_confidence": 0.9, "perceived_difficulty": 2},
            "gold_coverage": {
                "concepts_coverage": 1.0,
                "actions_coverage": 1.0,
                "facts_coverage": 1.0,
                "clarification_detected": False,
                "forbidden_claims_violated": [],
            },
            "qualitative": {
                "strengths": ["ok"],
                "weaknesses": ["ok"],
                "missing_elements": [],
            },
        }
        responses = [{"scores": {}}, valid_payload]

        async def fake_call_llm_json(**kwargs):
            return responses.pop(0)

        monkeypatch.setattr(module, "call_llm_json", fake_call_llm_json)

        case = CaseSpec(
            case_id="case1",
            family="diagnosis",
            difficulty="easy",
            query="Tengo un problema",
            context=CaseContext(user_role="agricultor"),
            gold_expectations=GoldExpectations(),
        )
        output = NormalizedOutput(parse_status="ok")

        metrics = asyncio.run(
            module.evaluate_multi_metrics(
                case,
                output,
                model="test-model",
                provider="openrouter",
                max_retries=1,
            )
        )

        assert metrics.overall_quality.score == 4
        assert responses == []

    def test_evaluate_multi_metrics_raises_on_exhausted_failures(self, monkeypatch):
        import asyncio
        from evaluation import llm_metrics as module
        from evaluation.schemas import CaseSpec, CaseContext, GoldExpectations

        async def fake_call_llm_json(**kwargs):
            raise RuntimeError("Connection error.")

        monkeypatch.setattr(module, "call_llm_json", fake_call_llm_json)

        case = CaseSpec(
            case_id="case1",
            family="diagnosis",
            difficulty="easy",
            query="Tengo un problema",
            context=CaseContext(user_role="agricultor"),
            gold_expectations=GoldExpectations(),
        )
        output = NormalizedOutput(parse_status="ok")

        with pytest.raises(RuntimeError, match="No se pudo obtener una evaluación válida del juez"):
            asyncio.run(
                module.evaluate_multi_metrics(
                    case,
                    output,
                    model="test-model",
                    provider="openrouter",
                    max_retries=1,
                )
            )
