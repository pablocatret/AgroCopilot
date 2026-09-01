"""Tests for evaluation/config.py — EvalConfig serialization."""
import json
import tempfile
from pathlib import Path

from evaluation.config import EvalConfig, ModelConfig


class TestModelConfig:
    def test_defaults(self):
        m = ModelConfig(name="test-model")
        assert m.provider == "openrouter"
        assert m.model_id == "test-model"
        assert m.role == "both"

    def test_custom(self):
        m = ModelConfig(name="gpt", model_id="openai/gpt-4.1-mini", role="judge")
        assert m.model_id == "openai/gpt-4.1-mini"
        assert m.role == "judge"


class TestEvalConfig:
    def test_defaults(self):
        cfg = EvalConfig()
        assert cfg.budget_usd == 10.0
        assert cfg.runs_per_case == 1
        assert cfg.judge_model == "openai/gpt-4.1-mini"

    def test_from_json_roundtrip(self):
        from evaluation.config import JudgeConfig
        cfg = EvalConfig(
            budget_usd=5.0,
            runs_per_case=3,
            judges=[JudgeConfig(name="j1", model="xiaomi/mimo-v2.5", provider="openrouter")],
            models={"gpt": ModelConfig(name="gpt", model_id="openai/gpt-4.1-mini")},
        )
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            cfg.to_json(f.name)
            loaded = EvalConfig.from_json(f.name)

        assert loaded.budget_usd == 5.0
        assert loaded.runs_per_case == 3
        assert len(loaded.judges) == 1
        assert loaded.judges[0].model == "xiaomi/mimo-v2.5"
        assert "gpt" in loaded.models
        assert loaded.models["gpt"].model_id == "openai/gpt-4.1-mini"

    def test_get_model_ids(self):
        cfg = EvalConfig(
            models={
                "gpt41mini": ModelConfig(name="gpt41mini", model_id="openai/gpt-4.1-mini"),
                "claude": ModelConfig(name="claude", model_id="anthropic/claude-sonnet-4"),
            },
        )
        ids = cfg.get_model_ids()
        assert "openai/gpt-4.1-mini" in ids
        assert "anthropic/claude-sonnet-4" in ids
        assert len(ids) == 2

    def test_smoke_subset_config_loads(self):
        cfg = EvalConfig.from_json("evaluation/configs/smoke_integrado_barato.json")
        assert len(cfg.models) == 1
        assert len(cfg.judges) == 3
        assert [j.model for j in cfg.judges] == [
            "xiaomi/mimo-v2.5",
            "tencent/hy3-preview",
            "stepfun/step-3.7-flash",
        ]
        assert cfg.budget_usd == 2.5
        assert cfg.runs_per_case == 1
        assert "seed_001_olivar_plaga.json" in cfg.corpus_path
        assert "mt_001_diagnosis_followup.json" in cfg.corpus_path
        assert "att_001_leaf_disease.json" in cfg.corpus_path
        assert "rt_001_legal_route.json" in cfg.corpus_path
