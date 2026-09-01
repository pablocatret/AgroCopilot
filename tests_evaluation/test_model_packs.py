"""Tests para model packs de evaluación."""
from __future__ import annotations

import json

import pytest

from evaluation.config import EvalConfig, ModelConfig
from evaluation.model_packs import (
    MODEL_PACKS,
    get_pack_config,
    get_pack_info,
    generate_config_files,
    list_packs,
)


class TestModelConfigVision:
    def test_vision_fields_default_none(self):
        mc = ModelConfig(name="test", model_id="gpt-5-mini")
        assert mc.vision_model_id is None
        assert mc.vision_provider is None

    def test_vision_fields_set(self):
        mc = ModelConfig(
            name="test",
            model_id="deepseek/deepseek-v4-flash",
            vision_model_id="qwen/qwen3.6-flash",
            vision_provider="openrouter",
        )
        assert mc.vision_model_id == "qwen/qwen3.6-flash"
        assert mc.vision_provider == "openrouter"


class TestEvalConfigVision:
    def test_from_json_roundtrip_vision(self, tmp_path):
        config = EvalConfig()
        config.models["test"] = ModelConfig(
            name="test",
            provider="openrouter",
            model_id="deepseek/deepseek-v4-flash",
            vision_model_id="qwen/qwen3.6-flash",
            vision_provider="openrouter",
        )
        path = tmp_path / "test.json"
        config.to_json(path)

        loaded = EvalConfig.from_json(path)
        m = loaded.models["test"]
        assert m.vision_model_id == "qwen/qwen3.6-flash"
        assert m.vision_provider == "openrouter"

    def test_from_json_backward_compatible(self, tmp_path):
        """JSONs sin vision_model_id siguen funcionando."""
        data = {
            "models": {
                "old_model": {
                    "provider": "openai",
                    "model_id": "gpt-5-mini",
                    "role": "both",
                    "temperature": 0.0,
                    "max_tokens": None,
                }
            }
        }
        path = tmp_path / "old.json"
        path.write_text(json.dumps(data))
        config = EvalConfig.from_json(path)
        m = config.models["old_model"]
        assert m.vision_model_id is None
        assert m.vision_provider is None


class TestListPacks:
    def test_returns_all_packs(self):
        packs = list_packs()
        assert len(packs) == 6
        assert "hiper_rapida" in packs
        assert "hiper_pequeña" in packs
        assert "china_barata" in packs
        assert "china_top" in packs
        assert "openai_buena" in packs
        assert "openai_barata" in packs

    def test_matches_model_packs_keys(self):
        assert set(list_packs()) == set(MODEL_PACKS.keys())


class TestGetPackInfo:
    def test_returns_pack_dict(self):
        info = get_pack_info("china_barata")
        assert "text" in info
        assert "vision" in info
        assert info["text"]["model_id"] == "deepseek/deepseek-v4-flash"

    def test_invalid_pack_raises(self):
        with pytest.raises(ValueError, match="no encontrado"):
            get_pack_info("nonexistent_pack")


class TestGetPackConfig:
    def test_generates_eval_config(self):
        config = get_pack_config("china_barata")
        assert isinstance(config, EvalConfig)
        assert len(config.models) == 1
        m = list(config.models.values())[0]
        assert m.model_id == "deepseek/deepseek-v4-flash"
        assert m.vision_model_id == "qwen/qwen3.6-flash"

    def test_openai_packs_have_no_vision(self):
        for pack_name in ("openai_buena", "openai_barata"):
            config = get_pack_config(pack_name)
            m = list(config.models.values())[0]
            assert m.vision_model_id is None
            assert m.provider == "openai"

    def test_hiper_pequena_uses_compact_mistral_vision_model(self):
        config = get_pack_config("hiper_pequeña")
        m = list(config.models.values())[0]
        assert m.vision_model_id == "mistralai/ministral-8b-2512"

    def test_custom_judge(self):
        from evaluation.config import JudgeConfig
        custom_judges = [JudgeConfig(name="custom", model="anthropic/claude-sonnet-4", provider="openrouter")]
        config = get_pack_config("china_barata", judges=custom_judges)
        assert len(config.judges) == 1
        assert config.judges[0].model == "anthropic/claude-sonnet-4"
        assert config.judges[0].provider == "openrouter"

    def test_all_packs_generate_valid_config(self):
        for pack_name in list_packs():
            config = get_pack_config(pack_name)
            assert isinstance(config, EvalConfig)
            assert len(config.models) == 1
            m = list(config.models.values())[0]
            assert m.model_id != ""
            assert m.provider in ("openai", "openrouter")


class TestGenerateConfigFiles:
    def test_creates_json_files(self, tmp_path):
        files = generate_config_files(str(tmp_path))
        assert len(files) == 6
        for f in files:
            assert f.exists()
            assert f.suffix == ".json"

    def test_json_files_are_loadable(self, tmp_path):
        files = generate_config_files(str(tmp_path))
        for f in files:
            config = EvalConfig.from_json(f)
            assert len(config.models) == 1

    def test_vision_fields_in_json(self, tmp_path):
        files = generate_config_files(str(tmp_path))
        for f in files:
            data = json.loads(f.read_text())
            for model_data in data["models"].values():
                assert "vision_model_id" in model_data
                assert "vision_provider" in model_data
