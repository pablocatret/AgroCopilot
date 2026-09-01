from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER_ORGANIZER", raising=False)
    monkeypatch.delenv("LLM_PROVIDER_WRITER", raising=False)
    monkeypatch.delenv("LLM_PROVIDER_LEGAL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER_STAC", raising=False)
    monkeypatch.delenv("LLM_PROVIDER_CASE_MANAGER", raising=False)
    monkeypatch.delenv("LLM_PROVIDER_DOCUMENT_ANALYST", raising=False)
    monkeypatch.delenv("LLM_PROVIDER_SPREADSHEET_ANALYST", raising=False)
    monkeypatch.delenv("LLM_PROVIDER_VISION_OCR", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


class TestProviderResolution:
    def test_global_provider_default(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        from backend.deps import Settings
        s = Settings()
        assert s.resolve_provider() == "openai"
        assert s.resolve_provider("LLM_PROVIDER") == "openai"

    def test_global_provider_openrouter(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        from backend.deps import Settings
        s = Settings()
        assert s.resolve_provider() == "openrouter"

    def test_per_agent_override_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_PROVIDER_WRITER", "openrouter")
        from backend.deps import Settings
        s = Settings()
        assert s.resolve_provider("LLM_PROVIDER_WRITER") == "openrouter"

    def test_per_agent_override_none_falls_back_to_global(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        from backend.deps import Settings
        s = Settings()
        assert s.resolve_provider("LLM_PROVIDER_WRITER") == "openrouter"

    def test_provider_stripped_and_lowercased(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "  OpenRouter  ")
        from backend.deps import Settings
        s = Settings()
        assert s.resolve_provider() == "openrouter"


class TestBaseUrlResolution:
    def test_openai_base_url(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        from backend.deps import Settings
        s = Settings()
        assert s.resolve_base_url("openai") == "https://api.openai.com/v1"

    def test_openrouter_base_url(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        from backend.deps import Settings
        s = Settings()
        assert s.resolve_base_url("openrouter") == "https://openrouter.ai/api/v1"


class TestApiKeyResolution:
    def test_openai_requires_key(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from backend.deps import Settings
        s = Settings()
        # Override the env_file to prevent reading from .env
        s.OPENAI_API_KEY = None
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            s.resolve_api_key("openai")

    def test_openrouter_requires_key(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        from backend.deps import Settings
        s = Settings()
        s.OPENROUTER_API_KEY = None
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            s.resolve_api_key("openrouter")

    def test_openrouter_uses_own_key(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        from backend.deps import Settings
        s = Settings()
        assert s.resolve_api_key("openrouter") == "sk-or-test"

    def test_openai_uses_own_key(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from backend.deps import Settings
        s = Settings()
        assert s.resolve_api_key("openai") == "sk-test"


class TestBuildClient:
    def test_build_client_openai(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        from backend.deps import Settings, settings
        from agents.base import _build_client
        # Patch the global settings to simulate no key
        original_key = settings.OPENAI_API_KEY
        settings.OPENAI_API_KEY = "sk-test"
        try:
            client = _build_client("openai")
            assert client.api_key == "sk-test"
            assert str(client.base_url).rstrip("/") == "https://api.openai.com/v1"
        finally:
            settings.OPENAI_API_KEY = original_key

    def test_build_client_openrouter(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        from backend.deps import settings
        from agents.base import _build_client
        original_key = settings.OPENROUTER_API_KEY
        settings.OPENROUTER_API_KEY = "sk-or-test"
        try:
            client = _build_client("openrouter")
            assert client.api_key == "sk-or-test"
            assert "openrouter.ai" in str(client.base_url)
        finally:
            settings.OPENROUTER_API_KEY = original_key

    def test_build_client_openrouter_headers(self, monkeypatch):
        from backend.deps import settings
        from agents.base import _build_client
        original_key = settings.OPENROUTER_API_KEY
        original_url = settings.OPENROUTER_APP_URL
        original_title = settings.OPENROUTER_APP_TITLE
        settings.OPENROUTER_API_KEY = "sk-or-test"
        settings.OPENROUTER_APP_URL = "http://localhost:5173"
        settings.OPENROUTER_APP_TITLE = "TestApp"
        try:
            client = _build_client("openrouter")
            assert client.default_headers.get("HTTP-Referer") == "http://localhost:5173"
            assert client.default_headers.get("X-OpenRouter-Title") == "TestApp"
        finally:
            settings.OPENROUTER_API_KEY = original_key
            settings.OPENROUTER_APP_URL = original_url
            settings.OPENROUTER_APP_TITLE = original_title


class TestFeatureDetection:
    def test_openai_supports_structured_outputs(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from agents.writer import WriterAgent
        agent = WriterAgent()
        assert agent.supports_structured_outputs is True
        assert agent.supports_tool_choice_none is True

    def test_openrouter_non_openai_no_structured_outputs(self, monkeypatch):
        from backend.deps import settings
        from agents.writer import WriterAgent
        original_provider = settings.LLM_PROVIDER
        original_model = settings.OPENAI_MODEL_WRITER
        monkeypatch.setattr(settings, "LLM_PROVIDER", "openrouter")
        monkeypatch.setattr(settings, "OPENAI_MODEL_WRITER", "anthropic/claude-3.5-sonnet")
        try:
            agent = WriterAgent()
            assert agent.supports_structured_outputs is False
            assert agent.supports_tool_choice_none is False
        finally:
            monkeypatch.setattr(settings, "LLM_PROVIDER", original_provider)
            monkeypatch.setattr(settings, "OPENAI_MODEL_WRITER", original_model)

    def test_openrouter_openai_model_supports_structured_outputs(self, monkeypatch):
        from backend.deps import settings
        from agents.writer import WriterAgent
        original_provider = settings.LLM_PROVIDER
        original_model = settings.OPENAI_MODEL_WRITER
        monkeypatch.setattr(settings, "LLM_PROVIDER", "openrouter")
        monkeypatch.setattr(settings, "OPENAI_MODEL_WRITER", "openai/gpt-4o-mini")
        try:
            agent = WriterAgent()
            assert agent.supports_structured_outputs is True
            assert agent.supports_tool_choice_none is True
        finally:
            monkeypatch.setattr(settings, "LLM_PROVIDER", original_provider)
            monkeypatch.setattr(settings, "OPENAI_MODEL_WRITER", original_model)


class TestAgentProviderKey:
    def test_writer_provider_key(self):
        from agents.writer import WriterAgent
        assert WriterAgent._provider_key == "LLM_PROVIDER_WRITER"

    def test_legal_provider_key(self):
        from agents.legal import LegalAgent
        assert LegalAgent._provider_key == "LLM_PROVIDER_LEGAL"

    def test_case_manager_provider_key(self):
        from agents.case_manager import CaseManagerAgent
        assert CaseManagerAgent._provider_key == "LLM_PROVIDER_CASE_MANAGER"

    def test_document_analyst_provider_key(self):
        from agents.document_analyst import DocumentAnalystAgent
        assert DocumentAnalystAgent._provider_key == "LLM_PROVIDER_DOCUMENT_ANALYST"

    def test_spreadsheet_analyst_provider_key(self):
        from agents.spreadsheet_analyst import SpreadsheetAnalystAgent
        assert SpreadsheetAnalystAgent._provider_key == "LLM_PROVIDER_SPREADSHEET_ANALYST"

    def test_vision_ocr_provider_key(self):
        from agents.vision_ocr import VisionOcrAgent
        assert VisionOcrAgent._provider_key == "LLM_PROVIDER_VISION_OCR"

    def test_stac_provider_key(self):
        from agents.stac_search import StacSearchAgent
        assert StacSearchAgent._provider_key == "LLM_PROVIDER_STAC"
