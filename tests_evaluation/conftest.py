from __future__ import annotations

import pytest

from backend.deps import settings


@pytest.fixture(autouse=True)
def _disable_paid_llm_calls(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "DISABLE_EXTERNALS", True)
    monkeypatch.setattr(settings, "OPENAI_API_KEY", None)
