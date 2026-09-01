# backend/deps.py
"""Dependencias y configuración (settings + logging)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from loguru import logger

# Carga .env desde la raíz del repo (un nivel por encima de backend/)
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

# Versión pública de la API
VERSION = "0.1.0"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    # OpenAI / LLM
    OPENAI_API_KEY: Optional[str] = Field(None, description="OpenAI API key (optional)")
    OPENAI_MODEL_BASELINE: str = "gpt-5-nano"
    OPENAI_MODEL_ORGANIZER: Optional[str] = "gpt-5-mini"
    OPENAI_MODEL_WRITER: Optional[str] = "gpt-5-mini"
    OPENAI_MODEL_STAC: Optional[str] = "gpt-5-mini"
    OPENAI_MODEL_LEGAL_WRITER: Optional[str] = "gpt-5-mini"
    OPENAI_MODEL_LEGAL: Optional[str] = "gpt-5-mini"
    OPENAI_MODEL_CASE_MANAGER: Optional[str] = "gpt-5-mini"
    OPENAI_MODEL_DOCUMENT_ANALYST: Optional[str] = "gpt-5-nano"
    OPENAI_MODEL_SPREADSHEET_ANALYST: Optional[str] = "gpt-5-nano"
    OPENAI_MODEL_VISION: Optional[str] = "gpt-4o-mini"
    OPENAI_MODEL_VISION_OCR: Optional[str] = "gpt-4o-mini"
    OPENAI_MODEL_FREE: Optional[str] = "gpt-5-mini"
    OPENAI_MODEL_QUERY_REWRITER: Optional[str] = "gpt-4o-mini"

    # LLM Provider
    LLM_PROVIDER: str = Field("openai", description="openai | openrouter")
    LLM_PROVIDER_ORGANIZER: Optional[str] = Field(None, description="Per-agent provider override")
    LLM_PROVIDER_WRITER: Optional[str] = Field(None, description="Per-agent provider override")
    LLM_PROVIDER_LEGAL: Optional[str] = Field(None, description="Per-agent provider override")
    LLM_PROVIDER_STAC: Optional[str] = Field(None, description="Per-agent provider override")
    LLM_PROVIDER_CASE_MANAGER: Optional[str] = Field(None, description="Per-agent provider override")
    LLM_PROVIDER_DOCUMENT_ANALYST: Optional[str] = Field(None, description="Per-agent provider override")
    LLM_PROVIDER_SPREADSHEET_ANALYST: Optional[str] = Field(None, description="Per-agent provider override")
    LLM_PROVIDER_VISION_OCR: Optional[str] = Field(None, description="Per-agent provider override")
    LLM_PROVIDER_FREE: Optional[str] = Field(None, description="Per-agent provider override")
    LLM_BASE_URL_OPENAI: str = "https://api.openai.com/v1"
    LLM_BASE_URL_OPENROUTER: str = "https://openrouter.ai/api/v1"
    OPENROUTER_API_KEY: Optional[str] = Field(None, description="OpenRouter API key")
    OPENROUTER_APP_URL: Optional[str] = Field(None, description="OpenRouter app URL for headers")
    OPENROUTER_APP_TITLE: Optional[str] = Field(None, description="OpenRouter app title for headers")

    # Web search
    SEARCH_PROVIDER: str = Field("serper", description="serper|tavily")
    SEARCH_API_KEY: Optional[str] = None
    ALLOWED_DOMAINS: str = "europa.eu,globalgap.org,boe.es"

    # RAG / vectores
    VECTOR_BACKEND: str = Field("sqlite", description="sqlite|pgvector|qdrant")
    DATABASE_URL: str = Field("sqlite:///./data.db")

    # Logging
    LOG_LEVEL: str = Field("INFO", description="DEBUG|INFO|WARNING|ERROR")
    SSE_LOG_MODE: str = Field("memory", description="memory|stdout|disk")
    SSE_TRACE: bool = Field(False, description="Habilita trazas SSE en memoria/disco")
    DISABLE_EXTERNALS: bool = Field(False, description="Bloquea llamadas a red/LLM (tests)")
    CORS_ORIGINS: str = Field(
        "http://localhost:5173",
        description="CSV de orígenes permitidos para CORS",
    )

    ENABLE_STAC: bool = True
    ENABLE_RS_ANALYST: bool = True

    QDRANT_PATH: str = "./qdrant_data"
    QDRANT_COLLECTION: str = "legal_chunks"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    LEGALIZE_DATA_DIR: str = "./data/legalize"
    LEGALIZE_DEFAULT_REPOS: str = "es,eu"
    LEGALIZE_INGEST_PROFILE: str = "agro"
    COST_TRACKING_ENABLED: bool = True
    COST_DB_PATH: str = "./data/costs.db"
    CONVERSATIONS_DB_PATH: str = "./data/conversations.db"
    COST_PRICING_MODE: str = Field("standard", description="standard|batch|flex|priority")
    COST_WARN_USD_PER_CONVERSATION: float = 0.25
    COST_WARN_USD_PER_DAY: float = 2.0
    WEB_SEARCH_COST_USD_PER_1K: float = 0.0
    OPENAI_WEB_SEARCH_COST_USD_PER_1K: float = 10.0
    ATTACHMENTS_DIR: str = "./backend/attachments"
    MEMORY_DIR: str = "./backend/memory"
    MEMORY_REMOTE_SENSING_TTL_DAYS: int = Field(
        21,
        description="Dias de frescura para reutilizar evidencia remota antes de forzar refresco.",
    )
    OCR_BACKEND: str = Field("tesseract", description="tesseract|none")
    LEGAL_RAG_STRATEGY: str = Field("hybrid", description="bm25|vector|hybrid")

    # --- STAC / Búsqueda semántica ---
    STAC_API_URL: str = Field(
        default="https://planetarycomputer.microsoft.com/api/stac/v1",
        description="Endpoint STAC base (p.ej. Planetary Computer)",
    )
    STAC_SEARCH_MODE: str = Field(
        default="direct", description="Modo de búsqueda STAC"  # direct | semantic_service
    )
    STAC_SEMANTIC_URL: str | None = Field(
        default=None, description="URL del microservicio semantic search (si se usa)"
    )
    STAC_DEFAULT_COLLECTIONS: str = Field(
        default="", description="Lista CSV de colecciones por defecto (opcional, mejor vacía)"
    )
    STAC_MAX_ITEMS: int = Field(
        default=12, description="Máximo de items a devolver en búsquedas STAC"
    )
    STAC_CONCURRENCY_LIMIT: int = Field(4, description="Concurrencia al consultar STAC")
    AGENT_CONCURRENCY_LIMIT: int = Field(
        4, description="Concurrencia máxima de agentes en modo multi"
    )
    REPLAN_MAX_COST_USD: float = Field(
        2.0, description="Coste máximo acumulado (USD) antes de desactivar replan automático"
    )

    # --- Geocodificación ---
    GEOCODER: str = "NOMINATIM"
    GEOCODER_URL: str = "https://nominatim.openstreetmap.org/search"
    GEOCODER_EMAIL: str | None = None
    GEOCODER_COUNTRY_BIAS: str = "ES"
    GEOCODER_VIEWBOX: str | None = None  # "minLon,minLat,maxLon,maxLat"
    GEOCODER_MAPBOX_TOKEN: str | None = None
    GEOCODER_GOOGLE_KEY: str | None = None

    def resolve_openai_model(self, *keys: str, default: Optional[str] = None) -> str:
        for key in keys:
            value = getattr(self, key, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        if isinstance(default, str) and default.strip():
            return default.strip()
        return self.OPENAI_MODEL_BASELINE

    def resolve_provider(self, *keys: str) -> str:
        """Resolve provider: per-agent overrides -> global."""
        for key in keys:
            value = getattr(self, key, None)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
        return (self.LLM_PROVIDER or "openai").strip().lower()

    def resolve_base_url(self, provider: str) -> str:
        if provider == "openrouter":
            return self.LLM_BASE_URL_OPENROUTER
        return self.LLM_BASE_URL_OPENAI

    def resolve_api_key(self, provider: str) -> str:
        if provider == "openrouter":
            if not self.OPENROUTER_API_KEY:
                raise RuntimeError(
                    "OPENROUTER_API_KEY is required when using OpenRouter provider."
                )
            return self.OPENROUTER_API_KEY
        return self.require_openai_key()

    def require_openai_key(self) -> str:
        if not self.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured. Set the environment variable to enable LLM functions."
            )
        return self.OPENAI_API_KEY


settings = Settings()


def require_openai_key() -> str:
    return settings.require_openai_key()


def configure_logging() -> None:
    """Configure loguru -> JSON logs to stdout, with queue (enqueue=True)."""
    logger.remove()
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        enqueue=False,  # more reliable for tests/demo local than async worker
        backtrace=False,
        diagnose=False,
        serialize=True,  # JSON output
    )
    logger.bind(component="bootstrap").info(
        json.dumps(
            {
                "event": "app_start",
                "version": VERSION,
                "vector_backend": settings.VECTOR_BACKEND,
                "search_provider": settings.SEARCH_PROVIDER,
            }
        )
    )
