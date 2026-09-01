"""Soporte LLM para evaluación con soporte OpenRouter y tracking de costes."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import httpx
from openai import AsyncOpenAI

from backend.deps import settings
from libs.costs.pricing import get_model_price
from libs.robust_json import JsonParseError, extract_llm_content, parse_json_content


from loguru import logger


def _read_positive_timeout(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number of seconds") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive number of seconds")
    return value


def _decode_json_content(content: str) -> dict[str, Any]:
    """Decodifica JSON directo o envuelto en un bloque Markdown fenced.

    Algunos endpoints de OpenRouter respetan el contenido del schema, pero
    añaden `````json`` y ````` alrededor. Ese envoltorio no cambia la carga y
    se puede eliminar de forma determinista antes de validar el contrato.
    """
    try:
        return parse_json_content(content, expected="object").value
    except JsonParseError as exc:
        info = exc.result.diagnostics()
        raise RuntimeError(
            f"JSON no interpretable ({info['status']}, {info['method']}): {info['preview'][:240]}"
        ) from exc


# Deliberately generous: a multi-agent execution can involve several provider
# calls and OpenRouter may queue a request before returning it.  The value is
# configurable for local probes without changing code.
EVALUATION_REQUEST_TIMEOUT_SECONDS = _read_positive_timeout(
    "EVALUATION_REQUEST_TIMEOUT_SECONDS", 900.0
)
EVALUATION_CATALOG_TIMEOUT_SECONDS = _read_positive_timeout(
    "EVALUATION_CATALOG_TIMEOUT_SECONDS", 30.0
)
_openrouter_capabilities: dict[str, set[str] | None] = {}


def llm_enabled() -> bool:
    """Verifica si las llamadas LLM están habilitadas para evaluación."""
    explicit_opt_in = os.environ.get("EVALUATION_ENABLE_LLM", "").strip().lower()
    return explicit_opt_in in {"1", "true", "yes", "on"} and (
        bool(settings.OPENAI_API_KEY) or bool(settings.OPENROUTER_API_KEY)
    )


def provider_enabled(provider: str) -> bool:
    """Return whether the explicitly selected provider has credentials."""
    normalized = (provider or "").strip().lower()
    if normalized == "openrouter":
        return bool(settings.OPENROUTER_API_KEY)
    if normalized == "openai":
        return bool(settings.OPENAI_API_KEY)
    return False


def _build_client(provider: str = "openrouter") -> AsyncOpenAI:
    """Construye un cliente AsyncOpenAI para el provider dado."""
    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        base_url = "https://openrouter.ai/api/v1"
        extra_headers = {}
        app_url = os.environ.get("OPENROUTER_APP_URL", "")
        app_title = os.environ.get("OPENROUTER_APP_TITLE", "")
        if app_url:
            extra_headers["HTTP-Referer"] = app_url
        if app_title:
            extra_headers["X-OpenRouter-Title"] = app_title
        return AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers=extra_headers or None,
            timeout=EVALUATION_REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
        )
    # Default: OpenAI
    api_key = os.environ.get("OPENAI_API_KEY", "")
    return AsyncOpenAI(
        base_url="https://api.openai.com/v1",
        api_key=api_key,
        timeout=EVALUATION_REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )


async def _get_openrouter_capabilities(model: str) -> set[str] | None:
    """Obtiene una vez los parámetros declarados por OpenRouter para un modelo."""
    if model in _openrouter_capabilities:
        return _openrouter_capabilities[model]

    try:
        async with httpx.AsyncClient(timeout=EVALUATION_CATALOG_TIMEOUT_SECONDS) as client:
            response = await client.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}"},
            )
            response.raise_for_status()
            entries = response.json().get("data", [])
        capabilities = next(
            (
                set(entry.get("supported_parameters", []))
                for entry in entries
                if entry.get("id") == model
            ),
            None,
        )
        _openrouter_capabilities[model] = capabilities
        return capabilities
    except Exception as exc:
        logger.warning(f"No se pudieron consultar capacidades de OpenRouter para {model}: {exc}")
        # Ante un fallo del catálogo, preferimos intentar structured outputs.
        _openrouter_capabilities[model] = None
        return None


def _resolve_provider(model_id: str) -> str:
    """Resuelve el provider basándose en el model_id."""
    if "/" in model_id and not model_id.startswith("gpt"):
        return "openrouter"
    return "openai"


@dataclass
class LLMCallMetrics:
    """Métricas de una llamada LLM individual."""

    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cost_known: bool = True
    latency_ms: float = 0.0
    operation: str = ""
    component: str = "other"
    timestamp: str = ""
    parse_status: str = "not_applicable"
    parse_method: str = ""
    parse_warning: str = ""
    retry: bool = False
    truncated: bool = False


@dataclass
class LLMCallTracker:
    """Acumula métricas de llamadas LLM durante una evaluación."""

    calls: list[LLMCallMetrics] = field(default_factory=list)

    def record(
        self,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        latency_ms: float,
        cost_known: bool = True,
        operation: str = "",
        component: str | None = None,
        parse_status: str = "not_applicable",
        parse_method: str = "",
        parse_warning: str = "",
        retry: bool = False,
        truncated: bool = False,
    ) -> None:
        import datetime as dt

        normalized_operation = operation or ""
        if component is None:
            if normalized_operation.startswith("eval."):
                component = "judge"
            elif any(token in normalized_operation.lower() for token in ("vision", "ocr")):
                component = "vision"
            elif normalized_operation.startswith("system."):
                component = "system"
            else:
                component = "other"

        self.calls.append(
            LLMCallMetrics(
                model=model,
                provider=provider,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost_usd,
                cost_known=cost_known,
                latency_ms=latency_ms,
                operation=operation,
                component=component,
                timestamp=dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds"),
                parse_status=parse_status,
                parse_method=parse_method,
                parse_warning=parse_warning,
                retry=retry,
                truncated=truncated,
            )
        )

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def total_latency_ms(self) -> float:
        return sum(c.latency_ms for c in self.calls)

    @property
    def cost_complete(self) -> bool:
        return all(c.cost_known for c in self.calls)

    @property
    def unknown_cost_calls(self) -> int:
        return sum(1 for c in self.calls if not c.cost_known)

    @property
    def unknown_cost_models(self) -> list[str]:
        return sorted({c.model for c in self.calls if not c.cost_known})

    @property
    def total_tokens(self) -> int:
        return sum(c.prompt_tokens + c.completion_tokens for c in self.calls)

    def parse_summary(self) -> dict[str, int]:
        """Aggregate parser outcomes without exposing full model responses."""
        summary: dict[str, int] = {}
        for call in self.calls:
            status = call.parse_status
            if status == "not_applicable":
                continue
            summary[status] = summary.get(status, 0) + 1
        return summary

    def breakdown(self, attribute: str) -> dict[str, float]:
        """Aggregate cost or latency by provider and logical component."""
        result: dict[str, float] = {}
        for call in self.calls:
            key = f"{call.provider}:{call.component}"
            result[key] = result.get(key, 0.0) + float(getattr(call, attribute, 0.0))
        return {key: round(value, 9) for key, value in sorted(result.items())}

    def component_breakdown(self, attribute: str) -> dict[str, float]:
        result: dict[str, float] = {}
        for call in self.calls:
            result[call.component] = result.get(call.component, 0.0) + float(
                getattr(call, attribute, 0.0)
            )
        return {key: round(value, 9) for key, value in sorted(result.items())}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> tuple[float, bool]:
    """Estima el coste en USD basándose en el catálogo de precios."""
    price = get_model_price(model, "text")
    if price is None:
        logger.debug(f"No se encontró precio para modelo '{model}', coste = 0.0")
        logger.warning(f"No se encontró precio para modelo '{model}'; coste no incluido en el total conocido")
        return 0.0, False
    cost = (
        prompt_tokens * price.input_per_million
        + completion_tokens * price.output_per_million
    ) / 1_000_000
    return cost, True


async def call_llm_json(
    *,
    system: str,
    user: str,
    schema_name: str,
    schema: Dict[str, Any],
    model: str | None = None,
    provider: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    reasoning_enabled: bool | None = None,
    tracker: LLMCallTracker | None = None,
) -> Dict[str, Any]:
    """Llamada LLM que devuelve JSON validado contra un schema."""
    if not llm_enabled():
        raise RuntimeError("LLM no disponible para evaluación. Establece EVALUATION_ENABLE_LLM=1.")

    effective_model = model or "openai/gpt-4.1-mini"
    effective_provider = (provider or _resolve_provider(effective_model)).strip().lower()
    if not provider_enabled(effective_provider):
        raise RuntimeError(f"No hay credenciales configuradas para el proveedor de evaluación '{effective_provider}'.")
    client = _build_client(effective_provider)

    # Preparar payload
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    # Usar structured outputs cuando el modelo lo declara. OpenRouter puede
    # enrutar a un proveedor incompatible, por eso require_parameters es clave.
    response_format: dict[str, Any] | None = None
    require_structured_parameters = False
    if effective_provider == "openai":
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        }
    elif effective_provider == "openrouter" and not effective_model.startswith("openai/"):
        capabilities = await _get_openrouter_capabilities(effective_model)
        supports_structured = capabilities is None or bool(
            {"response_format", "structured_outputs"} & capabilities
        )
        if supports_structured:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            }
            require_structured_parameters = True

    # Para modelos sin soporte declarado, conservar el fallback textual.
    if effective_provider == "openrouter" and response_format is None:
        messages[1]["content"] += (
            f"\n\nResponde SOLO con JSON válido que cumpla este esquema:\n{json.dumps(schema, ensure_ascii=False)}"
        )

    start = time.monotonic()
    try:
        kwargs: dict[str, Any] = {
            "model": effective_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format
        if require_structured_parameters:
            extra_body: dict[str, Any] = {"provider": {"require_parameters": True}}
            if reasoning_enabled is not None:
                extra_body["reasoning"] = {"enabled": reasoning_enabled}
            kwargs["extra_body"] = extra_body

        response = await client.chat.completions.create(**kwargs)
    except Exception as exc:
        raise RuntimeError(f"Error en llamada LLM ({effective_model}): {exc}") from exc

    latency_ms = (time.monotonic() - start) * 1000

    # Extraer uso y calcular coste
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    cost_usd, cost_known = _estimate_cost(effective_model, prompt_tokens, completion_tokens)

    # Registrar en tracker
    if tracker is not None:
        tracker.record(
            model=effective_model,
            provider=effective_provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            cost_known=cost_known,
            latency_ms=latency_ms,
            operation=f"eval.{schema_name}",
        )

    content, finish_reason, extraction_warning = extract_llm_content(response)
    if content is None:
        logger.warning(
            f"Respuesta estructurada no textual para {effective_model}: "
            f"tipo={type(content).__name__}"
        )
        raise RuntimeError(f"Respuesta estructurada no textual para {effective_model}")
    try:
        parsed = parse_json_content(content, expected="object", finish_reason=finish_reason)
        if tracker is not None and tracker.calls:
            tracker.calls[-1].parse_status = parsed.status
            tracker.calls[-1].parse_method = parsed.method
            tracker.calls[-1].parse_warning = ";".join(parsed.warnings + ([extraction_warning] if extraction_warning else []))
            tracker.calls[-1].truncated = parsed.status == "truncated"
        return parsed.value
    except JsonParseError as exc:
        preview = " ".join(str(content)[:500].split())
        logger.warning(f"JSON inválido del juez {effective_model} (preview={preview!r})")
        if tracker is not None and tracker.calls:
            diagnostics = exc.result.diagnostics()
            tracker.calls[-1].parse_status = diagnostics["status"]
            tracker.calls[-1].parse_method = diagnostics["method"]
            tracker.calls[-1].parse_warning = ";".join(diagnostics["warnings"])
            tracker.calls[-1].truncated = diagnostics["status"] == "truncated"
        raise RuntimeError(
            f"JSON no interpretable ({exc.result.status}, {exc.result.method})"
        ) from exc


async def call_llm_text(
    *,
    system: str,
    user: str,
    model: str | None = None,
    provider: str | None = None,
    temperature: float = 0.0,
    tracker: LLMCallTracker | None = None,
) -> str:
    """Llamada LLM que devuelve texto plano."""
    if not llm_enabled():
        raise RuntimeError("LLM no disponible para evaluación. Establece EVALUATION_ENABLE_LLM=1.")

    effective_model = model or "openai/gpt-4.1-mini"
    effective_provider = (provider or _resolve_provider(effective_model)).strip().lower()
    if not provider_enabled(effective_provider):
        raise RuntimeError(f"No hay credenciales configuradas para el proveedor de evaluación '{effective_provider}'.")
    client = _build_client(effective_provider)

    start = time.monotonic()
    response = await client.chat.completions.create(
        model=effective_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    latency_ms = (time.monotonic() - start) * 1000

    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    cost_usd, cost_known = _estimate_cost(effective_model, prompt_tokens, completion_tokens)

    if tracker is not None:
        tracker.record(
            model=effective_model,
            provider=effective_provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            cost_known=cost_known,
            latency_ms=latency_ms,
            operation="eval.text",
        )

    return (response.choices[0].message.content or "").strip()
