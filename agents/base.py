from __future__ import annotations

import json
import time
from copy import deepcopy
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type

from openai import AsyncOpenAI

from backend.deps import settings, require_openai_key
from libs.costs.tracker import (
    cost_context,
    finish_cost_capture,
    record_openai_chat_usage,
    start_cost_capture,
    summarize_captured_events,
)
from libs.openai_compat import (
    chat_temperature_kwargs,
    completion_token_kwargs,
    tool_reasoning_kwargs,
)
from libs.schemas import AgentInput, BaseAgentOutput, AgentTrace
from libs.robust_json import JsonParseError, parse_json_content


def _build_client(provider: str) -> AsyncOpenAI:
    """Create an AsyncOpenAI client for the given provider."""
    base_url = settings.resolve_base_url(provider)
    api_key = settings.resolve_api_key(provider)
    extra_headers = {}
    if provider == "openrouter":
        if settings.OPENROUTER_APP_URL:
            extra_headers["HTTP-Referer"] = settings.OPENROUTER_APP_URL
        if settings.OPENROUTER_APP_TITLE:
            extra_headers["X-OpenRouter-Title"] = settings.OPENROUTER_APP_TITLE
    return AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers=extra_headers or None,
    )


UNSUPPORTED_STRICT_SCHEMA_KEYS = {
    "default",
    "nullable",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minItems",
    "maxItems",
}


def _message_field(message: Any, field: str, default: Any = None) -> Any:
    """Read OpenAI SDK and OpenRouter dict messages uniformly."""
    if isinstance(message, dict):
        return message.get(field, default)
    return getattr(message, field, default)


def openai_strict_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Return an OpenAI Structured Outputs-compatible copy of a JSON schema."""

    def normalize(node: Any) -> Any:
        if isinstance(node, list):
            return [normalize(item) for item in node]
        if not isinstance(node, dict):
            return node

        normalized: Dict[str, Any] = {}
        nullable = bool(node.get("nullable"))
        for key, value in node.items():
            if key in UNSUPPORTED_STRICT_SCHEMA_KEYS:
                continue
            normalized[key] = normalize(value)

        node_type = normalized.get("type")
        if nullable and isinstance(node_type, str):
            normalized["type"] = [node_type, "null"]

        properties = normalized.get("properties")
        if isinstance(properties, dict):
            normalized["required"] = list(properties.keys())
            normalized["additionalProperties"] = False

        if normalized.get("type") == "object" and "additionalProperties" not in normalized:
            normalized["additionalProperties"] = False

        return normalized

    return normalize(deepcopy(schema))


def validate_openai_strict_json_schema(schema: Dict[str, Any]) -> list[str]:
    errors: list[str] = []

    def visit(node: Any, path: str) -> None:
        if isinstance(node, list):
            for index, item in enumerate(node):
                visit(item, f"{path}[{index}]")
            return
        if not isinstance(node, dict):
            return

        for key in UNSUPPORTED_STRICT_SCHEMA_KEYS:
            if key in node:
                errors.append(f"{path}: unsupported key '{key}'")

        properties = node.get("properties")
        if isinstance(properties, dict):
            if node.get("additionalProperties") is not False:
                errors.append(f"{path}: object schemas must set additionalProperties=false")
            required = node.get("required")
            if set(required or []) != set(properties.keys()):
                errors.append(f"{path}: all properties must be listed in required")
            for name, child in properties.items():
                visit(child, f"{path}.properties.{name}")

        for key in ("items", "anyOf", "oneOf", "allOf", "$defs"):
            if key in node:
                visit(node[key], f"{path}.{key}")

    visit(schema, "$")
    return errors


class BaseAgent:
    name: str = "base"
    output_model: Type[BaseAgentOutput] = BaseAgentOutput
    requires_llm: bool = False
    retry_limit: int = 1
    timeout_s: Optional[float] = None
    _provider_key: str = "LLM_PROVIDER"

    def __init__(self) -> None:
        self.model = settings.resolve_openai_model("OPENAI_MODEL_ORGANIZER")
        self._client: Optional[AsyncOpenAI] = None
        self._provider: Optional[str] = None
        self.evaluation_temperature: float | None = None
        self.evaluation_max_tokens: int | None = None

    def _effective_temperature(self, requested: float) -> float:
        return requested if self.evaluation_temperature is None else self.evaluation_temperature

    def _completion_limits(self) -> dict[str, int]:
        return completion_token_kwargs(self.model, self.provider, self.evaluation_max_tokens)

    @property
    def provider(self) -> str:
        if self._provider is None:
            self._provider = settings.resolve_provider(self._provider_key)
        return self._provider

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = _build_client(self.provider)
        return self._client

    @property
    def supports_structured_outputs(self) -> bool:
        if self.provider == "openai":
            return True
        if self.provider == "openrouter":
            return self.model.startswith("openai/")
        return False

    @property
    def supports_tool_choice_none(self) -> bool:
        if self.provider == "openai":
            return True
        if self.provider == "openrouter":
            return self.model.startswith("openai/")
        return False

    def external_enabled(self) -> bool:
        return not settings.DISABLE_EXTERNALS

    def _error_output(self, exc: Exception, *, duration_ms: float) -> BaseAgentOutput:
        summary = f"{type(exc).__name__}: {exc}"
        return self.output_model(
            agent=self.name,
            status="error",
            summary=summary,
            errors=[str(exc)],
            trace=AgentTrace(duration_ms=duration_ms),
        )

    async def run(self, agent_input: AgentInput) -> BaseAgentOutput:
        start = time.monotonic()
        capture_token = start_cost_capture()
        capture_finished = False
        try:
            with cost_context(agent=self.name):
                result = await self._run(agent_input)
            trace = result.trace or AgentTrace()
            trace.duration_ms = (time.monotonic() - start) * 1000.0
            captured = finish_cost_capture(capture_token)
            capture_finished = True
            cost_totals = summarize_captured_events(captured)
            trace.tokens_input = int(cost_totals["input_tokens"] or 0) or trace.tokens_input
            trace.tokens_output = int(cost_totals["output_tokens"] or 0) or trace.tokens_output
            trace.cost_usd = float(cost_totals["cost_usd"] or 0.0) or trace.cost_usd
            result.trace = trace
            return result
        except Exception as exc:
            if not capture_finished:
                finish_cost_capture(capture_token)
            return self._error_output(exc, duration_ms=(time.monotonic() - start) * 1000.0)

    async def _run(self, agent_input: AgentInput) -> BaseAgentOutput:
        raise NotImplementedError

    async def call_llm_json(
        self,
        *,
        system: str,
        user: str,
        schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        if not self.external_enabled():
            raise RuntimeError("LLM disabled by DISABLE_EXTERNALS")
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            **chat_temperature_kwargs(self.model, self._effective_temperature(temperature)),
            **self._completion_limits(),
        }
        if schema is not None:
            if self.supports_structured_outputs:
                payload["response_format"] = self._json_response_format(schema)
            else:
                payload["messages"][1]["content"] += (
                    f"\n\nReturn ONLY valid JSON matching this schema:\n{json.dumps(schema)}"
                )
        response = await self.client.chat.completions.create(**payload)
        if getattr(response, "usage", None) is not None:
            record_openai_chat_usage(
                self.model,
                response.usage,
                operation="chat.completions",
                metadata={"agent": self.name},
                provider=self.provider,
            )
        content = _message_field(response.choices[0].message, "content") or "{}"
        try:
            return parse_json_content(content, expected="object").value
        except JsonParseError as exc:
            raise RuntimeError(f"Respuesta JSON inválida del agente {self.name}: {exc}") from exc

    async def call_llm_json_with_tools(
        self,
        *,
        system: str,
        user: str,
        schema: Optional[Dict[str, Any]] = None,
        tools: Optional[list[dict]] = None,
        tool_map: Optional[Dict[str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]]] = None,
        temperature: float = 0.2,
        max_iterations: int = 5,
    ) -> Dict[str, Any]:
        if not self.external_enabled():
            raise RuntimeError("LLM disabled by DISABLE_EXTERNALS")
        tools = tools or []
        tool_map = tool_map or {}
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        for iteration in range(max_iterations):
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
                **chat_temperature_kwargs(self.model, self._effective_temperature(temperature)),
                **tool_reasoning_kwargs(self.model, self.provider),
                **self._completion_limits(),
            )
            if getattr(response, "usage", None) is not None:
                record_openai_chat_usage(
                    self.model,
                    response.usage,
                    operation=f"chat.completions.tools.iteration_{iteration}",
                    metadata={"agent": self.name},
                    provider=self.provider,
                )
            msg = response.choices[0].message
            messages.append(msg)
            
            tool_calls = _message_field(msg, "tool_calls")
            if not tool_calls:
                break
                
            for tool_call in tool_calls:
                function = _message_field(tool_call, "function", {})
                fn_name = _message_field(function, "name", "")
                handler = tool_map.get(fn_name)
                parse_error: str | None = None
                try:
                    args = parse_json_content(
                        _message_field(function, "arguments") or "{}", expected="object"
                    ).value
                except JsonParseError as exc:
                    args = {}
                    parse_error = f"Argumentos JSON inválidos: {exc}"
                if parse_error:
                    result = {"error": parse_error}
                elif handler is None:
                    result = {"error": f"Tool '{fn_name}' not registered."}
                else:
                    try:
                        result = await handler(args)
                    except Exception as e:
                        result = {"error": f"Error executing '{fn_name}': {str(e)}"}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": _message_field(tool_call, "id", ""),
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
        
        if schema is not None:
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                **chat_temperature_kwargs(self.model, self._effective_temperature(temperature)),
                **tool_reasoning_kwargs(self.model, self.provider),
                **self._completion_limits(),
            }
            if tools:
                payload["tools"] = tools
                if self.supports_tool_choice_none:
                    payload["tool_choice"] = "none"
            if self.supports_structured_outputs:
                payload["response_format"] = self._json_response_format(schema)
            else:
                last_content = _message_field(messages[-1], "content", "") or ""
                if isinstance(messages[-1], dict):
                    messages[-1]["content"] = last_content + (
                        f"\n\nReturn ONLY valid JSON matching this schema:\n{json.dumps(schema)}"
                    )
                else:
                    messages[-1].content = last_content + (
                    f"\n\nReturn ONLY valid JSON matching this schema:\n{json.dumps(schema)}"
                    )
                payload["messages"] = messages
            response = await self.client.chat.completions.create(**payload)
            if getattr(response, "usage", None) is not None:
                record_openai_chat_usage(
                    self.model,
                    response.usage,
                    operation="chat.completions.tools.final",
                    metadata={"agent": self.name},
                    provider=self.provider,
                )
            content = _message_field(response.choices[0].message, "content") or "{}"
        else:
            content = _message_field(messages[-1], "content", "") or "{}"
            
        try:
            return parse_json_content(content, expected="object").value
        except JsonParseError as exc:
            raise RuntimeError(f"Respuesta JSON inválida del agente {self.name}: {exc}") from exc

    async def call_llm_text_with_tools(
        self,
        *,
        system: str,
        user: str,
        tools: Optional[list[dict]] = None,
        tool_map: Optional[Dict[str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]]] = None,
        temperature: float = 0.2,
        max_iterations: int = 5,
    ) -> str:
        if not self.external_enabled():
            raise RuntimeError("LLM disabled by DISABLE_EXTERNALS")
        tools = tools or []
        tool_map = tool_map or {}
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        for iteration in range(max_iterations):
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
                **chat_temperature_kwargs(self.model, self._effective_temperature(temperature)),
                **tool_reasoning_kwargs(self.model, self.provider),
                **self._completion_limits(),
            )
            if getattr(response, "usage", None) is not None:
                record_openai_chat_usage(
                    self.model,
                    response.usage,
                    operation=f"chat.completions.text.tools.iteration_{iteration}",
                    metadata={"agent": self.name},
                    provider=self.provider,
                )
            msg = response.choices[0].message
            messages.append(msg)
            
            tool_calls = _message_field(msg, "tool_calls")
            if not tool_calls:
                break
                
            for tool_call in tool_calls:
                function = _message_field(tool_call, "function", {})
                fn_name = _message_field(function, "name", "")
                handler = tool_map.get(fn_name)
                parse_error: str | None = None
                try:
                    args = parse_json_content(
                        _message_field(function, "arguments") or "{}", expected="object"
                    ).value
                except JsonParseError as exc:
                    args = {}
                    parse_error = f"Argumentos JSON inválidos: {exc}"
                if parse_error:
                    result = {"error": parse_error}
                elif handler is None:
                    result = {"error": f"Tool '{fn_name}' not registered."}
                else:
                    try:
                        result = await handler(args)
                    except Exception as e:
                        result = {"error": f"Error executing '{fn_name}': {str(e)}"}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": _message_field(tool_call, "id", ""),
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
        
        if tools and _message_field(messages[-1], "role") == "tool":
            final_kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                **chat_temperature_kwargs(self.model, self._effective_temperature(temperature)),
                **tool_reasoning_kwargs(self.model, self.provider),
                **self._completion_limits(),
            }
            if self.supports_tool_choice_none:
                final_kwargs["tool_choice"] = "none"
            response = await self.client.chat.completions.create(**final_kwargs)
            if getattr(response, "usage", None) is not None:
                record_openai_chat_usage(
                    self.model,
                    response.usage,
                    operation="chat.completions.text.tools.final",
                    metadata={"agent": self.name},
                    provider=self.provider,
                )
            return (_message_field(response.choices[0].message, "content") or "").strip()
        else:
            return (_message_field(messages[-1], "content", "") or "").strip()


    async def call_llm_text(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
    ) -> str:
        if not self.external_enabled():
            raise RuntimeError("LLM disabled by DISABLE_EXTERNALS")
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **chat_temperature_kwargs(self.model, self._effective_temperature(temperature)),
            **self._completion_limits(),
        )
        if getattr(response, "usage", None) is not None:
            record_openai_chat_usage(
                self.model,
                response.usage,
                operation="chat.completions.text",
                metadata={"agent": self.name},
                provider=self.provider,
            )
        return (_message_field(response.choices[0].message, "content") or "").strip()

    async def call_llm_vision(
        self,
        *,
        system: str,
        images: List[str],
        question: str,
        model: Optional[str] = None,
        temperature: float = 0.2,
        detail: str = "low",
    ) -> str:
        if not self.external_enabled():
            raise RuntimeError("LLM disabled by DISABLE_EXTERNALS")
        use_model = model or self.model
        content: list[Dict[str, Any]] = [{"type": "text", "text": question}]
        for img_data in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": img_data, "detail": detail},
            })
        response = await self.client.chat.completions.create(
            model=use_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            **chat_temperature_kwargs(use_model, self._effective_temperature(temperature)),
            **self._completion_limits(),
        )
        if getattr(response, "usage", None) is not None:
            record_openai_chat_usage(
                use_model,
                response.usage,
                operation="chat.completions.vision",
                metadata={"agent": self.name},
                provider=self.provider,
            )
        return (_message_field(response.choices[0].message, "content") or "").strip()

    async def call_llm_vision_json(
        self,
        *,
        system: str,
        images: List[str],
        question: str,
        schema: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        detail: str = "low",
    ) -> Dict[str, Any]:
        if not self.external_enabled():
            raise RuntimeError("LLM disabled by DISABLE_EXTERNALS")
        use_model = model or self.model
        content: list[Dict[str, Any]] = [{"type": "text", "text": question}]
        for img_data in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": img_data, "detail": detail},
                }
            )
        payload: Dict[str, Any] = {
            "model": use_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            **chat_temperature_kwargs(use_model, self._effective_temperature(temperature)),
            **self._completion_limits(),
        }
        if schema is not None:
            if self.supports_structured_outputs:
                payload["response_format"] = self._json_response_format(schema)
            else:
                content[0]["text"] += (
                    f"\n\nReturn ONLY valid JSON matching this schema:\n{json.dumps(schema)}"
                )
        response = await self.client.chat.completions.create(**payload)
        if getattr(response, "usage", None) is not None:
            record_openai_chat_usage(
                use_model,
                response.usage,
                operation="chat.completions.vision.json",
                metadata={"agent": self.name},
                provider=self.provider,
            )
        raw = _message_field(response.choices[0].message, "content") or "{}"
        try:
            return parse_json_content(raw, expected="object").value
        except JsonParseError as exc:
            raise RuntimeError(f"Respuesta JSON inválida del agente {self.name}: {exc}") from exc

    def _json_response_format(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        strict_schema = openai_strict_json_schema(schema)
        errors = validate_openai_strict_json_schema(strict_schema)
        if errors:
            raise ValueError(f"Schema incompatible with Structured Outputs: {'; '.join(errors)}")
        return {
            "type": "json_schema",
            "json_schema": {
                "name": f"{self.name}_response",
                "strict": True,
                "schema": strict_schema,
            },
        }
