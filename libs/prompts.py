from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from jinja2 import Environment, StrictUndefined, Template

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
PROMPT_PROTOCOL_VERSION = "agro-prompting-2026-04-02"

_PROMPT_ENV = Environment(
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=StrictUndefined,
)


@lru_cache(maxsize=128)
def _load_template(path: Path) -> Template:
    return _PROMPT_ENV.from_string(path.read_text(encoding="utf-8"))


def render_prompt(name: str, **context: Dict[str, Any]) -> str:
    path = PROMPT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    template = _load_template(path)
    return template.render(**context).strip()


def compose_system_prompt(
    *,
    agent_name: str,
    body: str,
    output_contract: str | None = None,
) -> str:
    prefix = (
        f"# System context\n"
        f"You are part of a multi-agent agricultural copiloting system.\n"
        f"Prompt protocol version: {PROMPT_PROTOCOL_VERSION}.\n"
        f"Current agent: {agent_name}.\n"
        f"- Use only the context and tools available to this agent.\n"
        f"- Do not mention internal routing, transfers, retries or hidden agent mechanics.\n"
        f"- If evidence is weak, say so explicitly instead of filling gaps.\n"
        f"- Keep stable instructions here and task-specific evidence in the user prompt.\n"
    )
    contract = f"\n\n# Output contract\n{output_contract.strip()}" if output_contract else ""
    return f"{prefix}\n{body.strip()}{contract}".strip()
