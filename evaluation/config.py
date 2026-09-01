"""Configuration for comparing model backends in the evaluation framework.

Each model is evaluated as the "brain" of the same multi-agent system. There
is no separate monolithic baseline in the final dissertation benchmark.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    """Configuration for one evaluated model.

    If ``vision_model_id`` is specified, the ``vision_ocr`` agent uses it
    instead of ``model_id``. Other agents use ``model_id``.
    """

    name: str
    provider: str = "openrouter"
    model_id: str = ""
    role: str = "both"
    temperature: float = 0.0
    max_tokens: int | None = None
    vision_model_id: str | None = None
    vision_provider: str | None = None

    def __post_init__(self) -> None:
        if not self.model_id:
            self.model_id = self.name
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")


@dataclass
class JudgeConfig:
    """Configuration for one LLM judge.

    Each judge evaluates the same answer independently, allowing inter-judge
    consistency to be measured.
    """

    name: str          # identificador legible del juez
    model: str         # model_id del juez
    provider: str = "openrouter"
    max_tokens: int | None = 2048
    reasoning_enabled: bool | None = False

    def __post_init__(self) -> None:
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("judge max_tokens must be positive")


@dataclass
class EvalConfig:
    """Complete configuration for an evaluation run.

    Models are compared as backends for the multi-agent system. Each model
    invokes ``run_system()`` (``ChatOrchestratorService``). Multiple judges are
    supported for cross-evaluation.
    """

    models: dict[str, ModelConfig] = field(default_factory=dict)
    judges: list[JudgeConfig] = field(default_factory=list)
    judge_model: str = "openai/gpt-4.1-mini"  # LEGACY: se migra a judges[0]
    judge_provider: str = "openrouter"          # LEGACY
    budget_usd: float = 10.0
    runs_per_case: int = 1
    temperature: float = 0.0
    max_concurrent: int = 8
    corpus_path: str = "evaluation/cases/seed"
    adversarial_path: str = "evaluation/cases/adversarial"
    output_path: str = "evaluation/results"
    eval_stage: str = "full"
    max_cases: int | None = None

    def __post_init__(self) -> None:
        if self.runs_per_case < 1:
            raise ValueError("runs_per_case must be >= 1")
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        if self.budget_usd < 0:
            raise ValueError("budget_usd must be >= 0")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")

    @classmethod
    def from_json(cls, path: str | Path) -> EvalConfig:
        """Load configuration from a JSON file.

        Both the legacy ``judge_model``/``judge_provider`` format and the
        current ``judges`` list are supported.
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        models = {}
        for name, mdata in raw.get("models", {}).items():
            models[name] = ModelConfig(
                name=name,
                provider=mdata.get("provider", "openrouter"),
                model_id=mdata.get("model_id", name),
                role=mdata.get("role", "both"),
                temperature=mdata.get("temperature", 0.0),
                max_tokens=mdata.get("max_tokens"),
                vision_model_id=mdata.get("vision_model_id"),
                vision_provider=mdata.get("vision_provider"),
            )

        # Parse judges: nuevo formato (judges list) o legacy
        judges = []
        for jdata in raw.get("judges", []):
            judges.append(JudgeConfig(
                name=jdata["name"],
                model=jdata["model"],
                provider=jdata.get("provider", "openrouter"),
                max_tokens=jdata.get("max_tokens", 2048),
                reasoning_enabled=jdata.get("reasoning_enabled", False),
            ))

        # Legacy: si no hay judges pero sí judge_model, migrar
        judge_model = raw.get("judge_model", "")
        judge_provider = raw.get("judge_provider", "openrouter")
        if not judges and judge_model:
            judges.append(JudgeConfig(
                name=judge_model.split("/")[-1],
                model=judge_model,
                provider=judge_provider,
            ))

        return cls(
            models=models,
            judges=judges,
            judge_model=judge_model,
            judge_provider=judge_provider,
            budget_usd=raw.get("budget_usd", 10.0),
            runs_per_case=raw.get("runs_per_case", 1),
            temperature=raw.get("temperature", 0.0),
            max_concurrent=raw.get("max_concurrent", 8),
            corpus_path=raw.get("corpus_path", "evaluation/cases/seed"),
            adversarial_path=raw.get("adversarial_path", "evaluation/cases/adversarial"),
            output_path=raw.get("output_path", "evaluation/results"),
            eval_stage=raw.get("eval_stage", "full"),
            max_cases=raw.get("max_cases"),
        )

    def to_json(self, path: str | Path) -> None:
        """Guarda la configuración a un archivo JSON."""
        data = {
            "models": {
                name: {
                    "provider": m.provider,
                    "model_id": m.model_id,
                    "role": m.role,
                    "temperature": m.temperature,
                    "max_tokens": m.max_tokens,
                    "vision_model_id": m.vision_model_id,
                    "vision_provider": m.vision_provider,
                }
                for name, m in self.models.items()
            },
            "judges": [
                {
                    "name": j.name,
                    "model": j.model,
                    "provider": j.provider,
                    "max_tokens": j.max_tokens,
                    "reasoning_enabled": j.reasoning_enabled,
                }
                for j in self.judges
            ],
            "budget_usd": self.budget_usd,
            "runs_per_case": self.runs_per_case,
            "temperature": self.temperature,
            "max_concurrent": self.max_concurrent,
            "corpus_path": self.corpus_path,
            "adversarial_path": self.adversarial_path,
            "output_path": self.output_path,
            "eval_stage": self.eval_stage,
            "max_cases": self.max_cases,
        }
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def get_model_ids(self) -> list[str]:
        """Devuelve lista de model_ids configurados."""
        return [m.model_id for m in self.models.values()]

    def get_judge_names(self) -> list[str]:
        """Devuelve lista de nombres de jueces configurados."""
        return [j.name for j in self.judges]


DEFAULT_CONFIG = EvalConfig()
