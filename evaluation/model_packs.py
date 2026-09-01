"""Model packs predefinidos para evaluación.

Cada pack define un conjunto de modelos (texto + visión opcional)
que se usarán como cerebro del sistema multi-agente.

Uso:
    from evaluation.model_packs import get_pack_config, list_packs
    config = get_pack_config("china_barata")
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation.config import EvalConfig, JudgeConfig, ModelConfig


# ── Jueces por defecto ────────────────────────────────────────────────

DEFAULT_JUDGES: list[JudgeConfig] = [
    JudgeConfig(name="mimo-v2.5", model="xiaomi/mimo-v2.5", provider="openrouter", max_tokens=2048, reasoning_enabled=False),
    JudgeConfig(name="hy3-preview", model="tencent/hy3-preview", provider="openrouter", max_tokens=4096, reasoning_enabled=True),
    JudgeConfig(name="step-3.7-flash", model="stepfun/step-3.7-flash", provider="openrouter", max_tokens=4096, reasoning_enabled=True),
]

# ── Definición de packs ──────────────────────────────────────────────

MODEL_PACKS: dict[str, dict[str, Any]] = {
    "hiper_rapida": {
        "description": "Modelo rápido y barato (nitro)",
        "text": {
            "provider": "openrouter",
            "model_id": "openai/gpt-oss-120b:nitro",
        },
        "vision": {
            "provider": "openrouter",
            "model_id": "google/gemma-4-31b-it:nitro",
        },
    },
    "hiper_pequeña": {
        "description": "Modelo compacto multimodal con tools y structured outputs",
        "text": {
            "provider": "openrouter",
            "model_id": "mistralai/ministral-8b-2512",
        },
        "vision": {
            "provider": "openrouter",
            "model_id": "mistralai/ministral-8b-2512",
        },
    },
    "china_barata": {
        "description": "Modelos chinos baratos de buen nivel",
        "text": {
            "provider": "openrouter",
            "model_id": "deepseek/deepseek-v4-flash",
        },
        "vision": {
            "provider": "openrouter",
            "model_id": "qwen/qwen3.6-flash",
        },
    },
    "china_top": {
        "description": "Modelos chinos top",
        "text": {
            "provider": "openrouter",
            "model_id": "deepseek/deepseek-v4-pro",
        },
        "vision": {
            "provider": "openrouter",
            "model_id": "minimax/minimax-m3",
        },
    },
    "openai_buena": {
        "description": "OpenAI de calidad (vía API directa)",
        "text": {
            "provider": "openai",
            "model_id": "gpt-5.6-luna",
        },
        "vision": None,
    },
    "openai_barata": {
        "description": "OpenAI barata actual (vía API directa)",
        "text": {
            "provider": "openai",
            "model_id": "gpt-5-mini",
        },
        "vision": None,
    },
}


def list_packs() -> list[str]:
    """Devuelve lista de nombres de packs disponibles."""
    return list(MODEL_PACKS.keys())


def get_pack_info(pack_name: str) -> dict[str, Any]:
    """Devuelve la definición completa de un pack."""
    if pack_name not in MODEL_PACKS:
        raise ValueError(
            f"Pack '{pack_name}' no encontrado. "
            f"Packs disponibles: {', '.join(list_packs())}"
        )
    return MODEL_PACKS[pack_name]


def get_pack_config(
    pack_name: str,
    *,
    judges: list[JudgeConfig] | None = None,
    budget_usd: float = 10.0,
    runs_per_case: int = 1,
    temperature: float = 0.0,
    max_concurrent: int = 8,
    max_cases: int | None = None,
) -> EvalConfig:
    """Genera un EvalConfig completo desde un pack predefinido.

    Args:
        pack_name: Nombre del pack (ej: 'china_barata').
        judges: Lista de jueces (default: 3 jueces predefinidos).
        budget_usd: Presupuesto máximo en USD.
        runs_per_case: Veces que se ejecuta cada caso.
        temperature: Temperatura del modelo.
        max_concurrent: Máximo de ejecuciones concurrentes.
        max_cases: Máximo de casos a evaluar (None = todos).

    Returns:
        EvalConfig listo para usar con run_batch().
    """
    pack = get_pack_info(pack_name)
    text_cfg = pack["text"]
    vision_cfg = pack.get("vision")

    model_config = ModelConfig(
        name=pack_name,
        provider=text_cfg["provider"],
        model_id=text_cfg["model_id"],
        role="both",
        temperature=temperature,
        vision_model_id=vision_cfg["model_id"] if vision_cfg else None,
        vision_provider=vision_cfg["provider"] if vision_cfg else None,
    )

    effective_judges = judges if judges is not None else DEFAULT_JUDGES

    return EvalConfig(
        models={pack_name: model_config},
        judges=effective_judges,
        budget_usd=budget_usd,
        runs_per_case=runs_per_case,
        temperature=temperature,
        max_concurrent=max_concurrent,
        corpus_path="evaluation/cases/seed",
        adversarial_path="evaluation/cases/adversarial",
        output_path="evaluation/results",
        eval_stage="full",
        max_cases=max_cases,
    )


def generate_config_files(output_dir: str = "evaluation/configs") -> list[Path]:
    """Genera archivos JSON predefinidos para cada pack.

    Crea un directorio con un JSON por pack, listo para usar con
    `evaluation/cli.py run --config evaluation/configs/china_barata.json`.

    Args:
        output_dir: Directorio donde crear los JSONs.

    Returns:
        Lista de paths de archivos creados.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    created = []
    for pack_name in MODEL_PACKS:
        config = get_pack_config(pack_name)
        path = out / f"{pack_name}.json"
        config.to_json(path)
        created.append(path)

    return created
