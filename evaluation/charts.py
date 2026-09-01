"""Visualizaciones matplotlib para evaluación de chat."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib no instalado. Gráficas no disponibles.")


def create_radar_chart(
    data: dict[str, list[float]],
    output_path: str,
    title: str = "Radar Chart - Métricas por Modelo",
) -> str | None:
    """Crea un radar chart con las métricas del juez.

    Args:
        data: Dict model_name -> list de scores por dimensión.
        output_path: Ruta de salida de la imagen.
        title: Título del gráfico.

    Returns:
        Path de la imagen generada o None si falla.
    """
    if not HAS_MATPLOTLIB:
        return None

    # Dimensiones del juez
    dimensions = [
        "Factual", "Domain", "Responsible", "Actionable",
        "Decision", "Evidence", "Transparent", "Personalized",
        "Practical", "Overall"
    ]
    n_dims = len(dimensions)

    # Ángulos para el radar
    angles = [n / float(n_dims) * 2 * math.pi for n in range(n_dims)]
    angles += angles[:1]  # Cerrar el polígono

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    # Labels
    ax.set_xticklabels(dimensions, fontsize=9)

    # Escala 0-5
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8)

    colors = plt.cm.Set2.colors
    for i, (model, scores) in enumerate(data.items()):
        # Algunos resúmenes históricos incluyen métricas adicionales (p.ej.
        # coverage) además de las 10 dimensiones del juez. Normalizamos la
        # longitud para que el gráfico nunca rompa el guardado del batch.
        values = list(scores[:n_dims])
        if len(values) < n_dims:
            values.extend([0.0] * (n_dims - len(values)))
        values += values[:1]
        color = colors[i % len(colors)]
        ax.plot(angles, values, linewidth=2, label=model, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.title(title, size=14, fontweight="bold", pad=20)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def create_cost_quality_scatter(
    points: list[tuple[str, float, float]],
    output_path: str,
    title: str = "Coste vs Calidad (Pareto)",
) -> str | None:
    """Crea un scatter plot de coste vs calidad con frontera de Pareto.

    Args:
        points: Lista de (modelo, coste_usd, calidad).
        output_path: Ruta de salida.
        title: Título.

    Returns:
        Path o None.
    """
    if not HAS_MATPLOTLIB:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    names = [p[0] for p in points]
    costs = [p[1] for p in points]
    qualities = [p[2] for p in points]

    # Marcar puntos de Pareto
    from evaluation.stats import pareto_frontier
    pareto = pareto_frontier(points)

    colors = ["#e74c3c" if not p.is_pareto_optimal else "#2ecc71" for p in pareto]

    ax.scatter(costs, qualities, c=colors, s=100, alpha=0.7, edgecolors="black")

    # Etiquetas
    for name, cost, quality in points:
        ax.annotate(name, (cost, quality), xytext=(5, 5), textcoords="offset points", fontsize=8)

    # Línea de Pareto
    pareto_points = [(p.cost, p.quality) for p in pareto if p.is_pareto_optimal]
    pareto_points.sort()
    if len(pareto_points) > 1:
        xs, ys = zip(*pareto_points)
        ax.plot(xs, ys, "--", color="#27ae60", alpha=0.5, label="Frontera de Pareto")

    ax.set_xlabel("Coste (USD)", fontsize=11)
    ax.set_ylabel("Calidad (Judge Overall)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def create_forest_plot(
    comparison: dict[str, dict[str, float]],
    output_path: str,
    title: str = "Forest Plot - Comparación de Modelos",
) -> str | None:
    """Crea un forest plot con IC95%.

    Args:
        comparison: Dict model_name -> {mean, ci_low, ci_high}.
        output_path: Ruta de salida.
        title: Título.

    Returns:
        Path o None.
    """
    if not HAS_MATPLOTLIB:
        return None

    fig, ax = plt.subplots(figsize=(10, max(4, len(comparison) * 0.8)))

    models = list(comparison.keys())
    y_positions = range(len(models))

    for i, (model, stats) in enumerate(comparison.items()):
        mean = stats.get("mean", 0)
        ci_low = stats.get("ci_low", mean - 0.1)
        ci_high = stats.get("ci_high", mean + 0.1)

        # Línea de IC95%
        ax.plot([ci_low, ci_high], [i, i], "b-", linewidth=2)
        # Punto de la media
        ax.plot(mean, i, "bo", markersize=8)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlabel("Score", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    # Línea de referencia
    ax.axvline(x=3.0, color="gray", linestyle="--", alpha=0.5, label="Referencia (3.0)")
    ax.legend()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path
