"""CLI for model evaluation in AgroCopilot.

Main commands:
  compare  — compare model backends in the multi-agent system
  run      — run a complete evaluation
  report   — generate a report from saved results
  audit    — inspect evaluation quality
  routing  — analyse agent-routing patterns by model
"""
from __future__ import annotations

import asyncio
import csv
import json
import sys
from pathlib import Path

import click
from loguru import logger


def _setup_logging(verbose: bool = False) -> None:
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level, format="{time:HH:mm:ss} | {level:<7} | {message}")


def _resolve_config(config_path: str | None, model_pack: str | None):
    """Resolve config from file or model pack name."""
    from evaluation.config import EvalConfig
    from evaluation.model_packs import get_pack_config

    if model_pack:
        return get_pack_config(model_pack)
    if config_path:
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            logger.error(f"Config no encontrado: {config_path}")
            sys.exit(1)
        return EvalConfig.from_json(str(cfg_path))
    logger.error("Debes especificar --config o --model-pack")
    sys.exit(1)


def _parse_case_range(value: str | None, total: int) -> tuple[list[int], dict[str, int] | None]:
    """Parse an inclusive one-based range such as ``1:10``."""
    if not value:
        return list(range(total)), None
    try:
        start_text, end_text = value.split(":", 1)
        start = int(start_text)
        end = int(end_text)
    except (ValueError, AttributeError) as exc:
        raise click.BadParameter("El rango debe tener formato INICIO:FIN, por ejemplo 1:10") from exc
    if start < 1 or end < start or end > total:
        raise click.BadParameter(f"Rango inválido; debe estar entre 1 y {total}")
    return list(range(start - 1, end)), {"start": start, "end": end}


def _sort_cases(cases):
    return sorted(cases, key=lambda case: (1 if case.case_id.startswith("adv_") else 0, case.case_id))


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Salida detallada")
def cli(verbose: bool) -> None:
    """AgroCopilot Evaluación — Comparación de Modelos."""
    _setup_logging(verbose)


# ── compare ──────────────────────────────────────────────────────────


@cli.command()
@click.option("--config", "config_path", default=None, help="Archivo eval_config.json")
@click.option("--model-pack", default=None, type=click.Choice([
    "hiper_rapida", "hiper_pequeña", "china_barata", "china_top",
    "openai_buena", "openai_barata",
]), help="Pack de modelos predefinido")
@click.option("--corpus", default=None, help="Directorio con casos JSON (default: config)")
def preflight(config_path: str | None, model_pack: str | None, corpus: str | None) -> None:
    """Valida configuración y adjuntos sin ejecutar llamadas LLM."""
    from evaluation.baselines import validate_case_attachments
    from evaluation.llm_support import llm_enabled, provider_enabled
    from evaluation.loaders import load_cases

    config = _resolve_config(config_path, model_pack)
    cases = load_cases(corpus or config.corpus_path)
    attachment_issues = [
        issue for case in cases for issue in validate_case_attachments(case)
    ]
    unavailable = [
        f"{model.name} ({model.provider})"
        for model in config.models.values()
        if not provider_enabled(model.provider)
    ]
    # Verificar jueces
    for jp in {j.provider for j in config.judges}:
        if not provider_enabled(jp):
            jnames = [j.name for j in config.judges if j.provider == jp]
            unavailable.append(f"juez {jp} ({', '.join(jnames)})")

    click.echo(f"Casos válidos: {len(cases)}")
    click.echo(f"Problemas de adjuntos: {len(attachment_issues)}")
    click.echo(f"Opt-in LLM: {'activo' if llm_enabled() else 'inactivo'}")
    if unavailable:
        click.echo("Credenciales pendientes: " + ", ".join(unavailable))
    if attachment_issues:
        for item in attachment_issues:
            click.echo(f"  - {item}")
    if attachment_issues:
        raise click.ClickException("Hay fixtures de evaluación ausentes o con hash incorrecto.")
    click.echo("Preflight de corpus completada. Añade credenciales y EVALUATION_ENABLE_LLM=1 para ejecutar.")


@cli.command()
@click.option("--config", "config_path", default=None, help="Archivo eval_config.json")
@click.option("--model-pack", default=None, type=click.Choice([
    "hiper_rapida", "hiper_pequeña", "china_barata", "china_top",
    "openai_buena", "openai_barata",
]), help="Pack de modelos predefinido")
@click.option("--corpus", default=None, help="Directorio con casos JSON (default: config)")
@click.option("--output", "-o", default="./eval_results", help="Directorio de salida")
@click.option("--judge", "judge_filter", multiple=True, help="Filtrar jueces por nombre (ej: --judge hy3)")
@click.option("--max-concurrent", default=None, type=int, help="Ejecuciones concurrentes (por defecto: configuración)")
@click.option("--case-range", default=None, help="Rango inclusivo de casos, por ejemplo 1:10")
@click.option("--dry-run", is_flag=True, help="Solo muestra el plan, no ejecuta")
def compare(
    config_path: str | None,
    model_pack: str | None,
    corpus: str | None,
    output: str,
    judge_filter: tuple[str, ...],
    max_concurrent: int | None,
    case_range: str | None,
    dry_run: bool,
) -> None:
    """Compara modelos usando el sistema multi-agente."""
    from evaluation.config import JudgeConfig
    from evaluation.loaders import load_cases
    from evaluation.runners import run_batch

    config = _resolve_config(config_path, model_pack)
    # El runner publica el progreso usando EvalConfig.output_path; debe
    # coincidir con el destino final solicitado por el usuario.
    config.output_path = output

    # Filtrar jueces si se especifica
    if judge_filter:
        config.judges = [j for j in config.judges if j.name in judge_filter]
        if not config.judges:
            raise click.ClickException(
                f"Ningún juez coincide con los filtros: {', '.join(judge_filter)}. "
                f"Jueces disponibles: {', '.join(j.name for j in config.judges)}"
            )

    # Cargar casos
    corpus_path = corpus or config.corpus_path
    cases = _sort_cases(load_cases(corpus_path))
    if config.max_cases:
        cases = cases[:config.max_cases]
    selected_indexes, selected_range = _parse_case_range(case_range, len(cases))
    cases = [cases[index] for index in selected_indexes]
    if not cases:
        logger.error(f"No se encontraron casos en {corpus_path}")
        sys.exit(1)

    model_ids = config.get_model_ids()

    # Calcular plan
    n_cases = len(cases)
    n_models = len(model_ids)
    n_runs = config.runs_per_case
    total_executions = n_cases * n_models * n_runs

    # Estimar coste. Las respuestas reales del sistema incluyen bastante más
    # contexto que el prompt de la consulta, y los jueces pueden repetir una
    # llamada si el JSON estructurado no cumple el contrato. Por eso mostramos
    # una estimación central y una reserva de reintentos, no una cifra única
    # artificialmente optimista.
    unknown_price_models: set[str] = set()
    try:
        from libs.costs.pricing import get_model_price
        estimated_cost = 0.0
        retry_reserve = 0.0
        avg_input_tokens = 4000
        avg_output_tokens = 1500
        judge_input_tokens = 4500
        judge_output_tokens = 1000

        def _price_cost(price, input_tokens: int, output_tokens: int) -> float:
            return (
                price.input_per_million * input_tokens
                + price.output_per_million * output_tokens
            ) / 1_000_000

        for model_id in model_ids:
            price = get_model_price(model_id)
            if price:
                system_cost = _price_cost(price, avg_input_tokens, avg_output_tokens)
                estimated_cost += system_cost * n_cases * n_runs
            else:
                unknown_price_models.add(model_id)
        # Coste de jueces
        if estimated_cost is not None:
            for jcfg in config.judges:
                jprice = get_model_price(jcfg.model)
                if not jprice:
                    unknown_price_models.add(jcfg.model)
                    continue
                jcost = _price_cost(jprice, judge_input_tokens, judge_output_tokens)
                executions = n_cases * n_runs
                estimated_cost += jcost * executions
                # evaluate_multi_metrics permite hasta 3 intentos; la reserva
                # expresa el coste adicional si todos los jueces fallan una
                # vez antes de recuperar el formato.
                retry_reserve += jcost * executions * 2
    except ImportError:
        estimated_cost = None
        retry_reserve = 0.0

    click.echo(f"\nPlan de evaluacion:")
    click.echo(f"  Casos:         {n_cases}")
    click.echo(f"  Modelos:       {n_models}")
    click.echo(f"  Runs/caso:     {n_runs}")
    click.echo(f"  Ejecuciones:   {total_executions}")
    click.echo(f"  Jueces:        {len(config.judges)} ({', '.join(j.name for j in config.judges)})")
    click.echo(f"  LLM calls:     ~{total_executions * (1 + len(config.judges))}")
    if estimated_cost is None:
        click.echo("  Coste est.:    no disponible (falta precio de algún modelo)")
    else:
        suffix = " (mínimo conocido; faltan precios)" if unknown_price_models else ""
        click.echo(f"  Coste est.:    ~${estimated_cost:.3f}{suffix}")
        click.echo(f"  Reserva retry: +${retry_reserve:.3f} (hasta ~${estimated_cost + retry_reserve:.3f})")
    if unknown_price_models:
        click.echo(f"  Sin precio:    {', '.join(sorted(unknown_price_models))}")
    click.echo(f"  Presupuesto:   ${config.budget_usd:.2f}")
    click.echo(f"  Modelos:       {model_ids}")

    if dry_run:
        click.echo("\n[DRY RUN] No se ejecuta nada.")
        return

    click.echo("\nEjecutando...")
    batch = asyncio.run(
        run_batch(cases, config, max_concurrent=max_concurrent, case_range=selected_range)
    )

    batch_dir = batch.save(output)
    click.echo(f"\nResultados guardados en: {batch_dir}")
    if batch.tracker.cost_complete:
        click.echo(f"Coste total estimado: ${batch.total_cost_usd:.4f}")
    else:
        click.echo(
            f"Coste conocido: ${batch.total_cost_usd:.4f} "
            f"(incompleto; sin precio para: {', '.join(batch.tracker.unknown_cost_models)})"
        )
    click.echo(f"Latencia:    {batch.total_latency_ms/1000:.1f}s")

    click.echo("\nResumen por modelo:")
    for model_id, agg in batch._aggregate_by_model().items():
        click.echo(
            f"  {model_id:<45} "
            f"judge={agg['judge_overall_mean']:.2f}/5  "
            f"lat={agg['latency_mean_ms']:.0f}ms  "
            f"cost=${agg['cost_mean_usd']:.4f}  "
            f"(n={agg['n']})"
        )


# ── run ──────────────────────────────────────────────────────────────


@cli.command()
@click.option("--config", "config_path", default=None, help="Archivo eval_config.json")
@click.option("--model-pack", default=None, type=click.Choice([
    "hiper_rapida", "hiper_pequeña", "china_barata", "china_top",
    "openai_buena", "openai_barata",
]), help="Pack de modelos predefinido")
@click.option("--corpus", default=None, help="Directorio con casos JSON")
@click.option("--output", "-o", default="./eval_results", help="Directorio de salida")
@click.option("--judge", "judge_filter", multiple=True, help="Filtrar jueces por nombre")
@click.option("--models", multiple=True, help="Filtrar modelos especificos")
def run(
    config_path: str | None,
    model_pack: str | None,
    corpus: str | None,
    output: str,
    judge_filter: tuple[str, ...],
    models: tuple[str, ...],
) -> None:
    """Ejecuta evaluacion completa."""
    from evaluation.loaders import load_cases
    from evaluation.runners import run_batch

    config = _resolve_config(config_path, model_pack)

    if judge_filter:
        config.judges = [j for j in config.judges if j.name in judge_filter]
        if not config.judges:
            raise click.ClickException(f"Ningún juez coincide con: {', '.join(judge_filter)}")

    if models:
        config.models = {k: v for k, v in config.models.items() if v.model_id in models or k in models}

    corpus_path = corpus or config.corpus_path
    cases = load_cases(corpus_path)
    if not cases:
        logger.error(f"No se encontraron casos en {corpus_path}")
        sys.exit(1)

    click.echo(f"Ejecutando {len(cases)} casos con {len(config.models)} modelos...")

    batch = asyncio.run(run_batch(cases, config))
    batch_dir = batch.save(output)

    click.echo(f"\nCompletado: {batch_dir}")
    click.echo(
        f"Coste estimado: ${batch.total_cost_usd:.4f}"
        if batch.tracker.cost_complete
        else f"Coste conocido: ${batch.total_cost_usd:.4f} (incompleto)"
    )


# ── report ───────────────────────────────────────────────────────────


def _cases_for_manifest(config, manifest):
    from evaluation.loaders import load_cases

    candidates = []
    for source in (config.corpus_path, config.adversarial_path):
        try:
            candidates.extend(load_cases(source))
        except Exception:
            continue
    by_id = {case.case_id: case for case in _sort_cases(candidates)}
    missing = [case_id for case_id in manifest.get("case_ids", []) if case_id not in by_id]
    if missing:
        raise click.ClickException(f"No se pudieron cargar casos del batch: {', '.join(missing[:5])}")
    return [by_id[case_id] for case_id in manifest.get("case_ids", [])]


@cli.command()
@click.option("--batch", "batch_dir", required=True, help="Directorio del batch a reanudar")
@click.option("--max-concurrent", default=None, type=int, help="Concurrencia máxima para tareas pendientes")
def resume(batch_dir: str, max_concurrent: int | None) -> None:
    """Reanuda tareas incompletas de un batch."""
    from evaluation.config import EvalConfig
    from evaluation.persistence import load_manifest
    from evaluation.runners import run_batch

    root = Path(batch_dir)
    manifest = load_manifest(root)
    config_path = root / "eval_config.json"
    if not config_path.exists():
        raise click.ClickException("El batch no contiene eval_config.json")
    config = EvalConfig.from_json(config_path)
    config.output_path = str(root.parent)
    cases = _cases_for_manifest(config, manifest)
    click.echo(f"Reanudando {manifest['batch_id']}: {manifest.get('expected_tasks', 0)} tareas")
    batch = asyncio.run(run_batch(cases, config, max_concurrent=max_concurrent, resume_batch_dir=root))
    result_dir = batch.save(str(root.parent))
    click.echo(f"Resultados actualizados: {result_dir}")
    click.echo(f"Coste conocido: ${batch.total_cost_usd:.4f}")


@cli.command()
@click.option("--batch", "batch_dir", required=True, help="Directorio del batch a rebasear")
def rebase(batch_dir: str) -> None:
    """Actualiza metadatos de evaluación compatibles sin repetir tareas válidas."""
    from evaluation.config import EvalConfig
    from evaluation.persistence import load_manifest, rebase_manifest

    root = Path(batch_dir)
    manifest = load_manifest(root)
    config_path = root / "eval_config.json"
    if not config_path.exists():
        raise click.ClickException("El batch no contiene eval_config.json")
    config = EvalConfig.from_json(config_path)
    config.output_path = str(root.parent)
    cases = _cases_for_manifest(config, manifest)
    try:
        updated = rebase_manifest(root, manifest, config, cases)
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Batch rebaseado: {updated['batch_id']}")
    click.echo("Se conservaron los artefactos válidos y las tareas pendientes siguen pendientes.")


@cli.command()
@click.option("--batch", "batch_dir", required=True, help="Directorio del batch")
def status(batch_dir: str) -> None:
    """Muestra el estado persistido y el coste de un batch."""
    from evaluation.persistence import load_manifest, load_persisted_tracker

    root = Path(batch_dir)
    manifest = load_manifest(root)
    state_path = root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    ledger_path = root / "ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {}
    counts: dict[str, int] = {}
    judge_counts: dict[str, int] = {}
    for task in manifest.get("tasks", []):
        task_status = str(task.get("status", "pending"))
        counts[task_status] = counts.get(task_status, 0) + 1
        for judge, judge_status in (task.get("judges") or {}).items():
            key = f"{judge}:{judge_status}"
            judge_counts[key] = judge_counts.get(key, 0) + 1
    tracker = load_persisted_tracker(root)
    click.echo(f"Batch: {manifest.get('batch_id')}")
    click.echo(f"Estado: {state.get('status', 'unknown')}")
    click.echo(f"Tareas: {counts}")
    click.echo(f"Jueces: {judge_counts}")
    click.echo(f"Coste conocido: ${tracker.total_cost_usd:.4f}")
    click.echo(f"Reservado: ${float(ledger.get('reserved_cost_usd', 0.0)):.4f}")
    click.echo(f"Presupuesto restante: ${float(ledger.get('remaining_budget_usd', 0.0)):.4f}")
    if not tracker.cost_complete:
        click.echo(f"Coste incompleto; sin precio: {', '.join(tracker.unknown_cost_models)}")
    click.echo(f"Última actualización: {state.get('updated_at', 'desconocida')}")


@cli.command("repair-broken")
@click.option("--batch", "batch_dir", required=True, help="Directorio del batch")
def repair_broken(batch_dir: str) -> None:
    """Elimina ejecuciones con fallo técnico del aggregate y las deja pendientes."""
    from evaluation.persistence import load_manifest, reset_tasks_for_retry

    root = Path(batch_dir)
    manifest = load_manifest(root)
    aggregate = root / "aggregate.csv"
    if not aggregate.exists():
        raise click.ClickException("Falta aggregate.csv en el batch")
    broken_pairs = set()
    with aggregate.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("execution_status") or "").lower() == "failed":
                broken_pairs.add((row.get("case_id", ""), row.get("model", "")))
    keys = {
        str(task.get("task_key"))
        for task in manifest.get("tasks", [])
        if (task.get("case_id"), task.get("model_id")) in broken_pairs
    }
    if not keys:
        click.echo("No se encontraron ejecuciones técnicas rotas.")
        return
    reset = reset_tasks_for_retry(root, keys)
    click.echo(f"Ejecuciones eliminadas y pendientes de reintento: {len(reset)}")
    for key in sorted(reset):
        click.echo(f"  - {key}")

@cli.command()
@click.option("--results-dir", required=True, help="Directorio con resultados de batch")
@click.option("--output", "-o", default="./eval_report", help="Directorio de salida del reporte")
@click.option("--format", "fmt", type=click.Choice(["json", "html", "both"]), default="both")
def report(results_dir: str, output: str, fmt: str) -> None:
    """Genera un reporte desde resultados guardados."""
    from evaluation.reporting import generate_report

    results_path = Path(results_dir)
    if not results_path.exists():
        logger.error(f"Directorio no encontrado: {results_dir}")
        sys.exit(1)

    click.echo(f"Generando reporte desde {results_dir}...")
    report_path = generate_report(results_path, output, fmt)
    click.echo(f"\nReporte generado: {report_path}")


# ── audit ────────────────────────────────────────────────────────────


@cli.command()
@click.option("--results-dir", required=True, help="Directorio con resultados de batch")
def audit(results_dir: str) -> None:
    """Inspecciona la calidad de las evaluaciones."""
    results_path = Path(results_dir)
    if not results_path.exists():
        logger.error(f"Directorio no encontrado: {results_dir}")
        sys.exit(1)

    summary_path = results_path / "batch_summary.json"
    if not summary_path.exists():
        logger.error(f"No se encontro batch_summary.json en {results_dir}")
        sys.exit(1)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    click.echo(f"\nAuditoria del batch: {summary['batch_id']}")
    click.echo(f"  Total artefactos: {summary['total_artifacts']}")
    click.echo(f"  Coste total: ${summary['total_cost_usd']:.4f}")
    click.echo(f"  Latencia total: {summary['total_latency_ms']/1000:.1f}s")

    issues: list[str] = []
    total = 0
    no_judge = 0
    low_confidence = 0
    high_caution = 0
    empty_output = 0

    for case_dir in sorted(results_path.iterdir()):
        if not case_dir.is_dir():
            continue
        total += 1

        judge_path = case_dir / "judge_metrics.json"
        output_path = case_dir / "output.json"

        if not judge_path.exists():
            no_judge += 1
            issues.append(f"{case_dir.name}: Sin metricas del juez")
            continue

        if output_path.exists():
            output = json.loads(output_path.read_text(encoding="utf-8"))
            if output.get("parse_status") == "failed" or not output.get("executive_summary", "").strip():
                empty_output += 1
                issues.append(f"{case_dir.name}: Output vacio o fallido")

        judge = json.loads(judge_path.read_text(encoding="utf-8"))
        if judge.get("judge_confidence", 0) < 0.5:
            low_confidence += 1
            issues.append(f"{case_dir.name}: Baja confianza del juez ({judge.get('judge_confidence', 0):.2f})")

        if judge.get("responsible_action_quality", {}).get("score", 3) < 2:
            high_caution += 1
            issues.append(f"{case_dir.name}: Posible cautela excesiva")

    click.echo(f"\nEstadisticas:")
    click.echo(f"  Total casos:       {total}")
    click.echo(f"  Sin juez:          {no_judge}")
    click.echo(f"  Baja confianza:    {low_confidence}")
    click.echo(f"  Cautela excesiva:  {high_caution}")
    click.echo(f"  Output vacio:      {empty_output}")

    if issues:
        click.echo(f"\nProblemas encontrados ({len(issues)}):")
        for issue in issues[:20]:
            click.echo(f"  - {issue}")
        if len(issues) > 20:
            click.echo(f"  ... y {len(issues) - 20} mas")
    else:
        click.echo("\nNo se encontraron problemas significativos.")


# ── routing ──────────────────────────────────────────────────────────


@cli.command()
@click.option("--results-dir", required=True, help="Directorio con resultados de batch")
def routing(results_dir: str) -> None:
    """Analiza patrones de routing de agentes por modelo."""
    results_path = Path(results_dir)
    if not results_path.exists():
        logger.error(f"Directorio no encontrado: {results_dir}")
        sys.exit(1)

    # Cargar routing de todos los artifacts
    by_model: dict[str, dict[str, int]] = {}
    total = 0

    for case_dir in sorted(results_path.iterdir()):
        if not case_dir.is_dir():
            continue

        routing_path = case_dir / "routing.json"
        artifact_path = case_dir / "artifact.json"

        if not routing_path.exists():
            continue

        routing = json.loads(routing_path.read_text(encoding="utf-8"))
        model = "unknown"
        if artifact_path.exists():
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            model = artifact.get("model", "unknown")

        agents = routing.get("agents", {})
        if model not in by_model:
            by_model[model] = {}
        for agent_name in agents:
            by_model[model][agent_name] = by_model[model].get(agent_name, 0) + 1
            total += 1

    click.echo(f"\nAnalisis de Routing por Modelo")
    click.echo(f"  Total invocaciones: {total}")
    click.echo()

    for model, agent_counts in sorted(by_model.items()):
        click.echo(f"  Modelo: {model}")
        sorted_agents = sorted(agent_counts.items(), key=lambda x: -x[1])
        for agent_name, count in sorted_agents:
            pct = (count / total * 100) if total > 0 else 0
            click.echo(f"    {agent_name:<30} {count:>4} ({pct:.1f}%)")
        click.echo()


# ── list-packs ─────────────────────────────────────────────────────


@cli.command("list-packs")
def list_packs_cmd() -> None:
    """Muestra los packs de modelos disponibles."""
    from evaluation.model_packs import MODEL_PACKS

    click.echo("Packs de modelos disponibles:\n")
    for name, pack in MODEL_PACKS.items():
        text = pack["text"]
        vision = pack.get("vision")
        click.echo(f"  {name}")
        click.echo(f"    {pack['description']}")
        click.echo(f"    Text:   {text['provider']}/{text['model_id']}")
        if vision:
            click.echo(f"    Vision: {vision['provider']}/{vision['model_id']}")
        else:
            click.echo(f"    Vision: (usa el modelo de texto)")
        click.echo()


# ── list-judges ─────────────────────────────────────────────────────


@cli.command("list-judges")
def list_judges_cmd() -> None:
    """Muestra los jueces disponibles para evaluación."""
    from evaluation.model_packs import DEFAULT_JUDGES
    from libs.costs.pricing import get_model_price

    click.echo("Jueces predefinidos:\n")
    for j in DEFAULT_JUDGES:
        price = get_model_price(j.model)
        if price:
            cost = (price.input_per_million * 1500 + price.output_per_million * 600) / 1_000_000
            click.echo(f"  {j.name:<25} {j.provider:<12} {j.model}")
            click.echo(f"    Input:  ${price.input_per_million:.3f}/M")
            click.echo(f"    Output: ${price.output_per_million:.3f}/M")
            click.echo(f"    Coste/caso: ~${cost:.6f}")
        else:
            click.echo(f"  {j.name:<25} {j.provider:<12} {j.model}")
            click.echo(f"    (sin pricing en catálogo)")
        click.echo()


# ── init-config ──────────────────────────────────────────────────────


@cli.command()
@click.option("--output", "-o", default="eval_config.json", help="Archivo de salida")
def init_config(output: str) -> None:
    """Copy the canonical dissertation benchmark configuration."""
    from evaluation.config import EvalConfig

    canonical_path = Path(__file__).resolve().parent / "configs" / "benchmark_final.json"
    config = EvalConfig.from_json(canonical_path)
    config.to_json(output)
    click.echo(f"Canonical benchmark configuration written to: {output}")


# ── Entry point ──────────────────────────────────────────────────────


if __name__ == "__main__":
    cli()
