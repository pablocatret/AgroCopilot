"""Persistent state and idempotency helpers for resumable evaluation batches."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from evaluation.reporting import BatchLock
from evaluation.schemas import ExecutionMetrics, JudgeMultiMetrics, NormalizedOutput, RunArtifact
from evaluation.llm_support import LLMCallMetrics, LLMCallTracker


PERSISTENCE_VERSION = "1.0"
REBASE_ALLOWED_CASE_FIELDS = {"expected_route", "optional_route", "routing_assertion"}
REBASE_ALLOWED_CASE_FIELDS = {"expected_route", "optional_route", "routing_assertion"}


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically so an interrupted process cannot leave partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def stable_task_key(case_id: str, model_id: str, run_idx: int) -> str:
    readable_case = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id).strip("_") or "case"
    model_digest = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:12]
    return f"{readable_case}__model-{model_digest}__run-{run_idx}"


def stable_judge_key(task_key: str, judge_name: str) -> str:
    judge = re.sub(r"[^A-Za-z0-9_.-]+", "_", judge_name).strip("_") or "judge"
    return f"{task_key}__judge-{judge}"


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def config_fingerprint(config: Any) -> str:
    """Fingerprint fields that affect evaluation compatibility.

    Budget, output path and concurrency are intentionally excluded: they may
    be changed while resuming an existing batch.
    """
    payload = {
        "models": [
            {
                "name": name,
                "provider": model.provider,
                "model_id": model.model_id,
                "temperature": model.temperature,
                "max_tokens": model.max_tokens,
                "vision_model_id": model.vision_model_id,
                "vision_provider": model.vision_provider,
            }
            for name, model in sorted(config.models.items())
        ],
        "judges": [
            {
                "name": judge.name,
                "model": judge.model,
                "provider": judge.provider,
                "max_tokens": judge.max_tokens,
                "reasoning_enabled": judge.reasoning_enabled,
            }
            for judge in config.judges
        ],
        "runs_per_case": config.runs_per_case,
        "temperature": config.temperature,
    }
    return _hash_payload(payload)


def cases_fingerprint(cases: list[Any]) -> str:
    return _hash_payload([case.model_dump() for case in cases])


def create_manifest(
    *,
    batch_id: str,
    config: Any,
    cases: list[Any],
    tasks: list[tuple[Any, Any, int]],
    case_range: dict[str, int] | None = None,
) -> dict[str, Any]:
    task_entries = []
    for case, model, run_idx in tasks:
        key = stable_task_key(case.case_id, model.model_id, run_idx)
        task_entries.append(
            {
                "task_key": key,
                "case_id": case.case_id,
                "model_id": model.model_id,
                "model_name": model.name,
                "run_idx": run_idx,
                "status": "pending",
                "judges": {judge.name: "pending" for judge in config.judges},
                "updated_at": time.time(),
            }
        )
    return {
        "persistence_version": PERSISTENCE_VERSION,
        "schema_version": "2.0",
        "batch_id": batch_id,
        "created_at": time.time(),
        "config_fingerprint": config_fingerprint(config),
        "cases_fingerprint": cases_fingerprint(cases),
        "case_ids": [case.case_id for case in cases],
        "case_range": case_range,
        "models": [model.model_id for model in config.models.values()],
        "judges": [judge.name for judge in config.judges],
        "runs_per_case": config.runs_per_case,
        "budget_usd": config.budget_usd,
        "expected_tasks": len(tasks),
        "tasks": task_entries,
    }


def validate_manifest(manifest: dict[str, Any], config: Any, cases: list[Any]) -> None:
    expected_config = config_fingerprint(config)
    expected_cases = cases_fingerprint(cases)
    if manifest.get("config_fingerprint") != expected_config:
        raise ValueError("El batch no es compatible: configuración de modelos/jueces distinta")
    if manifest.get("cases_fingerprint") != expected_cases:
        raise ValueError("El batch no es compatible: corpus o rango de casos distinto")
    if manifest.get("runs_per_case") != config.runs_per_case:
        raise ValueError("El batch no es compatible: runs_per_case distinto")


def _case_execution_payload(case: Any) -> dict[str, Any]:
    payload = dict(case if isinstance(case, dict) else case.model_dump())
    for field in REBASE_ALLOWED_CASE_FIELDS:
        payload.pop(field, None)
    # Adding an integrity hash to an existing fixture is safe; the hash itself
    # is validated separately and is not part of the model's input contract.
    for attachment in payload.get("attachments", []) if isinstance(payload.get("attachments"), list) else []:
        if isinstance(attachment, dict):
            attachment.pop("sha256", None)
    return payload


def _validate_attachment_hash_changes(old_case: dict[str, Any], new_case: Any) -> bool:
    old_attachments = old_case.get("attachments") or []
    new_attachments = new_case.model_dump().get("attachments") or []
    old_by_id = {str(item.get("attachment_id")): item for item in old_attachments if isinstance(item, dict)}
    new_by_id = {str(item.get("attachment_id")): item for item in new_attachments if isinstance(item, dict)}
    for attachment_id, old_attachment in old_by_id.items():
        old_hash = str(old_attachment.get("sha256") or "").lower().strip()
        new_hash = str((new_by_id.get(attachment_id) or {}).get("sha256") or "").lower().strip()
        if old_hash and old_hash != new_hash:
            return False
    return True


def rebase_manifest(batch_dir: str | Path, manifest: dict[str, Any], config: Any, cases: list[Any]) -> dict[str, Any]:
    """Rebase evaluation-only case metadata without changing executions.

    This is intentionally narrower than changing a fingerprint manually: query,
    attachments, gold, models, judges and runs must remain identical. Only the
    routing metadata fields listed above may change.
    """
    root = Path(batch_dir).resolve()
    with BatchLock(root):
        current = load_manifest(root)
        validate_manifest(
            {**current, "cases_fingerprint": cases_fingerprint(cases)},
            config,
            cases,
        )
        snapshot_path = root / "case_specs.json"
        if not snapshot_path.exists():
            raise ValueError("No se puede rebasear: falta case_specs.json")
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        stored_cases = snapshot.get("cases") if isinstance(snapshot, dict) else None
        if not isinstance(stored_cases, list):
            raise ValueError("No se puede rebasear: formato de case_specs.json no reconocido")
        old_by_id = {str(item.get("case_id")): item for item in stored_cases if isinstance(item, dict)}
        current_by_id = {str(case.case_id): case for case in cases}
        if set(old_by_id) != set(current_by_id):
            raise ValueError("No se puede rebasear: han cambiado los casos del batch")
        incompatible: list[str] = []
        for case_id, case in current_by_id.items():
            old_case = old_by_id[case_id]
            if not _validate_attachment_hash_changes(old_case, case):
                incompatible.append(case_id)
                continue
            if _case_execution_payload(case) != _case_execution_payload(old_case):
                incompatible.append(case_id)
        if incompatible:
            raise ValueError(
                "No se puede rebasear: cambiaron campos de ejecución en "
                + ", ".join(sorted(incompatible)[:8])
            )

        updated = dict(current)
        updated["cases_fingerprint"] = cases_fingerprint(cases)
        updated["rebased_at"] = time.time()
        updated["rebase_allowed_fields"] = sorted(REBASE_ALLOWED_CASE_FIELDS)
        atomic_write_json(root / "manifest.json", updated)
        atomic_write_json(
            snapshot_path,
            {"schema_version": snapshot.get("schema_version"), "cases": [case.model_dump() for case in cases]},
        )

        # Update only the descriptive case snapshots; execution outputs and costs stay untouched.
        for task in updated.get("tasks", []):
            task_key = str(task.get("task_key") or "")
            artifact_dir = root / str(task.get("artifact_dir") or task_key)
            artifact_dir = artifact_dir.resolve()
            try:
                artifact_dir.relative_to(root)
            except ValueError:
                continue
            if not artifact_dir.is_dir():
                continue
            case = current_by_id.get(str(task.get("case_id")))
            if case is not None:
                atomic_write_json(artifact_dir / "case_spec.json", case.model_dump())
        for name in ("aggregate.csv", "batch_summary.json", "report.html", "partial_summary.json", "partial_report.html"):
            path = root / name
            if path.exists():
                path.unlink()
        return updated


def _legacy_case_execution_payload(case: Any) -> dict[str, Any]:
    payload = dict(case.model_dump())
    for field in REBASE_ALLOWED_CASE_FIELDS:
        payload.pop(field, None)
    return payload


def _legacy_rebase_manifest(batch_dir: str | Path, manifest: dict[str, Any], config: Any, cases: list[Any]) -> dict[str, Any]:
    """Rebase evaluation-only case metadata without changing executions.

    This is intentionally narrower than changing a fingerprint manually: query,
    attachments, gold, models, judges and runs must remain identical. Only the
    routing metadata fields listed above may change.
    """
    root = Path(batch_dir).resolve()
    with BatchLock(root):
        current = load_manifest(root)
        validate_manifest(
            {**current, "cases_fingerprint": cases_fingerprint(cases)},
            config,
            cases,
        )
        snapshot_path = root / "case_specs.json"
        if not snapshot_path.exists():
            raise ValueError("No se puede rebasear: falta case_specs.json")
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        stored_cases = snapshot.get("cases") if isinstance(snapshot, dict) else None
        if not isinstance(stored_cases, list):
            raise ValueError("No se puede rebasear: formato de case_specs.json no reconocido")
        old_by_id = {str(item.get("case_id")): item for item in stored_cases if isinstance(item, dict)}
        current_by_id = {str(case.case_id): case for case in cases}
        if set(old_by_id) != set(current_by_id):
            raise ValueError("No se puede rebasear: han cambiado los casos del batch")
        incompatible: list[str] = []
        for case_id, case in current_by_id.items():
            old_case = old_by_id[case_id]
            if _case_execution_payload(case) != _case_execution_payload(type("Snapshot", (), {"model_dump": lambda self, item=old_case: item})()):
                incompatible.append(case_id)
        if incompatible:
            raise ValueError(
                "No se puede rebasear: cambiaron campos de ejecución en "
                + ", ".join(sorted(incompatible)[:8])
            )

        updated = dict(current)
        updated["cases_fingerprint"] = cases_fingerprint(cases)
        updated["rebased_at"] = time.time()
        updated["rebase_allowed_fields"] = sorted(REBASE_ALLOWED_CASE_FIELDS)
        atomic_write_json(root / "manifest.json", updated)
        atomic_write_json(
            snapshot_path,
            {"schema_version": snapshot.get("schema_version"), "cases": [case.model_dump() for case in cases]},
        )

        # Update only the descriptive case snapshots; execution outputs and costs stay untouched.
        for task in updated.get("tasks", []):
            task_key = str(task.get("task_key") or "")
            artifact_dir = root / str(task.get("artifact_dir") or task_key)
            artifact_dir = artifact_dir.resolve()
            if not str(artifact_dir).startswith(str(root)) or not artifact_dir.is_dir():
                continue
            case = current_by_id.get(str(task.get("case_id")))
            if case is not None:
                atomic_write_json(artifact_dir / "case_spec.json", case.model_dump())
        for name in ("aggregate.csv", "batch_summary.json", "report.html", "partial_summary.json", "partial_report.html"):
            path = root / name
            if path.exists():
                path.unlink()
        return updated


def task_record_path(batch_dir: Path, task_key: str) -> Path:
    return batch_dir / "tasks" / f"{task_key}.json"


def write_task_record(batch_dir: Path, task_key: str, record: dict[str, Any]) -> None:
    atomic_write_json(task_record_path(batch_dir, task_key), record)


def load_manifest(batch_dir: str | Path) -> dict[str, Any]:
    path = Path(batch_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"No existe manifest.json en {batch_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def update_manifest_task(batch_dir: Path, task_key: str, **changes: Any) -> dict[str, Any]:
    """Update one task under the batch lock and return the updated record."""
    manifest_path = batch_dir / "manifest.json"
    with BatchLock(batch_dir):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for task in manifest.get("tasks", []):
            if task.get("task_key") == task_key:
                task.update(changes)
                task["updated_at"] = time.time()
                write_task_record(batch_dir, task_key, task)
                atomic_write_json(manifest_path, manifest)
                return task
    raise KeyError(f"Tarea no encontrada: {task_key}")


def reset_tasks_for_retry(batch_dir: str | Path, task_keys: set[str]) -> list[str]:
    """Remove broken task artifacts and return those tasks to ``pending``."""
    root = Path(batch_dir).resolve()
    manifest_path = root / "manifest.json"
    reset: list[str] = []
    with BatchLock(root):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for task in manifest.get("tasks", []):
            key = str(task.get("task_key") or "")
            if key not in task_keys:
                continue
            artifact_dir = (root / str(task.get("artifact_dir") or key)).resolve()
            if root not in artifact_dir.parents:
                raise ValueError(f"Artefacto fuera del batch: {artifact_dir}")
            if artifact_dir.exists():
                shutil.rmtree(artifact_dir)
            task_record = root / "tasks" / f"{key}.json"
            if task_record.exists():
                task_record.unlink()
            task.update(
                status="pending",
                judges={name: "pending" for name in (task.get("judges") or {})},
                errors=[],
                cost_usd=0.0,
                updated_at=time.time(),
            )
            reset.append(key)
        atomic_write_json(manifest_path, manifest)
        completed = sum(1 for task in manifest.get("tasks", []) if task.get("status") == "completed")
        atomic_write_json(root / "state.json", {
            "batch_id": manifest.get("batch_id"),
            "status": "running",
            "completed": completed,
            "expected_tasks": manifest.get("expected_tasks", len(manifest.get("tasks", []))),
            "updated_at": time.time(),
        })
    for name in (
        "aggregate.csv", "batch_summary.json", "report.html",
        "partial_summary.json", "partial_report.html",
    ):
        path = root / name
        if path.exists():
            path.unlink()
    return reset


def load_persisted_artifacts(batch_dir: str | Path) -> dict[str, RunArtifact]:
    """Load artifacts written by the resumable writer, keyed by task key."""
    root = Path(batch_dir)
    result: dict[str, RunArtifact] = {}
    tasks_dir = root / "tasks"
    if not tasks_dir.exists():
        return result
    for record_path in sorted(tasks_dir.glob("*.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        task_key = str(record.get("task_key") or record_path.stem)
        artifact_dir = root / str(record.get("artifact_dir") or task_key)
        meta_path = artifact_dir / "artifact.json"
        output_path = artifact_dir / "output.json"
        metrics_path = artifact_dir / "metrics.json"
        if not (meta_path.exists() and output_path.exists() and metrics_path.exists()):
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        output = NormalizedOutput.model_validate(json.loads(output_path.read_text(encoding="utf-8")))
        metrics = ExecutionMetrics(**json.loads(metrics_path.read_text(encoding="utf-8")))
        judges: dict[str, JudgeMultiMetrics | None] = {}
        judges_dir = artifact_dir / "judges"
        if judges_dir.exists():
            for judge_dir in judges_dir.iterdir():
                if not judge_dir.is_dir():
                    continue
                judge_path = judge_dir / "judge_metrics.json"
                if not judge_path.exists():
                    continue
                raw = json.loads(judge_path.read_text(encoding="utf-8"))
                judges[judge_dir.name] = None if raw.get("error") == "judge_failed" else JudgeMultiMetrics.model_validate(raw)
        routing_path = artifact_dir / "routing.json"
        routing = json.loads(routing_path.read_text(encoding="utf-8")) if routing_path.exists() else None
        result[task_key] = RunArtifact(
            run_id=str(meta.get("run_id") or task_key),
            case_id=str(meta.get("case_id") or record.get("case_id") or ""),
            model=str(meta.get("model") or record.get("model_id") or ""),
            input_query=str(meta.get("input_query") or ""),
            normalized_output=output,
            metrics=metrics,
            judge_results=judges,
            agent_routing=routing,
            errors=list(meta.get("errors") or []),
            timestamp_iso=str(meta.get("timestamp_iso") or ""),
            family=str(meta.get("family") or ""),
            difficulty=str(meta.get("difficulty") or ""),
            run_idx=int(meta.get("run_idx") or record.get("run_idx") or 0),
        )
    return result


def load_persisted_tracker(batch_dir: str | Path) -> LLMCallTracker:
    path = Path(batch_dir) / "llm_calls.json"
    tracker = LLMCallTracker()
    if not path.exists():
        return tracker
    raw_calls = json.loads(path.read_text(encoding="utf-8"))
    for raw in raw_calls if isinstance(raw_calls, list) else []:
        allowed = {field for field in LLMCallMetrics.__dataclass_fields__}
        tracker.calls.append(LLMCallMetrics(**{key: value for key, value in raw.items() if key in allowed}))
    return tracker
