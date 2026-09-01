import json

import pytest

from evaluation.cli import _parse_case_range
from evaluation.config import EvalConfig, JudgeConfig, ModelConfig
from evaluation.persistence import (
    cases_fingerprint,
    config_fingerprint,
    create_manifest,
    load_persisted_artifacts,
    stable_judge_key,
    stable_task_key,
    rebase_manifest,
)
from evaluation.schemas import CaseSpec, ExecutionMetrics, NormalizedOutput, RunArtifact
from evaluation.runners import (
    BatchResult,
    _load_execution_checkpoint,
    _write_execution_checkpoint,
)


def _config() -> EvalConfig:
    return EvalConfig(
        models={"m": ModelConfig(name="m", provider="openrouter", model_id="test/model")},
        judges=[JudgeConfig(name="judge-a", model="test/judge", provider="openrouter")],
        budget_usd=14.0,
    )


def _case(case_id: str = "seed_001") -> CaseSpec:
    return CaseSpec(case_id=case_id, family="general", query="test")


def test_case_range_is_inclusive_and_one_based():
    indexes, metadata = _parse_case_range("2:4", 6)
    assert indexes == [1, 2, 3]
    assert metadata == {"start": 2, "end": 4}


def test_case_range_rejects_invalid_values():
    with pytest.raises(Exception):
        _parse_case_range("4:2", 6)


def test_manifest_has_stable_task_and_judge_keys():
    config = _config()
    case = _case()
    manifest = create_manifest(
        batch_id="batch-test",
        config=config,
        cases=[case],
        tasks=[(case, config.models["m"], 0)],
        case_range={"start": 1, "end": 1},
    )
    task_key = stable_task_key(case.case_id, "test/model", 0)
    assert manifest["tasks"][0]["task_key"] == task_key
    assert stable_judge_key(task_key, "judge-a").endswith("judge-judge-a")
    assert manifest["config_fingerprint"] == config_fingerprint(config)
    assert manifest["cases_fingerprint"] == cases_fingerprint([case])


def test_saved_artifact_can_be_loaded_by_task_key(tmp_path):
    config = _config()
    case = _case()
    model = config.models["m"]
    task_key = stable_task_key(case.case_id, model.model_id, 0)
    artifact = RunArtifact(
        run_id="run-1",
        case_id=case.case_id,
        model=model.model_id,
        input_query=case.query,
        normalized_output=NormalizedOutput(message_md="respuesta", parse_status="ok"),
        metrics=ExecutionMetrics(success=True, execution_status="ok"),
        judge_results={},
        run_idx=0,
    )
    batch_dir = BatchResult(
        batch_id="batch-test",
        config=config,
        cases=[case],
        artifacts=[artifact],
    ).save(str(tmp_path))
    task_dir = batch_dir / task_key
    (batch_dir / "tasks").mkdir(exist_ok=True)
    (batch_dir / "tasks" / f"{task_key}.json").write_text(
        json.dumps({"task_key": task_key, "artifact_dir": task_key}),
        encoding="utf-8",
    )
    loaded = load_persisted_artifacts(batch_dir)
    assert task_dir.exists()
    assert loaded[task_key].normalized_output.message_md == "respuesta"


def test_execution_checkpoint_round_trip_preserves_output_and_state(tmp_path):
    artifact = RunArtifact(
        run_id="run-1",
        case_id="mt-1",
        model="test/model",
        input_query="test",
        normalized_output=NormalizedOutput(message_md="turno final", parse_status="ok"),
        metrics=ExecutionMetrics(success=True, execution_status="ok"),
        run_idx=0,
    )
    _write_execution_checkpoint(
        tmp_path,
        artifact=artifact,
        execution_report={"agents": {"writer": {"status": "ok"}}},
    )
    restored = _load_execution_checkpoint(tmp_path)
    assert restored is not None
    assert restored.normalized_output.message_md == "turno final"
    assert restored.agent_routing["agents"]["writer"]["status"] == "ok"


def test_rebase_allows_only_routing_metadata_changes(tmp_path):
    config = _config()
    old_case = _case()
    new_case = CaseSpec.model_validate({**old_case.model_dump(), "optional_route": ["free"]})
    manifest = create_manifest(
        batch_id="batch-rebase",
        config=config,
        cases=[old_case],
        tasks=[(old_case, config.models["m"], 0)],
    )
    batch_dir = tmp_path / "batch-rebase"
    batch_dir.mkdir()
    (batch_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (batch_dir / "case_specs.json").write_text(
        json.dumps({"schema_version": "test", "cases": [old_case.model_dump()]}),
        encoding="utf-8",
    )
    updated = rebase_manifest(batch_dir, manifest, config, [new_case])
    assert updated["cases_fingerprint"] == cases_fingerprint([new_case])


def test_rebase_rejects_query_changes(tmp_path):
    config = _config()
    old_case = _case()
    new_case = CaseSpec.model_validate({**old_case.model_dump(), "query": "different"})
    manifest = create_manifest(
        batch_id="batch-rebase",
        config=config,
        cases=[old_case],
        tasks=[(old_case, config.models["m"], 0)],
    )
    batch_dir = tmp_path / "batch-rebase"
    batch_dir.mkdir()
    (batch_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (batch_dir / "case_specs.json").write_text(
        json.dumps({"schema_version": "test", "cases": [old_case.model_dump()]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cambiaron campos de ejecución"):
        rebase_manifest(batch_dir, manifest, config, [new_case])
