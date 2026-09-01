import asyncio

from evaluation.baselines import _build_attachments, _configure_evaluation_agent, _extract_execution_report, validate_case_attachments
from evaluation.config import EvalConfig, JudgeConfig
from evaluation.loaders import load_cases
from evaluation.runners import (
    BatchResult,
    _is_saturation_error,
    _required_agents,
    _normalize_agent_names,
    _write_batch_progress,
    run_batch,
    run_single_case,
)
from evaluation.llm_metrics import _build_judge_user_prompt
from libs.context_engineering import summarize_attachments


def test_evaluation_timeout_defaults_are_generous_and_configurable(monkeypatch):
    import evaluation.llm_support as llm_support
    import evaluation.runners as runners

    assert llm_support.EVALUATION_REQUEST_TIMEOUT_SECONDS >= 600
    monkeypatch.setenv("EVALUATION_TASK_TIMEOUT_SECONDS", "1234")
    assert runners._batch_task_timeout_seconds() == 1234
from evaluation.schemas import ExecutionMetrics, NormalizedOutput, RunArtifact


def test_evaluation_attachment_filename_is_sanitized_for_model_prompts():
    case = load_cases("evaluation/cases/seed/att_001_leaf_disease.json")[0]

    attachments = _build_attachments(case)

    assert attachments[0].filename != "olive_peacock_spot.jpg"
    assert attachments[0].filename == "attachment_1.jpg"
    assert attachments[0].metadata["original_filename"] == "olive_peacock_spot.jpg"
    assert "olive_peacock_spot.jpg" not in summarize_attachments(attachments)
    assert "olive_peacock_spot.jpg" not in _build_judge_user_prompt(
        case, NormalizedOutput(message_md="respuesta", parse_status="ok"), {}
    )


def test_judge_prompt_contains_visible_conversation_and_parse_diagnostics():
    case = load_cases("evaluation/cases/seed/seed_001_olivar_plaga.json")[0]
    marker = "marker-visible-at-end-of-conversation"
    output = NormalizedOutput(message_md=("x" * 2500) + marker, parse_status="ok")

    prompt = _build_judge_user_prompt(case, output, {"finish_reason": "stop"})

    assert marker in prompt
    assert '"parse_status": "ok"' in prompt
    assert '"finish_reason": "stop"' in prompt
    assert "visible_text_transport_truncated" in prompt


class _Agent:
    model = "old-model"
    _provider = "openai"
    _client = object()


def test_extract_execution_report_prefers_current_answer_contract_and_keeps_legacy():
    result = {
        "execution": {"legacy": {"final_level": "ok"}},
        "answer": {
            "execution": {"free": {"final_level": "ok"}},
        },
    }

    assert _extract_execution_report(result) == {
        "agents": {"free": {"final_level": "ok"}},
    }


def test_routing_normalizes_writer_aliases_and_requires_vision_for_images():
    assert _normalize_agent_names(["direct_writer", "vision_ocr"]) == ["writer", "vision_ocr"]
    case = load_cases("evaluation/cases/seed/att_001_leaf_disease.json")[0]
    assert _required_agents(case) == {"vision_ocr", "writer"}


def test_attachment_capability_requirements_do_not_add_vision_to_documents_or_csv():
    pdf_case = load_cases("evaluation/cases/seed/att_002_compliance_doc.json")[0]
    csv_case = load_cases("evaluation/cases/seed/att_003_yield_spreadsheet.json")[0]
    assert "vision_ocr" not in _required_agents(pdf_case)
    assert "vision_ocr" not in _required_agents(csv_case)
    assert validate_case_attachments(pdf_case) == []
    assert validate_case_attachments(csv_case) == []


def test_evaluation_agent_override_sets_model_and_provider():
    agent = _Agent()
    _configure_evaluation_agent(
        agent,
        model_id="anthropic/claude-sonnet-4",
        provider="openrouter",
    )
    assert agent.model == "anthropic/claude-sonnet-4"
    assert agent._provider == "openrouter"
    assert agent._client is None


def test_run_batch_requires_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("EVALUATION_ENABLE_LLM", raising=False)
    config = EvalConfig.from_json("eval_config.json")
    case = load_cases("evaluation/cases/seed")[0]

    try:
        asyncio.run(run_batch([case], config))
    except RuntimeError as exc:
        assert "EVALUATION_ENABLE_LLM=1" in str(exc)
    else:
        raise AssertionError("run_batch must reject execution without explicit opt-in")


def test_failed_system_output_is_still_sent_to_quality_judge(monkeypatch):
    case = load_cases("evaluation/cases/seed")[0]
    judge_called = False

    async def fake_run_system(*args, **kwargs):
        return NormalizedOutput(parse_status="failed"), {"error": "upstream"}

    from evaluation.schemas import JudgeMultiMetrics

    async def fake_evaluate(*args, **kwargs):
        nonlocal judge_called
        judge_called = True
        assert kwargs["execution_context"]["error"] == "upstream"
        return JudgeMultiMetrics()

    monkeypatch.setattr("evaluation.runners.run_system", fake_run_system)
    monkeypatch.setattr("evaluation.runners.evaluate_multi_metrics", fake_evaluate)

    artifact = asyncio.run(
        run_single_case(
            case,
            "test-model",
            judges=[JudgeConfig(name="judge", model="test-judge", provider="openrouter")],
        )
    )

    assert judge_called is True
    assert artifact.judge_results["judge"] is not None
    assert artifact.metrics.success is False


def test_visual_case_without_vision_is_not_success_even_with_visible_text(monkeypatch):
    case = load_cases("evaluation/cases/seed/att_001_leaf_disease.json")[0]

    async def fake_run_system(*args, **kwargs):
        return (
            NormalizedOutput(message_md="respuesta visible", parse_status="ok"),
            {"plan": {"steps": ["writer"]}, "agents": {}, "final_level": "ok"},
        )

    monkeypatch.setattr("evaluation.runners.run_system", fake_run_system)

    artifact = asyncio.run(run_single_case(case, "test-model", judges=[]))

    assert artifact.metrics.success is False
    assert artifact.metrics.execution_status == "failed"
    assert "vision_ocr" in artifact.metrics.required_agents_missing


def test_visual_case_with_insufficient_vision_is_partial(monkeypatch):
    case = load_cases("evaluation/cases/seed/att_001_leaf_disease.json")[0]

    async def fake_run_system(*args, **kwargs):
        return (
            NormalizedOutput(message_md="respuesta visible", parse_status="ok"),
            {
                "plan": {"steps": ["vision_ocr", "writer"]},
                "agents": {
                    "vision_ocr": {"final_level": "ok"},
                    "writer": {"final_level": "ok"},
                },
                "visual_evidence": {"status": "insufficient", "used_in_final": False},
                "final_level": "ok",
            },
        )

    monkeypatch.setattr("evaluation.runners.run_system", fake_run_system)

    artifact = asyncio.run(run_single_case(case, "test-model", judges=[]))

    assert artifact.metrics.success is False
    assert artifact.metrics.execution_status == "partial"
    assert artifact.metrics.visual_evidence_status == "insufficient"


def test_batch_summary_exposes_judge_failures(tmp_path):
    config = EvalConfig.from_json("eval_config.json")
    artifact = RunArtifact(
        run_id="run-1",
        case_id="case-1",
        model="test-model",
        input_query="test",
        normalized_output=NormalizedOutput(parse_status="failed"),
        metrics=ExecutionMetrics(success=False),
        judge_results={"judge-a": None},
        errors=["Judge judge-a skipped: system execution failed"],
    )

    batch_dir = BatchResult(
        batch_id="batch-failures",
        config=config,
        artifacts=[artifact],
    ).save(str(tmp_path))

    summary = __import__("json").loads((batch_dir / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["quality_scored_artifacts"] == 0
    assert summary["judge_failure_counts"] == {"judge-a": 1}


def test_batch_progress_is_written_atomically_with_eta(tmp_path):
    _write_batch_progress(
        tmp_path,
        {
            "status": "running",
            "batch_id": "batch-progress",
            "completed": 2,
            "total": 10,
            "elapsed_seconds": 12.5,
            "eta_seconds": 50.0,
            "judge_valid": 5,
            "judge_failures": 1,
        },
    )

    progress = __import__("json").loads(
        (tmp_path / "batch_progress.json").read_text(encoding="utf-8")
    )
    assert progress["status"] == "running"
    assert progress["completed"] == 2
    assert progress["eta_seconds"] == 50.0
    assert not (tmp_path / "batch_progress.json.tmp").exists()


def test_connection_errors_trigger_adaptive_reduction():
    assert _is_saturation_error("Error en llamada LLM: Connection error.")
    assert _is_saturation_error("httpx.ReadTimeout")
