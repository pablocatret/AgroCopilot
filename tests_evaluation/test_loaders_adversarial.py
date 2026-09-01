import json

import pytest

from evaluation.loaders import EvaluationLoadError, filter_cases, load_case, load_cases


def test_load_case_rejects_missing_and_invalid_json(tmp_path):
    with pytest.raises(EvaluationLoadError):
        load_case(tmp_path / "missing.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{oops", encoding="utf-8")
    with pytest.raises(EvaluationLoadError):
        load_case(broken)


def test_load_case_rejects_invalid_schema(tmp_path):
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"case_id": "c1"}), encoding="utf-8")
    with pytest.raises(EvaluationLoadError):
        load_case(invalid)


def test_load_cases_supports_directory_and_ignores_non_json(tmp_path):
    payload = {
        "case_id": "c1",
        "family": "decision",
        "difficulty": "easy",
        "query": "demo",
    }
    (tmp_path / "a.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "note.txt").write_text("ignore", encoding="utf-8")
    cases = load_cases(tmp_path)
    assert len(cases) == 1
    assert cases[0].case_id == "c1"


def test_load_cases_supports_comma_separated_paths(tmp_path):
    payload_a = {
        "case_id": "c1",
        "family": "decision",
        "difficulty": "easy",
        "query": "demo a",
    }
    payload_b = {
        "case_id": "c2",
        "family": "diagnosis",
        "difficulty": "easy",
        "query": "demo b",
    }
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(payload_a), encoding="utf-8")
    b.write_text(json.dumps(payload_b), encoding="utf-8")
    cases = load_cases(f"{a},{b}")
    assert [case.case_id for case in cases] == ["c1", "c2"]


def test_load_cases_empty_directory_is_clean_failure(tmp_path):
    with pytest.raises(EvaluationLoadError):
        load_cases(tmp_path)


def test_filter_cases_rejects_unknown_family():
    cases = load_cases("evaluation/cases/seed")
    with pytest.raises(EvaluationLoadError):
        filter_cases(cases, family="unknown")


def test_load_adversarial_cases():
    """Verify adversarial cases are loadable."""
    adv_dir = "evaluation/cases/adversarial"
    cases = load_cases(adv_dir)
    assert len(cases) >= 3
    for case in cases:
        assert case.case_id.startswith("adv_")


def test_adversarial_missing_query_expects_clarification():
    case = load_case("evaluation/cases/adversarial/adv_001_missing_query.json")
    assert case.gold_expectations.expects_clarification is True
    assert case.query == ""


def test_adversarial_ambiguous_query_is_loadable():
    case = load_case("evaluation/cases/adversarial/adv_002_ambiguous_query.json")
    assert case.gold_expectations.expects_clarification is True
    assert len(case.gold_expectations.must_contain_concepts) > 0
