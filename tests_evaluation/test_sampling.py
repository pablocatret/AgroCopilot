from types import SimpleNamespace

from evaluation.loaders import load_cases
from evaluation.sampling import stratified_case_sample


def test_stratified_case_sample_balances_families():
    cases = load_cases("evaluation/cases/seed")

    sample = stratified_case_sample(cases, 8)
    families = [case.family for case in sample]

    assert len(sample) == 8
    assert families.count("diagnosis") >= 1
    assert families.count("compliance") >= 1
    assert families.count("decision") >= 1
    assert families.count("general") >= 1


def test_stratified_case_sample_balances_larger_sample():
    cases = load_cases("evaluation/cases/seed")

    sample = stratified_case_sample(cases, 20)
    families = [case.family for case in sample]

    assert len(sample) == 20
    assert families.count("diagnosis") >= 4
    assert families.count("compliance") >= 4
    assert families.count("decision") >= 4
    assert families.count("general") >= 4


def test_stratified_case_sample_keeps_remainder_deterministic():
    cases = [
        SimpleNamespace(case_id="a1", family="a"),
        SimpleNamespace(case_id="a2", family="a"),
        SimpleNamespace(case_id="b1", family="b"),
        SimpleNamespace(case_id="b2", family="b"),
        SimpleNamespace(case_id="c1", family="c"),
    ]

    sample = stratified_case_sample(cases, 4)

    assert [case.case_id for case in sample] == ["a1", "b1", "c1", "a2"]
