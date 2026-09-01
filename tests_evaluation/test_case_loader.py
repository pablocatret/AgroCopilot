from evaluation.loaders import load_cases


def test_load_seed_cases():
    cases = load_cases("evaluation/cases/seed")
    assert len(cases) >= 25
    assert len({case.case_id for case in cases}) == len(cases)
    families = {}
    for case in cases:
        families[case.family] = families.get(case.family, 0) + 1
    assert families["diagnosis"] >= 5
    assert families["compliance"] >= 4
    assert families["decision"] >= 5
    assert families["general"] >= 5
    for case in cases:
        gold = case.gold_expectations
        has_gold = (
            gold.must_contain_concepts
            or gold.must_mention_facts
            or gold.must_actions
            or gold.forbidden_claims
        )
        # Multi-turn cases have gold expectations per turn, not at top level
        if not has_gold and case.is_multiturn and case.turns:
            has_gold = any(
                turn.gold_expectations.must_contain_concepts
                or turn.gold_expectations.must_mention_facts
                or turn.gold_expectations.must_actions
                or turn.gold_expectations.forbidden_claims
                for turn in case.turns
            )
        assert has_gold, f"{case.case_id} must expose at least one evaluable gold signal"
