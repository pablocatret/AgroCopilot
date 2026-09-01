"""Tests para métricas de routing de agentes."""
from __future__ import annotations

import pytest

from evaluation.metrics import compute_routing_score, check_routing_assertion
from evaluation.schemas import CaseSpec, CaseContext, GoldExpectations


def _make_case(
    expected_route: list[str] | None = None,
    routing_assertion: str = "",
) -> CaseSpec:
    return CaseSpec(
        case_id="test_routing",
        family="general",
        difficulty="easy",
        query="test query",
        expected_route=expected_route or [],
        routing_assertion=routing_assertion,
    )


class TestComputeRoutingScore:
    def test_no_expected_route_returns_1(self):
        case = _make_case(expected_route=[])
        assert compute_routing_score(case, ["legal", "writer"]) == 1.0

    def test_perfect_match_returns_1(self):
        case = _make_case(expected_route=["legal", "writer"])
        assert compute_routing_score(case, ["legal", "writer"]) == 1.0

    def test_partial_match(self):
        case = _make_case(expected_route=["legal", "stac", "writer"])
        score = compute_routing_score(case, ["legal", "writer"])
        assert 0.0 < score < 1.0
        assert score == pytest.approx(2 / 3)

    def test_no_match_returns_0(self):
        case = _make_case(expected_route=["stac", "rs_analyst"])
        assert compute_routing_score(case, ["legal", "writer"]) == 0.0

    def test_empty_invoked_returns_0(self):
        case = _make_case(expected_route=["legal"])
        assert compute_routing_score(case, []) == 0.0

    def test_extra_agents_are_penalized(self):
        case = _make_case(expected_route=["legal"])
        assert compute_routing_score(case, ["legal", "stac", "writer"]) == pytest.approx(1 / 3)


class TestCheckRoutingAssertion:
    def test_no_assertion_returns_true(self):
        case = _make_case(routing_assertion="")
        assert check_routing_assertion(case, ["legal"]) is True

    def test_must_be_invoked_success(self):
        case = _make_case(routing_assertion="legal agent must be invoked")
        assert check_routing_assertion(case, ["legal", "writer"]) is True

    def test_must_be_invoked_failure(self):
        case = _make_case(routing_assertion="legal agent must be invoked")
        assert check_routing_assertion(case, ["stac", "writer"]) is False

    def test_no_specialized_agents_success(self):
        case = _make_case(routing_assertion="no specialized agents needed for simple definitional query")
        assert check_routing_assertion(case, ["free", "writer"]) is True

    def test_no_specialized_agents_failure(self):
        case = _make_case(routing_assertion="no specialized agents needed for simple definitional query")
        assert check_routing_assertion(case, ["stac", "writer"]) is False

    def test_must_not_hallucinate(self):
        case = _make_case(routing_assertion="system must not hallucinate treatment for non-existent disease")
        assert check_routing_assertion(case, ["free", "writer"]) is True

    def test_case_insensitive(self):
        case = _make_case(routing_assertion="LEGAL agent must be invoked")
        assert check_routing_assertion(case, ["legal", "writer"]) is True

    def test_stac_must_be_invoked(self):
        case = _make_case(routing_assertion="stac agent must be invoked for satellite imagery queries")
        assert check_routing_assertion(case, ["stac", "rs_analyst", "writer"]) is True
        assert check_routing_assertion(case, ["free", "writer"]) is False
