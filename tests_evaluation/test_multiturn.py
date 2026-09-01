"""Tests para evaluación de casos multi-turn."""
from __future__ import annotations

import json
import pytest

from evaluation.schemas import CaseSpec, ConversationTurn, GoldExpectations
from evaluation.loaders import load_case


class TestConversationTurn:
    def test_basic_creation(self):
        turn = ConversationTurn(
            turn=1,
            query="Test query",
            expected_behavior="test_behavior",
        )
        assert turn.turn == 1
        assert turn.query == "Test query"
        assert turn.expected_behavior == "test_behavior"
        assert turn.gold_expectations is not None
        assert turn.context_override == {}

    def test_with_gold_expectations(self):
        gold = GoldExpectations(
            must_contain_concepts=["concept1", "concept2"],
            must_actions=["action1"],
        )
        turn = ConversationTurn(
            turn=1,
            query="Test query",
            gold_expectations=gold,
        )
        assert len(turn.gold_expectations.must_contain_concepts) == 2
        assert len(turn.gold_expectations.must_actions) == 1


class TestCaseSpecMultiTurn:
    def test_multiturn_fields(self):
        case = CaseSpec(
            case_id="test_multiturn",
            family="diagnosis",
            difficulty="medium",
            query="Test query",
            is_multiturn=True,
            turns=[
                ConversationTurn(turn=1, query="Turn 1"),
                ConversationTurn(turn=2, query="Turn 2"),
            ],
            expected_route=["legal", "writer"],
            routing_assertion="legal agent must be invoked",
        )
        assert case.is_multiturn is True
        assert len(case.turns) == 2
        assert case.expected_route == ["legal", "writer"]
        assert case.routing_assertion == "legal agent must be invoked"

    def test_single_turn_defaults(self):
        case = CaseSpec(
            case_id="test_single",
            family="general",
            difficulty="easy",
            query="Test query",
        )
        assert case.is_multiturn is False
        assert case.turns == []
        assert case.expected_route == []
        assert case.routing_assertion == ""


class TestMultiTurnCaseLoading:
    def test_load_multiturn_case(self):
        case = load_case("evaluation/cases/seed/mt_001_diagnosis_followup.json")
        assert case.is_multiturn is True
        assert len(case.turns) == 2
        assert case.turns[0].query == "Tengo manchas oscuras en las hojas de mi olivar en Jaén."
        assert case.turns[1].query == "Las manchas son marrones y se caen las hojas. Es otoño."

    def test_load_all_multiturn_cases(self):
        import os
        cases_dir = "evaluation/cases/seed"
        for filename in os.listdir(cases_dir):
            if filename.startswith("mt_"):
                filepath = os.path.join(cases_dir, filename)
                case = load_case(filepath)
                assert case.is_multiturn is True
                assert len(case.turns) >= 2, f"{filename} should have at least 2 turns"

    def test_multiturn_has_gold_in_turns(self):
        case = load_case("evaluation/cases/seed/mt_001_diagnosis_followup.json")
        for turn in case.turns:
            assert turn.gold_expectations is not None
            assert len(turn.gold_expectations.must_contain_concepts) > 0


class TestMultiTurnRouting:
    def test_routing_assertion_present(self):
        case = load_case("evaluation/cases/seed/mt_001_diagnosis_followup.json")
        assert case.routing_assertion != ""

    def test_expected_route_present(self):
        case = load_case("evaluation/cases/seed/mt_001_diagnosis_followup.json")
        assert len(case.expected_route) > 0
