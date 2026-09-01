from libs.schemas import AgentInput, AgentPlan


def test_agent_input_defaults():
    ai = AgentInput(query="hola")
    assert ai.language == "es"


def test_agent_plan():
    ap = AgentPlan(steps=["legal", "writer"])
    assert len(ap.steps) == 2
    assert ap.policy.allow_retries is False
    assert ap.policy.writer_search_allowed is False
    assert ap.policy.fast_path.enabled is False
    assert ap.diagnostics.planner_source == "heuristic"
