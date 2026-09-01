from evaluation.metrics import compute_routing_metrics, compute_routing_score
from evaluation.runners import _required_agents
from evaluation.schemas import CaseSpec


def _case(**overrides):
    data = {
        "case_id": "routing-test",
        "family": "attachment_analysis",
        "query": "Analiza el adjunto",
        "expected_route": ["vision_ocr", "free", "writer"],
        "optional_route": ["free"],
    }
    data.update(overrides)
    return CaseSpec.model_validate(data)


def test_optional_route_is_not_a_hard_execution_requirement():
    case = _case()
    assert _required_agents(case) == {"vision_ocr", "writer"}


def test_optional_route_does_not_penalize_valid_alternative_execution():
    case = _case()
    precision, recall, order = compute_routing_metrics(case, ["vision_ocr", "free", "writer"])
    assert (precision, recall, order) == (1.0, 1.0, 1.0)
    assert compute_routing_score(case, ["vision_ocr", "writer"]) == 1.0


def test_missing_required_specialist_remains_a_routing_failure():
    case = _case()
    assert _required_agents(case) - {"writer"} == {"vision_ocr"}
    assert compute_routing_metrics(case, ["writer"]) == (1.0, 0.5, 0.0)
