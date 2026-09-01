import pytest

from agents.stac_search import STACSearchInput, StacItemResult, StacSearchAgent, STACToolkit
from libs.schemas import AgentInput


def _item(item_id: str, dt: str) -> StacItemResult:
    return StacItemResult(
        id=item_id,
        collection="sentinel-2-l2a",
        datetime=dt,
        bbox=[0, 0, 1, 1],
        assets={"thumbnail": "data:image/png;base64,abc"},
        properties={},
    )


def test_stac_search_temporal_strategy_can_force_temporal_mode():
    params = STACSearchInput(
        rationale="baseline",
        collections=["sentinel-2-l2a"],
        bbox=[0, 0, 1, 1],
        limit=2,
        temporal_strategy="annual_baseline",
    )

    assert StacSearchAgent._should_plan_temporal_search(
        params, AgentInput(query="Dame dos escenas para comparar")
    )


@pytest.mark.asyncio
async def test_stac_search_respects_requested_limit_in_temporal_mode(monkeypatch):
    agent = object.__new__(StacSearchAgent)
    agent.name = "stac"
    agent.toolkit = type("ToolkitStub", (), {})()

    async def fake_temporal(_params, _windows):
        return [
            _item("scene-1", "2026-04-01T00:00:00Z"),
            _item("scene-2", "2026-03-01T00:00:00Z"),
            _item("scene-3", "2026-02-01T00:00:00Z"),
        ]

    monkeypatch.setattr(agent, "_run_temporal_search_plan", fake_temporal)

    output = await agent._execute_search(
        {
            "rationale": "comparación anual",
            "collections": ["sentinel-2-l2a"],
            "bbox": [0, 0, 1, 1],
            "datetime": "2025-01-01/2026-04-30",
            "limit": 2,
            "temporal_strategy": "annual_baseline",
        },
        AgentInput(query="Necesito dos escenas separadas para comparar"),
    )

    assert len(output.data.items) == 2
    assert output.data.temporal_selection is not None
    assert output.data.temporal_selection.strategy.endswith("annual_baseline")


@pytest.mark.asyncio
async def test_stac_search_keeps_selected_temporal_pair_when_recent_items_would_truncate_it(monkeypatch):
    agent = object.__new__(StacSearchAgent)
    agent.name = "stac"
    agent.toolkit = type("ToolkitStub", (), {})()

    async def fake_temporal(_params, _windows):
        return [
            _item("scene-current", "2026-04-01T00:00:00Z"),
            _item("scene-recent", "2026-03-25T00:00:00Z"),
            _item("scene-baseline", "2025-04-01T00:00:00Z"),
        ]

    monkeypatch.setattr(agent, "_run_temporal_search_plan", fake_temporal)

    output = await agent._execute_search(
        {
            "rationale": "comparacion anual",
            "collections": ["sentinel-2-l2a"],
            "bbox": [0, 0, 1, 1],
            "datetime": "2025-01-01/2026-04-30",
            "limit": 2,
            "temporal_strategy": "annual_baseline",
        },
        AgentInput(query="Necesito una comparacion anual de la parcela"),
    )

    item_ids = [item.id for item in output.data.items]
    assert item_ids == ["scene-current", "scene-baseline"]
    assert output.data.temporal_selection is not None
    assert output.data.temporal_selection.previous_item_id == "scene-baseline"
    assert output.data.temporal_selection.current_item_id == "scene-current"


def test_validate_bbox_raises_on_invalid_length():
    with pytest.raises(ValueError, match="4 elementos"):
        STACToolkit._validate_bbox([0, 0, 1])


def test_validate_bbox_raises_on_five_elements():
    with pytest.raises(ValueError, match="4 elementos"):
        STACToolkit._validate_bbox([0, 0, 1, 1, 1])


def test_validate_bbox_swaps_inverted_coords():
    bbox, warnings = STACToolkit._validate_bbox([1, 1, 0, 0])
    assert bbox == [0, 0, 1, 1]
    assert any("invertido" in w.lower() for w in warnings)


def test_validate_bbox_expands_tiny_bbox():
    bbox, warnings = STACToolkit._validate_bbox([0, 0, 0.001, 0.001])
    assert bbox[2] - bbox[0] >= 0.01
    assert bbox[3] - bbox[1] >= 0.01


@pytest.mark.asyncio
async def test_search_images_returns_empty_on_invalid_bbox():
    toolkit = object.__new__(STACToolkit)
    toolkit.client = object()
    from agents.stac_search import STACSearchInput
    params = STACSearchInput(
        rationale="test",
        collections=["sentinel-2-l2a"],
        bbox=[0, 0, 1],
    )
    result = await toolkit.search_images(params)
    assert result == []
