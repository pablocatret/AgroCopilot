import os

import pytest

from agents.rs_analyst import RSAnalystAgent
from agents.stac_search import STACSearchInput, STACToolkit
from agents.stac_search import StacSearchAgent
from libs.schemas import AgentInput


@pytest.mark.integration
@pytest.mark.asyncio
async def test_planetary_computer_stac_ndvi_contract_live():
    if os.getenv("RUN_STAC_INTEGRATION") != "1":
        pytest.skip("Set RUN_STAC_INTEGRATION=1 to query Planetary Computer.")

    toolkit = STACToolkit()
    if toolkit.client is None:
        pytest.skip("STAC client dependencies are not available.")

    items = await toolkit.search_images(
        STACSearchInput(
            rationale="live contract validation",
            collections=["sentinel-2-l2a"],
            bbox=[-3.75, 40.35, -3.65, 40.45],
            datetime="2025-05-01/2025-06-01",
            max_cloud_cover=20,
            assets_filter=["B04", "B08", "SCL"],
            limit=2,
        )
    )

    assert len(items) >= 1
    item = items[0]
    assert item.collection == "sentinel-2-l2a"
    assert item.cloud_cover is None or item.cloud_cover < 20
    assert item.index_name == "NDVI"
    assert item.index_stats
    assert item.index_stats["valid_pixels"] > 0
    assert item.index_stats["quality_mask_applied"] is True
    assert item.assets.get("thumbnail", "").startswith("data:image/png;base64,")

    stac_results = StacSearchAgent._results_to_schema(object.__new__(StacSearchAgent), items)
    rs_output = await RSAnalystAgent().run(
        AgentInput(query="Seguimiento de vigor en parcela", context={"stac_results": stac_results})
    )

    assert rs_output.status == "ok"
    assert rs_output.data is not None
    assert rs_output.data.insights
    assert rs_output.data.insights[0].stats is not None
    assert "media NDVI" in rs_output.data.insights[0].summary
    if len(stac_results.items) >= 2:
        assert rs_output.data.temporal_changes
        change = rs_output.data.temporal_changes[0]
        assert change.metric == "NDVI"
        assert change.delta_mean is not None
        assert change.label
        assert change.preview_href is None or change.preview_href.startswith(
            "data:image/png;base64,"
        )
