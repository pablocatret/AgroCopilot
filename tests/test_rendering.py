import numpy as np
import pytest

from agents.stac_search import STACToolkit
from agents.stac_search import STACSearchInput
from agents.base import BaseAgent
from libs.rendering import (
    _apply_quality_mask,
    _evaluate_index_expression,
    _load_previews_as_arrays,
    categorical_statistics,
    render_spectral_index_difference,
    spectral_index_statistics,
)


def test_load_previews_aligns_different_resolution_bands(monkeypatch):
    class FakeImage:
        def __init__(self, value, shape):
            height, width = shape
            self.data = np.full((1, height, width), value, dtype="float32")
            self.mask = np.full((height, width), 255, dtype="uint8")
            self.width = width
            self.height = height

    class FakeReader:
        def __init__(self, url):
            self.url = url

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def part(self, bbox, bounds_crs):
            if self.url == "nir.tif":
                return FakeImage(1, (4, 4))
            return FakeImage(2, (2, 2))

        def preview(self, max_size=1024):
            return self.part([], "epsg:4326")

    monkeypatch.setattr("libs.rendering.Reader", FakeReader)

    arrays, mask = _load_previews_as_arrays(
        {"b8": "nir.tif", "b11": "swir.tif"}, bbox=[1, 2, 3, 4]
    )

    assert arrays["b8"].shape == (4, 4)
    assert arrays["b11"].shape == (4, 4)
    assert mask.shape == (4, 4)
from libs.schemas import AgentInput, BaseAgentOutput


def test_evaluate_index_expression_supports_expected_arithmetic():
    ctx = {
        "b4": np.array([[1.0, 2.0]], dtype="float32"),
        "b8": np.array([[3.0, 6.0]], dtype="float32"),
    }

    result = _evaluate_index_expression("(b8 - b4) / (b8 + b4)", ctx)

    assert result.shape == (1, 2)
    assert np.allclose(result, np.array([[0.5, 0.5]], dtype="float32"))


def test_evaluate_index_expression_rejects_unsafe_nodes():
    ctx = {"b4": np.array([[1.0]], dtype="float32")}

    with pytest.raises(ValueError):
        _evaluate_index_expression("__import__('os').system('id')", ctx)


def test_spectral_index_statistics_uses_valid_mask(monkeypatch):
    def fake_loader(_band_urls, bbox=None):
        arrays = {
            "b4": np.array([[1.0, 1.0], [1.0, 1.0]], dtype="float32"),
            "b8": np.array([[3.0, 3.0], [1.0, 5.0]], dtype="float32"),
        }
        mask = np.array([[255, 255], [0, 255]], dtype="uint8")
        return arrays, mask

    monkeypatch.setattr("libs.rendering._load_previews_as_arrays", fake_loader)

    stats = spectral_index_statistics({"b4": "red", "b8": "nir"}, "(b8 - b4) / (b8 + b4)")

    assert stats["valid_pixels"] == 3
    assert np.isclose(stats["mean"], np.mean([0.5, 0.5, 4 / 6]))


def test_categorical_statistics_counts_worldcover_classes(monkeypatch):
    class FakeImage:
        width = 3
        height = 2
        data = np.array([[[40, 40, 10], [80, 0, 40]]], dtype="uint8")
        mask = np.array([[255, 255, 255], [255, 0, 255]], dtype="uint8")

    class FakeReader:
        def __init__(self, url):
            self.url = url

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def preview(self, max_size=512):
            return FakeImage()

    monkeypatch.setattr("libs.rendering.Reader", FakeReader)

    stats = categorical_statistics("worldcover.tif")

    assert stats["valid_pixels"] == 5
    assert stats["masked_pixels"] == 1
    assert stats["class_stats"][0] == {
        "code": 40,
        "label": "cultivo",
        "pixels": 3,
        "percent": 60.0,
    }
    assert stats["class_stats"][1]["label"] == "arbolado"


def test_quality_mask_excludes_cloud_shadow_water_snow_and_nodata(monkeypatch):
    class DummyImage:
        data = np.array([[[4, 8], [6, 11]]], dtype="uint8")
        mask = np.array([[255, 255], [255, 255]], dtype="uint8")
        width = 2
        height = 2

    class DummyReader:
        def __init__(self, url):
            self.url = url

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def part(self, bbox, bounds_crs):
            return DummyImage()

    monkeypatch.setattr("libs.rendering.Reader", DummyReader)

    base = np.full((2, 2), 255, dtype="uint8")
    masked = _apply_quality_mask(
        base, quality_mask_url="scl.tif", bbox=[1, 2, 3, 4], excluded_classes=(6, 8, 11)
    )

    assert masked.tolist() == [[255, 0], [0, 0]]


@pytest.mark.asyncio
async def test_stac_preview_passes_bbox(monkeypatch):
    toolkit = object.__new__(STACToolkit)
    captured = {}

    class ImmediateLoop:
        async def run_in_executor(self, executor, func):
            return func()

    def fake_render_preview(url, *, colormap, rescale, bbox):
        captured["url"] = url
        captured["colormap"] = colormap
        captured["bbox"] = bbox
        return "preview"

    monkeypatch.setattr("agents.stac_search.rendering.render_preview", fake_render_preview)
    monkeypatch.setattr("agents.stac_search.asyncio.get_running_loop", lambda: ImmediateLoop())

    result = await toolkit._render_preview_async(
        "https://example.com/asset.tif", colormap="magma", bbox=[1, 2, 3, 4]
    )

    assert result == "preview"
    assert captured["bbox"] == [1, 2, 3, 4]


@pytest.mark.asyncio
async def test_stac_index_passes_bbox(monkeypatch):
    toolkit = object.__new__(STACToolkit)
    captured = {}

    class ImmediateLoop:
        async def run_in_executor(self, executor, func):
            return func()

    def fake_render_index(band_urls, expr, colormap, *, bbox, quality_mask_url=None):
        captured["band_urls"] = band_urls
        captured["expr"] = expr
        captured["colormap"] = colormap
        captured["bbox"] = bbox
        captured["quality_mask_url"] = quality_mask_url
        return "index"

    monkeypatch.setattr("agents.stac_search.rendering.render_spectral_index", fake_render_index)
    monkeypatch.setattr("agents.stac_search.asyncio.get_running_loop", lambda: ImmediateLoop())

    result = await toolkit._render_index_async(
        {"b4": "u1", "b8": "u2"}, "(b8-b4)/(b8+b4)", "rdylgn", bbox=[5, 6, 7, 8]
    )

    assert result == "index"
    assert captured["bbox"] == [5, 6, 7, 8]
    assert captured["quality_mask_url"] is None


def test_render_spectral_index_difference_uses_current_minus_previous(monkeypatch):
    captured = {}

    class DummyImageData:
        def __init__(self, data, mask=None):
            captured["delta"] = data[0].copy()
            captured["mask"] = mask.copy() if mask is not None else None
            self.data = data
            self.mask = mask

        def rescale(self, in_range):
            captured["rescale"] = in_range

        def render(self, img_format, colormap):
            return b"png"

    def fake_eval(
        band_urls, index_expression, bbox=None, quality_mask_url=None, excluded_quality_classes=()
    ):
        if band_urls["b4"].startswith("previous"):
            return np.array([[0.2, 0.4]], dtype="float32"), np.array([[255, 255]], dtype="uint8")
        return np.array([[0.5, 0.1]], dtype="float32"), np.array([[255, 0]], dtype="uint8")

    monkeypatch.setattr("libs.rendering.Reader", object())
    monkeypatch.setattr("libs.rendering.cmap", type("Cmap", (), {"get": lambda self, name: None})())
    monkeypatch.setattr("libs.rendering.ImageData", DummyImageData)
    monkeypatch.setattr("libs.rendering._evaluate_masked_index", fake_eval)

    rendered = render_spectral_index_difference(
        {"b4": "previous-red", "b8": "previous-nir"},
        {"b4": "current-red", "b8": "current-nir"},
        "(b8-b4)/(b8+b4)",
    )

    assert rendered.startswith("data:image/png;base64,")
    assert np.allclose(captured["delta"], np.array([[0.3, -0.3]], dtype="float32"))
    assert captured["mask"].tolist() == [[255, 0]]


@pytest.mark.asyncio
async def test_stac_search_uses_query_bbox_for_preview_and_limits_total(monkeypatch):
    toolkit = object.__new__(STACToolkit)
    captured = {}

    class DummyAsset:
        href = "https://example.com/B04.tif"

    class DummyItem:
        id = "scene"
        collection_id = "sentinel-2-l2a"
        datetime = "2026-01-01T00:00:00Z"
        bbox = [0, 0, 10, 10]
        assets = {"B04": DummyAsset()}
        properties = {"eo:cloud_cover": 3}

    class DummySearch:
        def items(self):
            return iter([DummyItem()])

    class DummyClient:
        def search(self, **kwargs):
            captured["search_params"] = kwargs
            return DummySearch()

    async def fake_auto_thumbnail(item, requested_assets, preview_bbox=None):
        captured["preview_bbox"] = preview_bbox
        return {"thumbnail": "data:image/png;base64,abc"}

    toolkit.client = DummyClient()
    monkeypatch.setattr(toolkit, "_auto_thumbnail", fake_auto_thumbnail)

    results = await toolkit.search_images(
        STACSearchInput(
            rationale="test",
            collections=["sentinel-2-l2a"],
            bbox=[1, 2, 3, 4],
            datetime="2026-01-01/2026-01-31",
            limit=1,
        )
    )

    assert len(results) == 1
    assert captured["preview_bbox"] == [1, 2, 3, 4]
    assert captured["search_params"]["max_items"] == 1
    assert captured["search_params"]["sortby"] == [
        {"field": "properties.datetime", "direction": "desc"}
    ]
    assert captured["search_params"]["query"] == {"eo:cloud_cover": {"lt": 20}}


@pytest.mark.asyncio
async def test_stac_ndvi_thumbnail_applies_scl_mask(monkeypatch):
    toolkit = object.__new__(STACToolkit)
    captured = {}

    class Asset:
        def __init__(self, href):
            self.href = href

    class Item:
        bbox = [0, 0, 1, 1]
        assets = {
            "B04": Asset("red.tif"),
            "B08": Asset("nir.tif"),
            "SCL": Asset("scl.tif"),
        }

    async def fake_render_index(band_urls, expr, colormap, bbox=None, quality_mask_url=None):
        captured["render_mask"] = quality_mask_url
        return "thumb"

    async def fake_stats(band_urls, expr, bbox=None, quality_mask_url=None):
        captured["stats_mask"] = quality_mask_url
        return {"mean": 0.4, "valid_pixels": 10, "quality_mask_applied": bool(quality_mask_url)}

    monkeypatch.setattr(toolkit, "_render_index_async", fake_render_index)
    monkeypatch.setattr(toolkit, "_index_stats_async", fake_stats)

    result = await toolkit._auto_thumbnail(Item(), ["B04", "B08", "SCL"], preview_bbox=[1, 2, 3, 4])

    assert result["thumbnail"] == "thumb"
    assert result["quality_mask"] == "SCL"
    assert captured["render_mask"] == "scl.tif"
    assert captured["stats_mask"] == "scl.tif"


@pytest.mark.asyncio
async def test_stac_ndwi_thumbnail_uses_green_and_nir(monkeypatch):
    toolkit = object.__new__(STACToolkit)
    captured = {}

    class Asset:
        def __init__(self, href):
            self.href = href

    class Item:
        bbox = [0, 0, 1, 1]
        assets = {"B03": Asset("green.tif"), "B08": Asset("nir.tif")}

    async def fake_render_index(band_urls, expr, colormap, bbox=None, quality_mask_url=None):
        captured["band_urls"] = band_urls
        captured["expr"] = expr
        captured["colormap"] = colormap
        return "ndwi-thumb"

    async def fake_stats(band_urls, expr, bbox=None, quality_mask_url=None):
        return {"mean": 0.2, "valid_pixels": 12}

    monkeypatch.setattr(toolkit, "_render_index_async", fake_render_index)
    monkeypatch.setattr(toolkit, "_index_stats_async", fake_stats)

    result = await toolkit._auto_thumbnail(Item(), ["B03", "B08"], preview_bbox=[1, 2, 3, 4])

    assert result["thumbnail"] == "ndwi-thumb"
    assert result["product_label"] == "NDWI recortado"
    assert result["index_name"] == "NDWI"
    assert captured["band_urls"] == {"b3": "green.tif", "b8": "nir.tif"}
    assert captured["expr"] == "(b3 - b8) / (b3 + b8)"
    assert captured["colormap"] == "brbg"


@pytest.mark.asyncio
async def test_stac_ndmi_thumbnail_uses_nir_and_swir(monkeypatch):
    toolkit = object.__new__(STACToolkit)
    captured = {}

    class Asset:
        def __init__(self, href):
            self.href = href

    class Item:
        bbox = [0, 0, 1, 1]
        assets = {"B08": Asset("nir.tif"), "B11": Asset("swir.tif")}

    async def fake_render_index(band_urls, expr, colormap, bbox=None, quality_mask_url=None):
        captured["band_urls"] = band_urls
        captured["expr"] = expr
        captured["colormap"] = colormap
        return "ndmi-thumb"

    async def fake_stats(band_urls, expr, bbox=None, quality_mask_url=None):
        return {"mean": 0.3, "valid_pixels": 14}

    monkeypatch.setattr(toolkit, "_render_index_async", fake_render_index)
    monkeypatch.setattr(toolkit, "_index_stats_async", fake_stats)

    result = await toolkit._auto_thumbnail(Item(), ["B08", "B11"], preview_bbox=[1, 2, 3, 4])

    assert result["thumbnail"] == "ndmi-thumb"
    assert result["product_label"] == "NDMI recortado"
    assert result["index_name"] == "NDMI"
    assert captured["band_urls"] == {"b8": "nir.tif", "b11": "swir.tif"}
    assert captured["expr"] == "(b8 - b11) / (b8 + b11)"
    assert captured["colormap"] == "rdylbu"


@pytest.mark.asyncio
async def test_stac_multispectral_thumbnail_returns_derived_index_products(monkeypatch):
    toolkit = object.__new__(STACToolkit)

    class Asset:
        def __init__(self, href):
            self.href = href

    class Item:
        bbox = [0, 0, 1, 1]
        assets = {
            "B03": Asset("green.tif"),
            "B04": Asset("red.tif"),
            "B08": Asset("nir.tif"),
            "B11": Asset("swir.tif"),
        }

    async def fake_render_index(band_urls, expr, colormap, bbox=None, quality_mask_url=None):
        return f"thumb:{expr}"

    async def fake_stats(band_urls, expr, bbox=None, quality_mask_url=None):
        return {"mean": 0.1, "valid_pixels": 10}

    monkeypatch.setattr(toolkit, "_render_index_async", fake_render_index)
    monkeypatch.setattr(toolkit, "_index_stats_async", fake_stats)

    result = await toolkit._auto_thumbnail(
        Item(), ["B03", "B04", "B08", "B11"], preview_bbox=[1, 2, 3, 4]
    )

    assert [item["index_name"] for item in result] == ["NDVI", "NDWI", "NDMI"]


@pytest.mark.asyncio
async def test_stac_radar_thumbnail_returns_vv_stats(monkeypatch):
    toolkit = object.__new__(STACToolkit)

    async def fake_render_preview(url, colormap="viridis", bbox=None):
        return f"preview:{url}:{colormap}:{bbox}"

    async def fake_band_stats(url, bbox=None):
        return {
            "min": -18.0,
            "max": -4.0,
            "mean": -11.2,
            "std": 2.1,
            "valid_pixels": 42,
            "masked_pixels": 3,
        }

    class Asset:
        def __init__(self, href):
            self.href = href

    class Item:
        bbox = [1, 2, 3, 4]
        assets = {"vv": Asset("vv.tif")}

    monkeypatch.setattr(toolkit, "_render_preview_async", fake_render_preview)
    monkeypatch.setattr(toolkit, "_band_stats_async", fake_band_stats)

    result = await toolkit._auto_thumbnail(Item(), ["vv"], preview_bbox=[1, 2, 3, 4])

    assert result["thumbnail"] == "preview:vv.tif:inferno:[1, 2, 3, 4]"
    assert result["product_type"] == "radar"
    assert result["product_label"] == "Radar Sentinel-1 VV recortado"
    assert result["index_name"] == "S1_VV"
    assert result["index_stats"]["mean"] == -11.2


@pytest.mark.asyncio
async def test_stac_radar_thumbnail_prefers_vh_vv_ratio_when_both_polarizations_are_requested(
    monkeypatch,
):
    toolkit = object.__new__(STACToolkit)

    async def fake_render_index(band_urls, expr, colormap, bbox=None, quality_mask_url=None):
        return f"ratio:{expr}:{colormap}:{bbox}"

    async def fake_index_stats(band_urls, expr, bbox=None, quality_mask_url=None):
        return {
            "min": 0.2,
            "max": 0.7,
            "mean": 0.42,
            "std": 0.08,
            "valid_pixels": 80,
            "masked_pixels": 5,
        }

    class Asset:
        def __init__(self, href):
            self.href = href

    class Item:
        bbox = [1, 2, 3, 4]
        assets = {"vv": Asset("vv.tif"), "vh": Asset("vh.tif")}

    monkeypatch.setattr(toolkit, "_render_index_async", fake_render_index)
    monkeypatch.setattr(toolkit, "_index_stats_async", fake_index_stats)

    result = await toolkit._auto_thumbnail(Item(), ["vv", "vh"], preview_bbox=[1, 2, 3, 4])

    assert result["thumbnail"] == "ratio:vh / vv:viridis:[1, 2, 3, 4]"
    assert result["product_label"] == "Radar Sentinel-1 ratio VH/VV recortado"
    assert result["index_name"] == "S1_VH_VV_RATIO"
    assert result["index_stats"]["mean"] == 0.42


@pytest.mark.asyncio
async def test_stac_worldcover_thumbnail_returns_class_stats(monkeypatch):
    toolkit = object.__new__(STACToolkit)

    async def fake_render_preview(url, colormap="viridis", bbox=None):
        return f"preview:{url}:{colormap}:{bbox}"

    async def fake_categorical_stats(url, bbox=None):
        return {
            "valid_pixels": 50,
            "masked_pixels": 5,
            "class_stats": [
                {"code": 40, "label": "cultivo", "pixels": 35, "percent": 70.0},
                {"code": 10, "label": "arbolado", "pixels": 15, "percent": 30.0},
            ],
        }

    class Asset:
        def __init__(self, href):
            self.href = href

    class Item:
        bbox = [1, 2, 3, 4]
        assets = {"map": Asset("worldcover.tif")}

    monkeypatch.setattr(toolkit, "_render_preview_async", fake_render_preview)
    monkeypatch.setattr(toolkit, "_categorical_stats_async", fake_categorical_stats)

    result = await toolkit._auto_thumbnail(Item(), ["map"], preview_bbox=[1, 2, 3, 4])

    assert result["thumbnail"] == "preview:worldcover.tif:tab20:[1, 2, 3, 4]"
    assert result["product_type"] == "landcover"
    assert result["product_label"] == "ESA WorldCover recortado"
    assert result["index_name"] == "ESA_WORLDCOVER"
    assert result["index_stats"]["class_stats"][0]["label"] == "cultivo"


class BrokenAgent(BaseAgent):
    name = "broken"

    async def _run(self, agent_input: AgentInput) -> BaseAgentOutput:
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_base_agent_returns_error_summary_with_exception_type():
    agent = BrokenAgent()

    result = await agent.run(AgentInput(query="demo"))

    assert result.status == "error"
    assert result.errors == ["boom"]
    assert result.summary.startswith("RuntimeError:")
