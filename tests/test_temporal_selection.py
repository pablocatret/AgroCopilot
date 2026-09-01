from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from libs.temporal_selection import (
    PREFERRED_MIN_GAP_DAYS,
    build_temporal_windows,
    detect_recent_override,
    detect_target_months,
    detect_temporal_intent,
    expand_window,
    select_temporal_pair,
)


def _item(
    item_id: str,
    *,
    days_ago: int,
    collection: str = "sentinel-2-l2a",
    index_name: str = "NDVI",
    quality: str = "alta",
):
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return SimpleNamespace(
        id=item_id,
        datetime=dt.isoformat(),
        collection=collection,
        index_name=index_name,
        quality=SimpleNamespace(label=quality),
        index_stats=SimpleNamespace(mean=0.3, valid_pixels=1000),
    )


class TemporalSelectionTests(unittest.TestCase):
    def test_detect_temporal_intent_and_recent_override(self):
        self.assertTrue(detect_temporal_intent("Compara la evolucion del cultivo en abril"))
        self.assertTrue(detect_recent_override("Compara las dos mas recientes"))
        self.assertFalse(detect_recent_override("Haz un diagnostico general"))

    def test_build_temporal_windows_uses_multiple_ranges_when_possible(self):
        now = datetime(2026, 4, 24, tzinfo=timezone.utc)
        windows = build_temporal_windows(
            "2026-01-01T00:00:00+00:00/2026-04-20T00:00:00+00:00",
            now=now,
            preferred_min_gap_days=PREFERRED_MIN_GAP_DAYS,
        )
        self.assertGreaterEqual(len(windows), 2)
        self.assertEqual(len({window.datetime_range for window in windows}), len(windows))

    def test_select_temporal_pair_prefers_meaningful_gap_over_nearly_consecutive_pair(self):
        items = [
            _item("a", days_ago=95),
            _item("b", days_ago=62),
            _item("c", days_ago=21),
            _item("d", days_ago=3),
        ]
        pair = select_temporal_pair(items, preferred_min_gap_days=PREFERRED_MIN_GAP_DAYS)
        self.assertIsNotNone(pair)
        self.assertGreaterEqual(pair.actual_gap_days or 0, 30)
        self.assertEqual(pair.previous.id, "a")
        self.assertEqual(pair.current.id, "d")

    def test_select_temporal_pair_can_reuse_explicit_ids(self):
        items = [
            _item("ref", days_ago=12),
            _item("cur", days_ago=2),
            _item("other", days_ago=1),
        ]
        pair = select_temporal_pair(
            items,
            preferred_min_gap_days=PREFERRED_MIN_GAP_DAYS,
            selected_previous_id="ref",
            selected_current_id="cur",
        )
        self.assertIsNotNone(pair)
        self.assertEqual(pair.previous.id, "ref")
        self.assertEqual(pair.current.id, "cur")

    def test_select_temporal_pair_rejects_mixed_collection(self):
        items = [
            _item("s2-new", days_ago=30, collection="sentinel-2-l2a"),
            _item("ls-old", days_ago=60, collection="landsat-c2-l2"),
        ]
        pair = select_temporal_pair(items)
        self.assertIsNone(pair)

    def test_select_temporal_pair_rejects_mixed_index(self):
        items = [
            _item("ndvi-new", days_ago=30, index_name="NDVI"),
            _item("ndwi-old", days_ago=60, index_name="NDWI"),
        ]
        pair = select_temporal_pair(items)
        self.assertIsNone(pair)

    def test_select_temporal_pair_allows_same_collection_and_index(self):
        items = [
            _item("a", days_ago=30, collection="sentinel-2-l2a", index_name="NDVI"),
            _item("b", days_ago=60, collection="sentinel-2-l2a", index_name="NDVI"),
        ]
        pair = select_temporal_pair(items)
        self.assertIsNotNone(pair)

    def test_build_temporal_windows_with_target_dates(self):
        now = datetime(2026, 7, 1, tzinfo=timezone.utc)
        windows = build_temporal_windows(
            None,
            now=now,
            target_dates=["2020-05-15", "2025-05-15"],
        )
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0].target_month, 5)
        self.assertEqual(windows[1].target_month, 5)
        self.assertIn("2020-04", windows[0].datetime_range)
        self.assertIn("2020-06", windows[0].datetime_range)
        self.assertIn("2025-04", windows[1].datetime_range)
        self.assertIn("2025-06", windows[1].datetime_range)

    def test_build_temporal_windows_target_dates_future_date(self):
        now = datetime(2026, 3, 1, tzinfo=timezone.utc)
        windows = build_temporal_windows(
            None,
            now=now,
            target_dates=["2026-07-15"],
        )
        self.assertEqual(len(windows), 1)
        self.assertIn("2026-03", windows[0].datetime_range)

    def test_expand_window(self):
        window = SimpleNamespace(
            label="test",
            datetime_range="2025-05-01T00:00:00+00:00/2025-05-31T00:00:00+00:00",
            limit=2,
            target_month=5,
        )
        expanded = expand_window(window, extra_days=15)
        self.assertIn("2025-04-16", expanded.datetime_range)
        self.assertIn("2025-06-15", expanded.datetime_range)
        self.assertEqual(expanded.target_month, 5)

    def test_detect_target_months(self):
        result = detect_target_months("compara mayo 2020 con mayo 2025")
        self.assertEqual(result, ["2020-05-15", "2025-05-15"])

    def test_detect_target_months_no_match(self):
        result = detect_target_months("diagnostico general del cultivo")
        self.assertIsNone(result)

    def test_select_temporal_pair_with_target_dates(self):
        now = datetime.now(timezone.utc)
        may_2020 = now.replace(year=2020, month=5, day=15)
        may_2025 = now.replace(year=2025, month=5, day=15)
        jan_2020 = now.replace(year=2020, month=1, day=15)
        items = [
            SimpleNamespace(
                id="may-2020",
                datetime=may_2020.isoformat(),
                collection="sentinel-2-l2a",
                index_name="NDVI",
                quality=SimpleNamespace(label="alta"),
                index_stats=SimpleNamespace(mean=0.45, valid_pixels=1000),
            ),
            SimpleNamespace(
                id="jan-2020",
                datetime=jan_2020.isoformat(),
                collection="sentinel-2-l2a",
                index_name="NDVI",
                quality=SimpleNamespace(label="alta"),
                index_stats=SimpleNamespace(mean=0.35, valid_pixels=1000),
            ),
            SimpleNamespace(
                id="may-2025",
                datetime=may_2025.isoformat(),
                collection="sentinel-2-l2a",
                index_name="NDVI",
                quality=SimpleNamespace(label="alta"),
                index_stats=SimpleNamespace(mean=0.40, valid_pixels=1000),
            ),
        ]
        pair = select_temporal_pair(
            items,
            target_dates=["2020-05-15", "2025-05-15"],
        )
        self.assertIsNotNone(pair)
        self.assertEqual(pair.previous.id, "may-2020")
        self.assertEqual(pair.current.id, "may-2025")


if __name__ == "__main__":
    unittest.main()
