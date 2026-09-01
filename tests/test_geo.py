import pytest

from libs.geo import _parse_viewbox


def test_parse_viewbox_returns_none_for_empty():
    assert _parse_viewbox(None) is None
    assert _parse_viewbox("") is None


def test_parse_viewbox_parses_four_floats():
    result = _parse_viewbox("-10.0,36.0,5.0,44.0")
    assert result == [-10.0, 36.0, 5.0, 44.0]


def test_parse_viewbox_returns_none_for_wrong_count():
    assert _parse_viewbox("1.0,2.0,3.0") is None
    assert _parse_viewbox("1.0,2.0,3.0,4.0,5.0") is None


def test_parse_viewbox_returns_none_for_invalid():
    assert _parse_viewbox("not,numbers,at,all") is None
