import pytest

from libs.search_tool import (
    WebSearchInput,
    _norm_url,
    _domain,
    _allowed,
    _dedupe_urls,
    _pick_title,
)


def test_norm_url_converts_http_to_https():
    assert _norm_url("http://example.com") == "https://example.com"


def test_norm_url_keeps_https():
    assert _norm_url("https://example.com") == "https://example.com"


def test_norm_url_strips_whitespace():
    assert _norm_url("  https://example.com  ") == "https://example.com"


def test_norm_url_returns_empty_for_non_string():
    assert _norm_url(None) == ""
    assert _norm_url(123) == ""


def test_domain_extracts_netloc():
    assert _domain("https://www.example.com/path") == "example.com"


def test_domain_strips_www():
    assert _domain("https://www.example.com") == "example.com"


def test_domain_returns_empty_on_error():
    assert _domain("") == ""


def test_allowed_returns_true_when_no_domains():
    assert _allowed("https://example.com", []) is True
    assert _allowed("https://example.com", None) is True


def test_allowed_checks_domain_suffix():
    allowed = ["europa.eu", "globalgap.org"]
    assert _allowed("https://www.europa.eu/policy", allowed) is True
    assert _allowed("https://sub.europa.eu/path", allowed) is True
    assert _allowed("https://example.com/path", allowed) is False


def test_dedupe_urls_removes_duplicates():
    urls = [
        "https://example.com/page",
        "https://example.com/Page",
        "https://other.com/path",
    ]
    result = _dedupe_urls(urls, threshold=92)
    assert len(result) == 2
    assert "https://example.com/page" in result
    assert "https://other.com/path" in result


def test_dedupe_urls_removes_empty():
    urls = ["", "https://example.com", ""]
    result = _dedupe_urls(urls)
    assert len(result) == 1


def test_pick_title_prefers_candidate():
    assert _pick_title("My Title", "Other", "https://url.com") == "My Title"


def test_pick_title_falls_back_to_fetched():
    assert _pick_title("", "Fetched Title", "https://url.com") == "Fetched Title"


def test_pick_title_falls_back_to_url():
    assert _pick_title("", "", "https://url.com") == "https://url.com"


def test_pick_title_returns_default_when_all_empty():
    assert _pick_title("", "", "") == "Web source"


def test_web_search_input_validation():
    data = WebSearchInput(query="test query")
    assert data.query == "test query"
    assert data.max_results == 8
    assert data.include_domains is None


def test_web_search_input_max_results_bounds():
    data = WebSearchInput(query="test", max_results=15)
    assert data.max_results == 15

    with pytest.raises(Exception):
        WebSearchInput(query="test", max_results=0)

    with pytest.raises(Exception):
        WebSearchInput(query="test", max_results=20)
