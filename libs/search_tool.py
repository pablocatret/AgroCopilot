from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field
from rapidfuzz import fuzz
import trafilatura

from backend.deps import settings
from libs.costs.tracker import record_web_search_call
from libs.tools import Tool


class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query.")
    max_results: int = Field(8, ge=1, le=15, description="Maximum number of results.")
    include_domains: Optional[List[str]] = Field(
        default=None, description="Allowed domains (optional)."
    )


def _norm_url(u: str) -> str:
    if not isinstance(u, str):
        return ""
    u = u.strip()
    return re.sub(r"^http://", "https://", u, flags=re.I)


def _domain(u: str) -> str:
    try:
        netloc = urlparse(u).netloc.lower()
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return ""


def _allowed(url: str, allow_domains: List[str]) -> bool:
    if not allow_domains:
        return True
    d = _domain(url)
    return any(d.endswith(ad) for ad in allow_domains)


def _dedupe_urls(urls: List[str], threshold: int = 92) -> List[str]:
    out: List[str] = []
    for u in urls:
        u = _norm_url(u)
        if not u:
            continue
        if not any(fuzz.ratio(u, v) >= threshold for v in out):
            out.append(u)
    return out


async def _fetch_extract(
    url: str, client: httpx.AsyncClient, timeout: float = 12.0
) -> Dict[str, str]:
    try:
        r = await client.get(
            url,
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (WebSearchTool)"},
        )
        r.raise_for_status()
        html = r.text
    except Exception:
        return {"title": "", "text": ""}

    loop = asyncio.get_running_loop()
    text = await loop.run_in_executor(
        None, lambda: trafilatura.extract(html, include_comments=False, favor_recall=True) or ""
    )
    title = ""
    try:
        title_match = re.search(r"<title>(.*?)</title>", html, flags=re.I | re.S)
        title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else ""
    except Exception:
        title = ""
    return {"title": title, "text": text or ""}


def _pick_title(candidate: str, fetched_title: str, fallback_url: str) -> str:
    for s in (candidate, fetched_title, fallback_url):
        if isinstance(s, str) and s.strip():
            return s.strip()
    return "Web source"


class BaseSearchProvider:
    async def search(
        self,
        query: str,
        include_domains: Optional[List[str]] = None,
        max_results: int = 8,
    ) -> List[Dict[str, str]]:
        raise NotImplementedError


class SerperProvider(BaseSearchProvider):
    URL = "https://google.serper.dev/search"

    async def search(
        self,
        query: str,
        include_domains: Optional[List[str]] = None,
        max_results: int = 8,
    ) -> List[Dict[str, str]]:
        key = settings.SEARCH_API_KEY
        if not key:
            raise RuntimeError("SEARCH_API_KEY required for Serper.")
        payload: Dict[str, Any] = {"q": query, "num": max_results}
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(
                self.URL,
                headers={"X-API-KEY": key, "Content-Type": "application/json"},
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
        organic = data.get("organic", []) or []
        out: List[Dict[str, str]] = []
        for item in organic:
            out.append(
                {
                    "url": item.get("link") or "",
                    "title": item.get("title") or "",
                    "snippet": item.get("snippet") or item.get("description") or "",
                }
            )
        return out[:max_results]


class TavilyProvider(BaseSearchProvider):
    URL = "https://api.tavily.com/search"

    async def search(
        self,
        query: str,
        include_domains: Optional[List[str]] = None,
        max_results: int = 8,
    ) -> List[Dict[str, str]]:
        key = settings.SEARCH_API_KEY
        if not key:
            raise RuntimeError("SEARCH_API_KEY required for Tavily.")
        payload: Dict[str, Any] = {
            "api_key": key,
            "query": query,
            "search_depth": "advanced",
            "max_results": max_results,
        }
        if include_domains:
            payload["include_domains"] = include_domains
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(self.URL, json=payload)
            r.raise_for_status()
            data = r.json()
        results = data.get("results", []) or []
        out: List[Dict[str, str]] = []
        for item in results:
            out.append(
                {
                    "url": item.get("url") or "",
                    "title": item.get("title") or "",
                    "snippet": item.get("content") or item.get("snippet") or "",
                }
            )
        return out[:max_results]


def _provider() -> BaseSearchProvider:
    p = (settings.SEARCH_PROVIDER or "serper").strip().lower()
    if p == "tavily":
        return TavilyProvider()
    return SerperProvider()


class WebSearchTool(Tool):
    def __init__(
        self,
        *,
        name: str = "web_search",
        description: str = "Search the web and return titles, URLs, and relevant snippets.",
        max_fetch: int = 5,
        do_fetch: bool = True,
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = WebSearchInput
        self.max_fetch = max_fetch
        self.do_fetch = do_fetch
        self.search = _provider()

    def tool_spec(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": WebSearchInput.model_json_schema(),
            },
        }

    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            data = WebSearchInput.model_validate(payload)
        except Exception:
            return {"results": [], "error": "Invalid search input."}
        if settings.DISABLE_EXTERNALS:
            return {"results": [], "error": "External search disabled."}
        query = data.query.strip()
        max_results = data.max_results or 8
        include_domains = data.include_domains or []
        raw_allow = [
            d.strip().lower() for d in (settings.ALLOWED_DOMAINS or "").split(",") if d.strip()
        ]
        allow = [d for d in raw_allow if "." in d and not d.startswith("#") and not d.startswith(" ")]
        merged_domains = sorted({d.lower() for d in include_domains + allow if d})
        try:
            hits = await self.search.search(
                query,
                include_domains=merged_domains or None,
                max_results=max_results,
            )
            record_web_search_call(
                settings.SEARCH_PROVIDER,
                operation="web_search.provider_call",
                metadata={
                    "query": query[:200],
                    "max_results": max_results,
                    "domains": merged_domains,
                },
            )
        except Exception as exc:
            return {"results": [], "error": str(exc)}

        if allow:
            hits = [h for h in hits if _allowed(h.get("url", ""), allow)]
        urls = _dedupe_urls([h.get("url", "") for h in hits])
        uniq_hits: List[Dict[str, str]] = []
        for u in urls:
            h = next((x for x in hits if _norm_url(x.get("url", "")) == u), None)
            if h:
                uniq_hits.append(h)

        fetched: Dict[str, Dict[str, str]] = {}
        if self.do_fetch and uniq_hits:
            async with httpx.AsyncClient(timeout=20) as client:
                tasks = []
                for h in uniq_hits[: self.max_fetch]:
                    u = h.get("url", "")
                    if not u:
                        continue
                    tasks.append(_fetch_extract(u, client))
                results = await asyncio.gather(*tasks, return_exceptions=True)
            j = 0
            for h in uniq_hits[: self.max_fetch]:
                u = h.get("url", "")
                if not u:
                    continue
                res = results[j]
                j += 1
                if isinstance(res, dict):
                    fetched[u] = res

        results: List[Dict[str, str]] = []
        for h in uniq_hits[:max_results]:
            url = h.get("url", "")
            fetched_meta = fetched.get(url, {}) if url else {}
            title = _pick_title(h.get("title", ""), fetched_meta.get("title", ""), url)
            snippet = (h.get("snippet") or "").strip()
            if fetched_meta.get("text"):
                txt = fetched_meta["text"].strip()
                if txt:
                    snippet = (txt[:300] + "…") if len(txt) > 320 else txt
            results.append({"title": title, "url": url, "snippet": snippet})
        return {"results": results}
