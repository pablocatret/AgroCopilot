# tests/test_health.py
import pytest
from httpx import ASGITransport, AsyncClient

from backend.api import app


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        r = await client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j.get("status") == "ok"
    assert "version" in j


@pytest.mark.asyncio
async def test_version():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        r = await client.get("/version")
    assert r.status_code == 200
    assert "version" in r.json()
