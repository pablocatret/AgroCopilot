import pytest

from libs.rag import ingest
from libs.rag import retriever


class FakeStore:
    def __init__(self, payloads=None) -> None:
        self.payloads = payloads or []
        self.scroll_calls = 0
        self.search_calls = 0

    def scroll(self, collection: str, limit: int = 2048):
        self.scroll_calls += 1
        return list(self.payloads)

    def search(self, collection: str, vector, limit: int = 8):
        self.search_calls += 1
        return [
            {
                "title": "Vector doc",
                "path": "/tmp/vector-doc",
                "text": "contenido vectorial",
            }
        ]

    def close(self) -> None:
        return None


class FakeUpsertStore(FakeStore):
    def __init__(self) -> None:
        super().__init__(payloads=[])
        self.points = []

    def ensure_collection(self, collection: str, dim: int):
        return None

    def upsert(self, collection: str, points):
        self.points.extend(points)


@pytest.mark.asyncio
async def test_retrieve_vector_strategy_skips_scroll(monkeypatch):
    store = FakeStore(payloads=[{"title": "bm25", "path": "/tmp/bm25", "text": "bm25"}])
    retriever.clear_retriever_cache()
    monkeypatch.setattr(retriever.settings, "LEGAL_RAG_STRATEGY", "vector")
    monkeypatch.setattr(retriever, "get_vector_store", lambda: store)

    async def fake_vector_rank(query, top_k=8):
        return [{"title": "Vector", "path": "/tmp/vector", "text": "abc"}]

    monkeypatch.setattr(retriever, "_vector_rank", fake_vector_rank)

    docs = await retriever.retrieve("consulta", k=2)

    assert store.scroll_calls == 0
    assert docs[0]["title"] == "Vector"


@pytest.mark.asyncio
async def test_retrieve_bm25_reuses_cached_corpus(monkeypatch):
    payloads = [
        {"title": "Doc A", "path": "/tmp/a", "text": "fertilizacion nitrogeno"},
        {"title": "Doc B", "path": "/tmp/b", "text": "riego sostenible"},
    ]
    store = FakeStore(payloads=payloads)
    retriever.clear_retriever_cache()
    monkeypatch.setattr(retriever.settings, "LEGAL_RAG_STRATEGY", "bm25")
    monkeypatch.setattr(retriever, "get_vector_store", lambda: store)

    first = await retriever.retrieve("nitrogeno", k=1)
    second = await retriever.retrieve("riego", k=1)

    assert store.scroll_calls == 1
    assert first
    assert second


@pytest.mark.asyncio
async def test_retrieve_bm25_preserves_legalize_metadata_and_prioritizes_current(monkeypatch):
    payloads = [
        {
            "title": "Norma derogada de agricultura",
            "path": "es/old.md",
            "text": "agricultura ayuda pac",
            "corpus": "legalize",
            "estado": "derogada",
            "source_url": "https://boe.es/old",
        },
        {
            "title": "Norma vigente de agricultura",
            "path": "es/current.md",
            "text": "agricultura ayuda pac",
            "corpus": "legalize",
            "estado": "vigente",
            "jurisdiccion": "es",
            "ultima_actualizacion": "2025-01-01",
            "source_url": "https://boe.es/current",
            "article": "Articulo 1",
        },
    ]
    store = FakeStore(payloads=payloads)
    retriever.clear_retriever_cache()
    monkeypatch.setattr(retriever.settings, "LEGAL_RAG_STRATEGY", "bm25")
    monkeypatch.setattr(retriever, "get_vector_store", lambda: store)

    docs = await retriever.retrieve("ayuda PAC agricultura Espana", k=2)

    assert docs[0]["title"] == "Norma vigente de agricultura"
    assert docs[0]["corpus"] == "legalize"
    assert docs[0]["source_status"] == "vigente"
    assert docs[0]["official_source"] == "https://boe.es/current"
    assert docs[0]["article"] == "Articulo 1"


@pytest.mark.asyncio
async def test_ingest_paths_uses_legalize_chunks_for_markdown(monkeypatch, tmp_path):
    law = tmp_path / "BOE-A-demo.md"
    law.write_text(
        """---
titulo: "Ley de agricultura sostenible"
identificador: "BOE-A-DEMO"
pais: "es"
jurisdiccion: "es"
estado: "vigente"
fuente: "https://www.boe.es/demo"
---
Articulo 1. Agricultura sostenible.
""",
        encoding="utf-8",
    )
    store = FakeUpsertStore()
    monkeypatch.setattr(ingest.settings, "DISABLE_EXTERNALS", False)
    monkeypatch.setattr(ingest, "client", object())
    monkeypatch.setattr(ingest, "get_vector_store", lambda: store)

    async def fake_embed(texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(ingest, "_embed_batched", fake_embed)

    count = await ingest.ingest_paths([str(law)])

    assert count == 1
    assert store.points
    payload = store.points[0].payload
    assert payload["corpus"] == "legalize"
    assert payload["official_source"] == "https://www.boe.es/demo"
