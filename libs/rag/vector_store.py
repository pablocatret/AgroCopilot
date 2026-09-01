from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List, Protocol, Tuple

import numpy as np
from pgvector.psycopg2 import register_vector
import psycopg2
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from backend.deps import settings


class VectorStore(Protocol):
    def ensure_collection(self, collection: str, dim: int) -> None: ...

    def upsert(self, collection: str, points: List[PointStruct]) -> None: ...

    def search(
        self, collection: str, vector: List[float], limit: int = 8
    ) -> List[Dict[str, Any]]: ...

    def scroll(self, collection: str, limit: int = 2048) -> List[Dict[str, Any]]: ...


class QdrantStore:
    def __init__(self) -> None:
        self.client = QdrantClient(path=settings.QDRANT_PATH)

    def ensure_collection(self, collection: str, dim: int) -> None:
        if collection not in [c.name for c in self.client.get_collections().collections]:
            self.client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, collection: str, points: List[PointStruct]) -> None:
        self.client.upsert(collection_name=collection, points=points)

    def search(self, collection: str, vector: List[float], limit: int = 8) -> List[Dict[str, Any]]:
        res = self.client.search(collection_name=collection, query_vector=vector, limit=limit)
        out = []
        for r in res:
            payload = r.payload or {}
            payload["_score"] = float(r.score)
            out.append(payload)
        return out

    def scroll(self, collection: str, limit: int = 2048) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        next_page = None
        while True:
            page, next_page = self.client.scroll(
                collection_name=collection, with_payload=True, limit=limit, offset=next_page
            )
            for p in page:
                results.append(p.payload or {})
            if not next_page:
                break
        return results

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass


class SQLiteStore:
    def __init__(self, path: str) -> None:
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS vectors (collection TEXT, id TEXT PRIMARY KEY, vector TEXT, payload TEXT)"
        )
        self.conn.commit()

    def ensure_collection(self, collection: str, dim: int) -> None:
        return None

    def upsert(self, collection: str, points: List[PointStruct]) -> None:
        for point in points:
            self.conn.execute(
                "INSERT OR REPLACE INTO vectors (collection, id, vector, payload) VALUES (?, ?, ?, ?)",
                (collection, str(point.id), json.dumps(point.vector), json.dumps(point.payload)),
            )
        self.conn.commit()

    def search(self, collection: str, vector: List[float], limit: int = 8) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT vector, payload FROM vectors WHERE collection = ?", (collection,)
        )
        candidates: List[Tuple[List[float], Dict[str, Any]]] = []
        for vec_json, payload_json in cursor.fetchall():
            candidates.append((json.loads(vec_json), json.loads(payload_json)))
        if not candidates:
            return []
        q = np.array(vector)
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for vec, payload in candidates:
            v = np.array(vec)
            denom = (np.linalg.norm(q) * np.linalg.norm(v)) or 1.0
            score = float(np.dot(q, v) / denom)
            payload["_score"] = score
            scored.append((score, payload))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [payload for _, payload in scored[:limit]]

    def scroll(self, collection: str, limit: int = 2048) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT payload FROM vectors WHERE collection = ? LIMIT ?", (collection, limit)
        )
        return [json.loads(row[0]) for row in cursor.fetchall()]

    def close(self) -> None:
        self.conn.close()


class PGVectorStore:
    def __init__(self, dsn: str) -> None:
        self.conn = psycopg2.connect(dsn)
        register_vector(self.conn)
        self.conn.autocommit = True

    def ensure_collection(self, collection: str, dim: int) -> None:
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {collection} (
                    id TEXT PRIMARY KEY,
                    vector VECTOR({dim}),
                    payload JSONB
                )
                """
            )

    def upsert(self, collection: str, points: List[PointStruct]) -> None:
        with self.conn.cursor() as cur:
            for point in points:
                cur.execute(
                    f"INSERT INTO {collection} (id, vector, payload) VALUES (%s, %s, %s)"
                    f" ON CONFLICT (id) DO UPDATE SET vector = EXCLUDED.vector, payload = EXCLUDED.payload",
                    (str(point.id), point.vector, json.dumps(point.payload)),
                )

    def search(self, collection: str, vector: List[float], limit: int = 8) -> List[Dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT payload, 1 - (vector <=> %s) AS score FROM {collection} ORDER BY vector <=> %s LIMIT %s",
                (vector, vector, limit),
            )
            rows = cur.fetchall()
        out = []
        for payload, score in rows:
            payload["_score"] = float(score)
            out.append(payload)
        return out

    def scroll(self, collection: str, limit: int = 2048) -> List[Dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT payload FROM {collection} LIMIT %s", (limit,))
            rows = cur.fetchall()
        return [row[0] for row in rows]

    def close(self) -> None:
        self.conn.close()


def get_vector_store() -> VectorStore:
    backend = (settings.VECTOR_BACKEND or "sqlite").lower()
    if backend == "qdrant":
        return QdrantStore()
    if backend == "pgvector":
        return PGVectorStore(settings.DATABASE_URL)
    db_path = settings.DATABASE_URL.replace("sqlite:///", "")
    return SQLiteStore(db_path)
