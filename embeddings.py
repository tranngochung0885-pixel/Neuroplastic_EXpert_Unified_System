"""
NEXUS Embedding & Vector Store Layer
=====================================
Abstractions for:
  - Embedding providers: SentenceTransformers, hash fallback
  - Cross-encoder reranker
  - Vector stores: LanceDB (embedded), Qdrant (server/in-memory), memory fallback
"""

from __future__ import annotations

import abc
import asyncio
import hashlib
import math
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nexus.core.config import NexusSettings, HAS_ST, HAS_QDRANT, HAS_LANCEDB
from nexus.core.math_utils import VectorOps
from nexus.core.observability import LOG

EPS = 1e-12


# ══════════════════════════════════════════════════════════════════════════════
# §1  EMBEDDING PROVIDERS
# ══════════════════════════════════════════════════════════════════════════════

class EmbeddingProvider(abc.ABC):
    """Abstract embedding backend."""

    @abc.abstractmethod
    async def embed(self, text: str) -> List[float]: ...

    async def embed_many(self, texts: List[str]) -> List[List[float]]:
        return [await self.embed(t) for t in texts]

    @property
    @abc.abstractmethod
    def dim(self) -> int: ...


class HashEmbedProvider(EmbeddingProvider):
    """Deterministic hash-based pseudo-embeddings (no dependencies)."""

    def __init__(self, dim: int = 384):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, text: str) -> List[float]:
        return VectorOps.hash_embed(text, self._dim)


class SentenceTransformerProvider(EmbeddingProvider):
    """
    SentenceTransformers backend — the best quality embedding without an API key.
    Model: all-MiniLM-L6-v2 (384-dim) is small and fast.
    Reference: https://www.sbert.net/
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 384):
        self._model_name = model_name
        self._dim = dim
        self._model: Any = None
        if HAS_ST:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(model_name)
                LOG.info("embed.init", backend="sentence_transformers", model=model_name)
            except Exception as exc:
                LOG.warning("embed.init_failed", backend="sentence_transformers", error=str(exc))

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, text: str) -> List[float]:
        if self._model is None:
            return VectorOps.hash_embed(text, self._dim)
        try:
            arr = await asyncio.to_thread(
                self._model.encode, text[:2048], normalize_embeddings=True
            )
            vec = arr.tolist() if hasattr(arr, "tolist") else list(arr)
            d = self._dim
            if len(vec) >= d:
                vec = vec[:d]
            else:
                vec = vec + [0.0] * (d - len(vec))
            return VectorOps.normalize(vec)
        except Exception as exc:
            LOG.warning("embed.encode_failed", error=str(exc))
            return VectorOps.hash_embed(text, self._dim)

    async def embed_many(self, texts: List[str]) -> List[List[float]]:
        if self._model is None:
            return [VectorOps.hash_embed(t, self._dim) for t in texts]
        try:
            truncated = [t[:2048] for t in texts]
            arrs = await asyncio.to_thread(
                self._model.encode, truncated, normalize_embeddings=True
            )
            result = []
            d = self._dim
            for arr in arrs:
                vec = arr.tolist() if hasattr(arr, "tolist") else list(arr)
                if len(vec) >= d:
                    vec = vec[:d]
                else:
                    vec = vec + [0.0] * (d - len(vec))
                result.append(VectorOps.normalize(vec))
            return result
        except Exception as exc:
            LOG.warning("embed_many.failed", error=str(exc))
            return [VectorOps.hash_embed(t, self._dim) for t in texts]


class CachedEmbedProvider(EmbeddingProvider):
    """LRU-cached wrapper around any EmbeddingProvider."""

    def __init__(self, inner: EmbeddingProvider, max_size: int = 15_000):
        self._inner = inner
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self._max = max_size
        self._lock = asyncio.Lock()
        self._hits = 0
        self._total = 0

    @property
    def dim(self) -> int:
        return self._inner.dim

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]

    async def embed(self, text: str) -> List[float]:
        k = self._key(text)
        async with self._lock:
            if k in self._cache:
                self._cache.move_to_end(k)
                self._hits += 1
                return self._cache[k]
        vec = await self._inner.embed(text)
        async with self._lock:
            self._cache[k] = vec
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)
        self._total += 1
        return vec

    async def embed_many(self, texts: List[str]) -> List[List[float]]:
        return [await self.embed(t) for t in texts]

    def cache_stats(self) -> Dict[str, Any]:
        rate = self._hits / max(self._total + self._hits, 1)
        return {"size": len(self._cache), "hits": self._hits,
                "total": self._total, "hit_rate": round(rate, 3)}


def build_embedder(cfg: NexusSettings) -> EmbeddingProvider:
    backend = cfg.embed_backend
    if backend == "auto":
        backend = "sentence_transformers" if HAS_ST else "hash"
    if backend == "sentence_transformers":
        raw: EmbeddingProvider = SentenceTransformerProvider(cfg.embed_model, cfg.embed_dim)
    else:
        raw = HashEmbedProvider(cfg.embed_dim)
    return CachedEmbedProvider(raw, cfg.embed_cache_size)


# ══════════════════════════════════════════════════════════════════════════════
# §2  RERANKER
# ══════════════════════════════════════════════════════════════════════════════

class Reranker(abc.ABC):
    """Abstract cross-encoder reranker."""
    @abc.abstractmethod
    async def rerank(self, query: str, docs: List[str]) -> List[Tuple[int, float]]: ...


class NoopReranker(Reranker):
    async def rerank(self, query: str, docs: List[str]) -> List[Tuple[int, float]]:
        return [(i, 0.0) for i in range(len(docs))]


class CrossEncoderReranker(Reranker):
    """
    Cross-encoder reranker (ms-marco-MiniLM-L-6-v2).
    Reference: https://www.sbert.net/docs/cross_encoder/pretrained_models.html
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model: Any = None
        if HAS_ST:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(model_name)
                LOG.info("reranker.init", model=model_name)
            except Exception as exc:
                LOG.warning("reranker.init_failed", error=str(exc))

    async def rerank(self, query: str, docs: List[str]) -> List[Tuple[int, float]]:
        if self._model is None or not docs:
            return [(i, 0.0) for i in range(len(docs))]
        pairs = [[query, d] for d in docs]
        try:
            scores = await asyncio.to_thread(self._model.predict, pairs)
            result = [(i, float(scores[i])) for i in range(len(docs))]
            result.sort(key=lambda x: x[1], reverse=True)
            return result
        except Exception as exc:
            LOG.warning("reranker.predict_failed", error=str(exc))
            return [(i, 0.0) for i in range(len(docs))]


def build_reranker(cfg: NexusSettings) -> Reranker:
    if cfg.enable_reranker and HAS_ST:
        return CrossEncoderReranker(cfg.reranker_model)
    return NoopReranker()


# ══════════════════════════════════════════════════════════════════════════════
# §3  VECTOR STORE ABSTRACTION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VectorHit:
    uid: str
    score: float
    payload: Dict[str, Any]


class VectorStore(abc.ABC):
    """Abstract vector store interface."""

    @abc.abstractmethod
    async def upsert(self, uid: str, vector: List[float],
                     payload: Dict[str, Any]) -> None: ...

    @abc.abstractmethod
    async def search(self, vector: List[float], limit: int = 10,
                     filters: Optional[Dict[str, Any]] = None) -> List[VectorHit]: ...

    async def delete(self, uid: str) -> None:
        pass

    async def count(self) -> int:
        return -1


# ── In-memory fallback ────────────────────────────────────────────────────────

class MemoryVectorStore(VectorStore):
    """Pure in-memory vector store with brute-force cosine search."""

    def __init__(self) -> None:
        self._items: Dict[str, Tuple[List[float], Dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    async def upsert(self, uid: str, vector: List[float],
                     payload: Dict[str, Any]) -> None:
        async with self._lock:
            self._items[uid] = (VectorOps.normalize(vector), payload)

    async def search(self, vector: List[float], limit: int = 10,
                     filters: Optional[Dict[str, Any]] = None) -> List[VectorHit]:
        qn = VectorOps.normalize(vector)
        async with self._lock:
            scored: List[VectorHit] = []
            for uid, (vec, payload) in self._items.items():
                if filters:
                    if not all(payload.get(k) == v for k, v in filters.items()):
                        continue
                scored.append(VectorHit(uid=uid,
                                        score=VectorOps.cosine(qn, vec),
                                        payload=payload))
            scored.sort(key=lambda h: h.score, reverse=True)
            return scored[:limit]

    async def delete(self, uid: str) -> None:
        async with self._lock:
            self._items.pop(uid, None)

    async def count(self) -> int:
        return len(self._items)


# ── LanceDB embedded vector store ────────────────────────────────────────────

class LanceDBVectorStore(VectorStore):
    """
    LanceDB embedded vector store — no external server needed.
    Uses IVF_PQ index for ANN search after initial data load.
    Reference: https://lancedb.github.io/lancedb/
    """

    def __init__(self, cfg: NexusSettings) -> None:
        self._cfg = cfg
        self._db: Any = None
        self._table: Any = None
        self._table_name = "nexus_episodic"
        self._dim = cfg.embed_dim
        self.ready = False

    async def initialize(self) -> None:
        if not HAS_LANCEDB:
            LOG.warning("lancedb.not_installed")
            return
        try:
            import lancedb
            import pyarrow as pa
            path = self._cfg.lancedb_dir()
            Path(path).mkdir(parents=True, exist_ok=True)
            # lancedb.connect is sync; offload to thread for async context
            self._db = await asyncio.to_thread(lancedb.connect, path)
            # Check/create table
            tables = await asyncio.to_thread(self._db.table_names)
            if self._table_name not in tables:
                schema = pa.schema([
                    pa.field("uid", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), self._dim)),
                    pa.field("text", pa.string()),
                    pa.field("importance", pa.float32()),
                    pa.field("intent", pa.string()),
                    pa.field("tier", pa.string()),
                    pa.field("created_at", pa.float64()),
                ])
                self._table = await asyncio.to_thread(
                    self._db.create_table, self._table_name, schema=schema
                )
            else:
                self._table = await asyncio.to_thread(
                    self._db.open_table, self._table_name
                )
            self.ready = True
            LOG.info("lancedb.initialized", path=path, table=self._table_name)
        except Exception as exc:
            LOG.warning("lancedb.init_failed", error=str(exc))
            self.ready = False

    async def upsert(self, uid: str, vector: List[float],
                     payload: Dict[str, Any]) -> None:
        if not self.ready or self._table is None:
            return
        try:
            import pyarrow as pa
            row = {
                "uid": uid,
                "vector": [float(x) for x in vector],
                "text": str(payload.get("text", ""))[:512],
                "importance": float(payload.get("importance", 0.5)),
                "intent": str(payload.get("intent", "unknown")),
                "tier": str(payload.get("tier", "episodic")),
                "created_at": float(payload.get("created_at", time.time())),
            }
            # LanceDB upsert: delete existing then add
            await asyncio.to_thread(
                self._table.delete, f"uid = '{uid}'"
            )
            await asyncio.to_thread(
                self._table.add, [row]
            )
        except Exception as exc:
            LOG.warning("lancedb.upsert_failed", uid=uid, error=str(exc))

    async def search(self, vector: List[float], limit: int = 10,
                     filters: Optional[Dict[str, Any]] = None) -> List[VectorHit]:
        if not self.ready or self._table is None:
            return []
        try:
            q = self._table.search(vector).limit(limit)
            if filters:
                # LanceDB filter syntax: SQL-like WHERE clause
                parts = []
                for k, v in filters.items():
                    if isinstance(v, str):
                        parts.append(f"{k} = '{v}'")
                    else:
                        parts.append(f"{k} = {v}")
                if parts:
                    q = q.where(" AND ".join(parts))
            results = await asyncio.to_thread(q.to_list)
            hits: List[VectorHit] = []
            for row in results:
                score = float(row.get("_distance", 0.0))
                # LanceDB returns L2 distance by default; convert to similarity
                score = max(0.0, 1.0 - score / 2.0)
                hits.append(VectorHit(
                    uid=str(row["uid"]),
                    score=score,
                    payload={k: v for k, v in row.items()
                             if k not in ("vector", "uid", "_distance")},
                ))
            return hits
        except Exception as exc:
            LOG.warning("lancedb.search_failed", error=str(exc))
            return []

    async def delete(self, uid: str) -> None:
        if not self.ready or self._table is None:
            return
        try:
            await asyncio.to_thread(self._table.delete, f"uid = '{uid}'")
        except Exception:
            pass

    async def count(self) -> int:
        if not self.ready or self._table is None:
            return 0
        try:
            return await asyncio.to_thread(lambda: len(self._table))
        except Exception:
            return 0


# ── Qdrant vector store ───────────────────────────────────────────────────────

class QdrantVectorStore(VectorStore):
    """
    Qdrant vector store using AsyncQdrantClient.
    Supports: remote server, local on-disk, and in-memory modes.
    Reference: https://qdrant.tech/documentation/
    """

    def __init__(self, cfg: NexusSettings) -> None:
        self._cfg = cfg
        self._client: Any = None
        self._collection = cfg.qdrant_collection
        self.ready = False

    async def initialize(self) -> None:
        if not HAS_QDRANT:
            return
        try:
            from qdrant_client import AsyncQdrantClient
            from qdrant_client import models as qm

            kw: Dict[str, Any] = {}
            if self._cfg.qdrant_url:
                kw["url"] = self._cfg.qdrant_url
                if self._cfg.qdrant_api_key:
                    kw["api_key"] = self._cfg.qdrant_api_key
            elif self._cfg.qdrant_local_path:
                kw["path"] = self._cfg.qdrant_local_path
            else:
                kw["location"] = ":memory:"

            self._client = AsyncQdrantClient(**kw)
            exists = await self._client.collection_exists(self._collection)
            if not exists:
                await self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=qm.VectorParams(
                        size=self._cfg.embed_dim,
                        distance=qm.Distance.COSINE,
                    ),
                )
            self.ready = True
            LOG.info("qdrant.initialized", collection=self._collection)
        except Exception as exc:
            LOG.warning("qdrant.init_failed", error=str(exc))
            self.ready = False
            self._client = None

    async def upsert(self, uid: str, vector: List[float],
                     payload: Dict[str, Any]) -> None:
        if not self.ready or self._client is None:
            return
        try:
            from qdrant_client import models as qm
            await self._client.upsert(
                collection_name=self._collection,
                points=[qm.PointStruct(id=uid, vector=vector, payload=payload)],
                wait=False,
            )
        except Exception as exc:
            LOG.warning("qdrant.upsert_failed", uid=uid, error=str(exc))

    async def search(self, vector: List[float], limit: int = 10,
                     filters: Optional[Dict[str, Any]] = None) -> List[VectorHit]:
        if not self.ready or self._client is None:
            return []
        try:
            f = None
            if filters and HAS_QDRANT:
                from qdrant_client import models as qm
                conds = [qm.FieldCondition(key=k, match=qm.MatchValue(value=v))
                         for k, v in filters.items()]
                if conds:
                    f = qm.Filter(must=conds)
            try:
                resp = await self._client.query_points(
                    collection_name=self._collection,
                    query=vector, limit=limit,
                    query_filter=f,
                )
                points = getattr(resp, "points", resp)
            except Exception:
                points = await self._client.search(
                    collection_name=self._collection,
                    query_vector=vector, limit=limit,
                    query_filter=f,
                )
            return [
                VectorHit(
                    uid=str(getattr(p, "id", "")),
                    score=float(getattr(p, "score", 0.0)),
                    payload=getattr(p, "payload", {}) or {},
                )
                for p in points
            ]
        except Exception as exc:
            LOG.warning("qdrant.search_failed", error=str(exc))
            return []

    async def delete(self, uid: str) -> None:
        if not self.ready or self._client is None:
            return
        try:
            from qdrant_client import models as qm
            await self._client.delete(
                collection_name=self._collection,
                points_selector=qm.PointIdsList(points=[uid]),
            )
        except Exception:
            pass

    async def count(self) -> int:
        if not self.ready or self._client is None:
            return 0
        try:
            info = await self._client.get_collection(self._collection)
            return int(getattr(info, "vectors_count", 0) or 0)
        except Exception:
            return 0


# ── Factory ───────────────────────────────────────────────────────────────────

async def build_vector_store(cfg: NexusSettings) -> VectorStore:
    """
    Auto-detect and initialise the best available vector store.
    Priority: Qdrant (if URL set) → LanceDB (embedded) → Memory (fallback).
    """
    backend = cfg.vector_backend
    if backend == "auto":
        if HAS_QDRANT and cfg.qdrant_url:
            backend = "qdrant"
        elif HAS_LANCEDB:
            backend = "lancedb"
        else:
            backend = "memory"

    if backend == "qdrant" and HAS_QDRANT:
        store = QdrantVectorStore(cfg)
        await store.initialize()
        if store.ready:
            return store
        LOG.warning("qdrant.fallback_to_lancedb")

    if (backend in ("lancedb", "qdrant")) and HAS_LANCEDB:
        store = LanceDBVectorStore(cfg)
        await store.initialize()
        if store.ready:
            return store
        LOG.warning("lancedb.fallback_to_memory")

    LOG.info("vector_store.using_memory")
    return MemoryVectorStore()
