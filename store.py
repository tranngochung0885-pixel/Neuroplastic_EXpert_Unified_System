"""
nexus/memory/store.py
=====================
Multi-tier memory architecture for NEXUS:
  SensoryBuffer     — 6-item FIFO with TTL expiry (Sperling 1960)
  WorkingMemory     — Cowan 4±1 with attention-weighted eviction (Cowan 2001)
  EpisodicStore     — SQLite-backed with interference modeling + SRS forgetting
  SemanticStore     — EWC-protected concept nodes (Kirkpatrick 2017)
  EngramLattice     — Unified multi-tier interface with Hebbian linking

All persistence is thread-safe via threading.local() DB connections.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import pickle
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from nexus.core.config import (
    AffectLabel,
    Engram,
    MemoryTier,
    NexusSettings,
    SemanticNode,
    now_ts,
)
from nexus.core.math_utils import HebbianPlasticity, SRS, VectorOps
from nexus.core.observability import LOG, METRICS
from nexus.memory.embeddings import EmbeddingProvider

EPS = 1e-12


# ---------------------------------------------------------------------------
# SQLite State Store  (persistence layer for all tiers)
# ---------------------------------------------------------------------------

class SQLiteStateStore:
    """
    Thread-safe SQLite persistence. Uses WAL mode + threading.local() for
    per-thread connection pooling. Stores engrams, semantic nodes, goals,
    and checkpoints.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    # ── Connection pooling ─────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-32000")  # 32 MB
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        self._conn().executescript(
            """
            CREATE TABLE IF NOT EXISTS engrams (
                uid TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                tier TEXT NOT NULL,
                created_at REAL,
                last_retrieved REAL,
                retrieval_count INTEGER DEFAULT 0,
                stability_h REAL,
                importance REAL,
                valence_encoded REAL,
                arousal_encoded REAL,
                keywords TEXT,
                embedding BLOB,
                entities TEXT,
                intent TEXT,
                links TEXT,
                consolidated INTEGER DEFAULT 0,
                abstract_concept TEXT,
                meta TEXT
            );

            CREATE TABLE IF NOT EXISTS semantic_nodes (
                uid TEXT PRIMARY KEY,
                concept_key TEXT UNIQUE,
                name TEXT,
                summary TEXT,
                embedding BLOB,
                members TEXT,
                confidence REAL,
                frequency INTEGER DEFAULT 0,
                updated_at REAL,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS goals (
                uid TEXT PRIMARY KEY,
                description TEXT,
                priority REAL,
                progress REAL DEFAULT 0.0,
                active INTEGER DEFAULT 1,
                created_at REAL,
                embedding BLOB,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL,
                payload TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_engrams_tier ON engrams(tier);
            CREATE INDEX IF NOT EXISTS idx_engrams_importance ON engrams(importance DESC);
            CREATE INDEX IF NOT EXISTS idx_engrams_created ON engrams(created_at DESC);
            """
        )
        self._conn().commit()

    # ── Engram persistence ─────────────────────────────────────────────────

    def upsert_engram(self, e: Engram) -> None:
        self._conn().execute(
            """
            INSERT OR REPLACE INTO engrams VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                e.uid,
                e.content,
                e.tier.value,
                e.created_at,
                e.updated_at,
                e.retrievals,
                e.stability_h,
                e.importance,
                e.valence_enc,
                e.arousal_enc,
                json.dumps(e.keywords),
                pickle.dumps(e.embedding) if e.embedding else None,
                json.dumps(e.entities),
                e.intent,
                json.dumps(e.links),
                int(e.consolidated),
                e.meta.get("abstract_concept"),
                json.dumps(e.meta),
            ),
        )
        self._conn().commit()

    def load_engrams(self, tier: Optional[MemoryTier] = None) -> List[Engram]:
        if tier:
            rows = self._conn().execute(
                "SELECT * FROM engrams WHERE tier=? ORDER BY created_at DESC",
                (tier.value,),
            ).fetchall()
        else:
            rows = self._conn().execute(
                "SELECT * FROM engrams ORDER BY created_at DESC"
            ).fetchall()
        result = []
        for r in rows:
            e = self._row_to_engram(r)
            if e:
                result.append(e)
        return result

    def delete_engram(self, uid: str) -> None:
        self._conn().execute("DELETE FROM engrams WHERE uid=?", (uid,))
        self._conn().commit()

    def count_engrams(self, tier: Optional[MemoryTier] = None) -> int:
        if tier:
            row = self._conn().execute(
                "SELECT COUNT(*) FROM engrams WHERE tier=?", (tier.value,)
            ).fetchone()
        else:
            row = self._conn().execute("SELECT COUNT(*) FROM engrams").fetchone()
        return int(row[0]) if row else 0

    @staticmethod
    def _row_to_engram(r: sqlite3.Row) -> Optional[Engram]:
        try:
            emb = pickle.loads(r["embedding"]) if r["embedding"] else []
            return Engram(
                uid=r["uid"],
                content=r["content"],
                tier=MemoryTier(r["tier"]),
                created_at=float(r["created_at"] or 0),
                updated_at=float(r["last_retrieved"] or 0),
                retrievals=int(r["retrieval_count"] or 0),
                stability_h=float(r["stability_h"] or 12.0),
                importance=float(r["importance"] or 0.5),
                valence_enc=float(r["valence_encoded"] or 0.0),
                arousal_enc=float(r["arousal_encoded"] or 0.4),
                keywords=json.loads(r["keywords"] or "[]"),
                embedding=emb,
                entities=json.loads(r["entities"] or "[]"),
                intent=r["intent"] or "unknown",
                links=json.loads(r["links"] or "{}"),
                consolidated=bool(r["consolidated"]),
                # abstract_concept stored in meta
                meta=json.loads(r["meta"] or "{}"),
            )
        except Exception as exc:
            LOG.warning("engram.row_parse_error", error=str(exc))
            return None

    # ── Semantic node persistence ──────────────────────────────────────────

    def upsert_semantic(self, ckey: str, node: SemanticNode) -> None:
        self._conn().execute(
            """
            INSERT OR REPLACE INTO semantic_nodes VALUES
            (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                node.uid,
                ckey,
                node.name,
                node.summary,
                pickle.dumps(node.embedding) if node.embedding else None,
                json.dumps(list(node.members)),
                node.confidence,
                node.frequency,
                node.updated_at,
                json.dumps(node.metadata),
            ),
        )
        self._conn().commit()

    def load_semantic_nodes(self) -> Dict[str, SemanticNode]:
        rows = self._conn().execute("SELECT * FROM semantic_nodes").fetchall()
        result: Dict[str, SemanticNode] = {}
        for r in rows:
            try:
                ckey = r["concept_key"]
                emb = pickle.loads(r["embedding"]) if r["embedding"] else []
                node = SemanticNode(
                    uid=r["uid"],
                    name=r["name"] or "",
                    summary=r["summary"] or "",
                    embedding=emb,
                    members=set(json.loads(r["members"] or "[]")),
                    confidence=float(r["confidence"] or 0.5),
                    frequency=int(r["frequency"] or 0),
                    updated_at=float(r["updated_at"] or 0),
                    metadata=json.loads(r["metadata"] or "{}"),
                )
                result[ckey] = node
            except Exception as exc:
                LOG.warning("semantic_node.row_parse_error", error=str(exc))
        return result

    # ── Goal persistence ───────────────────────────────────────────────────

    def upsert_goal(self, goal: Any) -> None:
        self._conn().execute(
            "INSERT OR REPLACE INTO goals VALUES (?,?,?,?,?,?,?,?)",
            (
                goal.uid,
                goal.description,
                goal.priority,
                goal.progress,
                1 if goal.active else 0,
                goal.created_at,
                pickle.dumps(goal.embedding) if goal.embedding else None,
                json.dumps(goal.notes),
            ),
        )
        self._conn().commit()

    def load_goals(self) -> List[Any]:
        from nexus.core.config import Goal
        rows = self._conn().execute(
            "SELECT * FROM goals WHERE active=1"
        ).fetchall()
        out = []
        for r in rows:
            try:
                out.append(
                    Goal(
                        uid=r["uid"],
                        description=r["description"],
                        priority=float(r["priority"]),
                        progress=float(r["progress"]),
                        active=bool(r["active"]),
                        created_at=float(r["created_at"]),
                        embedding=pickle.loads(r["embedding"]) if r["embedding"] else [],
                        notes=json.loads(r["notes"] or "[]"),
                    )
                )
            except Exception:
                pass
        return out

    # ── Checkpoint ────────────────────────────────────────────────────────

    def save_checkpoint(self, payload: Dict[str, Any]) -> None:
        self._conn().execute(
            "INSERT INTO checkpoints(ts, payload) VALUES (?,?)",
            (now_ts(), json.dumps(payload, default=str)),
        )
        self._conn().commit()

    def vacuum(self) -> None:
        """VACUUM periodically to reclaim space from deleted engrams."""
        self._conn().execute("VACUUM")


# ---------------------------------------------------------------------------
# Sensory Buffer  (Sperling 1960 — iconic store, ~3.5s TTL, 6 items)
# ---------------------------------------------------------------------------

class SensoryBuffer:
    """
    Pre-attentive sensory buffer: raw percepts with very short TTL.
    Capacity=6, FIFO displacement, millisecond-level TTL expiry.
    """

    def __init__(self, cfg: NexusSettings) -> None:
        self._capacity = cfg.sensory_capacity
        self._ttl_s = cfg.sensory_ttl_ms / 1000.0
        self._items: deque[Engram] = deque(maxlen=self._capacity)
        self._lock = threading.RLock()

    def push(self, engram: Engram) -> None:
        with self._lock:
            self._items.append(engram)

    def flush_expired(self) -> List[Engram]:
        """Remove expired items and return them for potential promotion."""
        now = now_ts()
        expired: List[Engram] = []
        with self._lock:
            alive: deque[Engram] = deque(maxlen=self._capacity)
            for e in self._items:
                if now - e.created_at > self._ttl_s:
                    expired.append(e)
                else:
                    alive.append(e)
            self._items = alive
        return expired

    def peek(self) -> List[Engram]:
        with self._lock:
            return list(self._items)

    def __len__(self) -> int:
        return len(self._items)


# ---------------------------------------------------------------------------
# Working Memory  (Cowan 2001 — 4±1 items, attention-weighted eviction)
# ---------------------------------------------------------------------------

class WorkingMemory:
    """
    Cowan's working memory model: 4±1 capacity with attention-weighted
    eviction. High NE sharpens attention (winner-take-more dynamics).
    Items decay after working_ttl_s without rehearsal.
    """

    def __init__(self, cfg: NexusSettings) -> None:
        self._capacity = cfg.working_memory_size  # default 4 (Cowan)
        self._ttl_s = cfg.working_ttl_s
        self._slots: OrderedDict[str, Engram] = OrderedDict()
        self._attention: Dict[str, float] = {}
        self._lock = threading.RLock()

    def put(self, engram: Engram, attention: float = 0.5) -> Optional[Engram]:
        """
        Insert into working memory.
        Returns evicted engram (lowest attention) if at capacity.
        """
        with self._lock:
            if engram.uid in self._slots:
                self._slots.move_to_end(engram.uid)
                self._attention[engram.uid] = max(
                    self._attention.get(engram.uid, 0.0), attention
                )
                return None

            evicted: Optional[Engram] = None
            if len(self._slots) >= self._capacity:
                evict_uid = min(
                    self._attention, key=lambda k: self._attention.get(k, 0.0)
                )
                evicted = self._slots.pop(evict_uid, None)
                self._attention.pop(evict_uid, None)

            self._slots[engram.uid] = engram
            self._attention[engram.uid] = attention
            return evicted

    def refresh(self, uid: str, new_attention: Optional[float] = None) -> None:
        with self._lock:
            if uid in self._slots:
                self._slots.move_to_end(uid)
                if new_attention is not None:
                    self._attention[uid] = new_attention

    def flush_expired(self) -> List[Engram]:
        """Remove items that haven't been rehearsed within TTL."""
        now = now_ts()
        flushed: List[Engram] = []
        with self._lock:
            expired = [
                uid
                for uid, e in self._slots.items()
                if now - e.updated_at >= self._ttl_s
            ]
            for uid in expired:
                e = self._slots.pop(uid, None)
                self._attention.pop(uid, None)
                if e:
                    flushed.append(e)
        return flushed

    def get_active(self) -> List[Engram]:
        now = now_ts()
        with self._lock:
            return [
                e
                for e in self._slots.values()
                if now - e.updated_at < self._ttl_s
            ]

    def search(self, query_emb: List[float], k: int = 3) -> List[Tuple[Engram, float]]:
        with self._lock:
            scored = [
                (VectorOps.cosine(e.embedding, query_emb), e)
                for e in self._slots.values()
                if e.embedding
            ]
        scored.sort(key=lambda x: -x[0])
        return [(e, s) for s, e in scored[:k]]

    def sharpen_attention(self, uid: str, ne_level: float) -> None:
        """NE gain modulation: high NE → focus on uid, suppress others."""
        with self._lock:
            if uid not in self._attention:
                return
            if ne_level > 0.65:
                for k in self._attention:
                    if k != uid:
                        self._attention[k] *= 1.0 - ne_level * 0.18
            self._attention[uid] = min(1.0, self._attention[uid] + 0.12)

    def __len__(self) -> int:
        return len(self._slots)


# ---------------------------------------------------------------------------
# Episodic Store  (SQLite-backed autobiographical memory)
# ---------------------------------------------------------------------------

class EpisodicStore:
    """
    Episodic memory with:
    - SQLite persistence
    - SRS forgetting curve (Wozniak SM-2)
    - Interference modeling (similar memories compete)
    - EWC protection for high-importance traces (Kirkpatrick 2017)
    - Mood-congruent retrieval bias (Bower 1981)
    """

    def __init__(self, cfg: NexusSettings, store: SQLiteStateStore) -> None:
        self.cfg = cfg
        self.store = store
        self._engrams: Dict[str, Engram] = {}
        self._lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        for e in self.store.load_engrams(MemoryTier.EPISODIC):
            self._engrams[e.uid] = e
        LOG.info(
            "episodic.loaded",
            count=len(self._engrams),
        )
        METRICS.set_memory("episodic", len(self._engrams))

    def save(self, engram: Engram) -> None:
        with self._lock:
            self._engrams[engram.uid] = engram
        self.store.upsert_engram(engram)
        METRICS.set_memory("episodic", len(self._engrams))

    def retrieve(
        self,
        query_emb: List[float],
        current_valence: float,
        k: int = 8,
        min_score: float = 0.15,
        keyword_boost: Optional[Set[str]] = None,
    ) -> List[Tuple[Engram, float]]:
        """
        Multi-factor retrieval:
          0.45 × semantic_similarity
          0.25 × SRS retrievability
          0.15 × recency
          0.10 × importance
          0.05 × access_frequency
        + mood-congruent valence bias (Bower 1981)
        + optional keyword overlap boost
        """
        with self._lock:
            snapshot = list(self._engrams.values())

        scored: List[Tuple[float, Engram]] = []
        for e in snapshot:
            if not e.embedding:
                continue
            sem = VectorOps.cosine(e.embedding, query_emb)
            elapsed_h = (now_ts() - e.updated_at) / 3600.0
            ret = SRS.retrievability(e.stability_h, elapsed_h)
            recency = math.exp(-elapsed_h / 48.0)
            freq_bonus = min(0.05, e.retrievals * 0.01)

            # Mood-congruent bias
            valence_bias = 1.0 + current_valence * e.valence_enc * 0.22

            score = (
                sem * 0.45
                + ret * 0.25
                + recency * 0.15
                + e.importance * 0.10
                + freq_bonus
            ) * max(0.4, valence_bias)

            # Keyword boost
            if keyword_boost and e.keywords:
                overlap = len(keyword_boost & set(e.keywords))
                score += min(0.15, overlap * 0.04)

            if score >= min_score:
                scored.append((score, e))

        scored.sort(key=lambda x: -x[0])
        top = scored[:k]

        # Trigger SRS update on retrieved items
        results: List[Tuple[Engram, float]] = []
        for score, e in top:
            e_copy = e
            quality = min(1.0, score * 1.2)
            e_copy.updated_at = now_ts()
            e_copy.retrievals += 1
            difficulty = SRS.difficulty_from_quality(quality)
            e_copy.stability_h = SRS.new_stability(
                e_copy.stability_h, difficulty, True, self.cfg.spacing_coeff
            )
            # Reconsolidation: makes memory temporarily labile
            e_copy.meta["reconsolidation_ts"] = now_ts()
            self.store.upsert_engram(e_copy)
            results.append((e_copy, score))

        return results

    def apply_interference(self, new_engram: Engram) -> None:
        """
        Proactive/retroactive interference: similar existing memories
        are weakened when a new similar memory is encoded.
        """
        if not new_engram.embedding:
            return
        with self._lock:
            for uid, existing in list(self._engrams.items()):
                if not existing.embedding or uid == new_engram.uid:
                    continue
                sim = VectorOps.cosine(new_engram.embedding, existing.embedding)
                if sim > self.cfg.interference_radius:
                    penalty = self.cfg.interference_penalty * sim
                    existing.stability_h = max(1.0, existing.stability_h * (1.0 - penalty))

    def prune(self, max_items: Optional[int] = None) -> int:
        """
        Remove weakest engrams when over capacity.
        EWC: protect engrams with importance >= ewc_threshold.
        """
        max_items = max_items or self.cfg.episodic_capacity
        with self._lock:
            n = len(self._engrams)
            if n <= max_items:
                return 0

            def prune_score(e: Engram) -> float:
                if e.importance >= self.cfg.ewc_threshold:
                    return 1.0  # protected
                elapsed_h = (now_ts() - e.updated_at) / 3600.0
                ret = SRS.retrievability(e.stability_h, elapsed_h)
                return ret * (1.0 + e.importance)

            sorted_e = sorted(self._engrams.values(), key=prune_score)
            n_remove = n - max_items + 200
            removed = 0
            for e in sorted_e[:n_remove]:
                if e.importance < self.cfg.ewc_threshold:
                    age_h = (now_ts() - e.created_at) / 3600.0
                    if age_h > 1.0:
                        del self._engrams[e.uid]
                        self.store.delete_engram(e.uid)
                        removed += 1

        METRICS.set_memory("episodic", len(self._engrams))
        return removed

    def all_engrams(self) -> List[Engram]:
        with self._lock:
            return list(self._engrams.values())

    def __len__(self) -> int:
        return len(self._engrams)


# ---------------------------------------------------------------------------
# Semantic Store  (EWC-protected concept nodes)
# ---------------------------------------------------------------------------

class SemanticStore:
    """
    Semantic memory: generalized knowledge extracted from episodic memory.
    Uses Elastic Weight Consolidation (EWC, Kirkpatrick 2017) to protect
    high-importance nodes from catastrophic overwriting.
    """

    def __init__(self, cfg: NexusSettings, store: SQLiteStateStore) -> None:
        self.cfg = cfg
        self.store = store
        self._nodes: Dict[str, SemanticNode] = {}  # ckey → SemanticNode
        self._lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        self._nodes = self.store.load_semantic_nodes()
        LOG.info("semantic.loaded", count=len(self._nodes))
        METRICS.set_memory("semantic", len(self._nodes))

    def upsert(self, ckey: str, node: SemanticNode) -> None:
        """EWC-aware upsert: high-confidence nodes blend rather than replace."""
        with self._lock:
            existing = self._nodes.get(ckey)
            if existing and existing.confidence >= self.cfg.ewc_threshold:
                # Protective blending: reinforce without overwriting
                existing.confidence = min(0.98, existing.confidence + 0.02)
                existing.stability_h = min(
                    getattr(existing, "stability_h", 8760.0) * 1.08, 8760.0 * 10
                )
                existing.frequency += 1
                existing.updated_at = now_ts()
                existing.members.update(node.members)
                self.store.upsert_semantic(ckey, existing)
            else:
                self._nodes[ckey] = node
                self.store.upsert_semantic(ckey, node)

        METRICS.set_memory("semantic", len(self._nodes))

    def search(
        self, query_emb: List[float], k: int = 5, min_score: float = 0.20
    ) -> List[Tuple[SemanticNode, float]]:
        with self._lock:
            snapshot = list(self._nodes.values())

        scored: List[Tuple[float, SemanticNode]] = []
        for node in snapshot:
            if not node.embedding:
                continue
            sim = VectorOps.cosine(node.embedding, query_emb)
            elapsed_h = (now_ts() - node.updated_at) / 3600.0
            ret = SRS.retrievability(
                getattr(node, "stability_h", 8760.0), elapsed_h
            )
            score = sim * 0.65 + ret * 0.20 + node.confidence * 0.15
            if score >= min_score:
                scored.append((score, node))

        scored.sort(key=lambda x: -x[0])
        return [(n, s) for s, n in scored[:k]]

    def add_hebbian_link(self, uid_a: str, uid_b: str, weight: float) -> None:
        """Add bidirectional semantic link between two concept nodes."""
        with self._lock:
            for ckey, n in self._nodes.items():
                if n.uid == uid_a:
                    n.metadata.setdefault("links", {})[uid_b] = min(
                        1.0, n.metadata["links"].get(uid_b, 0.0) + weight
                    )
                elif n.uid == uid_b:
                    n.metadata.setdefault("links", {})[uid_a] = min(
                        1.0, n.metadata["links"].get(uid_a, 0.0) + weight
                    )

    @property
    def nodes(self) -> Dict[str, SemanticNode]:
        return self._nodes

    def __len__(self) -> int:
        return len(self._nodes)


# ---------------------------------------------------------------------------
# Dream Consolidator  (background NREM/REM/pruning thread)
# ---------------------------------------------------------------------------

class DreamConsolidator:
    """
    Background consolidation thread modeled on sleep-phase memory processing:

    NREM phase (slow-wave):
      Episodic → Semantic promotion via CLS theory (McClelland 1995).
      High-importance episodic traces abstracted into semantic nodes.

    REM phase (associative):
      Random Hebbian linking of semantically nearby concepts.
      Models creative insight via unexpected co-activation.

    Pruning (synaptic homeostasis, Tononi 2003):
      Remove weak episodic engrams.
      Decay link weights for rarely co-activated nodes.
    """

    def __init__(
        self,
        cfg: NexusSettings,
        episodic: EpisodicStore,
        semantic: SemanticStore,
        embedder: EmbeddingProvider,
    ) -> None:
        self.cfg = cfg
        self.episodic = episodic
        self.semantic = semantic
        self.embedder = embedder
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cycles = 0
        self._last_stats: Dict[str, Any] = {}

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="DreamConsolidator",
        )
        self._thread.start()
        LOG.info("dream_consolidator.started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=6.0)

    def _loop(self) -> None:
        while not self._stop.wait(timeout=self.cfg.consolidation_interval_s):
            try:
                self._cycle()
            except Exception as exc:
                LOG.exception("dream_consolidator.cycle_error", error=str(exc))
                METRICS.inc_error("consolidation")

    def run_now(self) -> Dict[str, Any]:
        """Synchronous consolidation (for CLI /sleep command)."""
        return self._cycle()

    def _cycle(self) -> Dict[str, Any]:
        self._cycles += 1
        t0 = time.perf_counter()
        stats: Dict[str, Any] = {}

        # Phase 1: NREM — episodic → semantic
        nrem_count = self._nrem_phase()
        stats["nrem_promoted"] = nrem_count

        # Phase 2: REM — associative Hebbian linking
        rem_count = self._rem_phase()
        stats["rem_links"] = rem_count

        # Phase 3: Pruning
        pruned = self.episodic.prune()
        stats["pruned"] = pruned

        elapsed = time.perf_counter() - t0
        stats["elapsed_s"] = round(elapsed, 3)
        stats["cycle"] = self._cycles
        self._last_stats = stats

        LOG.info(
            "dream_consolidator.cycle_complete",
            **stats,
        )
        return stats

    def _nrem_phase(self) -> int:
        """Transfer high-importance episodic memories to semantic store."""
        threshold = self.cfg.ewc_threshold * 0.80
        candidates = [
            e
            for e in self.episodic.all_engrams()
            if not e.consolidated
            and e.importance >= threshold
            and e.retrievals >= self.cfg.semantic_promotion_threshold
        ]
        # Sort by importance × retrieval_count
        candidates.sort(
            key=lambda e: e.importance * e.retrievals, reverse=True
        )
        promoted = 0
        for e in candidates[: self.cfg.consolidation_batch_size]:
            if not e.embedding:
                continue
            # Average embedding with any existing semantic node
            concept = " ".join(e.content.split()[:5]).lower().strip(".,!?")
            ckey = hashlib.sha256(e.content[:100].encode()).hexdigest()[:14]

            node = SemanticNode(
                uid=uuid.uuid4().hex[:14],
                name=concept[:80],
                summary=e.content[:500],
                embedding=e.embedding[:],
                members={e.uid},
                confidence=min(0.95, 0.45 + 0.08 * e.retrievals),
                frequency=e.retrievals,
                updated_at=now_ts(),
                metadata={"source_tier": "episodic", "abstract_concept": concept},
            )
            node.stability_h = e.stability_h * 5.0  # semantic more stable

            self.semantic.upsert(ckey, node)
            e.consolidated = True
            e.meta["abstract_concept"] = concept
            self.episodic.save(e)
            promoted += 1

        return promoted

    def _rem_phase(self) -> int:
        """Random associative Hebbian linking of semantic concepts."""
        import random

        sem_nodes = list(self.semantic.nodes.values())
        if len(sem_nodes) < 4:
            return 0

        sample_size = min(24, len(sem_nodes))
        sample = random.sample(sem_nodes, sample_size)
        links = 0
        rem_prob = self.cfg.rem_link_probability

        for i, na in enumerate(sample[:-1]):
            if not na.embedding:
                continue
            for nb in sample[i + 1 :]:
                if not nb.embedding:
                    continue
                sim = VectorOps.cosine(na.embedding, nb.embedding)
                if sim > 0.70:
                    self.semantic.add_hebbian_link(na.uid, nb.uid, sim)
                    links += 1
                elif sim > 0.40 and random.random() < rem_prob:
                    # Creative REM link: connects moderately similar concepts
                    self.semantic.add_hebbian_link(na.uid, nb.uid, sim * 0.55)
                    links += 1

        return links

    def status(self) -> Dict[str, Any]:
        return {
            "cycles": self._cycles,
            "running": self._thread is not None and self._thread.is_alive(),
            "last_stats": self._last_stats,
        }


# ---------------------------------------------------------------------------
# Engram Lattice  (unified multi-tier interface)
# ---------------------------------------------------------------------------

class EngramLattice:
    """
    Unified interface to all memory tiers.
    Orchestrates: encoding, multi-tier retrieval, Hebbian strengthening,
    tier promotion, and consolidation scheduling.

    Hebbian plasticity: co-retrieved memories form stronger associations.
    """

    def __init__(
        self,
        cfg: NexusSettings,
        store: SQLiteStateStore,
        embedder: EmbeddingProvider,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.embedder = embedder

        self.sensory = SensoryBuffer(cfg)
        self.working = WorkingMemory(cfg)
        self.episodic = EpisodicStore(cfg, store)
        self.semantic = SemanticStore(cfg, store)
        self.dreamer = DreamConsolidator(cfg, self.episodic, self.semantic, embedder)

        self._total_encoded = 0
        self._hebbian = HebbianPlasticity()
        self._recent_kws: deque[str] = deque(maxlen=200)
        self._lock = threading.RLock()

    def start_consolidation(self) -> None:
        self.dreamer.start()

    def stop_consolidation(self) -> None:
        self.dreamer.stop()

    async def encode(
        self,
        content: str,
        tier: MemoryTier = MemoryTier.WORKING,
        keywords: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
        importance: float = 0.50,
        intent: str = "unknown",
        valence: float = 0.0,
        arousal: float = 0.4,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Engram:
        """
        Encode experience into memory.
        Tier routing: SENSORY → WORKING → EPISODIC → SEMANTIC.
        Importance is ACh-gated (learning gate modulates encoding strength).
        """
        embedding = await self.embedder.embed(content)
        kws = keywords or []
        ents = entities or []

        for kw in kws[:8]:
            self._recent_kws.append(kw)

        engram = Engram(
            uid=uuid.uuid4().hex[:14],
            content=content,
            tier=tier,
            created_at=now_ts(),
            updated_at=now_ts(),
            retrievals=0,
            stability_h=self.cfg.initial_stability_h,
            importance=min(1.0, max(0.0, importance)),
            valence_enc=valence,
            arousal_enc=arousal,
            keywords=kws[:12],
            embedding=embedding,
            entities=ents[:8],
            intent=intent,
            links={},
            consolidated=False,
            meta=meta or {},
        )

        if tier == MemoryTier.SENSORY:
            self.sensory.push(engram)

        elif tier == MemoryTier.WORKING:
            # Attention weight from arousal (high arousal → high attention)
            attention = 0.4 + arousal * 0.4
            evicted = self.working.put(engram, attention=attention)
            # Evicted working-memory items promoted to episodic if important
            if evicted and evicted.importance > 0.28:
                evicted.tier = MemoryTier.EPISODIC
                self.episodic.apply_interference(evicted)
                self.episodic.save(evicted)

        elif tier == MemoryTier.EPISODIC:
            self.episodic.apply_interference(engram)
            self.episodic.save(engram)

        elif tier == MemoryTier.SEMANTIC:
            ckey = hashlib.sha256(content[:100].encode()).hexdigest()[:14]
            node = SemanticNode(
                uid=engram.uid,
                name=content[:80],
                summary=content[:500],
                embedding=embedding,
                members={engram.uid},
                confidence=min(0.95, importance),
                frequency=1,
                updated_at=now_ts(),
                metadata={},
            )
            node.stability_h = self.cfg.initial_stability_h * 10.0
            self.semantic.upsert(ckey, node)

        self._total_encoded += 1
        return engram

    async def retrieve(
        self,
        query: str,
        k: int = 8,
        tiers: Optional[List[MemoryTier]] = None,
        goal_embedding: Optional[List[float]] = None,
        current_valence: float = 0.0,
    ) -> List[Tuple[Engram, float, MemoryTier]]:
        """
        Multi-tier retrieval with goal-biased query embedding.
        Returns (engram, score, tier) sorted by score descending.
        After retrieval, applies Hebbian strengthening to co-retrieved pairs.
        """
        tiers = tiers or [MemoryTier.WORKING, MemoryTier.EPISODIC, MemoryTier.SEMANTIC]
        q_emb = await self.embedder.embed(query)

        # Goal bias: blend query toward active goal embedding
        if goal_embedding and len(goal_embedding) == len(q_emb):
            q_emb = VectorOps.normalize(
                VectorOps.lerp(q_emb, goal_embedding, self.cfg.goal_bias_weight)
            )

        kws = set(query.lower().split())
        results: List[Tuple[Engram, float, MemoryTier]] = []

        if MemoryTier.WORKING in tiers:
            for e, s in self.working.search(q_emb, k=3):
                results.append((e, s, MemoryTier.WORKING))

        if MemoryTier.EPISODIC in tiers:
            ep_hits = self.episodic.retrieve(
                q_emb, current_valence, k=k, keyword_boost=kws
            )
            for e, s in ep_hits:
                results.append((e, s, MemoryTier.EPISODIC))

        if MemoryTier.SEMANTIC in tiers:
            sem_hits = self.semantic.search(q_emb, k=max(2, k // 2))
            for node, s in sem_hits:
                # Wrap SemanticNode as Engram-like for uniform interface
                e = Engram(
                    uid=node.uid,
                    content=f"{node.name}: {node.summary[:300]}",
                    tier=MemoryTier.SEMANTIC,
                    created_at=node.updated_at,
                    updated_at=node.updated_at,
                    retrievals=node.frequency,
                    stability_h=getattr(node, "stability_h", 8760.0),
                    importance=node.confidence,
                    embedding=node.embedding,
                    keywords=list(node.members)[:5],
                    intent="semantic",
                )
                results.append((e, s, MemoryTier.SEMANTIC))

        results.sort(key=lambda x: -x[1])

        # Hebbian strengthening: top-2 co-retrieved items form stronger links
        if len(results) >= 2:
            e1, _, _ = results[0]
            e2, _, _ = results[1]
            if e1.embedding and e2.embedding:
                delta_t = abs(e1.updated_at - e2.updated_at)
                co_score = self._hebbian.co_activation(
                    e1.embedding,
                    e2.embedding,
                    delta_t,
                    self.cfg.plasticity_window_s,
                )
                if co_score > 0.30:
                    new_w = self._hebbian.weight_update(
                        float(e1.links.get(e2.uid, 0.0)),
                        co_score,
                        co_score,
                        self.cfg.hebbian_lr,
                    )
                    e1.links[e2.uid] = new_w
                    e2.links[e1.uid] = new_w

        return results[:k]

    def stats(self) -> Dict[str, Any]:
        return {
            "sensory": len(self.sensory),
            "working": len(self.working),
            "episodic": len(self.episodic),
            "semantic": len(self.semantic),
            "total_encoded": self._total_encoded,
            "dreamer": self.dreamer.status(),
        }
