"""
NEXUS Cognitive Agent Framework — Configuration & Core Types
============================================================
Unified configuration system merging the best of OMEGA and SYNAPTIC,
with production-grade defaults and environment-variable overrides.
"""

from __future__ import annotations

import enum
import uuid
import time
import math
import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

# ── Optional pydantic_settings ────────────────────────────────────────────────
try:
    from pydantic import BaseModel, Field, ConfigDict
    from pydantic_settings import BaseSettings, SettingsConfigDict
    HAS_PYDANTIC = True
    HAS_PYDANTIC_SETTINGS = True
except ImportError:
    try:
        from pydantic import BaseModel, Field, ConfigDict  # type: ignore
        HAS_PYDANTIC = True
    except ImportError:
        HAS_PYDANTIC = False  # type: ignore
    HAS_PYDANTIC_SETTINGS = False

# ── Optional dependency flags (centralised) ──────────────────────────────────
def _has(pkg: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(pkg) is not None

HAS_NUMPY       = _has("numpy")
HAS_SCIPY       = _has("scipy")
HAS_NX          = _has("networkx")
HAS_ST          = _has("sentence_transformers")
HAS_QDRANT      = _has("qdrant_client")
HAS_LANCEDB     = _has("lancedb")
HAS_LITELLM     = _has("litellm")
HAS_ANTHROPIC   = _has("anthropic")
HAS_OPENAI      = _has("openai")
HAS_FASTAPI     = _has("fastapi")
HAS_UVICORN     = _has("uvicorn")
HAS_PROMETHEUS  = _has("prometheus_client")
HAS_STRUCTLOG   = _has("structlog")
HAS_TENACITY    = _has("tenacity")
HAS_PSUTIL      = _has("psutil")
HAS_ORJSON      = _has("orjson")
HAS_FAISS       = _has("faiss")
HAS_RICH        = _has("rich")

EPS = 1e-12


# ══════════════════════════════════════════════════════════════════════════════
# §0.5  TIME UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def now_ts() -> float:
    """Current time as a Unix timestamp float."""
    return time.time()


def utc_iso() -> str:
    """Current time as an ISO-8601 UTC string."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ══════════════════════════════════════════════════════════════════════════════
# §1  ENUMERATIONS
# ══════════════════════════════════════════════════════════════════════════════

class MemoryTier(str, enum.Enum):
    SENSORY    = "sensory"
    WORKING    = "working"
    EPISODIC   = "episodic"
    SEMANTIC   = "semantic"
    PROCEDURAL = "procedural"

class CognitionMode(str, enum.Enum):
    REFLEX  = "reflex"    # < 80ms, pattern-matched, no LLM call
    FAST    = "fast"      # single LLM call, 1–3 paragraph response
    DEEP    = "deep"      # multi-step CoT, deeper reasoning model
    TREE    = "tree"      # beam search over reasoning branches
    SOMATIC = "somatic"   # empathy-first, emotional priority
    CAUSAL  = "causal"    # causal reasoning, root-cause analysis

class Intent(str, enum.Enum):
    GREETING   = "greeting"
    FACTUAL    = "factual"
    ANALYTICAL = "analytical"
    CREATIVE   = "creative"
    EMOTIONAL  = "emotional"
    RECALL     = "recall"
    META       = "meta"
    PLANNING   = "planning"
    PROCEDURAL = "procedural"
    CAUSAL     = "causal"
    UNKNOWN    = "unknown"

class Modality(str, enum.Enum):
    TEXT  = "text"
    CODE  = "code"
    TOOL  = "tool"
    META  = "meta"

class AffectLabel(str, enum.Enum):
    ELATED   = "elated"
    ENGAGED  = "engaged"
    CONTENT  = "content"
    FOCUSED  = "focused"
    NEUTRAL  = "neutral"
    PENSIVE  = "pensive"
    STRESSED = "stressed"


# ══════════════════════════════════════════════════════════════════════════════
# §2  CORE DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class NeuroState:
    """Four-neuromodulator state machine (DA/NE/ACh/5-HT)."""
    da:   float = 0.50   # Dopamine   — reward, motivation, salience
    ne:   float = 0.45   # NE         — alertness, gain, stress
    ach:  float = 0.50   # ACh        — learning gate, attention
    sert: float = 0.55   # Serotonin  — patience, mood floor

    @property
    def exploration_rate(self) -> float:
        return min(1.0, max(0.0, self.da * 0.6 + (1 - self.ne) * 0.4))

    @property
    def learning_gate(self) -> float:
        return 0.5 + self.ach * 0.5

    @property
    def stress_level(self) -> float:
        return min(1.0, max(0.0, self.ne * 0.6 + max(0.0, 0.6 - self.sert) * 0.4))

    @property
    def creativity(self) -> float:
        return max(0.05, min(1.0, 0.4 * self.da + 0.3 * self.ach + 0.3 * (1.0 - self.ne)))

    @property
    def affect(self) -> AffectLabel:
        v = self.da - 0.5 + self.sert * 0.2 - self.ne * 0.1
        a = self.ne * 0.5 + self.da * 0.3
        if v > 0.25 and a > 0.55:  return AffectLabel.ELATED
        if v > 0.20 and a > 0.40:  return AffectLabel.ENGAGED
        if v > 0.15:               return AffectLabel.CONTENT
        if a > 0.65:               return AffectLabel.FOCUSED
        if v < -0.20 and a < 0.45: return AffectLabel.PENSIVE
        if v < -0.15 and a > 0.55: return AffectLabel.STRESSED
        return AffectLabel.NEUTRAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "da": round(self.da, 3), "ne": round(self.ne, 3),
            "ach": round(self.ach, 3), "sert": round(self.sert, 3),
            "exploration": round(self.exploration_rate, 3),
            "learning_gate": round(self.learning_gate, 3),
            "stress": round(self.stress_level, 3),
            "creativity": round(self.creativity, 3),
            "affect": self.affect.value,
        }


@dataclass
class PerceptualFeatures:
    """Rich perceptual analysis extracted from raw text."""
    text: str
    tokens: List[str]
    keywords: List[str]
    entities: List[str]
    intent: Intent
    valence: float       # −1..+1
    arousal: float       # 0..1
    complexity: float    # 0..1
    novelty: float       # 0..1
    salience: float      # composite 0..1
    urgency: float       # 0..1
    question_depth: int  # 0–3
    hedge_count: int
    negation_count: int
    sentence_count: int
    causal_markers: int
    lexical_density: float = 0.0   # unique_tokens / total_tokens


@dataclass
class Engram:
    """Atomic persistent memory unit — the physical trace of an experience."""
    uid: str            = field(default_factory=lambda: uuid.uuid4().hex[:14])
    content: str        = ""
    tier: MemoryTier    = MemoryTier.EPISODIC
    modality: Modality  = Modality.TEXT

    # Temporal
    created_at: float   = field(default_factory=time.time)
    updated_at: float   = field(default_factory=time.time)

    # SRS (Spaced-Repetition-System) model — stability-based forgetting
    stability_h: float  = 12.0   # hours until 50 % retrievability
    importance: float   = 0.50   # 0–1; EWC-protected above threshold
    retrievals: int     = 0

    # Emotional encoding context
    valence_enc: float  = 0.0
    arousal_enc: float  = 0.4

    # Semantic content
    keywords: List[str]        = field(default_factory=list)
    embedding: List[float]     = field(default_factory=list)
    entities: List[str]        = field(default_factory=list)
    intent: str                = "unknown"
    tags: List[str]            = field(default_factory=list)

    # Consolidation
    consolidated: bool         = False
    source_turn: int           = 0
    meta: Dict[str, Any]       = field(default_factory=dict)

    # Hebbian links: {uid: weight}
    links: Dict[str, float]    = field(default_factory=dict)

    def age_hours(self) -> float:
        return (time.time() - self.created_at) / 3600.0

    def hours_since_retrieval(self) -> float:
        return (time.time() - self.updated_at) / 3600.0

    def retrievability(self) -> float:
        """SRS forgetting curve: R = exp(−Δt / stability)."""
        elapsed = self.hours_since_retrieval()
        return math.exp(-elapsed / max(self.stability_h, EPS))

    def touch(self, quality: float = 0.7) -> None:
        """Retrieval event: strengthen stability (spacing effect)."""
        self.updated_at = time.time()
        self.retrievals += 1
        # SM-2-style growth (Wozniak 1994)
        difficulty = max(0.0, min(1.0, 1.0 - quality))
        spacing = 1.0 + 0.12 * math.log(max(1.0, self.stability_h))
        growth = 2.5 * (1 - difficulty) * spacing
        self.stability_h = min(8760.0, self.stability_h * max(1.1, growth))
        self.importance = min(1.0, self.importance + 0.015 * quality)

    def add_link(self, uid: str, weight: float) -> None:
        self.links[uid] = min(1.0, max(0.0, weight))

    def decay_links(self, rate: float = 0.002) -> None:
        self.links = {u: w - rate for u, w in self.links.items() if w - rate > 0.04}

    def retrieval_score(self, q_emb: List[float], current_valence: float) -> float:
        """Multi-factor retrieval combining semantics, SRS, recency, mood."""
        from nexus.core.math_utils import VectorOps
        sem = VectorOps.cosine(self.embedding, q_emb) if self.embedding else 0.25
        ret = self.retrievability()
        rec = math.exp(-self.hours_since_retrieval() / 48.0)
        # Mood-congruent bias (Bower 1981)
        mood_bias = 1.0 + current_valence * self.valence_enc * 0.25
        score = (sem * 0.45 + ret * 0.25 + rec * 0.15 + self.importance * 0.10
                 + min(self.retrievals * 0.02, 0.05))
        return max(0.0, min(1.0, score * max(0.4, mood_bias)))


@dataclass
class SemanticNode:
    """Consolidated concept node extracted from episodic memories."""
    uid: str
    name: str
    summary: str
    embedding: List[float]
    members: set               # engram uids
    confidence: float
    frequency: int
    updated_at: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Goal:
    """Autonomic goal with priority, progress, and decay."""
    uid: str          = field(default_factory=lambda: uuid.uuid4().hex[:8])
    description: str  = ""
    priority: float   = 0.5
    progress: float   = 0.0
    active: bool      = True
    horizon: str      = "immediate"   # immediate | session | lifetime
    created_at: float = field(default_factory=time.time)
    embedding: List[float] = field(default_factory=list)
    notes: List[str]  = field(default_factory=list)
    expected_reward: float = 0.5

    def decay(self, rate: float = 0.96) -> None:
        self.priority = max(0.03, self.priority * rate)

    def age_hours(self) -> float:
        return (time.time() - self.created_at) / 3600.0


@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]


@dataclass
class PlanCandidate:
    mode: CognitionMode
    score: float
    rationale: str
    tool_calls: List[ToolCall] = field(default_factory=list)


@dataclass
class TurnResult:
    """Structured output of one full cognitive cycle."""
    turn: int
    response: str
    mode: str
    confidence: float
    actual_quality: float
    surprise: float
    free_energy: float
    memory_hits: int
    tool_calls: List[Dict[str, Any]]
    latency_ms: float
    affect: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# §3  SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

if HAS_PYDANTIC_SETTINGS:
    class NexusSettings(BaseSettings):
        model_config = SettingsConfigDict(
            env_prefix="NEXUS_",
            env_file=".env",
            extra="ignore",
            case_sensitive=False,
        )

        # Identity
        agent_name: str          = "NEXUS"
        agent_version: str       = "1.0.0"
        app_env: str             = "dev"
        log_level: str           = "INFO"
        data_dir: str            = str(Path.home() / ".nexus_agent")

        # Embedding
        embed_backend: str       = "auto"   # auto|sentence_transformers|hash
        embed_model: str         = "all-MiniLM-L6-v2"
        embed_dim: int           = 384
        embed_cache_size: int    = 15_000

        # Reranker
        enable_reranker: bool    = True
        reranker_model: str      = "cross-encoder/ms-marco-MiniLM-L-6-v2"

        # Vector store
        vector_backend: str      = "auto"   # auto|qdrant|lancedb|memory
        qdrant_url: Optional[str]           = None
        qdrant_api_key: Optional[str]       = None
        qdrant_collection: str   = "nexus_episodic"
        qdrant_local_path: Optional[str]    = None
        lancedb_path: Optional[str]         = None

        # LLM
        llm_backend: str         = "auto"   # auto|litellm|anthropic|openai|mock
        llm_model: str           = "auto"
        llm_reasoning_model: str = "auto"
        anthropic_model: str     = "claude-sonnet-4-6"
        anthropic_reasoning_model: str = "claude-opus-4-6"
        openai_model: str        = "gpt-4o-mini"
        openai_reasoning_model: str = "gpt-4o"
        litellm_model: str       = "openai/gpt-4o-mini"
        litellm_reasoning_model: str = "anthropic/claude-sonnet-4-6"
        llm_temperature: float   = 0.65
        llm_max_tokens: int      = 2048
        llm_timeout_s: float     = 90.0

        # Memory architecture
        sensory_capacity: int    = 6
        sensory_ttl_ms: float    = 3500.0
        working_capacity: int    = 4        # Cowan 2001 (not Miller's 7±2)
        working_ttl_s: float     = 25.0
        episodic_capacity: int   = 80_000
        semantic_capacity: int   = 20_000
        episodic_top_k: int      = 10
        semantic_top_k: int      = 5
        retrieval_threshold: float = 0.15
        ewc_threshold: float     = 0.65     # EWC protection above this
        initial_stability_h: float = 12.0
        spacing_coeff: float     = 0.12

        # Hebbian plasticity
        hebbian_lr: float        = 0.06
        plasticity_window_s: float = 120.0

        # Neurochemistry (tonic baselines)
        da_tonic: float          = 0.50
        ne_tonic: float          = 0.45
        ach_tonic: float         = 0.50
        sert_tonic: float        = 0.55
        da_reuptake: float       = 0.04
        ne_reuptake: float       = 0.06
        ach_reuptake: float      = 0.05
        sert_reuptake: float     = 0.03

        # Predictive cortex
        prediction_lr: float     = 0.08
        prediction_momentum: float = 0.88
        surprise_archive_thresh: float = 0.68
        surprise_deep_thresh: float    = 0.42

        # Metacognition / routing thresholds
        reflex_confidence_min: float = 0.72
        fast_confidence_min: float   = 0.56
        deep_confidence_min: float   = 0.40
        optimal_arousal: float       = 0.58
        arousal_width: float         = 0.28
        goal_bias_weight: float      = 0.22

        # Planner
        planner_beam_width: int  = 4
        planner_max_tool_calls: int = 3

        # Consolidation (background)
        consolidation_interval_s: float = 45.0
        consolidation_batch: int = 24
        homeostasis_prune_threshold: float = 0.12
        semantic_promotion_min_retrievals: int = 2
        semantic_promotion_min_importance: float = 0.55

        # Calibration
        calibration_window: int  = 80

        # Identity / narrative
        chapter_size: int        = 12
        drift_threshold: float   = 0.25
        growth_log_maxsize: int  = 200

        # API
        api_host: str            = "0.0.0.0"
        api_port: int            = 8091
        api_key: Optional[str]   = None
        enable_cors: bool        = True

        # Ops
        enable_metrics: bool     = True
        save_every_n_turns: int  = 10

        # Extended fields — used by cognition/memory modules
        temporal_window_turns: int   = 10
        temporal_decay_lambda: float = 0.35
        goal_completion_threshold: float = 0.85
        goal_priority_decay: float   = 0.012
        max_active_goals: int        = 7
        interference_penalty: float  = 0.08
        interference_radius: float   = 0.22
        rem_link_probability: float  = 0.18
        semantic_promotion_threshold: float = 0.55
        da_reward_burst: float       = 0.18
        ne_novelty_burst: float      = 0.14
        ach_attention_burst: float   = 0.12
        surprise_threshold_archive: float = 0.68
        surprise_threshold_deep: float    = 0.42
        deep_reasoning_threshold: float   = 0.40

        @property
        def sqlite_path(self) -> str:
            return str(Path(self.data_dir) / "nexus.db")

        @property
        def lancedb_dir(self) -> str:
            return self.lancedb_path or str(Path(self.data_dir) / "lancedb")

        @property
        def working_memory_size(self) -> int:
            return self.working_capacity

        @property
        def consolidation_batch_size(self) -> int:
            return self.consolidation_batch


else:
    from dataclasses import dataclass as _dc

    @_dc
    class NexusSettings:  # type: ignore
        agent_name: str          = "NEXUS"
        agent_version: str       = "1.0.0"
        app_env: str             = "dev"
        log_level: str           = "INFO"
        data_dir: str            = str(Path.home() / ".nexus_agent")
        embed_backend: str       = "auto"
        embed_model: str         = "all-MiniLM-L6-v2"
        embed_dim: int           = 384
        embed_cache_size: int    = 15_000
        enable_reranker: bool    = True
        reranker_model: str      = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        vector_backend: str      = "auto"
        qdrant_url: Optional[str] = None
        qdrant_api_key: Optional[str] = None
        qdrant_collection: str   = "nexus_episodic"
        qdrant_local_path: Optional[str] = None
        lancedb_path: Optional[str] = None
        llm_backend: str         = "auto"
        llm_model: str           = "auto"
        llm_reasoning_model: str = "auto"
        anthropic_model: str     = "claude-sonnet-4-6"
        anthropic_reasoning_model: str = "claude-opus-4-6"
        openai_model: str        = "gpt-4o-mini"
        openai_reasoning_model: str = "gpt-4o"
        litellm_model: str       = "openai/gpt-4o-mini"
        litellm_reasoning_model: str = "anthropic/claude-sonnet-4-6"
        llm_temperature: float   = 0.65
        llm_max_tokens: int      = 2048
        llm_timeout_s: float     = 90.0
        sensory_capacity: int    = 6
        sensory_ttl_ms: float    = 3500.0
        working_capacity: int    = 4
        working_ttl_s: float     = 25.0
        episodic_capacity: int   = 80_000
        semantic_capacity: int   = 20_000
        episodic_top_k: int      = 10
        semantic_top_k: int      = 5
        retrieval_threshold: float = 0.15
        ewc_threshold: float     = 0.65
        initial_stability_h: float = 12.0
        spacing_coeff: float     = 0.12
        hebbian_lr: float        = 0.06
        plasticity_window_s: float = 120.0
        da_tonic: float          = 0.50
        ne_tonic: float          = 0.45
        ach_tonic: float         = 0.50
        sert_tonic: float        = 0.55
        da_reuptake: float       = 0.04
        ne_reuptake: float       = 0.06
        ach_reuptake: float      = 0.05
        sert_reuptake: float     = 0.03
        prediction_lr: float     = 0.08
        prediction_momentum: float = 0.88
        surprise_archive_thresh: float = 0.68
        surprise_deep_thresh: float    = 0.42
        reflex_confidence_min: float = 0.72
        fast_confidence_min: float   = 0.56
        deep_confidence_min: float   = 0.40
        optimal_arousal: float       = 0.58
        arousal_width: float         = 0.28
        goal_bias_weight: float      = 0.22
        planner_beam_width: int  = 4
        planner_max_tool_calls: int = 3
        consolidation_interval_s: float = 45.0
        consolidation_batch: int = 24
        homeostasis_prune_threshold: float = 0.12
        semantic_promotion_min_retrievals: int = 2
        semantic_promotion_min_importance: float = 0.55
        calibration_window: int  = 80
        chapter_size: int        = 12
        drift_threshold: float   = 0.25
        growth_log_maxsize: int  = 200
        api_host: str            = "0.0.0.0"
        api_port: int            = 8091
        api_key: Optional[str]   = None
        enable_cors: bool        = True
        enable_metrics: bool     = True
        save_every_n_turns: int  = 10

        # Extended fields — used by cognition/memory modules
        temporal_window_turns: int   = 10
        temporal_decay_lambda: float = 0.35
        goal_completion_threshold: float = 0.85
        goal_priority_decay: float   = 0.012
        max_active_goals: int        = 7
        interference_penalty: float  = 0.08
        interference_radius: float   = 0.22
        rem_link_probability: float  = 0.18
        semantic_promotion_threshold: float = 0.55
        da_reward_burst: float       = 0.18
        ne_novelty_burst: float      = 0.14
        ach_attention_burst: float   = 0.12
        surprise_threshold_archive: float = 0.68
        surprise_threshold_deep: float    = 0.42
        deep_reasoning_threshold: float   = 0.40

        @property
        def sqlite_path(self) -> str:
            return str(Path(self.data_dir) / "nexus.db")

        @property
        def lancedb_dir(self) -> str:
            return self.lancedb_path or str(Path(self.data_dir) / "lancedb")

        @property
        def working_memory_size(self) -> int:
            return self.working_capacity

        @property
        def consolidation_batch_size(self) -> int:
            return self.consolidation_batch

