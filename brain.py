"""
nexus/brain.py
===============
NexusBrain: the unified cognitive orchestrator for NEXUS.

Information resonance loop (bidirectional belief updating, not a pipeline):

  INPUT
    → PerceptionEngine         (feature extraction)
    → NeurochemicalBus         (neuromodulator integration)
    → PredictiveCortex         (surprise + belief update)
    → TemporalBinder           (temporal context fusion)
    → EngramLattice.retrieve   (multi-tier memory + Hebbian update)
    → MetacognitiveLoop        (routing decision + escalation)
    → AutonomicPlanner         (goal inference + tick)
    → IdentitySubstrate        (self-context block)
    → PromptArchitect          (compose system + user prompts)
    → ResonanceGateway         (LLM generation, possibly tree reasoning)
    → ProcessRewardModel       (quality scoring)
    → MetacognitiveLoop        (calibration update)
    → EngramLattice.encode     (multi-tier memory encoding)
    → NeurochemicalBus         (DA reward signal)
    → IdentitySubstrate        (self-model update)
    → TurnResult
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from nexus.cognition.metacognition import MetacognitiveLoop, ProcessRewardModel
from nexus.cognition.neurochemistry import NeurochemicalBus
from nexus.cognition.perception import PerceptionEngine
from nexus.cognition.planner import AutonomicPlanner, IdentitySubstrate, TemporalBinder
from nexus.cognition.predictive_cortex import PredictiveCortex
from nexus.core.config import (
    CognitionMode,
    Goal,
    Intent,
    MemoryTier,
    NexusSettings,
    PerceptualFeatures,
    TurnResult,
    now_ts,
    utc_iso,
)
from nexus.core.math_utils import VectorOps
from nexus.core.observability import LOG, METRICS
from nexus.llm.gateway import ResonanceGateway
from nexus.llm.prompts import PromptArchitect
from nexus.memory.embeddings import build_embedder, build_reranker, build_vector_store
from nexus.memory.store import EngramLattice, SQLiteStateStore
from nexus.tools.registry import ToolRegistry, build_default_registry


class NexusBrain:
    """
    Full cognitive agent. Initialize once; call .think() for each turn.
    Thread-safe: the async lock ensures sequential turn processing.
    """

    VERSION = "1.0.0"

    def __init__(self, cfg: Optional[NexusSettings] = None) -> None:
        self.cfg = cfg or NexusSettings()
        self._turn = 0
        self._started_at = now_ts()
        self._lock = asyncio.Lock()

        LOG.info("nexus.init_start", version=self.VERSION)

        # ── Ensure data directories exist ─────────────────────────────
        Path(self.cfg.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cfg.sqlite_path).parent.mkdir(parents=True, exist_ok=True)

        # ── Infrastructure ────────────────────────────────────────────
        self.embedder = build_embedder(self.cfg)
        self.reranker = build_reranker(self.cfg)
        self.store = SQLiteStateStore(self.cfg.sqlite_path)

        # ── Cognitive modules ─────────────────────────────────────────
        self.perception = PerceptionEngine()
        self.neuro = NeurochemicalBus(self.cfg)
        self.cortex = PredictiveCortex(self.cfg)
        self.metacog = MetacognitiveLoop(self.cfg)
        self.planner = AutonomicPlanner(self.cfg)
        self.identity = IdentitySubstrate(self.cfg)
        self.binder = TemporalBinder(self.cfg)
        self.gateway = ResonanceGateway(self.cfg)
        self.prm = ProcessRewardModel()

        # ── Memory (async init deferred to .initialize()) ─────────────
        self.lattice: Optional[EngramLattice] = None
        self._vector_store = None

        # ── Tools ─────────────────────────────────────────────────────
        self.tools = build_default_registry()

        # ── Causal registry (lightweight inline) ─────────────────────
        from nexus.core.math_utils import CausalRegistry
        self.causal = CausalRegistry()

    async def initialize(self) -> None:
        """Async initialization: vector store, memory lattice, tools."""
        self._vector_store = await build_vector_store(self.cfg)
        self.lattice = EngramLattice(self.cfg, self.store, self.embedder)
        self.lattice.start_consolidation()

        # Wire memory_search and system_status tools to live objects
        lattice = self.lattice

        async def memory_search_tool(query: str, k: int = 5) -> List[Dict[str, Any]]:
            hits = await lattice.retrieve(query, k=k)
            return [
                {
                    "uid": e.uid,
                    "score": round(s, 4),
                    "content": e.content[:200],
                    "tier": t.value,
                    "importance": round(e.importance, 4),
                }
                for e, s, t in hits
            ]

        async def system_status_tool() -> Dict[str, Any]:
            return self.status()

        self.tools.register(
            "memory_search",
            memory_search_tool,
            description="Search episodic and semantic memory for relevant context.",
            parameters={
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 5},
            },
        )
        self.tools.register(
            "system_status",
            system_status_tool,
            description="Return current NEXUS system status.",
            parameters={},
        )

        # Rebuild vector index from persisted engrams
        await self._rebuild_vector_index()

        LOG.info(
            "nexus.initialized",
            llm_provider=self.gateway._provider,
            embed_backend=self.embedder.__class__.__name__,
            episodic_count=len(self.lattice.episodic),
            semantic_count=len(self.lattice.semantic),
        )

    async def _rebuild_vector_index(self) -> None:
        """Re-index all loaded engrams into the vector store."""
        if not self.lattice or not self._vector_store:
            return
        for e in self.lattice.episodic.all_engrams():
            if e.embedding:
                await self._vector_store.upsert(
                    e.uid,
                    e.embedding,
                    {"content": e.content[:400], "importance": e.importance, "intent": e.intent},
                )

    # ── Main cognitive loop ────────────────────────────────────────────────

    async def think(self, user_input: str) -> TurnResult:
        """Execute one full cognitive resonance cycle."""
        assert self.lattice is not None, "Call .initialize() before .think()"

        t0 = time.perf_counter()

        async with self._lock:
            self._turn += 1
            METRICS.inc_turns()

            # ── 1. Tick periodic systems ───────────────────────────────
            self.neuro.tick()
            self.planner.tick()

            # ── 2. Perception ─────────────────────────────────────────
            percept = self.perception.parse(user_input)

            # ── 3. Neuromodulator integration ─────────────────────────
            self.neuro.integrate_percept(percept)

            # ── 4. Predictive cortex (surprise + belief update) ────────
            input_emb = await self.embedder.embed(user_input)
            surprise, free_energy = self.cortex.observe(input_emb, percept)
            METRICS.set_surprise(surprise)

            # ── 5. Topic shift detection ──────────────────────────────
            if self.binder.detect_topic_shift(surprise):
                self.binder.record_topic_shift(self._turn)

            # ── 6. Temporal binding ───────────────────────────────────
            self.binder.add_frame(
                role="user",
                text=user_input,
                embedding=input_emb,
                intent=percept.intent.value,
                importance=percept.salience,
            )

            # ── 7. Causal ingestion ───────────────────────────────────
            self.causal.ingest(user_input)

            # ── 8. Goal management ────────────────────────────────────
            new_goals = self.planner.infer_goals(percept)
            for g in new_goals:
                self.planner.add_goal(g)
                self.store.upsert_goal(g)

            # ── 9. Memory retrieval ───────────────────────────────────
            top_goal = self.planner.top_goal()
            goal_emb = top_goal.embedding if top_goal and top_goal.embedding else None
            ns = self.neuro.snapshot()
            current_valence = ns.da - 0.5

            memory_hits = await self.lattice.retrieve(
                user_input,
                k=self.cfg.episodic_top_k,
                goal_embedding=goal_emb,
                current_valence=current_valence,
            )

            # ── 10. Metacognitive routing ─────────────────────────────
            mode, pred_conf = self.metacog.select_mode(percept, surprise, ns)

            escalated = False
            if self.metacog.should_escalate(mode, pred_conf):
                mode = self.metacog.escalate(mode)
                escalated = True

            # ── 11. Tool routing ──────────────────────────────────────
            tool_outputs: List[Dict[str, Any]] = []
            planned_tools = self._plan_tools(percept, surprise)
            for tool_name, tool_kwargs in planned_tools:
                result = await self.tools.safe_call(tool_name, **tool_kwargs)
                tool_outputs.append({"tool": tool_name, "args": tool_kwargs, "result": result})
                METRICS.inc_tool(tool_name)

            # ── 12. Context assembly ──────────────────────────────────
            memory_ctx = PromptArchitect.build_memory_context(
                [(e, s, t) for e, s, t in memory_hits[:6]],
                [],
            )
            temporal_ctx = self.binder.get_context(
                query_emb=input_emb, n_frames=6
            )
            goal_ctx = self.planner.context_string()
            identity_block = self.identity.build_identity_block()

            # Causal context for CAUSAL mode
            causal_ctx = ""
            if percept.intent == Intent.CAUSAL or percept.causal_markers >= 2:
                causal_ctx = "\n".join(
                    self.causal.summary(kw)
                    for kw in percept.keywords[:3]
                    if self.causal.summary(kw)
                )

            # ── 13. Prompt construction ───────────────────────────────
            system_block = PromptArchitect.build_system(mode, ns)
            user_block = PromptArchitect.build_user_message(
                query=user_input,
                identity_block=identity_block,
                memory_context=memory_ctx,
                temporal_context=temporal_ctx,
                goal_context=goal_ctx,
                tool_outputs=tool_outputs if tool_outputs else None,
                causal_context=causal_ctx,
            )

            # ── 14. LLM generation ────────────────────────────────────
            temperature = self.neuro.response_temperature(self.cfg.llm_temperature)

            if mode == CognitionMode.TREE:
                response = await self.gateway.tree_reasoning(
                    user_input, user_block, system_block
                )
            else:
                response = await self.gateway.complete(
                    system=system_block,
                    user=user_block,
                    temperature=temperature,
                    mode=mode,
                )

            # ── 15. Quality assessment (PRM) ──────────────────────────
            actual_quality = self.prm.score_response(response, user_input)
            self.metacog.record_outcome(pred_conf, actual_quality)

            # ── 16. Memory encoding ───────────────────────────────────
            input_importance = min(
                1.0,
                percept.complexity * 0.40
                + percept.novelty * 0.30
                + percept.salience * 0.30,
            )
            # ACh-gate the encoding strength
            input_importance = self.neuro.encoding_strength(input_importance)

            await self.lattice.encode(
                content=f"User: {user_input[:500]}",
                tier=MemoryTier.WORKING,
                keywords=percept.keywords,
                entities=percept.entities,
                importance=input_importance,
                intent=percept.intent.value,
                valence=percept.valence,
                arousal=percept.arousal,
                meta={"role": "user", "turn": self._turn},
            )

            output_importance = min(
                1.0,
                actual_quality * 0.50 + surprise * 0.30 + percept.salience * 0.20,
            )
            is_archive = self.cortex.is_archive_worthy(surprise)
            out_tier = (
                MemoryTier.EPISODIC
                if (actual_quality > 0.55 or is_archive)
                else MemoryTier.WORKING
            )
            await self.lattice.encode(
                content=f"NEXUS: {response[:800]}",
                tier=out_tier,
                keywords=percept.keywords,
                importance=output_importance,
                intent=percept.intent.value,
                valence=percept.valence * actual_quality,
                arousal=percept.arousal,
                meta={
                    "role": "assistant",
                    "mode": mode.value,
                    "quality": actual_quality,
                    "turn": self._turn,
                },
            )

            # ── 17. Temporal binder: assistant frame ──────────────────
            resp_emb = await self.embedder.embed(response[:300])
            self.binder.add_frame(
                role="assistant",
                text=response[:300],
                embedding=resp_emb,
                intent=percept.intent.value,
                importance=output_importance,
            )

            # ── 18. Neurochemical reward signal ───────────────────────
            self.neuro.reward_signal(actual_quality)

            # ── 19. Identity update ───────────────────────────────────
            cognition_report = self.metacog.produce_report(
                percept,
                surprise,
                mode,
                pred_conf,
                actual_quality,
                escalated,
            )
            self.identity.record_turn(
                percept,
                calibration_error=cognition_report.calibration_error,
                affect_label=ns.affect.value,
                surprise=surprise,
            )
            if surprise > 0.65:
                self.identity.log_growth(
                    "predictive_horizon",
                    surprise,
                    f"Surprising input: {user_input[:60]}",
                )

            # ── 20. Finalize ──────────────────────────────────────────
            confidence = max(
                0.05,
                min(
                    0.95,
                    0.40
                    + 0.30 * actual_quality
                    + 0.20 * (1.0 - surprise)
                    + 0.10 * min(1.0, len(memory_hits) / 5.0),
                ),
            )
            METRICS.set_confidence(confidence)

            latency_ms = (time.perf_counter() - t0) * 1000.0
            METRICS.obs_latency(latency_ms / 1000.0)

            if self._turn % self.cfg.save_every_n_turns == 0:
                self.store.save_checkpoint(self.status())

            return TurnResult(
                turn=self._turn,
                response=response,
                mode=mode.value,
                confidence=round(confidence, 4),
                actual_quality=round(actual_quality, 4),
                surprise=round(surprise, 4),
                free_energy=round(free_energy, 4),
                memory_hits=len(memory_hits),
                tool_calls=tool_outputs,
                latency_ms=round(latency_ms, 2),
                affect=ns.affect.value,
                metadata={
                    "intent": percept.intent.value,
                    "complexity": round(percept.complexity, 4),
                    "novelty": round(percept.novelty, 4),
                    "salience": round(percept.salience, 4),
                    "urgency": round(percept.urgency, 4),
                    "question_depth": percept.question_depth,
                    "causal_markers": percept.causal_markers,
                    "quality": round(actual_quality, 4),
                    "escalated": escalated,
                    "is_archive": is_archive,
                    "goals": self.planner.summary(),
                    "neuro": ns.to_dict(),
                    "cortex": self.cortex.get_status(),
                    "metacog": self.metacog.get_stats(),
                    "memory": self.lattice.stats() if self.lattice else {},
                },
            )

    # ── Tool routing helper ────────────────────────────────────────────────

    def _plan_tools(
        self, percept: PerceptualFeatures, surprise: float
    ) -> List[tuple]:
        """Heuristic tool routing based on intent and content."""
        import re
        planned = []
        text_l = percept.text.lower()

        if re.search(r"\b(calc|calculate|compute|evaluate)\b", text_l):
            m = re.search(r"([0-9\+\-\*\/\(\)\.\s]{3,})", percept.text)
            if m and any(c.isdigit() for c in m.group(1)):
                planned.append(("calculator", {"expression": m.group(1).strip()}))

        if percept.intent == Intent.RECALL and self.tools.has("memory_search"):
            planned.append(("memory_search", {"query": percept.text, "k": 5}))

        if "status" in text_l and self.tools.has("system_status"):
            planned.append(("system_status", {}))

        return planned

    # ── Introspection ──────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        uptime = now_ts() - self._started_at
        return {
            "version": self.VERSION,
            "uptime_s": round(uptime, 2),
            "turn": self._turn,
            "memory": self.lattice.stats() if self.lattice else {},
            "goals": self.planner.summary(),
            "cortex": self.cortex.get_status(),
            "metacog": self.metacog.get_stats(),
            "identity": self.identity.get_status(),
            "neuro": self.neuro.snapshot().to_dict(),
            "gateway": self.gateway.get_stats(),
            "tools": self.tools.schema(),
        }

    async def consolidate(self) -> Dict[str, Any]:
        """Trigger an immediate consolidation cycle."""
        assert self.lattice is not None
        stats = self.lattice.dreamer.run_now()
        self.store.save_checkpoint(self.status())
        return stats

    async def shutdown(self) -> None:
        LOG.info("nexus.shutdown_start")
        if self.lattice:
            self.lattice.stop_consolidation()
        self.identity._save()
        self.store.save_checkpoint(self.status())
        LOG.info("nexus.shutdown_complete")
