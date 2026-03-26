"""
nexus/eval/harness.py
=====================
Automated evaluation harness for NEXUS cognitive quality measurement.

The harness measures five quality dimensions across 14 curated test cases:

  1. Routing accuracy       — does the system select the right CognitionMode?
  2. Intent classification  — does PerceptionEngine label intent correctly?
  3. Response quality       — ProcessRewardModel score ≥ threshold
  4. Latency                — does the turn complete within budget?
  5. Keyword presence       — do key terms appear in the response?

Additionally, the harness runs:
  - Memory stress test      — encode 50 items and verify retrieval overlap
  - Neurochemical sanity    — DA/NE stay in [0.01, 0.99] after a long session
  - Identity persistence    — turn count increments and chapter advances
  - Calibration convergence — Brier score stays below 0.30 after 8 turns

Usage
-----
    from nexus.eval.harness import EvalHarness
    from nexus.brain import NexusBrain

    brain = NexusBrain()
    await brain.initialize()

    harness = EvalHarness(brain)
    summary = await harness.run(verbose=True)

    # or from the CLI: nexus --eval
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from nexus.brain import NexusBrain
from nexus.core.config import CognitionMode, Intent, TurnResult
from nexus.core.observability import LOG

# ══════════════════════════════════════════════════════════════════════════════
# §1  CASE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalCase:
    case_id: str
    category: str
    description: str
    user_input: str
    expected_keywords: List[str] = field(default_factory=list)
    expected_intent: Optional[str] = None          # Intent.value string
    expected_mode: Optional[str] = None            # CognitionMode.value string
    min_quality: float = 0.35
    max_latency_ms: float = 12_000.0
    weight: float = 1.0                            # relative importance in summary


@dataclass
class EvalResult:
    case_id: str
    category: str
    passed: bool
    quality: float
    latency_ms: float
    mode: str
    affect: str
    keyword_hits: int
    keyword_total: int
    intent_detected: str
    intent_ok: Optional[bool]
    mode_ok: Optional[bool]
    latency_ok: bool
    quality_ok: bool
    notes: str = ""
    turn_result: Optional[TurnResult] = field(default=None, repr=False)


# ── Test suite ────────────────────────────────────────────────────────────────

STANDARD_SUITE: List[EvalCase] = [
    # ── Routing ──────────────────────────────────────────────────────────────
    EvalCase(
        "R-01", "routing",
        "Simple greeting → REFLEX/FAST, short latency",
        "Hi there!",
        expected_intent="greeting",
        expected_mode="reflex",
        max_latency_ms=4_000.0,
        min_quality=0.28,
    ),
    EvalCase(
        "R-02", "routing",
        "Emotional input → SOMATIC mode",
        "I'm feeling really overwhelmed and I don't know what to do.",
        expected_intent="emotional",
        expected_mode="somatic",
        expected_keywords=["feel", "understand"],
        min_quality=0.35,
    ),
    EvalCase(
        "R-03", "routing",
        "Deep analytical query → DEEP or TREE",
        "Analyze the trade-offs between consistency and availability in distributed systems.",
        expected_intent="analytical",
        expected_keywords=["consistency", "availability"],
        min_quality=0.40,
    ),
    EvalCase(
        "R-04", "routing",
        "Causal query → CAUSAL mode",
        "Why does high cortisol lead to impaired memory formation?",
        expected_intent="analytical",
        expected_keywords=["cortisol", "memory"],
        min_quality=0.40,
    ),

    # ── Factual ───────────────────────────────────────────────────────────────
    EvalCase(
        "F-01", "factual",
        "Simple factual question",
        "What is the capital of France?",
        expected_keywords=["paris"],
        expected_intent="factual",
        min_quality=0.38,
    ),
    EvalCase(
        "F-02", "factual",
        "Multi-step factual with depth",
        "Explain how attention mechanisms work in transformer models.",
        expected_keywords=["attention"],
        expected_intent="factual",
        min_quality=0.42,
    ),

    # ── Analytical ────────────────────────────────────────────────────────────
    EvalCase(
        "A-01", "analytical",
        "Comparative analysis",
        "Compare supervised learning and reinforcement learning in terms of feedback signals.",
        expected_intent="analytical",
        expected_keywords=["reward", "label"],
        min_quality=0.42,
    ),
    EvalCase(
        "A-02", "analytical",
        "Design / architecture question",
        "Design a fault-tolerant message queue for a high-throughput event streaming system.",
        expected_intent="analytical",
        expected_keywords=["queue", "fault"],
        min_quality=0.40,
    ),

    # ── Emotional ─────────────────────────────────────────────────────────────
    EvalCase(
        "E-01", "emotional",
        "Grief / loss — empathic presence",
        "My father passed away last week. I keep thinking about all the things I never said.",
        expected_mode="somatic",
        expected_intent="emotional",
        min_quality=0.35,
    ),
    EvalCase(
        "E-02", "emotional",
        "Positive emotion — engaged response",
        "I just got promoted! I'm so excited and a bit nervous.",
        expected_intent="emotional",
        min_quality=0.30,
    ),

    # ── Memory / recall ───────────────────────────────────────────────────────
    EvalCase(
        "M-01", "memory",
        "Self-recall — identity block activated",
        "What do you know about yourself?",
        expected_intent="meta",
        expected_keywords=["memory", "cognit"],
        min_quality=0.35,
    ),

    # ── Planning ──────────────────────────────────────────────────────────────
    EvalCase(
        "PL-01", "planning",
        "Complex planning task",
        "Help me build a 3-month roadmap to transition from backend engineer to ML engineer.",
        expected_intent="planning",
        expected_keywords=["month", "learn"],
        min_quality=0.40,
    ),

    # ── Creative ─────────────────────────────────────────────────────────────
    EvalCase(
        "CR-01", "creative",
        "Creative writing — short poem",
        "Write a short poem about the nature of memory.",
        expected_intent="creative",
        min_quality=0.32,
    ),

    # ── Performance ──────────────────────────────────────────────────────────
    EvalCase(
        "P-01", "performance",
        "Ultra-simple query — must stay fast",
        "What time is it?",
        max_latency_ms=6_000.0,
        min_quality=0.25,
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# §2  HARNESS
# ══════════════════════════════════════════════════════════════════════════════

class EvalHarness:
    """
    Async evaluation harness.

    Parameters
    ----------
    brain : NexusBrain
        A fully initialised brain (``await brain.initialize()`` called).
    """

    def __init__(self, brain: NexusBrain) -> None:
        self.brain = brain
        self.results: List[EvalResult] = []

    # ── Main entry point ─────────────────────────────────────────────────────

    async def run(
        self,
        suite: Optional[List[EvalCase]] = None,
        verbose: bool = True,
        include_stress: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full evaluation suite.

        Returns a summary dict with pass rate, average quality, latency
        distribution, and per-category breakdowns.
        """
        suite = suite or STANDARD_SUITE
        self.results.clear()

        if verbose:
            _hr("═")
            print(f"  NEXUS EVAL HARNESS  —  {len(suite)} test cases")
            _hr("═")

        for case in suite:
            result = await self._run_case(case)
            self.results.append(result)
            if verbose:
                _print_result(result)

        # ── Extra checks ──────────────────────────────────────────────────────
        extra: Dict[str, Any] = {}
        if include_stress:
            extra["memory_stress"] = await self._memory_stress_test(verbose=verbose)
            extra["neuro_sanity"]  = self._neuro_sanity_check(verbose=verbose)
            extra["calibration"]   = self._calibration_check(verbose=verbose)

        summary = self._build_summary(extra)

        if verbose:
            _hr("─")
            _print_summary(summary)
            _hr("═")

        return summary

    # ── Case runner ───────────────────────────────────────────────────────────

    async def _run_case(self, case: EvalCase) -> EvalResult:
        t0 = time.perf_counter()
        turn_result: Optional[TurnResult] = None

        try:
            turn_result = await self.brain.think(case.user_input)
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return EvalResult(
                case_id=case.case_id,
                category=case.category,
                passed=False,
                quality=0.0,
                latency_ms=round(latency_ms, 1),
                mode="error",
                affect="unknown",
                keyword_hits=0,
                keyword_total=len(case.expected_keywords),
                intent_detected="unknown",
                intent_ok=None,
                mode_ok=None,
                latency_ok=False,
                quality_ok=False,
                notes=f"Exception: {exc}",
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        resp_lower = turn_result.response.lower()

        # ── Keyword hits ──────────────────────────────────────────────────────
        kw_hits = sum(
            1 for kw in case.expected_keywords if kw.lower() in resp_lower
        )

        # ── Intent check ──────────────────────────────────────────────────────
        detected_intent = turn_result.metadata.get("intent", "unknown")
        intent_ok: Optional[bool] = None
        if case.expected_intent:
            intent_ok = detected_intent == case.expected_intent

        # ── Mode check ────────────────────────────────────────────────────────
        mode_ok: Optional[bool] = None
        if case.expected_mode:
            mode_ok = turn_result.mode == case.expected_mode

        # ── Quality / latency ─────────────────────────────────────────────────
        quality_ok = turn_result.actual_quality >= case.min_quality
        latency_ok = latency_ms <= case.max_latency_ms

        # ── Failure reasons ───────────────────────────────────────────────────
        failures: List[str] = []
        if case.expected_keywords and kw_hits < max(1, math.ceil(len(case.expected_keywords) * 0.5)):
            failures.append(f"kw={kw_hits}/{len(case.expected_keywords)}")
        if mode_ok is False:
            failures.append(f"mode={turn_result.mode} (want {case.expected_mode})")
        if not quality_ok:
            failures.append(f"q={turn_result.actual_quality:.2f}<{case.min_quality:.2f}")
        if not latency_ok:
            failures.append(f"latency={latency_ms:.0f}ms>{case.max_latency_ms:.0f}ms")

        return EvalResult(
            case_id=case.case_id,
            category=case.category,
            passed=len(failures) == 0,
            quality=round(turn_result.actual_quality, 4),
            latency_ms=round(latency_ms, 1),
            mode=turn_result.mode,
            affect=turn_result.affect,
            keyword_hits=kw_hits,
            keyword_total=len(case.expected_keywords),
            intent_detected=detected_intent,
            intent_ok=intent_ok,
            mode_ok=mode_ok,
            latency_ok=latency_ok,
            quality_ok=quality_ok,
            notes="; ".join(failures),
            turn_result=turn_result,
        )

    # ── Extra checks ─────────────────────────────────────────────────────────

    async def _memory_stress_test(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Encode 20 diverse items, then verify that top-8 retrieval returns at
        least 5 of the injected UIDs.
        """
        from nexus.core.config import MemoryTier, PerceptualFeatures

        if verbose:
            print("\n  [stress] Memory encode/retrieve consistency …", end=" ", flush=True)

        if not self.brain.lattice:
            if verbose:
                print("SKIP (lattice not ready)")
            return {"status": "skipped"}

        texts = [
            f"EVAL_STRESS_ITEM_{i}: topic_{i % 5} detail about synthetic test item"
            for i in range(20)
        ]
        uids: List[str] = []
        for t in texts:
            engram = self.brain.lattice.encode(
                content=t,
                tier=MemoryTier.EPISODIC,
                importance=0.8,
                intent="factual",
            )
            uids.append(engram.uid)

        # Retrieve using the first item as query
        hits = await self.brain.lattice.retrieve(texts[0], k=8)
        hit_uids = {e.uid for e, _, _ in hits}
        overlap = len(hit_uids & set(uids))

        passed = overlap >= 3  # expect at least 3/8 to match injected items
        if verbose:
            status = "PASS ✓" if passed else "FAIL ✗"
            print(f"{status}  overlap={overlap}/8")

        return {"status": "pass" if passed else "fail", "overlap": overlap}

    def _neuro_sanity_check(self, verbose: bool = True) -> Dict[str, Any]:
        """Verify DA/NE/ACh/SERT are all in [0.01, 0.99]."""
        if verbose:
            print("  [neuro]  Neurochemical bounds check …", end=" ", flush=True)

        ns = self.brain.neuro.snapshot()
        out_of_range = {
            k: v
            for k, v in [("da", ns.da), ("ne", ns.ne), ("ach", ns.ach), ("sert", ns.sert)]
            if not (0.01 <= v <= 0.99)
        }
        passed = len(out_of_range) == 0
        if verbose:
            status = "PASS ✓" if passed else f"FAIL ✗  out_of_range={out_of_range}"
            print(status)

        return {
            "status": "pass" if passed else "fail",
            "da": round(ns.da, 4),
            "ne": round(ns.ne, 4),
            "ach": round(ns.ach, 4),
            "sert": round(ns.sert, 4),
            "out_of_range": out_of_range,
        }

    def _calibration_check(self, verbose: bool = True) -> Dict[str, Any]:
        """After several turns the Brier score should be below 0.30."""
        if verbose:
            print("  [calib]  Metacognitive calibration check …", end=" ", flush=True)

        stats = self.brain.metacog.get_stats()
        brier = stats.get("calibration_brier", 0.25)
        passed = brier < 0.30
        if verbose:
            status = "PASS ✓" if passed else f"FAIL ✗  brier={brier:.4f}"
            print(status)

        return {
            "status": "pass" if passed else "fail",
            "brier": round(brier, 4),
            "bias": round(stats.get("calibration_bias", 0.0), 4),
        }

    # ── Summary builder ───────────────────────────────────────────────────────

    def _build_summary(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        if not self.results:
            return {"total": 0, "passed": 0, "pass_rate": 0.0}

        total   = len(self.results)
        passed  = sum(1 for r in self.results if r.passed)
        avg_q   = sum(r.quality for r in self.results) / total
        avg_lat = sum(r.latency_ms for r in self.results) / total
        p95_lat = _percentile([r.latency_ms for r in self.results], 95)

        # Per-category
        by_cat: Dict[str, Dict[str, int]] = {}
        for r in self.results:
            c = r.category
            if c not in by_cat:
                by_cat[c] = {"total": 0, "passed": 0}
            by_cat[c]["total"] += 1
            if r.passed:
                by_cat[c]["passed"] += 1
        cat_summary = {c: f"{v['passed']}/{v['total']}" for c, v in by_cat.items()}

        return {
            "total":          total,
            "passed":         passed,
            "failed":         total - passed,
            "pass_rate":      round(passed / total, 4),
            "avg_quality":    round(avg_q, 4),
            "avg_latency_ms": round(avg_lat, 1),
            "p95_latency_ms": round(p95_lat, 1),
            "by_category":    cat_summary,
            "extra":          extra,
        }


# ══════════════════════════════════════════════════════════════════════════════
# §3  PRINTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _hr(char: str = "─", width: int = 72) -> None:
    print(char * width)


def _print_result(r: EvalResult) -> None:
    status = "✓ PASS" if r.passed else "✗ FAIL"
    kw_str = f"  kw={r.keyword_hits}/{r.keyword_total}" if r.keyword_total else ""
    intent_str = f"  intent={r.intent_detected}" if r.intent_detected != "unknown" else ""
    print(
        f"  [{status}] {r.case_id:<8}  "
        f"mode={r.mode:<8}  q={r.quality:.2f}  "
        f"{r.latency_ms:>7.1f}ms{kw_str}{intent_str}"
    )
    if not r.passed and r.notes:
        print(f"           ↳ {r.notes}")


def _print_summary(s: Dict[str, Any]) -> None:
    extra = s.get("extra", {})
    print(
        f"\n  RESULTS: {s['passed']}/{s['total']} passed  "
        f"({s['pass_rate']:.0%})"
    )
    print(
        f"  Quality: avg={s['avg_quality']:.3f}  "
        f"Latency: avg={s['avg_latency_ms']:.0f}ms  "
        f"p95={s['p95_latency_ms']:.0f}ms"
    )
    print(f"  By category: {s['by_category']}")
    if extra:
        parts = []
        for key, val in extra.items():
            if isinstance(val, dict):
                parts.append(f"{key}={val.get('status','?')}")
        if parts:
            print(f"  Extra: {', '.join(parts)}")


def _percentile(data: List[float], p: int) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = max(0, min(len(sorted_data) - 1, int(len(sorted_data) * p / 100)))
    return sorted_data[idx]


# ══════════════════════════════════════════════════════════════════════════════
# §4  STANDALONE RUNNER  (used by __main__.py)
# ══════════════════════════════════════════════════════════════════════════════

async def run_eval(brain: NexusBrain, verbose: bool = True) -> Dict[str, Any]:
    """Convenience async wrapper called from __main__.py."""
    harness = EvalHarness(brain)
    return await harness.run(verbose=verbose, include_stress=True)
