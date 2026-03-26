"""
nexus/cognition/metacognition.py
==================================
MetacognitiveLoop: confidence tracking, routing, calibration.
ProcessRewardModel: step-level quality scoring via linguistic heuristics.

References:
  Yerkes & Dodson 1908 — arousal-performance relationship
  Bayesian calibration — Brier score, reliability diagram
  Process Reward Models — Lightman et al. 2023 (Let's Verify Step by Step)
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from nexus.core.config import (
    CognitionMode,
    Intent,
    NeuroState,
    NexusSettings,
    PerceptualFeatures,
)
from nexus.core.math_utils import BayesianCalibration, YerkesDodson

EPS = 1e-12


class ProcessRewardModel:
    """
    Step-level quality scoring for multi-step reasoning.
    Evaluates reasoning steps by: relevance, causal structure,
    logical coherence, epistemic hedging, and completeness.
    """

    _CAUSAL = frozenset({
        "because", "therefore", "hence", "thus", "since", "consequently",
        "implies", "leads", "causes", "results", "follows", "due", "given",
    })
    _HEDGES = frozenset({
        "however", "although", "but", "yet", "while", "whereas", "despite",
        "nevertheless", "alternatively", "on the other hand",
    })
    _CONCLUSION = frozenset({
        "therefore", "in conclusion", "thus", "hence", "so", "finally",
        "ultimately", "in summary", "this means", "this suggests",
    })

    @classmethod
    def score_step(cls, step: str, query: str, depth: int = 0) -> float:
        """Score a reasoning step ∈ [0, 1]."""
        score = 0.36
        text_lower = step.lower()
        q_words = set(re.findall(r"\b\w{4,}\b", query.lower()))
        s_words = set(re.findall(r"\b\w{4,}\b", text_lower))

        # Query–step semantic overlap
        overlap = len(q_words & s_words) / max(len(q_words), 1)
        score += overlap * 0.24

        # Causal structure bonus
        if any(m in text_lower for m in cls._CAUSAL):
            score += 0.10
        # Epistemic hedging (nuance)
        if any(m in text_lower for m in cls._HEDGES):
            score += 0.06
        # Conclusion markers at synthesis depth
        if depth >= 2 and any(m in text_lower for m in cls._CONCLUSION):
            score += 0.08
        # Length reward (up to 400 chars)
        score += min(0.08, len(step) / 4000.0)
        # Lingering question → continues inquiry
        if "?" in step:
            score += 0.04

        return min(1.0, max(0.0, score))

    @classmethod
    def score_response(cls, response: str, query: str) -> float:
        """Score a complete response ∈ [0, 1]."""
        return cls.score_step(response, query, depth=3)


@dataclass
class CognitionReport:
    """Metacognitive assessment of a single turn."""

    turn: int
    complexity: float
    surprise: float
    uncertainty: float
    selected_mode: CognitionMode
    predicted_confidence: float
    actual_quality: Optional[float]
    calibration_error: float
    yerkes_dodson_mod: float
    escalated: bool
    routing_rationale: str


class MetacognitiveLoop:
    """
    Metacognition: awareness of the system's own cognitive processes.

    1. Yerkes-Dodson arousal-calibrated mode routing
    2. Bayesian Brier-score confidence calibration
    3. Strategic escalation: REFLEX → FAST → DEEP → TREE
    4. Cognitive load tracking
    5. Produce CognitionReports for identity substrate learning
    """

    _MODE_BASE_CONF: Dict[CognitionMode, float] = {
        CognitionMode.REFLEX: 0.75,
        CognitionMode.FAST: 0.64,
        CognitionMode.DEEP: 0.72,
        CognitionMode.TREE: 0.80,
        CognitionMode.SOMATIC: 0.68,
        CognitionMode.CAUSAL: 0.70,
    }

    _MODE_COST: Dict[CognitionMode, float] = {
        CognitionMode.REFLEX: 0.08,
        CognitionMode.FAST: 0.28,
        CognitionMode.DEEP: 0.60,
        CognitionMode.TREE: 0.90,
        CognitionMode.SOMATIC: 0.38,
        CognitionMode.CAUSAL: 0.55,
    }

    def __init__(self, cfg: NexusSettings) -> None:
        self.cfg = cfg
        self.calibration = BayesianCalibration(cfg.calibration_window)
        self.prm = ProcessRewardModel()
        self._turn = 0
        self._mode_counts: Counter = Counter()
        self._cog_load: deque[float] = deque(maxlen=30)
        self._lock = threading.RLock()

    def select_mode(
        self,
        percept: PerceptualFeatures,
        surprise: float,
        ns: NeuroState,
    ) -> Tuple[CognitionMode, float]:
        """
        Select cognitive mode + predict confidence.
        Integrates: complexity, surprise, arousal (Yerkes-Dodson),
        serotonin patience, and calibration history.
        """
        with self._lock:
            self._turn += 1
            complexity = percept.complexity
            uncertainty = max(percept.novelty * 0.40, surprise * 0.60)

            # Yerkes-Dodson modifier (performance peaks at moderate arousal)
            yd_mod = YerkesDodson.performance(ns.ne)

            # Calibration-escalation boost
            cal_err = self.calibration.brier_score()
            esc_boost = max(0.0, cal_err - 0.20) * 0.38
            eff_complexity = min(1.0, complexity + esc_boost)

            # ── Mode selection ──────────────────────────────────────────
            if percept.intent == Intent.EMOTIONAL or percept.urgency > 0.60:
                mode = CognitionMode.SOMATIC
            elif percept.intent == Intent.CAUSAL and percept.causal_markers >= 2:
                mode = CognitionMode.CAUSAL
            elif eff_complexity < 0.18 and uncertainty < 0.28 and ns.sert > 0.42:
                mode = CognitionMode.REFLEX
            elif eff_complexity < 0.45 or ns.sert > 0.58:
                mode = CognitionMode.FAST
            elif eff_complexity < 0.70:
                mode = CognitionMode.DEEP
            else:
                mode = CognitionMode.TREE

            # Low serotonin (patience) avoids expensive TREE reasoning
            if ns.sert < 0.28 and mode == CognitionMode.TREE:
                mode = CognitionMode.DEEP

            # ── Confidence prediction ─────────────────────────────────
            base_conf = self._MODE_BASE_CONF.get(mode, 0.60)
            noise = uncertainty * -0.16
            raw_conf = max(0.10, min(0.95, base_conf + noise))
            pred_conf = self.calibration.calibrated(raw_conf) * yd_mod
            pred_conf = max(0.10, min(0.95, pred_conf))

            self._mode_counts[mode] += 1
            self._cog_load.append(self._MODE_COST.get(mode, 0.3))

            return mode, pred_conf

    def should_escalate(self, mode: CognitionMode, confidence: float) -> bool:
        thresholds = {
            CognitionMode.REFLEX: self.cfg.reflex_confidence_min,
            CognitionMode.FAST: self.cfg.fast_confidence_min,
            CognitionMode.DEEP: self.cfg.deep_confidence_min,
        }
        thresh = thresholds.get(mode)
        return thresh is not None and confidence < thresh

    def escalate(self, mode: CognitionMode) -> CognitionMode:
        ladder = {
            CognitionMode.REFLEX: CognitionMode.FAST,
            CognitionMode.FAST: CognitionMode.DEEP,
            CognitionMode.DEEP: CognitionMode.TREE,
            CognitionMode.CAUSAL: CognitionMode.TREE,
        }
        return ladder.get(mode, mode)

    def record_outcome(self, predicted: float, actual: float) -> None:
        self.calibration.record(predicted, actual)

    def produce_report(
        self,
        percept: PerceptualFeatures,
        surprise: float,
        mode: CognitionMode,
        pred_conf: float,
        actual_quality: Optional[float] = None,
        escalated: bool = False,
    ) -> CognitionReport:
        yd_mod = YerkesDodson.performance(0.45)
        return CognitionReport(
            turn=self._turn,
            complexity=percept.complexity,
            surprise=surprise,
            uncertainty=max(percept.novelty, surprise),
            selected_mode=mode,
            predicted_confidence=pred_conf,
            actual_quality=actual_quality,
            calibration_error=self.calibration.brier_score(),
            yerkes_dodson_mod=yd_mod,
            escalated=escalated,
            routing_rationale=(
                f"complexity={percept.complexity:.2f} surprise={surprise:.2f} "
                f"intent={percept.intent.value} escalated={escalated}"
            ),
        )

    def avg_cognitive_load(self) -> float:
        if not self._cog_load:
            return 0.3
        return sum(self._cog_load) / len(self._cog_load)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "turn": self._turn,
            "calibration_brier": round(self.calibration.brier_score(), 4),
            "calibration_bias": round(self.calibration.bias(), 4),
            "mode_counts": dict(self._mode_counts),
            "avg_cognitive_load": round(self.avg_cognitive_load(), 3),
        }
