"""
nexus/cognition/predictive_cortex.py
======================================
PredictiveCortex: Free Energy Principle implementation.

Every percept is a hypothesis. Surprise = prediction error.
Minimizing free energy drives both perception (belief update)
and action (make observations match predictions).

Reference: Friston 2010, Clark 2013, Rao & Ballard 1999.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from nexus.core.config import Intent, NexusSettings, PerceptualFeatures, now_ts
from nexus.core.math_utils import FreeEnergy, VectorOps

EPS = 1e-12


@dataclass
class BeliefState:
    """
    The system's posterior distribution over conversation states.
    Maintained as a compact vector + intent probability distribution.
    """

    topic_belief: List[float] = field(default_factory=list)
    intent_prior: Dict[str, float] = field(default_factory=dict)
    topic_history: List[str] = field(default_factory=list)
    last_surprise: float = 0.50
    cumulative_free_energy: float = 0.0
    uncertainty: float = 0.50
    turn: int = 0

    def update_intent_prior(self, observed: Intent) -> None:
        """Bayesian update: observed intent → prior for next turn (Markov model)."""
        transitions: Dict[Intent, Dict[str, float]] = {
            Intent.UNKNOWN:    {Intent.FACTUAL.value: 0.4, Intent.META.value: 0.3, Intent.UNKNOWN.value: 0.3},
            Intent.FACTUAL:    {Intent.FACTUAL.value: 0.5, Intent.ANALYTICAL.value: 0.3, Intent.UNKNOWN.value: 0.2},
            Intent.ANALYTICAL: {Intent.ANALYTICAL.value: 0.4, Intent.FACTUAL.value: 0.3, Intent.UNKNOWN.value: 0.3},
            Intent.EMOTIONAL:  {Intent.EMOTIONAL.value: 0.5, Intent.META.value: 0.2, Intent.UNKNOWN.value: 0.3},
            Intent.CREATIVE:   {Intent.CREATIVE.value: 0.4, Intent.FACTUAL.value: 0.3, Intent.UNKNOWN.value: 0.3},
            Intent.META:       {Intent.META.value: 0.3, Intent.FACTUAL.value: 0.4, Intent.UNKNOWN.value: 0.3},
            Intent.PLANNING:   {Intent.PLANNING.value: 0.4, Intent.ANALYTICAL.value: 0.3, Intent.UNKNOWN.value: 0.3},
            Intent.CAUSAL:     {Intent.CAUSAL.value: 0.35, Intent.ANALYTICAL.value: 0.35, Intent.UNKNOWN.value: 0.30},
            Intent.RECALL:     {Intent.RECALL.value: 0.3, Intent.FACTUAL.value: 0.4, Intent.UNKNOWN.value: 0.3},
        }
        next_dist = transitions.get(observed, {Intent.UNKNOWN.value: 1.0})
        alpha = 0.30
        for key, prob in next_dist.items():
            old = self.intent_prior.get(key, 1.0 / max(len(Intent), 1))
            self.intent_prior[key] = (1 - alpha) * prob + alpha * old
        # Normalize
        total = sum(self.intent_prior.values()) + EPS
        for k in self.intent_prior:
            self.intent_prior[k] /= total


class PredictiveCortex:
    """
    Predictive Processing loop:
    1. Generate top-down prediction from current beliefs
    2. Receive bottom-up observation (embedding)
    3. Compute prediction error (surprise = free energy proxy)
    4. Update beliefs to minimize free energy

    High surprise → strong learning signal → deeper encoding.
    Persistent low surprise → habituation (reduced novelty).
    """

    def __init__(self, cfg: NexusSettings) -> None:
        self.cfg = cfg
        self.beliefs = BeliefState()
        self._prediction: List[float] = []
        self._surprise_history: deque[float] = deque(maxlen=60)
        self._lock = threading.RLock()

    def observe(
        self, emb: List[float], percept: PerceptualFeatures
    ) -> Tuple[float, float]:
        """
        Observe new input embedding, compute surprise and free energy.
        Returns (surprise, free_energy) both ∈ [0, 1].
        """
        with self._lock:
            self.beliefs.turn += 1

            # Precision scales with perceptual salience
            precision = FreeEnergy.precision_from_salience(percept.salience)

            # Prediction error = surprise
            if self._prediction and emb:
                surprise = FreeEnergy.surprise(self._prediction, emb, precision)
            else:
                surprise = 0.55  # first turn: moderate prior

            # KL divergence component (complexity cost in VFE)
            if self.beliefs.topic_belief:
                kl = FreeEnergy.kl_approx(self.beliefs.topic_belief, emb)
                free_energy = 0.60 * surprise + 0.40 * kl
            else:
                free_energy = surprise

            # Variational inference: update beliefs
            lr = self.cfg.prediction_lr
            self.beliefs.topic_belief = FreeEnergy.update_beliefs(
                self.beliefs.topic_belief, emb, lr, precision
            )

            # Intent prior update (Markov transition)
            self.beliefs.update_intent_prior(percept.intent)

            # Topic history
            if percept.keywords:
                tag = " · ".join(percept.keywords[:3])
                self.beliefs.topic_history.append(tag)
                self.beliefs.topic_history = self.beliefs.topic_history[-18:]

            # Uncertainty via exponential smoothing
            self.beliefs.last_surprise = surprise
            self.beliefs.uncertainty = (
                self.cfg.prediction_momentum * self.beliefs.uncertainty
                + (1 - self.cfg.prediction_momentum) * surprise
            )
            self.beliefs.cumulative_free_energy += free_energy

            # Update prediction for next turn (blend belief with current obs)
            if emb and self.beliefs.topic_belief:
                self._prediction = VectorOps.normalize(
                    VectorOps.lerp(self.beliefs.topic_belief, emb, 0.35)
                )
            elif emb:
                self._prediction = emb[:]

            self._surprise_history.append(surprise)

            return surprise, free_energy

    def habituated(self) -> bool:
        """True if system has habituated (consistently low surprise → boring topic)."""
        if len(self._surprise_history) < 10:
            return False
        recent = list(self._surprise_history)[-5:]
        return sum(recent) / len(recent) < 0.18

    def warrants_deep_reasoning(self, surprise: float, complexity: float) -> bool:
        return (
            surprise >= self.cfg.surprise_threshold_deep
            or complexity > self.cfg.deep_reasoning_threshold
        )

    def is_archive_worthy(self, surprise: float) -> bool:
        return surprise >= self.cfg.surprise_threshold_archive

    def get_status(self) -> Dict[str, Any]:
        return {
            "turn": self.beliefs.turn,
            "uncertainty": round(self.beliefs.uncertainty, 4),
            "last_surprise": round(self.beliefs.last_surprise, 4),
            "cumulative_fe": round(self.beliefs.cumulative_free_energy, 2),
            "habituated": self.habituated(),
            "recent_topics": self.beliefs.topic_history[-4:],
            "intent_prior": {
                k: round(v, 3)
                for k, v in sorted(
                    self.beliefs.intent_prior.items(), key=lambda x: -x[1]
                )[:4]
            },
        }
