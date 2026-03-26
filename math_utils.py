"""
NEXUS Core Math Utilities
=========================
All numerical algorithms used throughout the framework.
No placeholder formulas — every function implements a documented algorithm.
"""

from __future__ import annotations

import math
import random
import hashlib
from collections import deque
from typing import List, Sequence, Tuple, Deque, Optional

EPS = 1e-12

# ── Optional numpy acceleration ───────────────────────────────────────────────
try:
    import numpy as np
    HAS_NP = True
except ImportError:
    np = None  # type: ignore
    HAS_NP = False


# ══════════════════════════════════════════════════════════════════════════════
# §1  VECTOR OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

class VectorOps:
    """Numpy-accelerated vector math with pure-Python fallbacks."""

    @staticmethod
    def cosine(a: Sequence[float], b: Sequence[float]) -> float:
        """Cosine similarity ∈ [−1, 1]."""
        if not a or not b:
            return 0.0
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        if HAS_NP:
            av = np.asarray(a, dtype=np.float32)
            bv = np.asarray(b, dtype=np.float32)
            na = float(np.linalg.norm(av))
            nb = float(np.linalg.norm(bv))
            if na < EPS or nb < EPS:
                return 0.0
            return float(np.dot(av, bv) / (na * nb))
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb + EPS)

    @staticmethod
    def norm(v: Sequence[float]) -> float:
        if HAS_NP:
            return float(np.linalg.norm(np.asarray(v, dtype=np.float32)))
        return math.sqrt(sum(x * x for x in v))

    @staticmethod
    def normalize(v: Sequence[float]) -> List[float]:
        n = VectorOps.norm(v)
        if n < EPS:
            return list(v)
        return [x / n for x in v]

    @staticmethod
    def lerp(a: Sequence[float], b: Sequence[float], t: float) -> List[float]:
        """Linear interpolation; t clamped to [0, 1]."""
        t = max(0.0, min(1.0, t))
        n = min(len(a), len(b))
        return [a[i] * (1 - t) + b[i] * t for i in range(n)]

    @staticmethod
    def weighted_mean(vecs: List[List[float]], weights: List[float]) -> List[float]:
        if not vecs or not weights:
            return []
        total_w = sum(abs(w) for w in weights) + EPS
        dim = len(vecs[0])
        out = [0.0] * dim
        for v, w in zip(vecs, weights):
            for i in range(min(dim, len(v))):
                out[i] += v[i] * w / total_w
        return out

    @staticmethod
    def hash_embed(text: str, dim: int = 384) -> List[float]:
        """Deterministic pseudo-embedding — reproducible, dimension-stable.
        Falls back to this when no encoder is available."""
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2 ** 32)
        rng = random.Random(seed)
        raw = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        return VectorOps.normalize(raw)

    @staticmethod
    def zeros(dim: int) -> List[float]:
        return [0.0] * dim

    @staticmethod
    def add(a: Sequence[float], b: Sequence[float]) -> List[float]:
        n = min(len(a), len(b))
        return [a[i] + b[i] for i in range(n)]

    @staticmethod
    def scale(v: Sequence[float], s: float) -> List[float]:
        return [x * s for x in v]


# ══════════════════════════════════════════════════════════════════════════════
# §2  FREE ENERGY / PREDICTIVE PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

class FreeEnergy:
    """
    Friston Free Energy Principle utilities (2010–2022).
    We use a tractable approximation: F ≈ surprise + KL(posterior || prior).
    Implemented as recommended in "Active Inference" (Parr & Friston 2019).
    """

    @staticmethod
    def surprise(prediction: List[float], observation: List[float],
                 precision: float = 1.0) -> float:
        """
        Bayesian surprise ∈ [0, 1].
        surprise = 1 − exp(−precision × (1 − cosine_sim))
        High precision → small deviations produce large surprise.
        """
        if not prediction:
            return 0.55
        sim = max(0.0, VectorOps.cosine(prediction, observation))
        raw = 1.0 - sim
        return min(1.0, max(0.0, 1.0 - math.exp(-precision * raw)))

    @staticmethod
    def update_beliefs(prior: List[float], likelihood: List[float],
                       lr: float = 0.08, precision: float = 1.0) -> List[float]:
        """
        Variational Bayes belief update:
          posterior ← prior + lr * precision * (likelihood − prior)
        Normalised onto unit sphere.
        """
        if not prior:
            return VectorOps.normalize(list(likelihood))
        if not likelihood:
            return list(prior)
        n = min(len(prior), len(likelihood))
        updated = [prior[i] + lr * precision * (likelihood[i] - prior[i]) for i in range(n)]
        return VectorOps.normalize(updated)

    @staticmethod
    def kl_approx(p: List[float], q: List[float]) -> float:
        """KL(p||q) approximated via 1 − cosine²(p, q)."""
        sim = VectorOps.cosine(p, q)
        return max(0.0, 1.0 - sim * sim)

    @staticmethod
    def precision_from_salience(salience: float) -> float:
        """Higher salience → higher precision → more sensitive to surprise."""
        return 0.3 + 0.7 * min(1.0, salience * 2.0)

    @staticmethod
    def free_energy(surprise: float, kl: float,
                    alpha: float = 0.6) -> float:
        """F = α·surprise + (1−α)·KL[q||p]"""
        return alpha * surprise + (1.0 - alpha) * kl


# ══════════════════════════════════════════════════════════════════════════════
# §3  SPACED REPETITION SYSTEM (SM-2 + Stability-Consolidation)
# ══════════════════════════════════════════════════════════════════════════════

class SRS:
    """
    SM-2 / SuperMemo-inspired spaced repetition.
    Stability-Consolidation model (Bjork 1992, Wozniak 1994).
    """

    @staticmethod
    def retrievability(stability_h: float, elapsed_h: float) -> float:
        """R = exp(−elapsed / stability); probability of successful recall."""
        return math.exp(-elapsed_h / max(stability_h, EPS))

    @staticmethod
    def new_stability(old: float, difficulty: float, success: bool,
                      spacing_coeff: float = 0.12) -> float:
        """
        On success: stability grows — spacing effect amplifies with delay.
        On failure: partial reset (reconsolidation lability, Nader 2000).
        """
        if success:
            spacing = 1.0 + spacing_coeff * math.log(max(1.0, old))
            growth = 2.5 * (1 - difficulty) * spacing
            return min(8760.0, old * max(1.1, growth))
        else:
            return max(1.0, old * 0.35 * (1 - difficulty * 0.5))

    @staticmethod
    def difficulty_from_quality(quality: float) -> float:
        return max(0.0, min(1.0, 1.0 - quality))


# ══════════════════════════════════════════════════════════════════════════════
# §4  HEBBIAN PLASTICITY
# ══════════════════════════════════════════════════════════════════════════════

class HebbianPlasticity:
    """
    'Cells that fire together, wire together' (Hebb 1949).
    Anti-Hebbian: lateral inhibition between competing memories.
    """

    @staticmethod
    def weight_update(w_cur: float, pre: float, post: float,
                      lr: float = 0.06, anti_lr: float = 0.02) -> float:
        """
        ΔW = lr·pre·post − anti_lr·(1−pre)·post − ε·W  (weight decay)
        """
        hebb  = lr * pre * post
        anti  = anti_lr * (1.0 - pre) * post
        decay = 0.001 * w_cur
        return max(0.0, min(1.0, w_cur + hebb - anti - decay))

    @staticmethod
    def co_activation(emb_a: List[float], emb_b: List[float],
                      dt_s: float, window_s: float) -> float:
        """
        Co-activation score ∈ [0, 1] based on:
          semantic similarity × temporal proximity
        """
        sem = max(0.0, VectorOps.cosine(emb_a, emb_b))
        temporal = math.exp(-dt_s / max(window_s, EPS))
        return sem * temporal


# ══════════════════════════════════════════════════════════════════════════════
# §5  YERKES-DODSON AROUSAL CURVE
# ══════════════════════════════════════════════════════════════════════════════

class YerkesDodson:
    """
    Yerkes-Dodson law (1908): performance peaks at moderate arousal.
    P(a) = P_max · exp(−(a − a_opt)² / (2σ²))
    Used by metacognitive routing to adjust cognitive mode selection.
    """

    @staticmethod
    def performance(arousal: float, optimal: float = 0.58,
                    width: float = 0.28) -> float:
        """Returns performance modifier ∈ [0, 1]."""
        return math.exp(-((arousal - optimal) ** 2) / (2 * width ** 2 + EPS))

    @staticmethod
    def temperature(arousal: float, base: float = 0.65,
                    optimal: float = 0.58) -> float:
        """High arousal → lower temperature (focused). Low → higher (exploratory)."""
        delta = (arousal - optimal) * -0.15
        return max(0.10, min(1.50, base + delta))


# ══════════════════════════════════════════════════════════════════════════════
# §6  TEMPORAL-DIFFERENCE (TD) LEARNING
# ══════════════════════════════════════════════════════════════════════════════

class TDLearning:
    """
    TD(λ) algorithm for value-function estimation.
    Used by the world model to track reward prediction errors.
    Reference: Sutton & Barto "Reinforcement Learning" Ch.7.
    """

    def __init__(self, gamma: float = 0.92, lam: float = 0.7,
                 lr: float = 0.12):
        self.gamma = gamma
        self.lam = lam
        self.lr = lr
        self.v: float = 0.5          # expected value estimate
        self.elig: float = 0.0       # eligibility trace
        self.errors: Deque[float] = deque(maxlen=128)

    def update(self, reward: float) -> float:
        """
        TD(λ) update:
          δ = r + γV − V
          e ← γλe + 1
          V ← V + α·δ·e
          e ← γλe
        Returns TD error δ.
        """
        td = reward + self.gamma * self.v - self.v
        self.elig = self.gamma * self.lam * self.elig + 1.0
        self.v += self.lr * td * self.elig
        self.elig *= self.gamma * self.lam
        self.errors.append(td)
        return td

    def avg_abs_error(self) -> float:
        if not self.errors:
            return 0.0
        return sum(abs(e) for e in self.errors) / len(self.errors)


# ══════════════════════════════════════════════════════════════════════════════
# §7  BAYESIAN CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

class BayesianCalibration:
    """
    Online Brier-score calibration (Murphy 1973).
    Tracks predicted vs actual confidence and corrects systematic bias.
    """

    def __init__(self, window: int = 80):
        self._pairs: Deque[Tuple[float, float]] = deque(maxlen=window)

    def record(self, predicted: float, actual: float) -> None:
        self._pairs.append((
            max(0.0, min(1.0, predicted)),
            max(0.0, min(1.0, actual)),
        ))

    def brier_score(self) -> float:
        """Lower is better. Perfect calibration → 0."""
        if not self._pairs:
            return 0.25
        return sum((p - a) ** 2 for p, a in self._pairs) / len(self._pairs)

    def bias(self) -> float:
        """Positive = overconfident, negative = underconfident."""
        if not self._pairs:
            return 0.0
        return sum(p - a for p, a in self._pairs) / len(self._pairs)

    def calibrated(self, raw: float) -> float:
        """Adjust raw confidence by historical bias."""
        return max(0.05, min(0.95, raw - self.bias() * 0.5))

    def reliability_bins(self, n_bins: int = 10) -> List[Tuple[float, float, int]]:
        """Reliability diagram data: [(bin_center, mean_actual, count)]."""
        bins: List[List[float]] = [[] for _ in range(n_bins)]
        for p, a in self._pairs:
            idx = min(n_bins - 1, int(p * n_bins))
            bins[idx].append(a)
        result = []
        for i, acts in enumerate(bins):
            if acts:
                center = (i + 0.5) / n_bins
                result.append((center, sum(acts) / len(acts), len(acts)))
        return result


# ══════════════════════════════════════════════════════════════════════════════
# §8  CAUSAL PATTERN REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

import re
from collections import defaultdict
from typing import Dict
import threading

class CausalRegistry:
    """
    Lightweight causal chain extractor using syntactic patterns.
    Builds a directed graph of cause→effect relationships.
    """
    PATTERNS = [
        (re.compile(r"\b(.{3,60}?)\s+causes?\s+(.{3,60}?)\b", re.I), 0.80),
        (re.compile(r"\b(.{3,60}?)\s+leads?\s+to\s+(.{3,60}?)\b", re.I), 0.72),
        (re.compile(r"\b(.{3,60}?)\s+results?\s+in\s+(.{3,60}?)\b", re.I), 0.72),
        (re.compile(r"\b(.{3,60}?)\s+because\s+(.{3,60}?)\b", re.I), 0.65),
        (re.compile(r"\b(.{3,60}?)\s+enables?\s+(.{3,60}?)\b", re.I), 0.60),
        (re.compile(r"\b(.{3,60}?)\s+implies?\s+(.{3,60}?)\b", re.I), 0.55),
    ]

    def __init__(self) -> None:
        self._forward: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self._backward: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self._lock = threading.RLock()

    def ingest(self, text: str) -> int:
        """Extract causal pairs from text. Returns number of pairs extracted."""
        found = 0
        lower = text.strip()
        with self._lock:
            for pat, w in self.PATTERNS:
                for m in pat.finditer(lower):
                    a = m.group(1).strip()[:80]
                    b = m.group(2).strip()[:80]
                    if len(a) >= 3 and len(b) >= 3:
                        self._forward[a].append((b, w))
                        self._backward[b].append((a, w))
                        found += 1
        return found

    def effects_of(self, cause: str, limit: int = 5) -> List[Tuple[str, float]]:
        with self._lock:
            items = sorted(self._forward.get(cause.lower().strip(), []),
                           key=lambda x: x[1], reverse=True)
            return items[:limit]

    def causes_of(self, effect: str, limit: int = 5) -> List[Tuple[str, float]]:
        with self._lock:
            items = sorted(self._backward.get(effect.lower().strip(), []),
                           key=lambda x: x[1], reverse=True)
            return items[:limit]

    def summary(self, topic: str) -> str:
        t = topic.lower().strip()
        causes  = self.causes_of(t)
        effects = self.effects_of(t)
        parts: List[str] = []
        if causes:
            parts.append("Caused by: " + ", ".join(f"{c}({w:.2f})" for c, w in causes))
        if effects:
            parts.append("Leads to: " + ", ".join(f"{e}({w:.2f})" for e, w in effects))
        return " | ".join(parts) if parts else f"No causal links for '{topic}'."
