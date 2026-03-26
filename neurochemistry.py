"""
nexus/cognition/neurochemistry.py
===================================
NeurochemicalBus: global neuromodulator state machine.

Models 4 key systems with phasic bursts, tonic reuptake, cross-modulator
coupling, and perceptual integration. Read by ALL downstream modules.
"""

from __future__ import annotations

import copy
import threading
import time
from collections import deque
from typing import Any, Dict, Optional

from nexus.core.config import (
    AffectLabel,
    CognitionMode,
    NeuroState,
    NexusSettings,
    PerceptualFeatures,
    now_ts,
)
from nexus.core.math_utils import YerkesDodson
from nexus.core.observability import LOG


class NeurochemicalBus:
    """
    Global neuromodulator state machine.

    Phasic bursts: rapid, stimulus-locked release.
    Tonic reuptake: exponential decay to baseline (pharmacokinetics).
    Cross-modulator coupling:
      DA↑ → NE↑ (via locus coeruleus)
      NE↑ → ACh↑ (arousal → attention narrowing)
    """

    def __init__(self, cfg: NexusSettings) -> None:
        self.cfg = cfg
        self.state = NeuroState(
            da=cfg.da_tonic,
            ne=cfg.ne_tonic,
            ach=cfg.ach_tonic,
            sert=cfg.sert_tonic,
        )
        self._history: deque[Dict] = deque(maxlen=300)
        self._last_tick = time.time()
        self._lock = threading.RLock()

    def phasic_burst(
        self, neurotransmitter: str, magnitude: float, source: str = "unknown"
    ) -> None:
        """Stimulus-locked phasic release with cross-modulator coupling."""
        with self._lock:
            mag = max(0.0, min(0.5, magnitude))
            s = self.state

            if neurotransmitter == "dopamine":
                s.da = min(0.99, s.da + mag)
                # DA → NE coupling (LC-NE pathway)
                s.ne = min(0.99, s.ne + mag * 0.30)
            elif neurotransmitter == "norepinephrine":
                s.ne = min(0.99, s.ne + mag)
                # NE → ACh coupling
                s.ach = min(0.99, s.ach + mag * 0.18)
            elif neurotransmitter == "acetylcholine":
                s.ach = min(0.99, s.ach + mag)
            elif neurotransmitter == "serotonin":
                s.sert = min(0.99, s.sert + mag)

    def integrate_percept(self, features: PerceptualFeatures) -> None:
        """
        Integrate perceptual features into neurochemical state.
        Primary pathway by which the external world affects cognition.
        """
        with self._lock:
            cfg = self.cfg
            s = self.state

            # Novelty → NE burst (locus coeruleus)
            if features.novelty > 0.45:
                s.ne = min(
                    0.99, s.ne + cfg.ne_novelty_burst * features.novelty
                )

            # Salience → DA (reward prediction / saliency gating)
            if features.salience > 0.50:
                s.da = min(
                    0.99, s.da + cfg.da_reward_burst * features.salience * 0.60
                )

            # Question depth → ACh (attention)
            if features.question_depth > 0:
                boost = cfg.ach_attention_burst * (features.question_depth / 3.0)
                s.ach = min(0.99, s.ach + boost)

            # Negative valence + high arousal → NE stress response
            if features.valence < -0.30 and features.arousal > 0.50:
                s.ne = min(0.99, s.ne + 0.10)
                s.sert = max(0.05, s.sert - 0.05)

            # Positive valence → SERT + DA lift
            if features.valence > 0.30:
                s.sert = min(0.99, s.sert + 0.04)
                s.da = min(0.99, s.da + 0.05)

            # Urgency → NE spike
            if features.urgency > 0.5:
                s.ne = min(
                    0.99, s.ne + features.urgency * 0.12
                )

    def tick(self, dt_s: Optional[float] = None) -> None:
        """
        Tonic reuptake: monoexponential pharmacokinetic return to baseline.
        """
        now = time.time()
        if dt_s is None:
            dt_s = now - self._last_tick
        self._last_tick = now

        cfg = self.cfg
        with self._lock:
            s = self.state
            r = min(1.0, dt_s)
            s.da += (cfg.da_tonic - s.da) * cfg.da_reuptake * r
            s.ne += (cfg.ne_tonic - s.ne) * cfg.ne_reuptake * r
            s.ach += (cfg.ach_tonic - s.ach) * cfg.ach_reuptake * r
            s.sert += (cfg.sert_tonic - s.sert) * cfg.sert_reuptake * r

            # Clamp all values
            for attr in ("da", "ne", "ach", "sert"):
                setattr(s, attr, max(0.01, min(0.99, getattr(s, attr))))

            self._history.append({"t": now, **s.to_dict()})

    def reward_signal(self, quality: float) -> None:
        """Post-response DA burst proportional to quality (RPE signal)."""
        if quality >= 0.70:
            self.phasic_burst("dopamine", quality * 0.14, "response_quality")
            self.phasic_burst("serotonin", 0.04, "response_quality")
        elif quality < 0.38:
            self.phasic_burst("norepinephrine", 0.07, "response_quality_low")

    def snapshot(self) -> NeuroState:
        with self._lock:
            return copy.copy(self.state)

    def response_temperature(self, base: float = 0.65) -> float:
        """Derive LLM temperature from arousal via Yerkes-Dodson."""
        return YerkesDodson.temperature(self.state.ne, base)

    def encoding_strength(self, base_importance: float) -> float:
        """ACh gates encoding; DA adds salience boost."""
        gate = self.state.learning_gate
        da_boost = (self.state.da - 0.5) * 0.22
        return min(1.0, max(0.0, base_importance * gate + da_boost))
