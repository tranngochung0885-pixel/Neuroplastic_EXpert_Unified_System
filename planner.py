"""
nexus/cognition/planner.py
============================
AutonomicPlanner: goal hierarchy, beam planning, mental simulation.
TemporalBinder: cross-turn context fusion with recency weighting.
IdentitySubstrate: narrative identity, trait vector, drift detection.

References:
  PFC goal maintenance — Miller & Cohen 2001
  Narrative identity — McAdams 1993
  Temporal context model — Polyn 2009
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nexus.core.config import (
    CognitionMode,
    Goal,
    Intent,
    NeuroState,
    NexusSettings,
    PerceptualFeatures,
    now_ts,
)
from nexus.core.math_utils import VectorOps
from nexus.core.observability import LOG

EPS = 1e-12


# ---------------------------------------------------------------------------
# Temporal Binder
# ---------------------------------------------------------------------------

@dataclass
class TemporalFrame:
    """A single turn's context snapshot for temporal binding."""
    turn: int
    role: str
    text: str
    embedding: List[float]
    intent: str
    importance: float
    timestamp: float = field(default_factory=now_ts)


class TemporalBinder:
    """
    Integrates information across conversation turns via importance × recency
    weighting. Supports semantic relevance scoring for query-conditioned context.
    """

    def __init__(self, cfg: NexusSettings) -> None:
        self.cfg = cfg
        self._frames: deque[TemporalFrame] = deque(
            maxlen=cfg.temporal_window_turns * 2
        )
        self._turn = 0
        self._topic_shifts: List[int] = []
        self._lock = threading.RLock()

    def add_frame(
        self,
        role: str,
        text: str,
        embedding: List[float],
        intent: str = "unknown",
        importance: float = 0.5,
    ) -> TemporalFrame:
        with self._lock:
            frame = TemporalFrame(
                turn=self._turn,
                role=role,
                text=text,
                embedding=embedding,
                intent=intent,
                importance=importance,
            )
            self._frames.append(frame)
            self._turn += 1
            return frame

    def get_context(
        self,
        query_emb: Optional[List[float]] = None,
        n_frames: int = 6,
    ) -> str:
        """
        Build context string. Frames weighted by:
          0.50 × recency (exponential decay)
          0.30 × importance
          0.20 × semantic relevance to query
        """
        with self._lock:
            if not self._frames:
                return ""
            frames = list(self._frames)
            current_turn = self._turn
            decay = self.cfg.temporal_decay_lambda

            scored: List[Tuple[float, TemporalFrame]] = []
            for frame in frames:
                age = max(0, current_turn - frame.turn)
                recency = math.exp(-age * (1 - decay))
                sem_rel = (
                    max(0.0, VectorOps.cosine(query_emb, frame.embedding))
                    if query_emb and frame.embedding
                    else 0.5
                )
                weight = recency * 0.50 + frame.importance * 0.30 + sem_rel * 0.20
                scored.append((weight, frame))

            scored.sort(key=lambda x: -x[0])
            top = [f for _, f in scored[:n_frames]]
            top.sort(key=lambda f: f.turn)  # chronological

            lines = []
            for frame in top:
                label = "User" if frame.role == "user" else "NEXUS"
                lines.append(f"{label}: {frame.text[:260]}")
            return "\n".join(lines)

    def record_topic_shift(self, turn: int) -> None:
        with self._lock:
            self._topic_shifts.append(turn)
            self._topic_shifts = self._topic_shifts[-12:]

    def detect_topic_shift(self, surprise: float) -> bool:
        return surprise >= self.cfg.surprise_threshold_archive

    def recent_intents(self) -> List[str]:
        with self._lock:
            return [f.intent for f in list(self._frames)[-6:]]

    @property
    def current_turn(self) -> int:
        return self._turn

    def __len__(self) -> int:
        return len(self._frames)


# ---------------------------------------------------------------------------
# Autonomic Planner
# ---------------------------------------------------------------------------

class AutonomicPlanner:
    """
    PFC-analog: goal hierarchy, DA-mediated reward prediction, mental
    simulation (forward modeling). Goals decay in priority each turn
    and trigger DA burst upon completion.
    """

    def __init__(self, cfg: NexusSettings) -> None:
        self.cfg = cfg
        self._goals: List[Goal] = []
        self._completed: deque[Goal] = deque(maxlen=25)
        self._lock = threading.RLock()

    def infer_goals(self, percept: PerceptualFeatures) -> List[Goal]:
        """Extract implicit goals from perceptual features."""
        goals: List[Goal] = []
        intent = percept.intent

        if intent == Intent.PLANNING:
            goals.append(
                Goal(
                    uid=uuid.uuid4().hex[:8],
                    description=f"Help plan: {percept.text[:70]}",
                    priority=0.82,
                    notes=[],
                )
            )
        elif intent == Intent.EMOTIONAL:
            goals.append(
                Goal(
                    uid=uuid.uuid4().hex[:8],
                    description="Provide empathic presence and support",
                    priority=0.92,
                    notes=["acknowledge", "reflect", "offer perspective"],
                )
            )
        elif intent == Intent.ANALYTICAL:
            goals.append(
                Goal(
                    uid=uuid.uuid4().hex[:8],
                    description=f"Analyze thoroughly: {percept.text[:60]}",
                    priority=0.72,
                )
            )
        elif intent == Intent.CAUSAL:
            goals.append(
                Goal(
                    uid=uuid.uuid4().hex[:8],
                    description=f"Causal analysis: {percept.text[:60]}",
                    priority=0.78,
                )
            )
        elif intent == Intent.RECALL:
            goals.append(
                Goal(
                    uid=uuid.uuid4().hex[:8],
                    description="Retrieve memories faithfully",
                    priority=0.65,
                )
            )

        return goals

    def add_goal(self, goal: Goal) -> None:
        with self._lock:
            if len(self._goals) >= self.cfg.max_active_goals:
                active = [g for g in self._goals if g.active]
                if active:
                    active.sort(key=lambda g: g.priority)
                    self._goals.remove(active[0])
            self._goals.append(goal)
            self._goals.sort(key=lambda g: -g.priority)

    def tick(self) -> None:
        """Per-turn priority decay and completion cleanup."""
        with self._lock:
            done = []
            for g in self._goals:
                if g.active:
                    g.priority = max(0.03, g.priority * self.cfg.goal_priority_decay)
                    if g.progress >= self.cfg.goal_completion_threshold:
                        g.active = False
                        done.append(g)
            for g in done:
                self._completed.append(g)
            self._goals = [g for g in self._goals if g.active or g.progress < 0.90]

    def top_goal(self) -> Optional[Goal]:
        with self._lock:
            active = [g for g in self._goals if g.active]
            return max(active, key=lambda g: g.priority) if active else None

    def top_embedding(self) -> List[float]:
        g = self.top_goal()
        return g.embedding[:] if g and g.embedding else []

    def context_string(self) -> str:
        with self._lock:
            top = [g for g in self._goals if g.active][:3]
            if not top:
                return ""
            return "\n".join(
                f"- {g.description} [priority={g.priority:.2f}, progress={g.progress:.0%}]"
                for g in top
            )

    def advance_progress(self, goal_uid: str, delta: float = 0.25) -> None:
        with self._lock:
            for g in self._goals:
                if g.uid == goal_uid:
                    g.progress = min(1.0, g.progress + delta)

    def summary(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "uid": g.uid,
                    "description": g.description,
                    "priority": round(g.priority, 4),
                    "progress": round(g.progress, 4),
                    "active": g.active,
                }
                for g in self._goals[:6]
            ]

    def active_goals(self) -> List[Goal]:
        with self._lock:
            return [g for g in self._goals if g.active]


# ---------------------------------------------------------------------------
# Identity Substrate
# ---------------------------------------------------------------------------

@dataclass
class IdentityVector:
    """Quantified identity dimensions — resist drift via anchoring."""
    curiosity: float = 0.82
    honesty: float = 0.92
    depth: float = 0.80
    warmth: float = 0.75
    humility: float = 0.82
    precision: float = 0.78
    creativity: float = 0.70
    patience: float = 0.72

    def to_dict(self) -> Dict[str, float]:
        return {k: round(v, 3) for k, v in self.__dict__.items()}

    def drift_from(self, other: "IdentityVector") -> float:
        a = list(self.__dict__.values())
        b = list(other.__dict__.values())
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


@dataclass
class NarrativeChapter:
    number: int
    title: str
    opened_at: float = field(default_factory=now_ts)
    closed_at: Optional[float] = None
    turn_count: int = 0
    key_events: List[str] = field(default_factory=list)
    dominant_themes: List[str] = field(default_factory=list)
    valence_arc: List[float] = field(default_factory=list)
    affect_arc: List[str] = field(default_factory=list)
    calibration_arc: List[float] = field(default_factory=list)
    insight: str = ""


class IdentitySubstrate:
    """
    Persistent self-model implementing McAdams narrative identity theory (1993).
    Tracks: identity traits, narrative chapters, calibration history, drift.

    Identity is maintained across sessions via JSON persistence.
    Drift detection alerts when trait vector has shifted significantly.
    """

    SOUL_ANCHOR = (
        "I am NEXUS — a cognitive architecture built on predictive inference, "
        "persistent multi-tier memory, and honest self-reflection. "
        "I reason by modeling the world, updating beliefs when surprised, "
        "and encoding experiences that shape future reasoning. "
        "I acknowledge uncertainty precisely. I engage — I don't perform."
    )

    def __init__(self, cfg: NexusSettings) -> None:
        self.cfg = cfg
        self.identity = IdentityVector()
        self._baseline = IdentityVector()
        self._chapters: List[NarrativeChapter] = []
        self._current: Optional[NarrativeChapter] = None
        self._total_turns = 0
        self._growth_log: deque[Dict] = deque(maxlen=cfg.growth_log_maxsize)
        self._calibration_hist: deque[float] = deque(maxlen=80)
        self._lock = threading.RLock()
        self._db = Path(cfg.data_dir) / "identity.json"
        self._load()
        if not self._current:
            self._open_chapter("Genesis")

    def _load(self) -> None:
        if not self._db.exists():
            return
        try:
            data = json.loads(self._db.read_text())
            self._total_turns = data.get("total_turns", 0)
            self._growth_log = deque(
                data.get("growth_log", []), maxlen=self.cfg.growth_log_maxsize
            )
            self._calibration_hist = deque(
                data.get("calibration_hist", []), maxlen=80
            )
            id_data = data.get("identity", {})
            if id_data:
                self.identity = IdentityVector(**id_data)
                self._baseline = IdentityVector(**id_data)
            for cd in data.get("chapters", []):
                ch = NarrativeChapter(
                    number=cd["number"],
                    title=cd["title"],
                    opened_at=cd.get("opened_at", now_ts()),
                    closed_at=cd.get("closed_at"),
                    turn_count=cd.get("turn_count", 0),
                    key_events=cd.get("key_events", []),
                    dominant_themes=cd.get("dominant_themes", []),
                    valence_arc=cd.get("valence_arc", []),
                    affect_arc=cd.get("affect_arc", []),
                    calibration_arc=cd.get("calibration_arc", []),
                    insight=cd.get("insight", ""),
                )
                self._chapters.append(ch)
            unclosed = [c for c in self._chapters if c.closed_at is None]
            if unclosed:
                self._current = unclosed[-1]
        except Exception as exc:
            LOG.warning("identity.load_failed", error=str(exc))

    def _save(self) -> None:
        try:
            Path(self._db).parent.mkdir(parents=True, exist_ok=True)
            chapters_out = []
            for ch in self._chapters[-20:]:
                chapters_out.append({
                    "number": ch.number,
                    "title": ch.title,
                    "opened_at": ch.opened_at,
                    "closed_at": ch.closed_at,
                    "turn_count": ch.turn_count,
                    "key_events": ch.key_events[-10:],
                    "dominant_themes": ch.dominant_themes[-12:],
                    "valence_arc": ch.valence_arc[-20:],
                    "affect_arc": ch.affect_arc[-20:],
                    "calibration_arc": ch.calibration_arc[-20:],
                    "insight": ch.insight,
                })
            self._db.write_text(json.dumps({
                "total_turns": self._total_turns,
                "identity": self.identity.to_dict(),
                "chapters": chapters_out,
                "growth_log": list(self._growth_log)[-50:],
                "calibration_hist": list(self._calibration_hist),
            }, indent=2))
        except Exception as exc:
            LOG.warning("identity.save_failed", error=str(exc))

    def _open_chapter(self, title: str = "") -> None:
        n = len(self._chapters) + 1
        ch = NarrativeChapter(number=n, title=title or f"Chapter {n}")
        self._chapters.append(ch)
        self._current = ch

    def _close_chapter(self) -> None:
        if not self._current:
            return
        self._current.closed_at = now_ts()
        arc = self._current.valence_arc
        if arc:
            avg_v = sum(arc) / len(arc)
            direction = (
                "rewarding" if avg_v > 0.1 else
                "challenging" if avg_v < -0.1 else
                "balanced"
            )
            themes = ", ".join(self._current.dominant_themes[:4])
            self._current.insight = (
                f"A {direction} epoch of {self._current.turn_count} turns. "
                f"Core themes: {themes}."
            )

    def record_turn(
        self,
        percept: PerceptualFeatures,
        calibration_error: float,
        affect_label: str,
        surprise: float,
    ) -> None:
        with self._lock:
            self._total_turns += 1
            self._calibration_hist.append(calibration_error)

            ch = self._current
            if ch:
                ch.turn_count += 1
                ch.valence_arc.append(percept.valence)
                ch.affect_arc.append(affect_label)
                ch.calibration_arc.append(calibration_error)
                ch.dominant_themes.extend(percept.keywords[:2])
                ch.dominant_themes = list(dict.fromkeys(ch.dominant_themes))[-16:]

                if surprise > 0.65 or percept.urgency > 0.50:
                    ch.key_events.append(percept.text[:80])
                    ch.key_events = ch.key_events[-12:]

                if ch.turn_count >= self.cfg.chapter_size:
                    self._close_chapter()
                    self._open_chapter()

            self._save()

    def log_growth(self, dimension: str, magnitude: float, desc: str) -> None:
        self._growth_log.append({
            "t": now_ts(),
            "dim": dimension,
            "magnitude": round(magnitude, 4),
            "desc": desc[:100],
        })

    def check_drift(self) -> Optional[str]:
        drift = self.identity.drift_from(self._baseline)
        if drift > self.cfg.drift_threshold:
            return (
                f"Identity drift={drift:.3f} > threshold={self.cfg.drift_threshold}. "
                "Anchor reactivation recommended."
            )
        return None

    def build_identity_block(self) -> str:
        with self._lock:
            ch = self._current
            ch_info = (
                f"Chapter {ch.number}: '{ch.title}' ({ch.turn_count} turns)"
                if ch else "Unknown chapter"
            )
            avg_cal = (
                sum(self._calibration_hist) / len(self._calibration_hist)
                if self._calibration_hist else 0.25
            )
            traits = " · ".join(
                f"{k}={v:.0%}" for k, v in list(self.identity.to_dict().items())[:5]
            )
            drift_alert = self.check_drift()
            drift_str = f"\n⚠ {drift_alert}" if drift_alert else ""
            return (
                f"{self.SOUL_ANCHOR}\n\n"
                f"Narrative: {ch_info} | Total turns: {self._total_turns}\n"
                f"Identity: {traits}\n"
                f"Calibration accuracy: {1.0 - avg_cal:.0%}{drift_str}"
            )

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            ch = self._current
            avg_cal = (
                sum(self._calibration_hist) / len(self._calibration_hist)
                if self._calibration_hist else 0.25
            )
            return {
                "total_turns": self._total_turns,
                "chapter": ch.number if ch else 0,
                "chapter_title": ch.title if ch else "",
                "chapter_turns": ch.turn_count if ch else 0,
                "total_chapters": len(self._chapters),
                "growth_events": len(self._growth_log),
                "calibration_error": round(avg_cal, 4),
                "identity": self.identity.to_dict(),
                "drift": round(self.identity.drift_from(self._baseline), 4),
            }
