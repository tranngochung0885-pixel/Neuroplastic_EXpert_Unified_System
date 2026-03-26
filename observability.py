"""
NEXUS Observability — Logging & Prometheus Metrics
===================================================
Production-safe structured logger + Prometheus counter/gauge/histogram.
Works whether or not structlog / prometheus_client are installed.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any, Optional

# ── structlog (optional) ──────────────────────────────────────────────────────
try:
    import structlog
    _HAS_STRUCTLOG = True
except ImportError:
    structlog = None  # type: ignore
    _HAS_STRUCTLOG = False

# ── prometheus_client (optional) ──────────────────────────────────────────────
try:
    from prometheus_client import (
        Counter as _PCounter,
        Gauge as _PGauge,
        Histogram as _PHisto,
        generate_latest as _prom_latest,
    )
    _HAS_PROM = True
except ImportError:
    _HAS_PROM = False


# ══════════════════════════════════════════════════════════════════════════════
# SAFE LOGGER
# ══════════════════════════════════════════════════════════════════════════════

class SafeLogger:
    """
    Wraps structlog when available, falls back to stdlib logging.
    Accepts arbitrary keyword arguments (unlike stdlib Logger.xxx).
    """

    def __init__(self, name: str, level: str = "INFO"):
        self.name = name
        self._lvl = getattr(logging, level.upper(), logging.INFO)
        self._std = logging.getLogger(name)
        self._std.setLevel(self._lvl)
        if not self._std.handlers:
            h = logging.StreamHandler(sys.stdout)
            h.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            ))
            self._std.addHandler(h)
        self._sl: Any = None
        if _HAS_STRUCTLOG:
            try:
                structlog.configure(
                    processors=[
                        structlog.stdlib.filter_by_level,
                        structlog.stdlib.add_logger_name,
                        structlog.stdlib.add_log_level,
                        structlog.processors.TimeStamper(fmt="iso", utc=True),
                        structlog.processors.format_exc_info,
                        structlog.processors.JSONRenderer(),
                    ],
                    wrapper_class=structlog.stdlib.BoundLogger,
                    logger_factory=structlog.stdlib.LoggerFactory(),
                    cache_logger_on_first_use=True,
                )
                self._sl = structlog.get_logger(name)
            except Exception:
                self._sl = None

    def _emit(self, level: str, msg: str, **kw: Any) -> None:
        if self._sl is not None:
            try:
                getattr(self._sl, level)(msg, **kw)
                return
            except Exception:
                pass
        if kw:
            extra = " | ".join(f"{k}={v!r}" for k, v in kw.items())
            msg = f"{msg} | {extra}"
        getattr(self._std, level)(msg)

    def debug(self, msg: str, **kw: Any) -> None:   self._emit("debug",   msg, **kw)
    def info(self, msg: str, **kw: Any) -> None:    self._emit("info",    msg, **kw)
    def warning(self, msg: str, **kw: Any) -> None: self._emit("warning", msg, **kw)
    def error(self, msg: str, **kw: Any) -> None:   self._emit("error",   msg, **kw)
    def exception(self, msg: str, **kw: Any) -> None:
        kw.setdefault("exc_info", True)
        self._emit("error", msg, **kw)


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

class _Noop:
    """No-op metric that accepts any call without raising."""
    def labels(self, **_: Any) -> "_Noop": return self
    def inc(self, _: float = 1) -> None: pass
    def observe(self, _: float) -> None: pass
    def set(self, _: float) -> None: pass


class NexusMetrics:
    """Prometheus metrics with graceful no-op fallback."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and _HAS_PROM
        N = _Noop()
        if self.enabled:
            self.turns         = _PCounter("nexus_turns_total", "Total agent turns")
            self.tool_calls    = _PCounter("nexus_tool_calls_total", "Tool calls", ["tool"])
            self.llm_calls     = _PCounter("nexus_llm_calls_total", "LLM calls", ["model"])
            self.errors        = _PCounter("nexus_errors_total", "Errors", ["kind"])
            self.latency       = _PHisto("nexus_turn_latency_s", "Turn latency")
            self.memory_size   = _PGauge("nexus_memory_items", "Memory count", ["tier"])
            self.surprise      = _PGauge("nexus_surprise", "Current surprise")
            self.confidence    = _PGauge("nexus_confidence", "Current confidence")
            self.free_energy   = _PGauge("nexus_free_energy", "Cumulative free energy")
        else:
            self.turns = self.tool_calls = self.llm_calls = self.errors = N
            self.latency = self.memory_size = self.surprise = self.confidence = N
            self.free_energy = N

    def inc_turns(self) -> None:           self.turns.inc()
    def inc_tool(self, t: str) -> None:    self.tool_calls.labels(tool=t).inc()
    def inc_llm(self, m: str) -> None:     self.llm_calls.labels(model=m).inc()
    def inc_error(self, k: str) -> None:   self.errors.labels(kind=k).inc()
    def obs_latency(self, s: float) -> None: self.latency.observe(s)
    def set_memory(self, tier: str, v: int) -> None: self.memory_size.labels(tier=tier).set(v)
    def set_surprise(self, v: float) -> None: self.surprise.set(v)
    def set_confidence(self, v: float) -> None: self.confidence.set(v)
    def set_free_energy(self, v: float) -> None: self.free_energy.set(v)

    def export(self) -> bytes:
        if not _HAS_PROM:
            return b""
        return _prom_latest()


# ── Module-level singletons ───────────────────────────────────────────────────
LOG = SafeLogger("nexus")
METRICS = NexusMetrics(enabled=True)
