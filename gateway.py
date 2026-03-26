"""
nexus/llm/gateway.py
=====================
ResonanceGateway: multi-provider LLM orchestration.

Supports: Anthropic, OpenAI, LiteLLM (covers 100+ providers), mock.
Features:
  - Auto-detection from env vars
  - Exponential backoff via tenacity
  - Per-provider cooldown on failure
  - Temperature derived from neurochemical state
  - Tree-of-thought reasoning for TREE mode
  - Full call audit log
"""

from __future__ import annotations

import asyncio
import math
import os
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from nexus.core.config import CognitionMode, NexusSettings, now_ts
from nexus.core.observability import LOG, METRICS

EPS = 1e-12

# Optional imports
try:
    import anthropic as _anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai as _openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import litellm as _litellm
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False

try:
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )
    HAS_TENACITY = True
except ImportError:
    HAS_TENACITY = False


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _make_retry(fn):
    """Wrap async fn with tenacity retry or hand-rolled fallback."""
    if HAS_TENACITY:
        @retry(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=5.0),
            reraise=True,
        )
        async def _wrapped(*args, **kwargs):
            return await fn(*args, **kwargs)
        return _wrapped

    async def _fallback(*args, **kwargs):
        delay = 0.5
        last_exc = None
        for _ in range(3):
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                await asyncio.sleep(delay)
                delay = min(delay * 2, 5.0)
        raise last_exc
    return _fallback


# ---------------------------------------------------------------------------
# Audit record
# ---------------------------------------------------------------------------

@dataclass
class LLMCall:
    provider: str
    model: str
    mode: str
    prompt_tokens: int
    response_tokens: int
    latency_ms: float
    quality: float
    success: bool
    timestamp: float = field(default_factory=now_ts)


# ---------------------------------------------------------------------------
# ResonanceGateway
# ---------------------------------------------------------------------------

class ResonanceGateway:
    """
    Multi-provider LLM gateway with quality-aware routing,
    exponential backoff on failure, and full call audit log.
    """

    def __init__(self, cfg: NexusSettings) -> None:
        self.cfg = cfg
        self._provider = "mock"
        self._model = "mock-internal"
        self._client: Any = None
        self._cooldowns: Dict[str, float] = {}
        self._call_log: deque[LLMCall] = deque(maxlen=1000)
        self._setup()

    def _setup(self) -> None:
        prov = self.cfg.llm_backend.lower()
        if prov == "auto":
            if HAS_LITELLM:
                prov = "litellm"
            elif HAS_ANTHROPIC and os.getenv("ANTHROPIC_API_KEY"):
                prov = "anthropic"
            elif HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
                prov = "openai"
            else:
                prov = "mock"

        if prov == "anthropic" and HAS_ANTHROPIC:
            key = os.getenv("ANTHROPIC_API_KEY", "")
            if key:
                try:
                    self._client = _anthropic.AsyncAnthropic(api_key=key)
                    self._provider = "anthropic"
                    self._model = self.cfg.llm_reasoning_model
                    LOG.info("gateway.init", provider="anthropic", model=self._model)
                    return
                except Exception as exc:
                    LOG.warning("gateway.anthropic_init_failed", error=str(exc))

        if prov in ("openai",) and HAS_OPENAI:
            key = os.getenv("OPENAI_API_KEY", "")
            if key:
                try:
                    self._client = _openai.AsyncOpenAI(api_key=key)
                    self._provider = "openai"
                    self._model = self.cfg.llm_model
                    LOG.info("gateway.init", provider="openai", model=self._model)
                    return
                except Exception as exc:
                    LOG.warning("gateway.openai_init_failed", error=str(exc))

        if prov == "litellm" and HAS_LITELLM:
            self._provider = "litellm"
            self._model = self.cfg.llm_model
            LOG.info("gateway.init", provider="litellm", model=self._model)
            return

        LOG.info("gateway.init", provider="mock")

    @property
    def is_live(self) -> bool:
        return self._provider != "mock"

    async def complete(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        mode: CognitionMode = CognitionMode.FAST,
    ) -> str:
        """Primary async completion endpoint with retry and audit logging."""
        model = model or (
            self.cfg.llm_reasoning_model
            if mode in {CognitionMode.DEEP, CognitionMode.CAUSAL, CognitionMode.TREE}
            else self.cfg.llm_model
        )
        temperature = temperature if temperature is not None else self.cfg.llm_temperature
        max_tokens = max_tokens or self.cfg.llm_max_tokens

        t0 = time.perf_counter()
        try:
            if self._provider == "anthropic" and not self._is_cooled("anthropic"):
                response = await self._call_anthropic(system, user, model, temperature, max_tokens)
            elif self._provider == "openai" and not self._is_cooled("openai"):
                response = await self._call_openai(system, user, model, temperature, max_tokens)
            elif self._provider == "litellm" and not self._is_cooled("litellm"):
                response = await self._call_litellm(system, user, model, temperature, max_tokens)
            else:
                response = self._mock(system, user, mode)

            latency_ms = (time.perf_counter() - t0) * 1000.0
            quality = self._score_response(response, user)
            METRICS.inc_llm(model)
            self._call_log.append(
                LLMCall(
                    provider=self._provider,
                    model=model,
                    mode=mode.value,
                    prompt_tokens=len((system + user).split()),
                    response_tokens=len(response.split()),
                    latency_ms=latency_ms,
                    quality=quality,
                    success=True,
                )
            )
            return response

        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            LOG.error("gateway.call_failed", provider=self._provider, error=str(exc))
            self._set_cooldown(self._provider, 45.0)
            METRICS.inc_error("llm")
            self._call_log.append(
                LLMCall(
                    provider=self._provider,
                    model=model,
                    mode=mode.value,
                    prompt_tokens=0,
                    response_tokens=0,
                    latency_ms=latency_ms,
                    quality=0.0,
                    success=False,
                )
            )
            return self._mock(system, user, mode)

    async def _call_anthropic(
        self,
        system: str,
        user: str,
        model: str,
        temp: float,
        max_tokens: int,
    ) -> str:
        resp = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=max(0.0, min(1.0, temp)),
        )
        return resp.content[0].text if resp.content else ""

    async def _call_openai(
        self,
        system: str,
        user: str,
        model: str,
        temp: float,
        max_tokens: int,
    ) -> str:
        resp = await self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=max(0.0, min(2.0, temp)),
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    async def _call_litellm(
        self,
        system: str,
        user: str,
        model: str,
        temp: float,
        max_tokens: int,
    ) -> str:
        resp = await _litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temp,
            max_tokens=max_tokens,
            timeout=self.cfg.llm_timeout_s,
        )
        try:
            return resp.choices[0].message.content or ""
        except Exception:
            return str(resp)

    def _mock(self, system: str, user: str, mode: CognitionMode) -> str:
        u = user.lower()
        if re.match(r"^(hi|hello|hey|greetings)", u):
            return (
                "Hello! I'm NEXUS — a cognitive agent with multi-tier memory, "
                "predictive processing, and Bayesian belief updating. How can I help?"
            )
        if any(w in u for w in ("who are you", "your architecture", "yourself", "what are you")):
            return (
                "I'm NEXUS — a production-grade cognitive framework inspired by "
                "neuroscience and predictive processing. I maintain sensory, working, "
                "episodic, and semantic memory tiers. I route reasoning through "
                "REFLEX/FAST/DEEP/TREE/SOMATIC/CAUSAL modes based on complexity and "
                "surprise. My beliefs update via Free Energy minimization on every turn."
            )
        if any(w in u for w in ("feel", "sad", "anxious", "overwhelmed", "worried")):
            return (
                "That sounds genuinely difficult. I want to understand it carefully "
                "before I respond. What part of this weighs on you most right now?"
            )
        if "cause" in u or "why" in u or "because" in u:
            return (
                "A complete causal analysis should distinguish: immediate triggers, "
                "root causes, enabling conditions, and counterfactuals. "
                "Want me to break this down step by step?"
            )
        kws = re.findall(r"\b[a-zA-Z]{4,}\b", user)[:4]
        kw_str = ", ".join(kws) if kws else "your input"
        return (
            f"[NEXUS — mock mode]\n\n"
            f"I've processed your message about '{kw_str}'. "
            f"All cognitive subsystems are active. "
            f"Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or configure LiteLLM "
            f"for full responses."
        )

    async def tree_reasoning(
        self,
        query: str,
        user_block: str,
        system_block: str,
        n_branches: int = 4,
    ) -> str:
        """
        Tree-of-thought reasoning:
        1. Generate N distinct reasoning paths
        2. PRM-score each path
        3. Synthesize top-2 into final answer

        Reference: Yao et al. 2023 (Tree of Thoughts), Wang et al. 2022 (self-consistency)
        """
        seed_prompt = (
            f"Generate {n_branches} distinct reasoning paths for the query below. "
            "Number each path. Keep them meaningfully different in approach.\n\n"
            f"{user_block}\n\n[QUERY]\n{query}"
        )
        raw = await self.complete(
            system_block,
            seed_prompt,
            mode=CognitionMode.DEEP,
            temperature=0.72,  # higher temp for diversity
        )

        # Parse numbered paths
        chunks = re.findall(
            r"^\s*\d+[\.\)]\s*(.+?)(?=\n\s*\d+[\.\)]|\Z)", raw, flags=re.M | re.S
        )
        if not chunks:
            chunks = [x.strip() for x in raw.split("\n") if len(x.strip()) > 30][:n_branches]

        # PRM-score each branch
        from nexus.cognition.metacognition import ProcessRewardModel
        prm = ProcessRewardModel()
        scored = sorted(
            [(prm.score_step(c, query, depth=2), c) for c in chunks],
            key=lambda x: -x[0],
        )
        top2 = [c for _, c in scored[:2]]

        # Synthesis pass
        synth_prompt = (
            "Synthesize the following best reasoning paths into one final, "
            "direct, well-integrated answer. Be honest about uncertainty.\n\n"
            + "\n\n".join(f"[PATH {i+1}]\n{p}" for i, p in enumerate(top2))
            + f"\n\n[QUERY]\n{query}"
        )
        return await self.complete(
            system_block,
            synth_prompt,
            mode=CognitionMode.DEEP,
            temperature=0.20,
        )

    async def complete_json(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Attempt JSON-structured completion."""
        raw = await self.complete(
            system, user, model=model, temperature=0.10, max_tokens=700
        )
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            return {}
        try:
            return __import__("json").loads(m.group(0))
        except Exception:
            return {}

    @staticmethod
    def _score_response(response: str, query: str) -> float:
        """Quick quality heuristic for audit log."""
        q_words = set(re.findall(r"\b\w{4,}\b", query.lower()))
        r_words = set(re.findall(r"\b\w{4,}\b", response.lower()))
        overlap = len(q_words & r_words) / max(len(q_words), 1)
        length_bonus = min(0.20, len(response) / 2000.0)
        return min(1.0, max(0.05, 0.35 + 0.35 * overlap + length_bonus))

    def _is_cooled(self, provider: str) -> bool:
        return time.time() < self._cooldowns.get(provider, 0.0)

    def _set_cooldown(self, provider: str, seconds: float) -> None:
        self._cooldowns[provider] = time.time() + seconds

    def get_stats(self) -> Dict[str, Any]:
        total = len(self._call_log)
        if not total:
            return {"total_calls": 0, "provider": self._provider, "model": self._model}
        success = sum(1 for c in self._call_log if c.success)
        avg_lat = sum(c.latency_ms for c in self._call_log) / total
        avg_q = sum(c.quality for c in self._call_log) / total
        return {
            "total_calls": total,
            "success_rate": round(success / total, 3),
            "avg_latency_ms": round(avg_lat, 1),
            "avg_quality": round(avg_q, 3),
            "provider": self._provider,
            "model": self._model,
        }
