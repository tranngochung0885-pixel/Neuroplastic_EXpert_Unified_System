"""
nexus/__main__.py
=================
Entry point for NEXUS cognitive agent framework.

Usage
-----
    python -m nexus                      # interactive CLI (default)
    python -m nexus --mode cli           # explicit CLI
    python -m nexus --mode api           # FastAPI server
    python -m nexus --mode demo          # scripted demonstration
    python -m nexus --eval               # run evaluation suite + exit
    python -m nexus --help

CLI commands (once inside the REPL)
------------------------------------
    /status          full cognitive status JSON
    /memory [query]  retrieve memories matching query
    /reflect         self-reflection on recent performance
    /goals           active goal stack
    /sleep           explicit dream-consolidation cycle
    /eval            run eval harness
    /neuro           neurochemical snapshot
    /cortex          predictive cortex status
    /identity        identity substrate status
    /tools           available tool list
    /quit            clean shutdown + exit
    /help            this help

Environment
-----------
    NEXUS_API_KEY          optional bearer token for API mode
    NEXUS_LLM_MODEL        override LLM model name
    NEXUS_EMBED_BACKEND    auto | sentence_transformers | hash
    ANTHROPIC_API_KEY      enables Anthropic backend
    OPENAI_API_KEY         enables OpenAI backend
    LITELLM_*              enables LiteLLM routing
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import Optional

from nexus.brain import NexusBrain
from nexus.core.config import NexusSettings
from nexus.core.observability import LOG

# ══════════════════════════════════════════════════════════════════════════════
# §1  BANNER
# ══════════════════════════════════════════════════════════════════════════════

BANNER = r"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   N E X U S  —  Neuroplastic EXpert Unified System  v1.0.0                   ║
║   Production-grade cognitive agent framework                                  ║
║                                                                               ║
║   Synthesises: Free Energy Principle · Hebbian plasticity · multi-tier       ║
║   memory · neurochemical modulation · Bayesian metacognition · narrative      ║
║   identity · goal hierarchies · multi-LLM orchestration                      ║
║                                                                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Commands:                                                                    ║
║   /status    /memory [q]    /reflect    /goals     /sleep                     ║
║   /eval      /neuro         /cortex     /identity  /tools                     ║
║   /quit      /help                                                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════════════════
# §2  HELPER UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _detect_provider() -> str:
    """Detect which LLM provider is available from environment variables."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("LITELLM_API_KEY") or os.getenv("LITELLM_MODEL"):
        return "litellm"
    return "mock"


def _print_feature_table(brain: NexusBrain) -> None:
    """Print a compact feature-availability table at startup."""
    try:
        import sentence_transformers  # noqa: F401
        st_ok = True
    except ImportError:
        st_ok = False
    try:
        import lancedb  # noqa: F401
        ldb_ok = True
    except ImportError:
        ldb_ok = False
    try:
        import qdrant_client  # noqa: F401
        qd_ok = True
    except ImportError:
        qd_ok = False
    try:
        import numpy  # noqa: F401
        np_ok = True
    except ImportError:
        np_ok = False

    provider = brain.gateway._provider
    embed    = brain.embedder.__class__.__name__

    rows = [
        ("sentence-transformers", "✓" if st_ok else "○"),
        ("lancedb",               "✓" if ldb_ok else "○"),
        ("qdrant-client",         "✓" if qd_ok else "○"),
        ("numpy",                 "✓" if np_ok else "○"),
        ("LLM provider",          f"✓  {provider}"),
        ("Embedder",              f"✓  {embed}"),
    ]
    print("Capabilities:")
    for name, status in rows:
        print(f"  {status}  {name}")
    print()

    if provider == "mock":
        print(
            "⚠  No LLM API key found. Running in mock mode.\n"
            "   Set ANTHROPIC_API_KEY or OPENAI_API_KEY for full responses.\n"
        )


def _json_pp(obj) -> str:
    try:
        import dataclasses
        if dataclasses.is_dataclass(obj):
            obj = dataclasses.asdict(obj)
        return json.dumps(obj, indent=2, default=str)
    except Exception:
        return str(obj)


# ══════════════════════════════════════════════════════════════════════════════
# §3  CLI LOOP
# ══════════════════════════════════════════════════════════════════════════════

async def run_cli(brain: NexusBrain, verbose: bool = False) -> None:
    """
    Interactive REPL loop.  Exits on /quit or EOF.
    """
    print(BANNER)
    _print_feature_table(brain)
    print(f"NEXUS  ready  (turn 0)  —  type a message or /help\n")

    while True:
        try:
            raw = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        # ── Commands ─────────────────────────────────────────────────────────
        if raw.startswith("/"):
            await _handle_command(raw, brain, verbose)
            continue

        # ── Cognitive turn ────────────────────────────────────────────────────
        try:
            result = await brain.think(raw)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {exc}\n")
            continue

        print(f"\nNEXUS> {result.response}\n")

        if verbose:
            print(
                f"  mode={result.mode}  q={result.actual_quality:.2f}  "
                f"surprise={result.surprise:.3f}  affect={result.affect}  "
                f"mem={result.memory_hits}  {result.latency_ms:.0f}ms\n"
            )

    await brain.shutdown()


async def _handle_command(raw: str, brain: NexusBrain, verbose: bool) -> None:
    parts = raw.strip().split(None, 1)
    cmd  = parts[0].lower()
    args = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/quit", "/exit", "/q"):
        print("Shutting down NEXUS…")
        await brain.shutdown()
        sys.exit(0)

    elif cmd == "/help":
        print(BANNER)

    elif cmd == "/status":
        print(_json_pp(brain.status()))

    elif cmd == "/memory":
        q = args or "recent conversation"
        if not brain.lattice:
            print("Memory subsystem not ready.\n")
            return
        hits = await brain.lattice.retrieve(q, k=8)
        if not hits:
            print("(no relevant memories)\n")
            return
        print(f"\nMemory results for '{q}':")
        for engram, score, tier in hits:
            print(
                f"  [{tier.value:<10}  score={score:.3f}  age={engram.age_hours():.1f}h"
                f"  R={engram.retrievability():.2f}]  {engram.content[:160]}"
            )
        print()

    elif cmd == "/reflect":
        print("\nNEXUS (reflecting)…")
        system = (
            "You are NEXUS reflecting honestly on your own cognitive performance. "
            "Be specific, identify patterns, acknowledge limitations. "
            "First person. No hollow self-praise."
        )
        sm   = brain.identity.get_status()
        mc   = brain.metacog.get_stats()
        topics = list(brain.cortex.beliefs.topic_history)[-5:]
        prompt = (
            f"Reflect on your recent performance.\n"
            f"Identity status: {json.dumps(sm)}\n"
            f"Metacognition stats: {json.dumps(mc)}\n"
            f"Recent topics: {', '.join(topics)}\n\n"
            "What patterns do you notice? Where are your gaps?"
        )
        try:
            from nexus.core.config import CognitionMode
            response = brain.gateway.complete(
                system_prompt=system,
                user_message=prompt,
                mode=CognitionMode.DEEP,
                temperature=0.62,
            )
            print(f"\nNEXUS> {response}\n")
        except Exception as exc:
            print(f"[reflect error] {exc}\n")

    elif cmd == "/goals":
        goals = brain.planner.summary()
        if not goals:
            print("(no active goals)\n")
            return
        print("\nActive Goals:")
        for g in goals:
            print(
                f"  [{g['priority']:.2f}]  {g['description']}"
                f"  (progress={g.get('progress',0):.0%})"
            )
        print()

    elif cmd == "/sleep":
        print("\nRunning dream consolidation cycle…")
        try:
            stats = await brain.consolidate()
            print(f"Done: {_json_pp(stats)}\n")
        except Exception as exc:
            print(f"[sleep error] {exc}\n")

    elif cmd == "/eval":
        from nexus.eval.harness import EvalHarness
        harness = EvalHarness(brain)
        await harness.run(verbose=True, include_stress=True)

    elif cmd == "/neuro":
        ns = brain.neuro.snapshot()
        print(_json_pp(ns.to_dict()))

    elif cmd == "/cortex":
        print(_json_pp(brain.cortex.get_status()))

    elif cmd == "/identity":
        print(_json_pp(brain.identity.get_status()))

    elif cmd == "/tools":
        print(_json_pp(brain.tools.schema()))

    elif cmd == "/verbose":
        verbose = not verbose
        print(f"Verbose mode: {'ON' if verbose else 'OFF'}\n")

    else:
        print(f"Unknown command: '{cmd}'.  Type /help for options.\n")


# ══════════════════════════════════════════════════════════════════════════════
# §4  DEMO
# ══════════════════════════════════════════════════════════════════════════════

async def run_demo(brain: NexusBrain) -> None:
    """
    Scripted demonstration covering all major cognitive subsystems.
    Useful for smoke-testing a fresh installation.
    """
    print(BANNER)
    print("Running NEXUS demonstration…\n")
    _print_feature_table(brain)

    exchanges = [
        (
            "Hello! What are you?",
            "Greeting + meta → identity block, REFLEX/FAST routing, low latency",
        ),
        (
            "How does memory consolidation work in the brain?",
            "Deep factual → predictive cortex update, DEEP reasoning, episodic encoding",
        ),
        (
            "Compare episodic and semantic memory — which matters more for identity?",
            "Analytical → world model, memory retrieval, DEEP/TREE reasoning",
        ),
        (
            "I've been feeling really anxious about an upcoming deadline.",
            "Emotional → SOMATIC mode, NE/SERT modulation, empathic response",
        ),
        (
            "What do you actually know about yourself?",
            "Meta → identity substrate, narrative context, self-model activation",
        ),
        (
            "Returning to memory — how does sleep affect consolidation?",
            "RECALL + factual → temporal binding, callback detection, multi-turn coherence",
        ),
    ]

    for i, (user_msg, note) in enumerate(exchanges, 1):
        print(f"─── Turn {i}: {note}")
        print(f"You: {user_msg}")
        result = await brain.think(user_msg)
        preview = result.response[:500] + ("…" if len(result.response) > 500 else "")
        print(f"NEXUS: {preview}")
        ns = brain.neuro.snapshot()
        print(
            f"  ↳  mode={result.mode}  q={result.actual_quality:.2f}  "
            f"surprise={result.surprise:.3f}  affect={ns.affective_state.value}  "
            f"DA={ns.da:.2f}  NE={ns.ne:.2f}  {result.latency_ms:.0f}ms\n"
        )

    print("─── Predictive Cortex Status")
    print(_json_pp(brain.cortex.get_status()))

    print("\n─── Memory Stats")
    if brain.lattice:
        print(_json_pp(brain.lattice.stats()))

    print("\n─── Identity Status")
    print(_json_pp(brain.identity.get_status()))

    print("\n─── Full Status")
    print(_json_pp(brain.status()))

    print("\nDemo complete.")


# ══════════════════════════════════════════════════════════════════════════════
# §5  ARGUMENT PARSING + MAIN
# ══════════════════════════════════════════════════════════════════════════════

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nexus",
        description="NEXUS — Neuroplastic EXpert Unified System v1.0.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--mode",
        choices=["cli", "api", "demo"],
        default="cli",
        help="Run mode (default: cli)",
    )
    p.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation suite and exit",
    )
    p.add_argument(
        "--host",
        default=None,
        help="API server host (overrides NEXUS_API_HOST)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=None,
        help="API server port (overrides NEXUS_API_PORT)",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        dest="data_dir",
        help="Override data directory path",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-turn cognitive metadata in CLI mode",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="log_level",
        help="Logging verbosity (default: INFO)",
    )
    return p


async def _amain() -> int:
    parser = _build_arg_parser()
    args   = parser.parse_args()

    # ── Logging setup ─────────────────────────────────────────────────────────
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Settings ──────────────────────────────────────────────────────────────
    cfg = NexusSettings()
    if args.data_dir:
        cfg.data_dir = args.data_dir  # type: ignore[assignment]
    if args.host:
        cfg.api_host = args.host      # type: ignore[assignment]
    if args.port:
        cfg.api_port = args.port      # type: ignore[assignment]

    # ── Initialise brain ──────────────────────────────────────────────────────
    brain = NexusBrain(cfg=cfg)
    await brain.initialize()

    # ── Eval mode ─────────────────────────────────────────────────────────────
    if args.eval:
        from nexus.eval.harness import EvalHarness
        harness = EvalHarness(brain)
        summary = await harness.run(verbose=True, include_stress=True)
        await brain.shutdown()
        failed = summary.get("failed", 0)
        return 0 if failed == 0 else 1

    # ── Dispatch by mode ──────────────────────────────────────────────────────
    if args.mode == "cli":
        await run_cli(brain, verbose=args.verbose)
        return 0

    if args.mode == "demo":
        await run_demo(brain)
        await brain.shutdown()
        return 0

    if args.mode == "api":
        try:
            from nexus.api.server import run_server
        except ImportError as exc:
            print(
                f"FastAPI / uvicorn not available: {exc}\n"
                "Install with: pip install fastapi uvicorn",
                file=sys.stderr,
            )
            return 1
        await run_server(brain)
        return 0

    return 0


def main() -> int:
    try:
        return asyncio.run(_amain())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
