# NEXUS — Neuroplastic EXpert Unified System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Version](https://img.shields.io/badge/version-1.0.0-green?style=flat-square)
![Status](https://img.shields.io/badge/status-research%20prototype-yellow?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-purple?style=flat-square)

**A research exploration into neuroscience-inspired cognitive agent design.**

*What if we designed an LLM agent the way the brain actually works — not as a pipeline, but as a resonance loop?*

</div>

---

## What is this?

NEXUS is a personal research project exploring one question:

> **Can principles from computational neuroscience — free energy, Hebbian plasticity, neurochemical modulation, spaced repetition — be meaningfully applied to the design of an LLM agent?**

This is not a replacement for LangChain, LlamaIndex, or any production framework. It is an experiment in cognitive architecture — an attempt to take neuroscience ideas seriously and see what happens when you implement them.

The result is a framework built around a **19-step cognitive resonance loop** that processes each turn through perception, neurochemical state, multi-tier memory, metacognitive routing, goal tracking, and identity narrative — before ever reaching the LLM.

Whether this approach is actually *useful* is an open question. That is part of what I hope to find out.

---

## Theoretical Inspirations

These are the ideas NEXUS tries to implement — not perfectly, but sincerely:

| Idea | Source | What it does in NEXUS |
|------|--------|----------------------|
| Free Energy Principle | Friston (2010–2022) | Surprise computation & belief updating each turn |
| Complementary Learning Systems | McClelland et al. (1995) | Background episodic → semantic consolidation |
| Spaced Repetition / SM-2 | Wozniak (1994) | Stability-based forgetting curve on memories |
| Yerkes-Dodson Law | Yerkes & Dodson (1908) | Arousal-based routing of reasoning depth |
| Cowan's Working Memory | Cowan (2001) | Capacity limit of 4 ± 1 (not Miller's 7 ± 2) |
| Hebbian Plasticity | Hebb (1949) | Memory links strengthen through co-activation |
| Elastic Weight Consolidation | Kirkpatrick et al. (2017) | Protects important memories from interference |
| Temporal Difference Learning | Sutton (1988) | Dopamine-like reward signal after each response |
| Narrative Identity | McAdams (1993) | Agent maintains a self-model across chapters |

I am not a neuroscientist. Some of these mappings are approximate, some are speculative, and all of them are open to critique. Feedback from people who know these fields better than I do is very welcome.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    NexusBrain                            │
│             Cognitive Resonance Loop (19 steps)          │
└──────────────────────────────┬───────────────────────────┘
                               │
       ┌───────────────────────┼────────────────────┐
       ▼                       ▼                    ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│ Perception  │     │ Neurochemistry   │     │ Predictive Cortex   │
│             │     │                  │     │                     │
│ Intent      │     │ DA · NE · ACh    │     │ Surprise (FEP)      │
│ Valence     │────▶│ SERT reuptake    │────▶│ Belief updating     │
│ Arousal     │     │ Phasic bursts    │     │ Topic transitions   │
│ Complexity  │     │ Cross-coupling   │     │ Habituation         │
└─────────────┘     └──────────────────┘     └─────────────────────┘
       │
       ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│   Memory    │     │  Metacognition   │     │ Planner + Identity  │
│             │     │                  │     │                     │
│ Sensory     │     │ Yerkes-Dodson    │     │ Goal tracking       │
│ Working     │◀────│ Mode routing     │     │ Temporal binding    │
│ Episodic    │     │ Bayesian calib.  │     │ Narrative chapters  │
│ Semantic    │     │ PRM scoring      │     │ Drift detection     │
└─────────────┘     └──────────────────┘     └─────────────────────┘
       │
       ▼
┌─────────────┐     ┌──────────────────┐
│ LLM Gateway │     │  Tool Registry   │
│             │     │                  │
│ Anthropic   │     │ calculator       │
│ OpenAI      │     │ memory_search    │
│ LiteLLM     │     │ system_status    │
│ Mock        │     │ (extensible)     │
└─────────────┘     └──────────────────┘
```

---

## Project Structure

```
nexus/
├── __init__.py                  # Public exports
├── __main__.py                  # CLI — REPL, API server, demo, eval
├── brain.py                     # NexusBrain — main orchestrator
│
├── core/
│   ├── config.py                # Settings, dataclasses, enums
│   ├── math_utils.py            # FreeEnergy, Hebbian, SRS, YerkesDodson, TD...
│   └── observability.py         # Logging, Prometheus metrics
│
├── memory/
│   ├── embeddings.py            # Embedding, VectorStore, Reranker backends
│   └── store.py                 # SensoryBuffer, WorkingMemory, EpisodicStore,
│                                # SemanticStore, DreamConsolidator, EngramLattice
│
├── cognition/
│   ├── perception.py            # Feature extraction from raw text
│   ├── neurochemistry.py        # DA/NE/ACh/SERT state machine
│   ├── predictive_cortex.py     # Free energy surprise & belief update
│   ├── metacognition.py         # Mode routing, Process Reward Model
│   └── planner.py               # Goals, temporal context, identity substrate
│
├── llm/
│   ├── gateway.py               # Multi-provider LLM + Tree-of-Thought
│   └── prompts.py               # Mode-aware system prompt builder
│
├── tools/
│   └── registry.py              # Async tool dispatch and schema
│
├── api/
│   └── server.py                # FastAPI — REST + WebSocket
│
└── eval/
    └── harness.py               # 14 test cases + stress tests
```

---

## Getting Started

### Requirements

Python 3.10+. Every dependency is optional — NEXUS degrades gracefully if something is missing.

```bash
# Minimal — uses mock LLM and hash embeddings, everything in memory
pip install pydantic pydantic-settings numpy structlog tenacity

# Recommended — adds real embeddings and persistent storage
pip install -r requirements.txt
```

### Set up an LLM

```bash
# Pick one (or none — mock mode still works for exploring the architecture)
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
```

### Run

```bash
# Interactive REPL
python -m nexus

# Scripted demo (6 turns across different cognitive modes)
python -m nexus --mode demo

# REST API + WebSocket (http://localhost:8091/docs)
python -m nexus --mode api

# Evaluation suite
python -m nexus --eval
```

### Basic Python usage

```python
import asyncio
from nexus import NexusBrain

async def main():
    brain = NexusBrain()
    await brain.initialize()

    result = await brain.think("Explain the free energy principle.")
    print(result.response)
    print(f"mode={result.mode}  quality={result.actual_quality:.2f}  {result.latency_ms:.0f}ms")

    # Emotional input — should route to SOMATIC mode
    result = await brain.think("I've been feeling overwhelmed lately.")
    print(result.mode)   # → "somatic"

    await brain.shutdown()

asyncio.run(main())
```

---

## CLI Commands

Once inside the REPL:

| Command | Description |
|---------|-------------|
| `/status` | Full cognitive state as JSON |
| `/memory [query]` | Retrieve memories matching a query |
| `/reflect` | Self-reflection on recent performance |
| `/goals` | Active goal stack |
| `/sleep` | Trigger dream consolidation |
| `/neuro` | Current neurochemical snapshot |
| `/cortex` | Predictive cortex status |
| `/identity` | Identity substrate and narrative arc |
| `/eval` | Run evaluation harness |
| `/quit` | Clean shutdown |

---

## Configuration

All settings use environment variables with the `NEXUS_` prefix, or a `.env` file.

```bash
NEXUS_AGENT_NAME=NEXUS
NEXUS_DATA_DIR=~/.nexus_agent      # SQLite, LanceDB, identity JSON

NEXUS_LLM_MODEL=auto               # auto-detects from available API keys
NEXUS_EMBED_BACKEND=auto           # auto | sentence_transformers | hash
NEXUS_VECTOR_BACKEND=auto          # auto | lancedb | qdrant | memory

NEXUS_API_HOST=0.0.0.0
NEXUS_API_PORT=8091
NEXUS_API_KEY=                     # empty = no auth
```

---

## What works, what doesn't

Being honest about the current state:

**Things that work reasonably well:**
- The cognitive routing — emotional inputs do land in SOMATIC mode, complex analytical queries do escalate to DEEP
- Multi-tier memory with background consolidation runs correctly
- Neurochemical state does shift in response to inputs and stabilises via reuptake
- The eval harness gives a repeatable quality signal

**Things that are approximate or unvalidated:**
- Neurochemical parameters (tonic levels, reuptake rates) are hand-tuned heuristics, not calibrated from data
- The "free energy" computation is a simplified approximation — not a rigorous FEP implementation
- Whether any of this actually improves agent behaviour over a simpler baseline is an open empirical question
- No systematic benchmark comparison against other frameworks has been done yet

---

## Ideas for contribution

If any part of this interests you, here are directions worth exploring:

- **Calibration** — benchmark the neurochemical parameters against something measurable
- **Ablation studies** — does removing the neurochemical layer change output quality? What about memory consolidation?
- **Comparison** — how does NEXUS memory retrieval compare to MemGPT or a standard RAG setup on long-context tasks?
- **Better FEP** — the free energy implementation is simplified; a more rigorous version would be interesting
- **Tool use** — native function-calling through Anthropic/OpenAI tool-use API
- **Anything I haven't thought of** — open an issue and let's talk

---

## Honest caveats

- This is a **research prototype**, not a production system
- I am not a neuroscientist — the theoretical mappings are made in good faith but are likely imperfect
- The code has been tested to run and produce reasonable behaviour, but not battle-hardened
- If you find something wrong — scientifically, architecturally, or in the code — please open an issue. I genuinely want to know

---

## License

MIT — free to use, modify, and build on.

---

<div align="center">

*Standing on the shoulders of:*
**Friston · McClelland · Hebb · Wozniak · Cowan · Kirkpatrick · Sutton · McAdams**

*Built with curiosity. Feedback welcome.*

</div>
