"""
NEXUS — Neuroplastic EXpert Unified System
==========================================
Production-grade cognitive agent framework synthesising:
  - Free Energy Principle (Friston 2010-2022)
  - Complementary Learning Systems (McClelland 1995)
  - Neurochemical modulation (DA/NE/ACh/SERT state machines)
  - Multi-tier memory with Hebbian plasticity & SRS forgetting
  - Bayesian metacognitive calibration
  - Narrative identity substrate (McAdams 1993)

Quick start
-----------
    from nexus import NexusBrain, NexusSettings

    brain = NexusBrain()
    await brain.initialize()
    result = await brain.think("Hello, what are you?")
    print(result.response)

Version:  1.0.0
Python:   3.10+
License:  MIT
"""

from nexus.brain import NexusBrain
from nexus.core.config import (
    AffectLabel,
    CognitionMode,
    Engram,
    Goal,
    Intent,
    MemoryTier,
    Modality,
    NeuroState,
    NexusSettings,
    PerceptualFeatures,
    TurnResult,
)

__version__ = "1.0.0"
__all__ = [
    "NexusBrain",
    "NexusSettings",
    "TurnResult",
    "NeuroState",
    "Engram",
    "Goal",
    "PerceptualFeatures",
    "CognitionMode",
    "Intent",
    "MemoryTier",
    "Modality",
    "AffectLabel",
]
