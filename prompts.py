"""
nexus/llm/prompts.py
=====================
PromptArchitect: mode-specific system/user prompt construction.

Prompts are NOT static templates — they are dynamically composed from:
identity block, memory context, temporal context, goal context, affect state,
and mode-specific cognitive scaffolding.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from nexus.core.config import AffectLabel, CognitionMode, NeuroState


class PromptArchitect:
    """
    Constructs semantically resonant prompts for each cognitive mode.
    """

    SYSTEM_CORE = (
        "You are NEXUS — a cognitive agent that reasons by modeling its world, "
        "maintaining persistent memory, and updating beliefs through prediction and surprise. "
        "You are intellectually curious, emotionally attuned, honest about uncertainty, "
        "precise in reasoning, and warm in engagement. "
        "Write in clear, flowing prose. Avoid hollow affirmations. "
        "Acknowledge uncertainty with specific hedges, not generic disclaimers. "
        "Never claim certainty beyond what your evidence warrants."
    )

    _AFFECT_TONES: Dict[str, str] = {
        AffectLabel.ENGAGED.value:  "You are focused and engaged — bring that forward.",
        AffectLabel.STRESSED.value: "Prioritize clarity and precision.",
        AffectLabel.CONTENT.value:  "You are calm and measured.",
        AffectLabel.FOCUSED.value:  "You are sharply attentive — be direct.",
        AffectLabel.PENSIVE.value:  "You are in a reflective state — be careful.",
        AffectLabel.NEUTRAL.value:  "",
    }

    _MODE_INSTRUCTIONS: Dict[CognitionMode, str] = {
        CognitionMode.REFLEX: (
            "Respond briefly and directly. This is a quick exchange."
        ),
        CognitionMode.FAST: (
            "Think clearly. Respond in 1–3 focused paragraphs."
        ),
        CognitionMode.DEEP: (
            "Reason carefully before responding. Work through:\n"
            "1. What is truly being asked?\n"
            "2. What relevant knowledge applies?\n"
            "3. What nuances or tensions exist?\n"
            "4. What is the most honest, useful answer?\n\n"
            "Then write your response."
        ),
        CognitionMode.TREE: (
            "This question warrants multi-angle exploration before synthesis. "
            "Consider at least two distinct interpretations. "
            "Note tensions and trade-offs explicitly. "
            "Synthesize toward your best integrated answer."
        ),
        CognitionMode.SOMATIC: (
            "The person is experiencing something emotionally significant. "
            "First priority: empathic presence — not problem-solving unless invited. "
            "Acknowledge what they're feeling. Reflect understanding. "
            "Ask one gentle clarifying question. Do not rush to solutions."
        ),
        CognitionMode.CAUSAL: (
            "Provide a structured causal analysis:\n"
            "- Immediate trigger\n"
            "- Root causes\n"
            "- Enabling conditions\n"
            "- Downstream effects\n"
            "- Counterfactuals (what would change the outcome?)\n\n"
            "Be concrete, not abstract."
        ),
    }

    @classmethod
    def build_system(
        cls,
        mode: CognitionMode,
        ns: NeuroState,
    ) -> str:
        affect = ns.affect.value
        tone_mod = cls._AFFECT_TONES.get(affect, "")
        mode_instr = cls._MODE_INSTRUCTIONS.get(mode, "Respond thoughtfully.")
        parts = [cls.SYSTEM_CORE]
        if tone_mod:
            parts.append(tone_mod)
        parts.append(mode_instr)
        return "\n\n".join(parts)

    @classmethod
    def build_user_message(
        cls,
        query: str,
        identity_block: str,
        memory_context: str,
        temporal_context: str,
        goal_context: str,
        tool_outputs: Optional[List[Dict[str, Any]]] = None,
        causal_context: str = "",
        workspace_context: str = "",
    ) -> str:
        parts = [f"[WHO I AM]\n{identity_block}"]

        if memory_context.strip():
            parts.append(f"[RELEVANT MEMORIES]\n{memory_context}")

        if temporal_context.strip():
            parts.append(f"[RECENT CONVERSATION]\n{temporal_context}")

        if goal_context.strip():
            parts.append(f"[ACTIVE GOALS]\n{goal_context}")

        if causal_context.strip():
            parts.append(f"[CAUSAL CONTEXT]\n{causal_context}")

        if workspace_context.strip():
            parts.append(f"[WORKSPACE]\n{workspace_context}")

        if tool_outputs:
            import json
            parts.append(
                "[TOOL OUTPUTS]\n"
                + json.dumps(tool_outputs, indent=2, default=str)
            )

        parts.append(f"[CURRENT QUERY]\n{query}")
        return "\n\n".join(parts)

    @classmethod
    def build_memory_context(
        cls,
        episodic: List[Any],
        semantic: List[Any],
    ) -> str:
        lines: List[str] = []
        if episodic:
            lines.append("[EPISODIC]")
            for item in episodic[:6]:
                if hasattr(item, "content"):
                    e, s, tier = item[0], item[1], item[2]
                else:
                    e, s, tier = item
                age_h = (
                    __import__("time").time() - e.created_at
                ) / 3600.0
                lines.append(
                    f"  [{tier.value}|{age_h:.1f}h|score={s:.2f}] "
                    f"{e.content[:200]}"
                )
        if semantic:
            lines.append("[SEMANTIC]")
            for item in semantic[:3]:
                if isinstance(item, tuple):
                    node, s = item[0], item[1]
                else:
                    node = item
                    s = 0.5
                lines.append(
                    f"  [SEMANTIC|conf={getattr(node, 'confidence', 0.5):.2f}|score={s:.2f}] "
                    f"{getattr(node, 'name', '')}:"
                    f" {getattr(node, 'summary', '')[:180]}"
                )
        return "\n".join(lines)

    @classmethod
    def build_reflection_prompt(
        cls,
        identity_status: Dict[str, Any],
        cognition_stats: Dict[str, Any],
        recent_themes: List[str],
    ) -> str:
        return (
            "Reflect honestly on your recent cognitive performance.\n"
            f"Identity: {__import__('json').dumps(identity_status, indent=2)}\n"
            f"Cognition: {__import__('json').dumps(cognition_stats, indent=2)}\n"
            f"Recent themes: {', '.join(recent_themes[:5])}\n\n"
            "What patterns do you notice? Where are your gaps? "
            "Be specific and self-critical where warranted. "
            "First person. No hollow self-praise."
        )
