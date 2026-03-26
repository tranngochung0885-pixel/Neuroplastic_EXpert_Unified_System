"""
nexus/cognition/perception.py
==============================
PerceptionEngine: multi-feature text analysis without external NLP libraries.

Extracts: intent, valence, arousal, complexity, novelty, salience, urgency,
causal markers, hedge count, negation count, entities, keywords.

Biologically inspired by sensory cortex pre-processing before thalamic relay.
"""

from __future__ import annotations

import math
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import ClassVar, Dict, FrozenSet, List, Optional, Set

from nexus.core.config import Intent, PerceptualFeatures


class PerceptionEngine:
    """
    Rich feature extraction from raw text.
    All features have functional downstream consequences in routing,
    neurochemistry integration, and memory encoding.
    """

    _STOPWORDS: ClassVar[FrozenSet[str]] = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "have",
        "has", "had", "do", "does", "did", "will", "would", "could", "should",
        "can", "may", "might", "to", "of", "in", "on", "at", "by", "for",
        "with", "about", "from", "into", "i", "you", "he", "she", "it", "we",
        "they", "my", "your", "our", "their", "its", "and", "but", "or", "so",
        "yet", "not", "just", "also", "very", "more", "than", "that", "this",
        "these", "those", "what", "which", "who", "then", "there", "if", "as",
        "when", "where", "how", "all", "any", "each", "both", "few", "some",
        "said", "say", "get", "go", "make", "know", "think", "see", "come",
        "want", "use", "find", "give", "tell", "feel", "put", "seem", "ask",
    })

    _POSITIVE: ClassVar[FrozenSet[str]] = frozenset({
        "good", "great", "excellent", "amazing", "wonderful", "love", "happy",
        "joy", "perfect", "fantastic", "brilliant", "beautiful", "awesome",
        "helpful", "clear", "success", "positive", "benefit", "enjoy", "thank",
        "interesting", "inspiring", "impressive", "creative", "elegant",
        "grateful", "appreciate", "excited", "thrilled", "confident",
        "optimistic", "proud", "satisfied", "delightful",
    })

    _NEGATIVE: ClassVar[FrozenSet[str]] = frozenset({
        "bad", "terrible", "awful", "hate", "sad", "fail", "error", "wrong",
        "broken", "problem", "issue", "confused", "frustrated", "disappoint",
        "never", "impossible", "afraid", "worried", "angry", "boring",
        "useless", "incorrect", "mislead", "painful", "difficult", "struggle",
        "anxious", "depressed", "hopeless", "helpless", "trapped", "scared",
        "fear", "regret", "guilt", "shame", "lonely", "hurt", "tired",
        "overwhelmed", "exhausted", "desperate",
    })

    _ANALYTICAL: ClassVar[FrozenSet[str]] = frozenset({
        "analyze", "analyse", "compare", "evaluate", "design", "synthesize",
        "prove", "derive", "argue", "explain", "demonstrate", "contrast",
        "distinguish", "investigate", "examine", "assess", "interpret",
        "reason", "deduce", "infer", "hypothesize", "formulate", "construct",
        "critique", "elaborate", "justify", "validate", "refute",
        "extrapolate",
    })

    _EMOTIONAL: ClassVar[FrozenSet[str]] = frozenset({
        "feel", "feeling", "felt", "emotion", "emotional", "mood", "sense",
        "sad", "happy", "anxious", "worried", "love", "lonely", "hurt",
        "afraid", "excited", "angry", "frustrated", "overwhelmed", "panic",
        "grief", "loss", "depressed", "heartbroken", "devastated", "joyful",
        "peaceful", "calm",
    })

    _CREATIVE: ClassVar[FrozenSet[str]] = frozenset({
        "create", "write", "imagine", "invent", "story", "poem", "compose",
        "novel", "fiction", "imagine", "design", "craft", "paint", "build",
        "generate", "brainstorm",
    })

    _CAUSAL: ClassVar[FrozenSet[str]] = frozenset({
        "because", "therefore", "hence", "thus", "since", "consequently",
        "causes", "leads", "results", "enables", "implies", "due", "owing",
        "reason", "mechanism", "confounder", "effect", "influence",
    })

    _PLANNING: ClassVar[FrozenSet[str]] = frozenset({
        "plan", "roadmap", "strategy", "goal", "should", "decision", "choose",
        "option", "next", "step", "approach", "best", "recommend",
    })

    _HEDGE: ClassVar[FrozenSet[str]] = frozenset({
        "maybe", "perhaps", "possibly", "might", "could", "probably",
        "likely", "seem", "appears", "apparently", "uncertain", "unclear",
        "roughly", "approximately", "sort", "somewhat", "rather",
    })

    _NEGATION_RE: ClassVar[List[re.Pattern]] = [
        re.compile(p)
        for p in [
            r"\bnot\b", r"\bnever\b", r"\bno\b", r"\bnor\b", r"\bneither\b",
            r"\bcannot\b", r"\bcan't\b", r"\bwon't\b", r"\bdon't\b",
            r"\bisn't\b", r"\baren't\b", r"\bwasn't\b", r"\bweren't\b",
            r"\bdidn't\b", r"\bwouldn't\b", r"\bcouldn't\b", r"\bshouldn't\b",
        ]
    ]

    _GREETING_RE = re.compile(
        r"^(hi|hello|hey|good\s*(morning|afternoon|evening)|greetings|howdy)\b",
        re.I,
    )
    _FAREWELL_RE = re.compile(
        r"^(bye|goodbye|see\s*you|farewell|later|take\s*care|good\s*night)\b",
        re.I,
    )
    _RECALL_RE = re.compile(
        r"\b(remember|recall|mentioned|said\s+earlier|last\s+time|before|previously|told\s+you)\b",
        re.I,
    )
    _ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")
    _WORD_RE = re.compile(r"\b[a-zA-Z]{2,}\b")

    def __init__(self) -> None:
        self._recent_kws: deque[str] = deque(maxlen=150)

    def parse(self, text: str) -> PerceptualFeatures:
        text = text.strip()
        text_lower = text.lower()
        tokens = [w.lower() for w in self._WORD_RE.findall(text)]
        content = [t for t in tokens if t not in self._STOPWORDS]
        word_set = set(tokens)
        content_set = set(content)

        # ── Keywords (TF, content-only) ────────────────────────────────
        freq: Dict[str, int] = Counter(w for w in content if len(w) >= 3)
        keywords = [w for w, _ in freq.most_common(15)]

        # ── Named entities ─────────────────────────────────────────────
        stop_titles = {
            "The", "A", "An", "In", "On", "At", "For", "To", "Of",
            "And", "Or", "But", "I", "You",
        }
        entities = list(
            dict.fromkeys(
                e for e in self._ENTITY_RE.findall(text) if e not in stop_titles
            )
        )[:8]

        # ── Sentiment ─────────────────────────────────────────────────
        pos = len(word_set & self._POSITIVE)
        neg = len(word_set & self._NEGATIVE)
        denom = max(pos + neg, 1)
        valence = max(-1.0, min(1.0, (pos - neg) / denom))

        # ── Arousal ───────────────────────────────────────────────────
        excl = text.count("!")
        emo_count = len(word_set & self._EMOTIONAL)
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        arousal = min(1.0, 0.18 + excl * 0.10 + emo_count * 0.04 + caps_ratio * 0.25)

        # ── Complexity ────────────────────────────────────────────────
        n_words = len(text.split())
        length_f = min(n_words / 60.0, 1.0)
        anal_f = min(0.40, len(content_set & self._ANALYTICAL) * 0.07)
        clause_f = len(re.findall(r"[,;:—–]", text)) * 0.025
        complexity = min(1.0, length_f * 0.55 + anal_f + clause_f)

        # ── Novelty ───────────────────────────────────────────────────
        kw_set = set(keywords[:10])
        seen = set(self._recent_kws)
        new_kws = kw_set - seen
        novelty = len(new_kws) / max(len(kw_set), 1) if kw_set else 0.4

        # ── Causal markers ────────────────────────────────────────────
        causal_count = len(content_set & self._CAUSAL)

        # ── Hedge count ───────────────────────────────────────────────
        hedge_count = sum(1 for h in self._HEDGE if h in text_lower)

        # ── Negation count ────────────────────────────────────────────
        negation_count = sum(1 for p in self._NEGATION_RE if p.search(text))

        # ── Question depth ────────────────────────────────────────────
        qdepth = 0
        if re.search(r"\?", text):
            qdepth = 1
            if any(w in content_set for w in ("why", "explain", "describe", "elaborate")):
                qdepth = 2
            if any(w in content_set for w in self._ANALYTICAL):
                qdepth = 3

        # ── Sentence count ────────────────────────────────────────────
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        sentence_count = max(1, len(sentences))

        # ── Lexical density ───────────────────────────────────────────
        lexical_density = len(content) / max(len(tokens), 1)

        # ── Salience ──────────────────────────────────────────────────
        salience = min(
            1.0,
            novelty * 0.30
            + complexity * 0.25
            + abs(valence) * 0.20
            + min(arousal, 0.9) * 0.15
            + min(qdepth / 3.0, 1.0) * 0.10,
        )

        # ── Urgency ───────────────────────────────────────────────────
        urgency_sigs = sum([
            1 if arousal > 0.70 else 0,
            1 if valence < -0.40 else 0,
            1 if excl > 1 else 0,
            1 if emo_count > 2 else 0,
            2 if any(w in text_lower for w in ("urgent", "emergency", "immediately", "help me")) else 0,
        ])
        urgency = min(1.0, urgency_sigs * 0.20)

        # ── Intent ────────────────────────────────────────────────────
        intent = self._classify_intent(
            text_lower, content_set, qdepth, causal_count
        )

        # Update recency buffer
        self._recent_kws.extend(keywords[:8])

        return PerceptualFeatures(
            text=text,
            tokens=tokens,
            keywords=keywords,
            entities=entities,
            intent=intent,
            valence=valence,
            arousal=arousal,
            complexity=complexity,
            novelty=novelty,
            salience=salience,
            urgency=urgency,
            question_depth=qdepth,
            causal_markers=causal_count,
            hedge_count=hedge_count,
            negation_count=negation_count,
            sentence_count=sentence_count,
            lexical_density=lexical_density,
        )

    def _classify_intent(
        self,
        text_lower: str,
        content_set: Set[str],
        qdepth: int,
        causal_count: int,
    ) -> Intent:
        if self._GREETING_RE.match(text_lower) or self._FAREWELL_RE.match(text_lower):
            return Intent.META if "who are you" in text_lower else Intent.UNKNOWN
        if self._RECALL_RE.search(text_lower):
            return Intent.RECALL
        if any(w in content_set for w in self._EMOTIONAL):
            return Intent.EMOTIONAL
        if any(w in content_set for w in self._CREATIVE):
            return Intent.CREATIVE
        if any(w in text_lower for w in ("who am i", "who are you", "your architecture",
                                          "how do you work", "consciousness", "yourself")):
            return Intent.META
        if any(w in content_set for w in self._PLANNING):
            return Intent.PLANNING
        if causal_count >= 2 or "why" in content_set:
            return Intent.CAUSAL
        if any(w in content_set for w in self._ANALYTICAL) or qdepth >= 2:
            return Intent.ANALYTICAL
        if "how to" in text_lower or "steps" in content_set:
            return Intent.PROCEDURAL
        if qdepth >= 1:
            return Intent.FACTUAL
        return Intent.UNKNOWN
