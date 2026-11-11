"""DifficultyRouter: maps mastery scores to difficulty tiers and selects questions."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default path to the question bank YAML relative to this file's directory
_DEFAULT_BANK_PATH = Path(__file__).parent.parent / "data" / "question_bank.yaml"

# Mastery score thresholds for tier assignment
_INTERMEDIATE_THRESHOLD = 0.4   # score >= this → intermediate (if not advanced)
_ADVANCED_THRESHOLD = 0.75      # score >= this → advanced


# ---------------------------------------------------------------------------
# Difficulty tier enum
# ---------------------------------------------------------------------------

class DifficultyTier(str, Enum):
    """The three difficulty tiers supported by the router."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


# ---------------------------------------------------------------------------
# Question dataclass
# ---------------------------------------------------------------------------

@dataclass
class Question:
    """
    A structured question returned by the DifficultyRouter.

    Attributes
    ----------
    topic:
        The topic this question covers.
    tier:
        The difficulty tier of this question.
    text:
        The question text presented to the learner.
    answer_keywords:
        Keywords that indicate a correct or high-quality answer.
    hint:
        An optional hint shown to the learner on request.
    metadata:
        Any extra information from the question bank entry.
    """
    topic: str
    tier: DifficultyTier
    text: str
    answer_keywords: List[str] = field(default_factory=list)
    hint: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover
        return f"[{self.tier.value.upper()}] {self.text}"


# ---------------------------------------------------------------------------
# DifficultyRouter
# ---------------------------------------------------------------------------

class DifficultyRouter:
    """
    Routes a topic + mastery score to an appropriate :class:`Question`.

    The router:
    1. Loads a YAML question bank on first use (or on construction).
    2. Maps the mastery score to a :class:`DifficultyTier`.
    3. Selects a question template from the bank for that topic/tier.
    4. Returns a structured :class:`Question` object.

    Parameters
    ----------
    bank_path:
        Path to the YAML question bank.  Defaults to
        ``data/question_bank.yaml`` relative to the project root.
    seed:
        Optional random seed for deterministic question selection (useful
        in tests).
    """

    def __init__(
        self,
        bank_path: Optional[Path | str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._bank_path = Path(bank_path) if bank_path else _DEFAULT_BANK_PATH
        self._bank: Dict[str, Any] = {}
        self._rng = random.Random(seed)
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tier(self, mastery_score: float) -> DifficultyTier:
        """
        Map a mastery score (0.0–1.0) to a :class:`DifficultyTier`.

        Thresholds
        ----------
        * ``mastery_score < 0.4``  → BEGINNER
        * ``0.4 <= mastery_score < 0.75`` → INTERMEDIATE
        * ``mastery_score >= 0.75`` → ADVANCED

        Parameters
        ----------
        mastery_score:
            A float in [0.0, 1.0].

        Returns
        -------
        DifficultyTier
        """
        if mastery_score < _INTERMEDIATE_THRESHOLD:
            return DifficultyTier.BEGINNER
        if mastery_score < _ADVANCED_THRESHOLD:
            return DifficultyTier.INTERMEDIATE
        return DifficultyTier.ADVANCED

    def get_question(
        self,
        topic: str,
        mastery_score: float,
        exclude_texts: Optional[List[str]] = None,
    ) -> Question:
        """
        Select and return a :class:`Question` for *topic* at the appropriate
        difficulty tier.

        If the question bank does not contain the given topic, a generic
        fallback question is returned.  If the tier has no questions, the
        router falls back to the nearest available tier.

        Parameters
        ----------
        topic:
            The topic name to query (case-insensitive match attempted).
        mastery_score:
            The learner's current mastery score for the topic (0.0–1.0).
        exclude_texts:
            Optional list of question texts to exclude (avoids repeating
            questions in the same session).

        Returns
        -------
        Question
        """
        self._ensure_loaded()
        tier = self.get_tier(mastery_score)
        exclude_texts = exclude_texts or []

        # Try to find a matching topic key (case-insensitive)
        bank_key = self._resolve_topic_key(topic)

        if bank_key is None:
            return self._fallback_question(topic, tier)

        topic_bank = self._bank[bank_key]
        template = self._select_template(topic_bank, tier, exclude_texts)

        if template is None:
            return self._fallback_question(topic, tier)

        return Question(
            topic=topic,
            tier=tier,
            text=template["template"],
            answer_keywords=list(template.get("answer_keywords", [])),
            hint=template.get("hint", ""),
            metadata={"bank_key": bank_key},
        )

    def list_topics(self) -> List[str]:
        """Return the list of topics available in the question bank."""
        self._ensure_loaded()
        return list(self._bank.keys())

    def available_tiers(self, topic: str) -> List[DifficultyTier]:
        """Return the difficulty tiers available for *topic* in the bank."""
        self._ensure_loaded()
        bank_key = self._resolve_topic_key(topic)
        if bank_key is None:
            return []
        tiers = []
        topic_bank = self._bank[bank_key]
        for tier in DifficultyTier:
            if tier.value in topic_bank and topic_bank[tier.value]:
                tiers.append(tier)
        return tiers

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Lazy-load the question bank from the YAML file."""
        if not self._loaded:
            self._load_bank()

    def _load_bank(self) -> None:
        """Load and validate the YAML question bank."""
        if not self._bank_path.exists():
            raise FileNotFoundError(
                f"Question bank not found at: {self._bank_path}"
            )
        with open(self._bank_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ValueError(
                f"Question bank must be a YAML mapping, got {type(data).__name__}"
            )
        self._bank = data
        self._loaded = True

    def _resolve_topic_key(self, topic: str) -> Optional[str]:
        """
        Find the question bank key that matches *topic*.

        Matching strategy (in order):
        1. Exact match.
        2. Case-insensitive exact match.
        3. Substring match (topic contained in key or vice-versa).

        Returns ``None`` if no match is found.
        """
        # Exact match
        if topic in self._bank:
            return topic
        # Case-insensitive exact match
        topic_lower = topic.lower()
        for key in self._bank:
            if key.lower() == topic_lower:
                return key
        # Substring match
        for key in self._bank:
            if topic_lower in key.lower() or key.lower() in topic_lower:
                return key
        return None

    def _select_template(
        self,
        topic_bank: Dict[str, Any],
        tier: DifficultyTier,
        exclude_texts: List[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Select a question template from *topic_bank* for *tier*.

        Falls back to adjacent tiers if the preferred tier has no unused
        questions.  Fallback order:
        * BEGINNER → BEGINNER, INTERMEDIATE, ADVANCED
        * INTERMEDIATE → INTERMEDIATE, BEGINNER, ADVANCED
        * ADVANCED → ADVANCED, INTERMEDIATE, BEGINNER
        """
        fallback_order = self._tier_fallback_order(tier)
        for candidate_tier in fallback_order:
            templates = topic_bank.get(candidate_tier.value, [])
            if not templates:
                continue
            # Filter out excluded questions
            available = [
                t for t in templates
                if t.get("template", "") not in exclude_texts
            ]
            if not available:
                # All questions in this tier are excluded; try next
                continue
            return self._rng.choice(available)
        return None

    @staticmethod
    def _tier_fallback_order(tier: DifficultyTier) -> List[DifficultyTier]:
        """Return tiers in fallback priority order starting from *tier*."""
        all_tiers = [
            DifficultyTier.BEGINNER,
            DifficultyTier.INTERMEDIATE,
            DifficultyTier.ADVANCED,
        ]
        idx = all_tiers.index(tier)
        # Preferred tier first, then alternating outward
        order = [tier]
        for offset in range(1, len(all_tiers)):
            lower = idx - offset
            upper = idx + offset
            if upper < len(all_tiers):
                order.append(all_tiers[upper])
            if lower >= 0:
                order.append(all_tiers[lower])
        return order

    @staticmethod
    def _fallback_question(topic: str, tier: DifficultyTier) -> Question:
        """Return a generic fallback question when no bank entry exists."""
        return Question(
            topic=topic,
            tier=tier,
            text=f"Explain what you know about {topic} at a {tier.value} level.",
            answer_keywords=[topic.lower()],
            hint=f"Think about the key concepts related to {topic}.",
            metadata={"fallback": True},
        )
