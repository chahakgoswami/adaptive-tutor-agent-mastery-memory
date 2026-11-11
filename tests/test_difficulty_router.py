"""Unit tests for the DifficultyRouter module."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from tutor_agent.difficulty_router import (
    DifficultyRouter,
    DifficultyTier,
    Question,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_bank(tmp_path: Path) -> Path:
    """Write a minimal question bank YAML and return its path."""
    data = {
        "python": {
            "beginner": [
                {
                    "template": "What is a variable?",
                    "answer_keywords": ["variable", "store", "value"],
                    "hint": "It stores data.",
                }
            ],
            "intermediate": [
                {
                    "template": "Explain list comprehensions.",
                    "answer_keywords": ["list", "comprehension", "loop"],
                    "hint": "Concise list creation.",
                }
            ],
            "advanced": [
                {
                    "template": "What is the GIL?",
                    "answer_keywords": ["GIL", "thread", "lock"],
                    "hint": "It's about Python's threading model.",
                }
            ],
        },
        "math": {
            "beginner": [
                {
                    "template": "What is a prime number?",
                    "answer_keywords": ["prime", "divisible"],
                    "hint": "It has exactly two factors.",
                }
            ],
            "intermediate": [
                {
                    "template": "Explain derivatives.",
                    "answer_keywords": ["derivative", "slope", "rate"],
                    "hint": "Rate of change.",
                }
            ],
            "advanced": [
                {
                    "template": "State the fundamental theorem of calculus.",
                    "answer_keywords": ["integral", "derivative", "antiderivative"],
                    "hint": "Connects differentiation and integration.",
                }
            ],
        },
    }
    bank_path = tmp_path / "question_bank.yaml"
    with open(bank_path, "w") as fh:
        yaml.dump(data, fh)
    return bank_path


@pytest.fixture
def router(minimal_bank: Path) -> DifficultyRouter:
    """A DifficultyRouter backed by the minimal test bank."""
    return DifficultyRouter(bank_path=minimal_bank, seed=42)


@pytest.fixture
def single_tier_bank(tmp_path: Path) -> Path:
    """A bank where only beginner questions exist for a topic."""
    data = {
        "history": {
            "beginner": [
                {
                    "template": "When did WWII end?",
                    "answer_keywords": ["1945"],
                    "hint": "Mid-1940s.",
                }
            ]
        }
    }
    bank_path = tmp_path / "question_bank.yaml"
    with open(bank_path, "w") as fh:
        yaml.dump(data, fh)
    return bank_path


# ---------------------------------------------------------------------------
# DifficultyTier mapping tests
# ---------------------------------------------------------------------------

class TestGetTier:
    """Tests for mastery score → DifficultyTier mapping."""

    def test_zero_mastery_is_beginner(self, router):
        assert router.get_tier(0.0) == DifficultyTier.BEGINNER

    def test_low_mastery_is_beginner(self, router):
        assert router.get_tier(0.1) == DifficultyTier.BEGINNER
        assert router.get_tier(0.39) == DifficultyTier.BEGINNER

    def test_threshold_0_4_is_intermediate(self, router):
        assert router.get_tier(0.4) == DifficultyTier.INTERMEDIATE

    def test_mid_mastery_is_intermediate(self, router):
        assert router.get_tier(0.5) == DifficultyTier.INTERMEDIATE
        assert router.get_tier(0.74) == DifficultyTier.INTERMEDIATE

    def test_threshold_0_75_is_advanced(self, router):
        assert router.get_tier(0.75) == DifficultyTier.ADVANCED

    def test_high_mastery_is_advanced(self, router):
        assert router.get_tier(0.9) == DifficultyTier.ADVANCED
        assert router.get_tier(1.0) == DifficultyTier.ADVANCED

    def test_exactly_at_intermediate_threshold(self, router):
        assert router.get_tier(0.4) == DifficultyTier.INTERMEDIATE

    def test_just_below_intermediate_threshold(self, router):
        assert router.get_tier(0.399) == DifficultyTier.BEGINNER

    def test_just_below_advanced_threshold(self, router):
        assert router.get_tier(0.749) == DifficultyTier.INTERMEDIATE


# ---------------------------------------------------------------------------
# get_question basic behaviour
# ---------------------------------------------------------------------------

class TestGetQuestion:
    def test_returns_question_object(self, router):
        q = router.get_question("python", mastery_score=0.0)
        assert isinstance(q, Question)

    def test_question_has_correct_topic(self, router):
        q = router.get_question("python", mastery_score=0.0)
        assert q.topic == "python"

    def test_beginner_mastery_gives_beginner_question(self, router):
        q = router.get_question("python", mastery_score=0.1)
        assert q.tier == DifficultyTier.BEGINNER
        assert q.text == "What is a variable?"

    def test_intermediate_mastery_gives_intermediate_question(self, router):
        q = router.get_question("python", mastery_score=0.5)
        assert q.tier == DifficultyTier.INTERMEDIATE
        assert q.text == "Explain list comprehensions."

    def test_advanced_mastery_gives_advanced_question(self, router):
        q = router.get_question("python", mastery_score=0.9)
        assert q.tier == DifficultyTier.ADVANCED
        assert q.text == "What is the GIL?"

    def test_question_has_answer_keywords(self, router):
        q = router.get_question("python", mastery_score=0.0)
        assert isinstance(q.answer_keywords, list)
        assert len(q.answer_keywords) > 0

    def test_question_has_hint(self, router):
        q = router.get_question("python", mastery_score=0.0)
        assert isinstance(q.hint, str)
        assert len(q.hint) > 0

    def test_question_text_is_non_empty(self, router):
        q = router.get_question("math", mastery_score=0.5)
        assert len(q.text) > 0


# ---------------------------------------------------------------------------
# Topic key resolution
# ---------------------------------------------------------------------------

class TestTopicResolution:
    def test_exact_match(self, router):
        q = router.get_question("python", mastery_score=0.0)
        assert q.topic == "python"

    def test_case_insensitive_match(self, router):
        q = router.get_question("Python", mastery_score=0.0)
        assert q.tier == DifficultyTier.BEGINNER  # found and routed correctly

    def test_uppercase_topic(self, router):
        q = router.get_question("MATH", mastery_score=0.0)
        assert q.tier == DifficultyTier.BEGINNER

    def test_unknown_topic_returns_fallback(self, router):
        q = router.get_question("quantum_physics", mastery_score=0.5)
        assert "quantum_physics" in q.text
        assert q.metadata.get("fallback") is True

    def test_fallback_question_has_topic_name(self, router):
        q = router.get_question("nonexistent_topic", mastery_score=0.0)
        assert "nonexistent_topic" in q.text


# ---------------------------------------------------------------------------
# Tier fallback when only some tiers exist
# ---------------------------------------------------------------------------

class TestTierFallback:
    def test_advanced_mastery_falls_back_to_beginner(self, single_tier_bank):
        """If only beginner questions exist, advanced mastery falls back."""
        router = DifficultyRouter(bank_path=single_tier_bank, seed=0)
        q = router.get_question("history", mastery_score=0.9)
        # Should fall back to beginner
        assert q.tier == DifficultyTier.ADVANCED  # tier assigned by mastery
        # But the actual question text came from beginner pool
        assert q.text == "When did WWII end?"

    def test_intermediate_mastery_falls_back_to_beginner(self, single_tier_bank):
        router = DifficultyRouter(bank_path=single_tier_bank, seed=0)
        q = router.get_question("history", mastery_score=0.5)
        assert q.text == "When did WWII end?"

    def test_beginner_mastery_uses_beginner(self, single_tier_bank):
        router = DifficultyRouter(bank_path=single_tier_bank, seed=0)
        q = router.get_question("history", mastery_score=0.1)
        assert q.text == "When did WWII end?"


# ---------------------------------------------------------------------------
# exclude_texts deduplication
# ---------------------------------------------------------------------------

class TestExcludeTexts:
    def test_excluded_text_is_not_returned_when_alternative_exists(self, tmp_path):
        """If a question text is excluded, another should be chosen."""
        data = {
            "chemistry": {
                "beginner": [
                    {"template": "Q1", "answer_keywords": ["a"], "hint": ""},
                    {"template": "Q2", "answer_keywords": ["b"], "hint": ""},
                ]
            }
        }
        bank_path = tmp_path / "bank.yaml"
        with open(bank_path, "w") as fh:
            yaml.dump(data, fh)
        router = DifficultyRouter(bank_path=bank_path, seed=0)

        # First call gets one question
        q1 = router.get_question("chemistry", mastery_score=0.0)
        assert q1.text in ("Q1", "Q2")

        # Second call excludes the first
        q2 = router.get_question(
            "chemistry", mastery_score=0.0, exclude_texts=[q1.text]
        )
        assert q2.text != q1.text

    def test_all_excluded_returns_fallback_question(self, tmp_path):
        """When all questions in all tiers are excluded, fallback is returned."""
        data = {
            "biology": {
                "beginner": [
                    {"template": "Only Q", "answer_keywords": [], "hint": ""}
                ]
            }
        }
        bank_path = tmp_path / "bank.yaml"
        with open(bank_path, "w") as fh:
            yaml.dump(data, fh)
        router = DifficultyRouter(bank_path=bank_path, seed=0)

        q = router.get_question(
            "biology", mastery_score=0.0, exclude_texts=["Only Q"]
        )
        assert q.metadata.get("fallback") is True


# ---------------------------------------------------------------------------
# list_topics and available_tiers
# ---------------------------------------------------------------------------

class TestBankMetadata:
    def test_list_topics_returns_all_topics(self, router):
        topics = router.list_topics()
        assert "python" in topics
        assert "math" in topics

    def test_available_tiers_all_present(self, router):
        tiers = router.available_tiers("python")
        assert DifficultyTier.BEGINNER in tiers
        assert DifficultyTier.INTERMEDIATE in tiers
        assert DifficultyTier.ADVANCED in tiers

    def test_available_tiers_unknown_topic(self, router):
        tiers = router.available_tiers("does_not_exist")
        assert tiers == []

    def test_available_tiers_single_tier(self, single_tier_bank):
        router = DifficultyRouter(bank_path=single_tier_bank)
        tiers = router.available_tiers("history")
        assert tiers == [DifficultyTier.BEGINNER]


# ---------------------------------------------------------------------------
# File loading error handling
# ---------------------------------------------------------------------------

class TestBankLoading:
    def test_missing_bank_raises_file_not_found(self, tmp_path):
        router = DifficultyRouter(bank_path=tmp_path / "nonexistent.yaml")
        with pytest.raises(FileNotFoundError):
            router.list_topics()

    def test_invalid_yaml_structure_raises_value_error(self, tmp_path):
        bank_path = tmp_path / "bad.yaml"
        bank_path.write_text("- item1\n- item2\n")  # list, not dict
        router = DifficultyRouter(bank_path=bank_path)
        with pytest.raises(ValueError):
            router.list_topics()

    def test_bank_loaded_lazily(self, tmp_path):
        """Router can be constructed even if bank file doesn't exist yet."""
        router = DifficultyRouter(bank_path=tmp_path / "future.yaml")
        # No exception at construction time
        assert not router._loaded


# ---------------------------------------------------------------------------
# Tier fallback order helper
# ---------------------------------------------------------------------------

class TestTierFallbackOrder:
    def test_beginner_fallback_order(self):
        order = DifficultyRouter._tier_fallback_order(DifficultyTier.BEGINNER)
        assert order[0] == DifficultyTier.BEGINNER

    def test_advanced_fallback_order(self):
        order = DifficultyRouter._tier_fallback_order(DifficultyTier.ADVANCED)
        assert order[0] == DifficultyTier.ADVANCED

    def test_intermediate_fallback_order(self):
        order = DifficultyRouter._tier_fallback_order(DifficultyTier.INTERMEDIATE)
        assert order[0] == DifficultyTier.INTERMEDIATE

    def test_all_tiers_covered_in_fallback(self):
        for tier in DifficultyTier:
            order = DifficultyRouter._tier_fallback_order(tier)
            assert set(order) == set(DifficultyTier)


# ---------------------------------------------------------------------------
# Integration: router uses real question bank
# ---------------------------------------------------------------------------

class TestRealQuestionBank:
    """Smoke tests against the actual question_bank.yaml in the data/ folder."""

    def test_real_bank_loads(self):
        router = DifficultyRouter()  # uses default bank path
        topics = router.list_topics()
        assert len(topics) > 0

    def test_real_bank_python_all_tiers(self):
        router = DifficultyRouter(seed=0)
        for score, expected_tier in [
            (0.0, DifficultyTier.BEGINNER),
            (0.5, DifficultyTier.INTERMEDIATE),
            (0.9, DifficultyTier.ADVANCED),
        ]:
            q = router.get_question("python", mastery_score=score)
            assert q.tier == expected_tier
            assert len(q.text) > 0
            assert len(q.answer_keywords) > 0

    def test_real_bank_math_advanced(self):
        router = DifficultyRouter(seed=1)
        q = router.get_question("math", mastery_score=1.0)
        assert q.tier == DifficultyTier.ADVANCED
        assert isinstance(q, Question)

    def test_real_bank_unknown_topic_fallback(self):
        router = DifficultyRouter()
        q = router.get_question("underwater_basket_weaving", mastery_score=0.5)
        assert q.metadata.get("fallback") is True
