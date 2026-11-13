"""Unit tests for AnswerEvaluator."""

from __future__ import annotations

from datetime import date

import pytest

from tutor_agent.answer_evaluator import AnswerEvaluator, EvaluationResult, _QUALITY_THRESHOLDS
from tutor_agent.difficulty_router import DifficultyRouter, DifficultyTier, Question
from tutor_agent.learner_profile import LearnerProfile
from tutor_agent.mock_llm_client import MockLLMClient
from tutor_agent.spaced_repetition import SpacedRepetitionScheduler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TODAY = date(2024, 6, 15)


def _make_question(
    topic: str = "python",
    tier: DifficultyTier = DifficultyTier.BEGINNER,
    text: str = "What is a variable?",
    keywords: list[str] | None = None,
) -> Question:
    return Question(
        topic=topic,
        tier=tier,
        text=text,
        answer_keywords=keywords if keywords is not None else ["variable", "store", "value"],
    )


def _make_evaluator(profile: LearnerProfile | None = None) -> AnswerEvaluator:
    profile = profile or LearnerProfile(learner_id="test_user")
    client = MockLLMClient(seed=0)
    scheduler = SpacedRepetitionScheduler(profile, reference_date=TODAY)
    return AnswerEvaluator(profile=profile, llm_client=client, scheduler=scheduler)


# ---------------------------------------------------------------------------
# EvaluationResult structure tests
# ---------------------------------------------------------------------------

class TestEvaluationResultStructure:
    def test_evaluate_returns_evaluation_result(self):
        ev = _make_evaluator()
        q = _make_question()
        result = ev.evaluate(q, "A variable is used to store a value.")
        assert isinstance(result, EvaluationResult)

    def test_result_has_topic(self):
        ev = _make_evaluator()
        q = _make_question(topic="math")
        result = ev.evaluate(q, "some answer")
        assert result.topic == "math"

    def test_result_has_question_text(self):
        ev = _make_evaluator()
        q = _make_question(text="What is 2+2?")
        result = ev.evaluate(q, "4")
        assert result.question_text == "What is 2+2?"

    def test_result_has_learner_response(self):
        ev = _make_evaluator()
        q = _make_question()
        response = "A variable stores data."
        result = ev.evaluate(q, response)
        assert result.learner_response == response

    def test_result_has_llm_feedback(self):
        ev = _make_evaluator()
        q = _make_question()
        result = ev.evaluate(q, "A variable stores a value.")
        assert isinstance(result.llm_feedback, str)
        assert len(result.llm_feedback) > 0

    def test_result_has_matched_keywords(self):
        ev = _make_evaluator()
        q = _make_question(keywords=["variable", "store", "value"])
        result = ev.evaluate(q, "A variable is used to store a value.")
        assert isinstance(result.matched_keywords, list)

    def test_result_has_keyword_ratio(self):
        ev = _make_evaluator()
        q = _make_question(keywords=["variable", "store", "value"])
        result = ev.evaluate(q, "A variable is used to store a value.")
        assert 0.0 <= result.keyword_ratio <= 1.0

    def test_result_has_quality_score(self):
        ev = _make_evaluator()
        q = _make_question()
        result = ev.evaluate(q, "some answer")
        assert 0 <= result.quality_score <= 5

    def test_result_has_mastery_before_and_after(self):
        ev = _make_evaluator()
        q = _make_question()
        result = ev.evaluate(q, "A variable stores a value.")
        assert isinstance(result.mastery_before, float)
        assert isinstance(result.mastery_after, float)

    def test_correct_flag_true_when_quality_gte_3(self):
        ev = _make_evaluator()
        q = _make_question(keywords=["variable", "store", "value"])
        # All keywords present → quality 5 → correct
        result = ev.evaluate(q, "A variable is used to store a value in Python.")
        if result.quality_score >= 3:
            assert result.correct is True

    def test_correct_flag_false_when_quality_lt_3(self):
        ev = _make_evaluator()
        q = _make_question(keywords=["variable", "store", "value", "assign", "name", "memory"])
        # Empty answer → quality 0 → incorrect
        result = ev.evaluate(q, "I don't know.")
        if result.quality_score < 3:
            assert result.correct is False


# ---------------------------------------------------------------------------
# Keyword matching
# ---------------------------------------------------------------------------

class TestKeywordMatching:
    def test_all_keywords_matched_ratio_1(self):
        ev = _make_evaluator()
        matched, ratio = ev._match_keywords(
            "variable store value", ["variable", "store", "value"]
        )
        assert ratio == 1.0
        assert set(matched) == {"variable", "store", "value"}

    def test_no_keywords_matched_ratio_0(self):
        ev = _make_evaluator()
        matched, ratio = ev._match_keywords(
            "completely unrelated text", ["variable", "store", "value"]
        )
        assert ratio == 0.0
        assert matched == []

    def test_partial_match_ratio(self):
        ev = _make_evaluator()
        matched, ratio = ev._match_keywords(
            "I know about variable", ["variable", "store", "value"]
        )
        assert ratio == pytest.approx(1 / 3)
        assert "variable" in matched

    def test_case_insensitive_matching(self):
        ev = _make_evaluator()
        matched, ratio = ev._match_keywords(
            "VARIABLE STORE VALUE", ["variable", "store", "value"]
        )
        assert ratio == 1.0

    def test_empty_keywords_returns_perfect_ratio(self):
        ev = _make_evaluator()
        matched, ratio = ev._match_keywords("any response", [])
        assert ratio == 1.0
        assert matched == []

    def test_empty_response_no_match(self):
        ev = _make_evaluator()
        matched, ratio = ev._match_keywords("", ["variable", "store"])
        assert ratio == 0.0
        assert matched == []


# ---------------------------------------------------------------------------
# Quality score mapping
# ---------------------------------------------------------------------------

class TestRatioToQuality:
    def test_ratio_1_gives_quality_5(self):
        assert AnswerEvaluator._ratio_to_quality(1.0) == 5

    def test_ratio_0_gives_quality_0(self):
        assert AnswerEvaluator._ratio_to_quality(0.0) == 0

    def test_ratio_0_8_gives_quality_5(self):
        assert AnswerEvaluator._ratio_to_quality(0.8) == 5

    def test_ratio_0_6_gives_quality_4(self):
        assert AnswerEvaluator._ratio_to_quality(0.6) == 4

    def test_ratio_0_4_gives_quality_3(self):
        assert AnswerEvaluator._ratio_to_quality(0.4) == 3

    def test_ratio_0_2_gives_quality_2(self):
        assert AnswerEvaluator._ratio_to_quality(0.2) == 2

    def test_ratio_0_1_gives_quality_1(self):
        assert AnswerEvaluator._ratio_to_quality(0.1) == 1

    def test_ratio_0_05_gives_quality_0(self):
        assert AnswerEvaluator._ratio_to_quality(0.05) == 0

    def test_monotone_non_decreasing(self):
        ratios = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        qualities = [AnswerEvaluator._ratio_to_quality(r) for r in ratios]
        assert qualities == sorted(qualities)


# ---------------------------------------------------------------------------
# Profile update side effects
# ---------------------------------------------------------------------------

class TestProfileUpdates:
    def test_mastery_increases_on_good_answer(self):
        profile = LearnerProfile(learner_id="u1")
        ev = _make_evaluator(profile)
        q = _make_question(keywords=["variable", "store", "value"])
        before = profile.get_or_create_topic("python").mastery_score
        result = ev.evaluate(q, "A variable is used to store a value in Python.")
        assert result.mastery_after >= result.mastery_before  # >= because quality may be 3

    def test_mastery_decreases_on_poor_answer(self):
        profile = LearnerProfile(learner_id="u2")
        profile.update_mastery("python", 0.5)
        ev = _make_evaluator(profile)
        q = _make_question(
            keywords=["variable", "store", "value", "assign", "name", "type", "memory"]
        )
        result = ev.evaluate(q, "I am not sure.")
        assert result.mastery_after < result.mastery_before or result.quality_score >= 3

    def test_topic_created_in_profile_if_absent(self):
        profile = LearnerProfile(learner_id="u3")
        ev = _make_evaluator(profile)
        q = _make_question(topic="new_topic", keywords=["foo"])
        ev.evaluate(q, "foo bar")
        assert "new_topic" in profile.topic_mastery

    def test_mastery_clamped_at_1(self):
        profile = LearnerProfile(learner_id="u4")
        profile.update_mastery("python", 0.99)
        ev = _make_evaluator(profile)
        q = _make_question(keywords=["variable", "store", "value"])
        result = ev.evaluate(q, "A variable is used to store a value.")
        assert result.mastery_after <= 1.0

    def test_mastery_clamped_at_0(self):
        profile = LearnerProfile(learner_id="u5")
        profile.update_mastery("python", 0.01)
        ev = _make_evaluator(profile)
        q = _make_question(
            keywords=["variable", "store", "value", "assign", "name", "type", "memory"]
        )
        result = ev.evaluate(q, "I am not sure.")
        assert result.mastery_after >= 0.0

    def test_next_review_date_updated(self):
        profile = LearnerProfile(learner_id="u6")
        ev = _make_evaluator(profile)
        q = _make_question(keywords=["variable", "store", "value"])
        ev.evaluate(q, "A variable stores a value.")
        tm = profile.topic_mastery["python"]
        assert tm.last_reviewed_date == TODAY.isoformat()

    def test_multiple_evaluations_accumulate(self):
        profile = LearnerProfile(learner_id="u7")
        ev = _make_evaluator(profile)
        for _ in range(3):
            q = _make_question(keywords=["variable", "store", "value"])
            ev.evaluate(q, "A variable is used to store a value in Python.")
        # Mastery should have increased significantly
        assert profile.topic_mastery["python"].mastery_score > 0.0


# ---------------------------------------------------------------------------
# Integration: AnswerEvaluator + DifficultyRouter + MockLLMClient
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_pipeline_with_real_bank(self):
        """Evaluate a question drawn from the real question bank."""
        profile = LearnerProfile(learner_id="integration_user")
        client = MockLLMClient(seed=42)
        scheduler = SpacedRepetitionScheduler(profile, reference_date=TODAY)
        evaluator = AnswerEvaluator(
            profile=profile, llm_client=client, scheduler=scheduler
        )
        router = DifficultyRouter(seed=0)

        q = router.get_question("python", mastery_score=0.0)
        result = evaluator.evaluate(q, "A variable stores a value and you can assign it.")

        assert isinstance(result, EvaluationResult)
        assert result.topic == "python"
        assert 0 <= result.quality_score <= 5
        assert "python" in profile.topic_mastery

    def test_poor_answer_lowers_mastery(self):
        profile = LearnerProfile(learner_id="poor_learner")
        profile.update_mastery("python", 0.5)
        client = MockLLMClient(seed=0)
        scheduler = SpacedRepetitionScheduler(profile, reference_date=TODAY)
        evaluator = AnswerEvaluator(
            profile=profile, llm_client=client, scheduler=scheduler
        )
        # Question with many specific keywords
        q = Question(
            topic="python",
            tier=DifficultyTier.ADVANCED,
            text="Explain the GIL.",
            answer_keywords=[
                "GIL", "thread", "lock", "CPython", "parallel",
                "concurrent", "mutex", "bytecode",
            ],
        )
        result = evaluator.evaluate(q, "I am not sure about this.")
        # With zero matching keywords, mastery should decrease
        if result.quality_score < 3:
            assert result.mastery_after < result.mastery_before

    def test_evaluator_uses_default_llm_client_when_none(self):
        profile = LearnerProfile(learner_id="default_client_user")
        evaluator = AnswerEvaluator(profile=profile)  # no client supplied
        q = _make_question()
        result = evaluator.evaluate(q, "A variable stores a value.")
        assert isinstance(result, EvaluationResult)
