"""AnswerEvaluator: scores free-text learner responses and updates LearnerProfile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from tutor_agent.difficulty_router import Question
from tutor_agent.learner_profile import LearnerProfile
from tutor_agent.mock_llm_client import MockLLMClient
from tutor_agent.spaced_repetition import SpacedRepetitionScheduler


# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

# Fraction of answer_keywords that must appear in the learner's response
# to reach each quality threshold (SM-2 scale 0-5).
_QUALITY_THRESHOLDS = [
    # (min_keyword_ratio, quality_score)
    (0.80, 5),  # ≥ 80 % keywords → perfect
    (0.60, 4),  # ≥ 60 %          → good
    (0.40, 3),  # ≥ 40 %          → barely correct
    (0.20, 2),  # ≥ 20 %          → incorrect but recall easy
    (0.10, 1),  # ≥ 10 %          → incorrect, seemed easy
    (0.00, 0),  # < 10 %          → complete blackout
]


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """
    Result of evaluating a single learner response.

    Attributes
    ----------
    topic:
        The topic that was evaluated.
    question_text:
        The question that was asked.
    learner_response:
        The raw text provided by the learner.
    llm_feedback:
        The mock LLM's feedback / model answer.
    matched_keywords:
        Keywords from the question's answer_keywords list found in the
        learner's response.
    keyword_ratio:
        Fraction of answer_keywords matched (0.0 – 1.0).
    quality_score:
        SM-2 quality score assigned (0–5).
    mastery_before:
        Mastery score prior to this evaluation.
    mastery_after:
        Mastery score after updating the profile.
    correct:
        True when quality_score >= 3 (SM-2 convention for "correct").
    """
    topic: str
    question_text: str
    learner_response: str
    llm_feedback: str
    matched_keywords: List[str]
    keyword_ratio: float
    quality_score: int
    mastery_before: float
    mastery_after: float
    correct: bool


# ---------------------------------------------------------------------------
# AnswerEvaluator
# ---------------------------------------------------------------------------

class AnswerEvaluator:
    """
    Evaluates a learner's free-text answer to a :class:`~tutor_agent.difficulty_router.Question`.

    Scoring pipeline
    ----------------
    1. Ask the :class:`MockLLMClient` to generate contextual feedback / a
       model answer by passing the question + learner response.
    2. Count how many of the question's ``answer_keywords`` appear in the
       *learner's* response (case-insensitive).
    3. Map the keyword coverage ratio to an SM-2 quality score (0–5).
    4. Apply the review to the :class:`SpacedRepetitionScheduler`, which
       updates the :class:`LearnerProfile` in place.
    5. Return an :class:`EvaluationResult` with all details.

    Parameters
    ----------
    profile:
        The learner's profile to update.
    llm_client:
        A :class:`MockLLMClient` (or compatible object) for generating
        feedback.
    scheduler:
        An optional pre-built :class:`SpacedRepetitionScheduler`.  If
        ``None``, one is constructed from *profile*.
    """

    def __init__(
        self,
        profile: LearnerProfile,
        llm_client: Optional[MockLLMClient] = None,
        scheduler: Optional[SpacedRepetitionScheduler] = None,
    ) -> None:
        self.profile = profile
        self.llm_client = llm_client or MockLLMClient()
        self.scheduler = scheduler or SpacedRepetitionScheduler(profile)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        question: Question,
        learner_response: str,
    ) -> EvaluationResult:
        """
        Evaluate *learner_response* against *question* and update the profile.

        Parameters
        ----------
        question:
            The question that was asked (contains topic, tier, keywords).
        learner_response:
            The free-text answer provided by the learner.

        Returns
        -------
        EvaluationResult
        """
        topic = question.topic
        mastery_before = self.profile.get_or_create_topic(topic).mastery_score

        # 1. Get LLM feedback
        llm_feedback = self._get_llm_feedback(question, learner_response)

        # 2. Keyword matching
        matched, ratio = self._match_keywords(
            learner_response, question.answer_keywords
        )

        # 3. Determine quality score
        quality = self._ratio_to_quality(ratio)

        # 4. Apply SM-2 review (updates profile in place)
        self.scheduler.apply_review(topic, quality=quality)

        mastery_after = self.profile.topic_mastery[topic].mastery_score

        return EvaluationResult(
            topic=topic,
            question_text=question.text,
            learner_response=learner_response,
            llm_feedback=llm_feedback,
            matched_keywords=matched,
            keyword_ratio=ratio,
            quality_score=quality,
            mastery_before=mastery_before,
            mastery_after=mastery_after,
            correct=(quality >= 3),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_llm_feedback(self, question: Question, learner_response: str) -> str:
        """
        Call the mock LLM to generate contextual feedback.

        The prompt includes the question, tier, and learner response so the
        canned-response bank can pick something relevant.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful tutor. Given a question and a learner's "
                    "answer, provide brief constructive feedback and a model answer."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question.text}\n"
                    f"Topic: {question.topic}\n"
                    f"Difficulty: {question.tier.value}\n"
                    f"Learner's answer: {learner_response}\n\n"
                    "Please provide feedback and the correct answer."
                ),
            },
        ]
        completion = self.llm_client.chat.completions.create(
            model=self.llm_client.model,
            messages=messages,
        )
        return completion.first_content

    @staticmethod
    def _match_keywords(
        response: str,
        keywords: List[str],
    ) -> tuple[List[str], float]:
        """
        Find which *keywords* appear in *response* (case-insensitive).

        Returns
        -------
        (matched_keywords, ratio)
            *matched_keywords*: list of keywords found in the response.
            *ratio*: fraction of total keywords matched (0.0 if no keywords).
        """
        if not keywords:
            # No keywords → treat as a perfect answer (we can't judge)
            return [], 1.0

        response_lower = response.lower()
        matched = [
            kw for kw in keywords
            if kw.lower() in response_lower
        ]
        ratio = len(matched) / len(keywords)
        return matched, ratio

    @staticmethod
    def _ratio_to_quality(ratio: float) -> int:
        """
        Map a keyword-match ratio (0.0–1.0) to an SM-2 quality score (0–5).

        Thresholds
        ----------
        ratio >= 0.80 → 5 (perfect)
        ratio >= 0.60 → 4 (good)
        ratio >= 0.40 → 3 (barely correct)
        ratio >= 0.20 → 2 (incorrect, easy recall)
        ratio >= 0.10 → 1 (incorrect, seemed easy)
        ratio <  0.10 → 0 (complete blackout)
        """
        for min_ratio, quality in _QUALITY_THRESHOLDS:
            if ratio >= min_ratio:
                return quality
        return 0  # unreachable but satisfies type checkers
