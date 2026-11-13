"""TutorAgent: orchestrator for the full adaptive tutoring session loop."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

from tutor_agent.answer_evaluator import AnswerEvaluator, EvaluationResult
from tutor_agent.difficulty_router import DifficultyRouter, Question
from tutor_agent.learner_profile import LearnerProfile
from tutor_agent.mock_llm_client import MockLLMClient
from tutor_agent.spaced_repetition import SpacedRepetitionScheduler


# ---------------------------------------------------------------------------
# Session summary
# ---------------------------------------------------------------------------

@dataclass
class SessionSummary:
    """
    Summary of a completed tutoring session.

    Attributes
    ----------
    session_id:        Unique identifier for the session.
    topics_covered:    Topics that were practiced.
    questions_asked:   Total number of questions presented.
    correct_answers:   Number of questions answered correctly (quality >= 3).
    evaluations:       Full list of EvaluationResult objects, one per question.
    """
    session_id: str
    topics_covered: List[str] = field(default_factory=list)
    questions_asked: int = 0
    correct_answers: int = 0
    evaluations: List[EvaluationResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Fraction of correctly answered questions (0.0 if none asked)."""
        if self.questions_asked == 0:
            return 0.0
        return self.correct_answers / self.questions_asked


# ---------------------------------------------------------------------------
# TutorAgent
# ---------------------------------------------------------------------------

class TutorAgent:
    """
    Orchestrates a full adaptive tutoring session.

    Session loop
    ------------
    1. Fetch topics due for review from the :class:`SpacedRepetitionScheduler`.
    2. For each due topic, route to the appropriate difficulty tier via
       :class:`DifficultyRouter`.
    3. Present the question to the learner (via ``input()`` by default, but
       injectable for tests).
    4. Evaluate the answer with :class:`AnswerEvaluator` and update the
       :class:`LearnerProfile`.
    5. Save the profile after each answer.
    6. Require explicit human confirmation before any destructive operation
       (profile reset / deletion).

    Parameters
    ----------
    profile:
        The learner's persistent profile.
    profile_path:
        File path where the profile is saved after updates.
    router:
        A :class:`DifficultyRouter` instance (uses default question bank if
        not supplied).
    llm_client:
        A :class:`MockLLMClient` (or compatible) for generating feedback.
    reference_date:
        Date treated as "today" for scheduling.  Defaults to
        ``date.today()``.
    max_questions_per_session:
        Maximum number of questions asked per ``run_session()`` call.
    input_fn:
        Callable used to read learner input.  Defaults to the built-in
        ``input``.  Inject a custom callable in tests.
    print_fn:
        Callable used to display output.  Defaults to ``print``.
    """

    def __init__(
        self,
        profile: LearnerProfile,
        profile_path: str,
        router: Optional[DifficultyRouter] = None,
        llm_client: Optional[MockLLMClient] = None,
        reference_date: Optional[date] = None,
        max_questions_per_session: int = 10,
        input_fn=None,
        print_fn=None,
    ) -> None:
        self.profile = profile
        self.profile_path = profile_path
        self.router = router or DifficultyRouter()
        self.llm_client = llm_client or MockLLMClient()
        self._today = reference_date or date.today()
        self.max_questions_per_session = max_questions_per_session
        self._input = input_fn or input
        self._print = print_fn or print

        self.scheduler = SpacedRepetitionScheduler(
            self.profile, reference_date=self._today
        )
        self.evaluator = AnswerEvaluator(
            profile=self.profile,
            llm_client=self.llm_client,
            scheduler=self.scheduler,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_session(self) -> SessionSummary:
        """
        Run a complete tutoring session and return a :class:`SessionSummary`.

        The session ends when:
        * All due topics have been practised, or
        * ``max_questions_per_session`` questions have been asked.
        """
        session_id = str(uuid.uuid4())
        session_record = self.profile.start_session(session_id=session_id)
        summary = SessionSummary(session_id=session_id)

        self._print(f"\n{'='*60}")
        self._print(f"  Tutoring Session  |  Learner: {self.profile.display_name or self.profile.learner_id}")
        self._print(f"{'='*60}")

        due_topics = self.scheduler.due_topics()

        if not due_topics:
            self._print("\nNo topics due for review today. Great work — come back later!")
            self.profile.end_session(session_id)
            self.profile.save(self.profile_path)
            return summary

        self._print(f"\nTopics due for review: {', '.join(due_topics)}")
        self._print(f"Questions this session: up to {self.max_questions_per_session}\n")

        asked_texts: List[str] = []
        questions_asked = 0

        for topic in due_topics:
            if questions_asked >= self.max_questions_per_session:
                break

            mastery = self.profile.get_or_create_topic(topic).mastery_score
            question = self.router.get_question(
                topic, mastery_score=mastery, exclude_texts=asked_texts
            )
            asked_texts.append(question.text)

            self._print(f"Topic: {topic}  |  Difficulty: {question.tier.value.upper()}")
            self._print(f"Mastery: {mastery:.0%}")
            self._print(f"\nQ: {question.text}")

            # Offer a hint on request
            if question.hint:
                hint_resp = self._input("Would you like a hint? (y/n): ").strip().lower()
                if hint_resp == "y":
                    self._print(f"Hint: {question.hint}")

            learner_response = self._input("\nYour answer: ").strip()

            evaluation = self.evaluator.evaluate(question, learner_response)

            self._print(f"\n--- Feedback ---")
            self._print(evaluation.llm_feedback)
            self._print(
                f"\nKeywords matched: {evaluation.matched_keywords} "
                f"({evaluation.keyword_ratio:.0%})"
            )
            self._print(
                f"Quality score: {evaluation.quality_score}/5  |  "
                f"{'✓ Correct' if evaluation.correct else '✗ Incorrect'}"
            )
            self._print(
                f"Mastery: {evaluation.mastery_before:.0%} → {evaluation.mastery_after:.0%}"
            )
            self._print("-" * 40 + "\n")

            # Update summary
            summary.questions_asked += 1
            if topic not in summary.topics_covered:
                summary.topics_covered.append(topic)
            if evaluation.correct:
                summary.correct_answers += 1
            summary.evaluations.append(evaluation)

            # Update session record
            session_record.questions_asked += 1
            if topic not in session_record.topics_practiced:
                session_record.topics_practiced.append(topic)
            if evaluation.correct:
                session_record.correct_answers += 1

            questions_asked += 1

            # Persist after each question
            self.profile.save(self.profile_path)

        # End session
        self.profile.end_session(session_id)
        self.profile.save(self.profile_path)

        self._print_session_summary(summary)
        return summary

    def reset_profile(self) -> bool:
        """
        Reset the learner profile to a blank state after human confirmation.

        Returns
        -------
        bool
            ``True`` if the reset was confirmed and performed, ``False`` if
            the learner cancelled.
        """
        self._print(
            "\n⚠️  WARNING: This will erase ALL mastery scores and session history."
        )
        confirmation = self._input(
            "Type 'RESET' to confirm, or anything else to cancel: "
        ).strip()

        if confirmation != "RESET":
            self._print("Reset cancelled.")
            return False

        # Preserve learner identity, wipe learning data
        self.profile.topic_mastery.clear()
        self.profile.session_history.clear()
        self.profile.save(self.profile_path)
        self._print("Profile reset. All mastery scores and session history cleared.")
        return True

    def delete_profile(self) -> bool:
        """
        Delete the learner profile file from disk after human confirmation.

        Returns
        -------
        bool
            ``True`` if deletion was confirmed and performed, ``False`` if
            the learner cancelled.
        """
        self._print(
            f"\n⚠️  WARNING: This will permanently delete the profile file: {self.profile_path}"
        )
        confirmation = self._input(
            "Type 'DELETE' to confirm, or anything else to cancel: "
        ).strip()

        if confirmation != "DELETE":
            self._print("Deletion cancelled.")
            return False

        if os.path.exists(self.profile_path):
            os.remove(self.profile_path)
            self._print(f"Profile deleted: {self.profile_path}")
        else:
            self._print("Profile file not found; nothing to delete.")
        return True

    def add_topics(self, topics: List[str]) -> None:
        """
        Register one or more new topics in the learner profile, scheduling
        them for immediate review.

        Parameters
        ----------
        topics:
            List of topic names to add.
        """
        for topic in topics:
            self.scheduler.schedule_new_topic(topic)
        self.profile.save(self.profile_path)
        self._print(f"Added topics: {', '.join(topics)}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _print_session_summary(self, summary: SessionSummary) -> None:
        """Render a brief end-of-session summary to the console."""
        self._print(f"{'='*60}")
        self._print("  Session Complete")
        self._print(f"{'='*60}")
        self._print(f"Questions asked : {summary.questions_asked}")
        self._print(f"Correct answers : {summary.correct_answers}")
        self._print(f"Accuracy        : {summary.accuracy:.0%}")
        self._print(f"Topics covered  : {', '.join(summary.topics_covered) or 'None'}")
        self._print(f"{'='*60}\n")
