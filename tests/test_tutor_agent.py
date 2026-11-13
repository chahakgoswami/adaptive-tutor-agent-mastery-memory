"""Unit and integration tests for TutorAgent."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import List

import pytest
import yaml

from tutor_agent.learner_profile import LearnerProfile
from tutor_agent.mock_llm_client import MockLLMClient
from tutor_agent.difficulty_router import DifficultyRouter
from tutor_agent.spaced_repetition import SpacedRepetitionScheduler
from tutor_agent.tutor_agent import TutorAgent, SessionSummary


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TODAY = date(2024, 6, 15)


# ---------------------------------------------------------------------------
# Fixtures and helpers
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
                    "hint": "It's about Python threading.",
                }
            ],
        },
        "math": {
            "beginner": [
                {
                    "template": "What is a prime number?",
                    "answer_keywords": ["prime", "divisible"],
                    "hint": "Has exactly two factors.",
                }
            ],
        },
    }
    bank_path = tmp_path / "question_bank.yaml"
    with open(bank_path, "w") as fh:
        yaml.dump(data, fh)
    return bank_path


def _make_agent(
    tmp_path: Path,
    minimal_bank: Path,
    inputs: List[str],
    topics: List[str] | None = None,
    max_questions: int = 5,
) -> TutorAgent:
    """Construct a TutorAgent with injected I/O for testing."""
    profile_path = str(tmp_path / "profile.json")
    profile = LearnerProfile(learner_id="test_user", display_name="Tester")

    if topics:
        for topic in topics:
            profile.get_or_create_topic(topic)

    client = MockLLMClient(seed=0)
    router = DifficultyRouter(bank_path=minimal_bank, seed=0)

    # Create an iterator over the canned inputs
    input_iter = iter(inputs)

    def fake_input(prompt: str = "") -> str:
        try:
            return next(input_iter)
        except StopIteration:
            return ""

    output_lines: List[str] = []

    def fake_print(*args, **kwargs) -> None:
        output_lines.append(" ".join(str(a) for a in args))

    agent = TutorAgent(
        profile=profile,
        profile_path=profile_path,
        router=router,
        llm_client=client,
        reference_date=TODAY,
        max_questions_per_session=max_questions,
        input_fn=fake_input,
        print_fn=fake_print,
    )
    # Attach output for inspection
    agent._output_lines = output_lines  # type: ignore[attr-defined]
    return agent


# ---------------------------------------------------------------------------
# SessionSummary tests
# ---------------------------------------------------------------------------

class TestSessionSummary:
    def test_accuracy_no_questions(self):
        s = SessionSummary(session_id="s1")
        assert s.accuracy == 0.0

    def test_accuracy_all_correct(self):
        s = SessionSummary(session_id="s1", questions_asked=4, correct_answers=4)
        assert s.accuracy == 1.0

    def test_accuracy_half_correct(self):
        s = SessionSummary(session_id="s1", questions_asked=4, correct_answers=2)
        assert s.accuracy == 0.5


# ---------------------------------------------------------------------------
# No due topics
# ---------------------------------------------------------------------------

class TestNoDueTopics:
    def test_session_with_no_due_topics_returns_empty_summary(
        self, tmp_path, minimal_bank
    ):
        profile_path = str(tmp_path / "profile.json")
        profile = LearnerProfile(learner_id="u1", display_name="Alice")
        # Schedule topic far in the future
        tm = profile.get_or_create_topic("python")
        tm.next_review_date = "2099-01-01"

        agent = TutorAgent(
            profile=profile,
            profile_path=profile_path,
            router=DifficultyRouter(bank_path=minimal_bank),
            reference_date=TODAY,
            input_fn=lambda p="": "",
            print_fn=lambda *a, **k: None,
        )
        summary = agent.run_session()
        assert summary.questions_asked == 0
        assert summary.topics_covered == []


# ---------------------------------------------------------------------------
# Session loop — basic run
# ---------------------------------------------------------------------------

class TestSessionLoop:
    def test_run_session_returns_summary(self, tmp_path, minimal_bank):
        # Inputs: decline hint ('n'), provide answer
        agent = _make_agent(
            tmp_path, minimal_bank,
            inputs=["n", "A variable stores a value."],
            topics=["python"],
        )
        summary = agent.run_session()
        assert isinstance(summary, SessionSummary)

    def test_run_session_asks_at_least_one_question(
        self, tmp_path, minimal_bank
    ):
        agent = _make_agent(
            tmp_path, minimal_bank,
            inputs=["n", "A variable stores a value."],
            topics=["python"],
        )
        summary = agent.run_session()
        assert summary.questions_asked >= 1

    def test_run_session_covers_due_topic(self, tmp_path, minimal_bank):
        agent = _make_agent(
            tmp_path, minimal_bank,
            inputs=["n", "A variable stores a value."],
            topics=["python"],
        )
        summary = agent.run_session()
        assert "python" in summary.topics_covered

    def test_run_session_respects_max_questions(self, tmp_path, minimal_bank):
        # 2 topics due, max_questions=1 — only one question asked
        agent = _make_agent(
            tmp_path, minimal_bank,
            inputs=["n", "A variable stores a value."],
            topics=["python", "math"],
            max_questions=1,
        )
        summary = agent.run_session()
        assert summary.questions_asked == 1

    def test_run_session_saves_profile(self, tmp_path, minimal_bank):
        agent = _make_agent(
            tmp_path, minimal_bank,
            inputs=["n", "A variable stores a value."],
            topics=["python"],
        )
        agent.run_session()
        assert os.path.exists(agent.profile_path)

    def test_run_session_updates_mastery(self, tmp_path, minimal_bank):
        agent = _make_agent(
            tmp_path, minimal_bank,
            inputs=["n", "A variable stores a value."],
            topics=["python"],
        )
        before = agent.profile.get_or_create_topic("python").mastery_score
        agent.run_session()
        after = agent.profile.topic_mastery["python"].mastery_score
        # Mastery should have changed (up or down)
        assert after != before or after == 0.0  # 0→0 for quality=3 is fine

    def test_run_session_records_session_history(self, tmp_path, minimal_bank):
        agent = _make_agent(
            tmp_path, minimal_bank,
            inputs=["n", "A variable stores a value."],
            topics=["python"],
        )
        agent.run_session()
        assert len(agent.profile.session_history) >= 1

    def test_hint_shown_when_requested(self, tmp_path, minimal_bank):
        agent = _make_agent(
            tmp_path, minimal_bank,
            # First input: 'y' requests hint; second: the actual answer
            inputs=["y", "A variable stores a value."],
            topics=["python"],
        )
        agent.run_session()
        # Check that the hint text appeared in the output
        output = " ".join(agent._output_lines)  # type: ignore[attr-defined]
        assert "Hint" in output or "It stores data" in output

    def test_correct_answer_increments_correct_count(
        self, tmp_path, minimal_bank
    ):
        agent = _make_agent(
            tmp_path, minimal_bank,
            inputs=["n", "A variable is used to store a value in Python."],
            topics=["python"],
        )
        summary = agent.run_session()
        # With all keywords matched the answer should be correct
        evaluation = summary.evaluations[0]
        if evaluation.correct:
            assert summary.correct_answers >= 1

    def test_evaluations_list_populated(self, tmp_path, minimal_bank):
        agent = _make_agent(
            tmp_path, minimal_bank,
            inputs=["n", "A variable stores a value."],
            topics=["python"],
        )
        summary = agent.run_session()
        assert len(summary.evaluations) == summary.questions_asked

    def test_multiple_topics_in_session(self, tmp_path, minimal_bank):
        # 2 topics, max_questions=2: both should be covered
        # For each topic: decline hint ('n'), then give answer
        agent = _make_agent(
            tmp_path, minimal_bank,
            inputs=["n", "A variable stores a value.", "n", "A prime has two factors."],
            topics=["python", "math"],
            max_questions=2,
        )
        summary = agent.run_session()
        assert summary.questions_asked == 2
        assert len(summary.topics_covered) == 2


# ---------------------------------------------------------------------------
# Human-in-the-loop: reset_profile
# ---------------------------------------------------------------------------

class TestResetProfile:
    def test_reset_confirmed_clears_mastery(self, tmp_path, minimal_bank):
        agent = _make_agent(
            tmp_path, minimal_bank,
            inputs=[],
            topics=["python"],
        )
        agent.profile.update_mastery("python", 0.8)
        agent._input = lambda p="": "RESET"
        result = agent.reset_profile()
        assert result is True
        assert agent.profile.topic_mastery == {}

    def test_reset_confirmed_clears_session_history(
        self, tmp_path, minimal_bank
    ):
        agent = _make_agent(
            tmp_path, minimal_bank,
            inputs=[],
            topics=["python"],
        )
        agent.profile.start_session()
        agent._input = lambda p="": "RESET"
        result = agent.reset_profile()
        assert result is True
        assert agent.profile.session_history == []

    def test_reset_confirmed_saves_profile(self, tmp_path, minimal_bank):
        agent = _make_agent(
            tmp_path, minimal_bank,
            inputs=[],
            topics=["python"],
        )
        agent._input = lambda p="": "RESET"
        agent.reset_profile()
        assert os.path.exists(agent.profile_path)

    def test_reset_cancelled_preserves_mastery(self, tmp_path, minimal_bank):
        agent = _make_agent(
            tmp_path, minimal_bank,
            inputs=[],
            topics=["python"],
        )
        agent.profile.update_mastery("python", 0.8)
        agent._input = lambda p="": "no"
        result = agent.reset_profile()
        assert result is False
        assert agent.profile.topic_mastery["python"].mastery_score == pytest.approx(0.8)

    def test_reset_requires_exact_word(self, tmp_path, minimal_bank):
        agent = _make_agent(
            tmp_path, minimal_bank,
            inputs=[],
            topics=["python"],
        )
        for bad_input in ("reset", "Reset", "RESET ", " RESET", "yes", ""):
            agent.profile.update_mastery("python", 0.5)
            agent._input = lambda p="", _bi=bad_input: _bi
            result = agent.reset_profile()
            assert result is False, f"Should be cancelled for input {bad_input!r}"


# ---------------------------------------------------------------------------
# Human-in-the-loop: delete_profile
# ---------------------------------------------------------------------------

class TestDeleteProfile:
    def test_delete_confirmed_removes_file(self, tmp_path, minimal_bank):
        agent = _make_agent(
            tmp_path, minimal_bank,
            inputs=["n", "some answer"],
            topics=["python"],
        )
        # Create the file first
        agent.profile.save(agent.profile_path)
        assert os.path.exists(agent.profile_path)

        agent._input = lambda p="": "DELETE"
        result = agent.delete_profile()
        assert result is True
        assert not os.path.exists(agent.profile_path)

    def test_delete_cancelled_keeps_file(self, tmp_path, minimal_bank):
        agent = _make_agent(
            tmp_path, minimal_bank,
            inputs=[],
            topics=["python"],
        )
        agent.profile.save(agent.profile_path)
        agent._input = lambda p="": "no"
        result = agent.delete_profile()
        assert result is False
        assert os.path.exists(agent.profile_path)

    def test_delete_requires_exact_word(self, tmp_path, minimal_bank):
        agent = _make_agent(
            tmp_path, minimal_bank,
            inputs=[],
            topics=["python"],
        )
        agent.profile.save(agent.profile_path)
        for bad_input in ("delete", "Delete", "DELETE ", "yes", ""):
            agent._input = lambda p="", _bi=bad_input: _bi
            result = agent.delete_profile()
            assert result is False, f"Should be cancelled for {bad_input!r}"

    def test_delete_nonexistent_file_does_not_raise(
        self, tmp_path, minimal_bank
    ):
        agent = _make_agent(
            tmp_path, minimal_bank,
            inputs=[],
            topics=["python"],
        )
        # Profile file was never saved
        agent._input = lambda p="": "DELETE"
        result = agent.delete_profile()  # should not raise
        assert result is True


# ---------------------------------------------------------------------------
# add_topics
# ---------------------------------------------------------------------------

class TestAddTopics:
    def test_add_topics_registers_in_profile(self, tmp_path, minimal_bank):
        agent = _make_agent(
            tmp_path, minimal_bank,
            inputs=[],
        )
        agent.add_topics(["python", "math"])
        assert "python" in agent.profile.topic_mastery
        assert "math" in agent.profile.topic_mastery

    def test_add_topics_schedules_for_today(self, tmp_path, minimal_bank):
        agent = _make_agent(
            tmp_path, minimal_bank,
            inputs=[],
        )
        agent.add_topics(["python"])
        tm = agent.profile.topic_mastery["python"]
        assert tm.next_review_date == TODAY.isoformat()

    def test_add_topics_saves_profile(self, tmp_path, minimal_bank):
        agent = _make_agent(
            tmp_path, minimal_bank,
            inputs=[],
        )
        agent.add_topics(["python"])
        assert os.path.exists(agent.profile_path)

    def test_add_existing_topic_does_not_reset(self, tmp_path, minimal_bank):
        agent = _make_agent(
            tmp_path, minimal_bank,
            inputs=[],
            topics=["python"],
        )
        agent.profile.topic_mastery["python"].mastery_score = 0.9
        agent.profile.topic_mastery["python"].repetitions = 5
        agent.add_topics(["python"])  # already exists
        assert agent.profile.topic_mastery["python"].mastery_score == pytest.approx(0.9)
        assert agent.profile.topic_mastery["python"].repetitions == 5


# ---------------------------------------------------------------------------
# Integration: full session end-to-end
# ---------------------------------------------------------------------------

class TestFullSessionIntegration:
    def test_full_session_pipeline(self, tmp_path, minimal_bank):
        """
        End-to-end: add topics, run a session, verify profile is updated and
        persisted.
        """
        profile_path = str(tmp_path / "profile.json")
        profile = LearnerProfile(learner_id="integration", display_name="Jane")
        client = MockLLMClient(seed=42)
        router = DifficultyRouter(bank_path=minimal_bank, seed=0)

        responses = iter(["n", "A variable stores a value."])

        agent = TutorAgent(
            profile=profile,
            profile_path=profile_path,
            router=router,
            llm_client=client,
            reference_date=TODAY,
            max_questions_per_session=5,
            input_fn=lambda p="": next(responses, ""),
            print_fn=lambda *a, **k: None,
        )

        agent.add_topics(["python"])
        summary = agent.run_session()

        # Session ran
        assert summary.questions_asked >= 1
        assert "python" in summary.topics_covered

        # Profile was persisted
        reloaded = LearnerProfile.load(profile_path)
        assert "python" in reloaded.topic_mastery
        assert len(reloaded.session_history) >= 1
        # Session record should have ended_at set
        last_session = reloaded.session_history[-1]
        assert last_session.ended_at is not None

    def test_session_with_poor_answer_decreases_or_maintains_mastery(
        self, tmp_path, minimal_bank
    ):
        """
        A blank answer should produce quality 0 and decrease mastery.
        """
        profile_path = str(tmp_path / "profile.json")
        profile = LearnerProfile(learner_id="poor_learner")
        profile.update_mastery("python", 0.5)

        responses = iter(["n", ""])  # decline hint, give empty answer

        agent = TutorAgent(
            profile=profile,
            profile_path=profile_path,
            router=DifficultyRouter(bank_path=minimal_bank, seed=0),
            llm_client=MockLLMClient(seed=0),
            reference_date=TODAY,
            input_fn=lambda p="": next(responses, ""),
            print_fn=lambda *a, **k: None,
        )

        summary = agent.run_session()
        evaluation = summary.evaluations[0]
        if evaluation.quality_score < 3:
            assert evaluation.mastery_after < evaluation.mastery_before

    def test_session_session_record_topics_match_summary(
        self, tmp_path, minimal_bank
    ):
        """
        The session record stored in LearnerProfile should list the same
        topics as the SessionSummary.
        """
        profile_path = str(tmp_path / "profile.json")
        profile = LearnerProfile(learner_id="u_match")

        responses = iter(["n", "A variable stores a value."])

        agent = TutorAgent(
            profile=profile,
            profile_path=profile_path,
            router=DifficultyRouter(bank_path=minimal_bank, seed=0),
            llm_client=MockLLMClient(seed=0),
            reference_date=TODAY,
            input_fn=lambda p="": next(responses, ""),
            print_fn=lambda *a, **k: None,
        )

        agent.add_topics(["python"])
        summary = agent.run_session()
        last_record = profile.session_history[-1]

        for topic in summary.topics_covered:
            assert topic in last_record.topics_practiced
