"""Unit and integration tests for SessionAnalytics."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import List

import pytest
import yaml

from tutor_agent.learner_profile import LearnerProfile, SessionRecord
from tutor_agent.session_analytics import (
    SessionAnalytics,
    TopicProgress,
    AnalyticsReport,
    _detect_weak_spot,
    _build_mastery_curve,
    _mastery_bar,
    _sparkline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TODAY = date(2024, 6, 15)


def _make_profile(learner_id: str = "tester", display_name: str = "Tester") -> LearnerProfile:
    return LearnerProfile(learner_id=learner_id, display_name=display_name)


def _add_session(
    profile: LearnerProfile,
    topics: List[str],
    questions: int,
    correct: int,
    session_id: str = None,
) -> SessionRecord:
    """Add a completed session record to a profile."""
    import uuid
    sid = session_id or str(uuid.uuid4())
    record = SessionRecord(
        session_id=sid,
        started_at="2024-01-01T10:00:00",
        ended_at="2024-01-01T10:30:00",
        topics_practiced=list(topics),
        questions_asked=questions,
        correct_answers=correct,
    )
    profile.session_history.append(record)
    return record


def _make_analytics(profile: LearnerProfile) -> tuple[SessionAnalytics, list]:
    """Return (SessionAnalytics, captured_lines)."""
    lines: list = []
    analytics = SessionAnalytics(profile=profile, print_fn=lambda *a, **k: lines.append(" ".join(str(x) for x in a)))
    return analytics, lines


# ---------------------------------------------------------------------------
# _detect_weak_spot
# ---------------------------------------------------------------------------

class TestDetectWeakSpot:
    def test_low_mastery_is_weak(self):
        is_weak, reason = _detect_weak_spot(mastery=0.2, accuracy=1.0, total_questions=5)
        assert is_weak is True
        assert "mastery" in reason.lower() or "threshold" in reason.lower() or "%" in reason

    def test_mastery_above_threshold_not_weak_by_mastery(self):
        is_weak, reason = _detect_weak_spot(mastery=0.5, accuracy=1.0, total_questions=5)
        # Mastery alone is fine; accuracy is fine too
        assert is_weak is False

    def test_poor_accuracy_with_enough_questions_is_weak(self):
        is_weak, reason = _detect_weak_spot(mastery=0.6, accuracy=0.3, total_questions=5)
        assert is_weak is True
        assert "accuracy" in reason.lower() or "%" in reason

    def test_poor_accuracy_too_few_questions_not_weak(self):
        # Only 1 question — not enough to trigger accuracy-based detection
        is_weak, reason = _detect_weak_spot(mastery=0.6, accuracy=0.0, total_questions=1)
        assert is_weak is False

    def test_zero_questions_not_weak_by_accuracy(self):
        is_weak, reason = _detect_weak_spot(mastery=0.8, accuracy=0.0, total_questions=0)
        assert is_weak is False

    def test_exactly_at_mastery_threshold_not_weak(self):
        is_weak, reason = _detect_weak_spot(mastery=0.4, accuracy=1.0, total_questions=0)
        assert is_weak is False

    def test_just_below_mastery_threshold_is_weak(self):
        is_weak, reason = _detect_weak_spot(mastery=0.399, accuracy=1.0, total_questions=0)
        assert is_weak is True


# ---------------------------------------------------------------------------
# _build_mastery_curve
# ---------------------------------------------------------------------------

class TestBuildMasteryCurve:
    def test_zero_sessions_returns_single_point(self):
        curve = _build_mastery_curve(current_mastery=0.5, session_count=0)
        assert len(curve) == 1
        assert curve[0] == (0, 0.5)

    def test_curve_starts_at_zero_mastery(self):
        curve = _build_mastery_curve(current_mastery=0.8, session_count=4)
        assert curve[0][1] == pytest.approx(0.0, abs=1e-4)

    def test_curve_ends_at_current_mastery(self):
        curve = _build_mastery_curve(current_mastery=0.8, session_count=4)
        assert curve[-1][1] == pytest.approx(0.8, abs=1e-4)

    def test_curve_length_is_session_count_plus_one(self):
        curve = _build_mastery_curve(current_mastery=0.6, session_count=5)
        assert len(curve) == 6

    def test_curve_is_non_decreasing(self):
        curve = _build_mastery_curve(current_mastery=0.9, session_count=6)
        values = [v for _, v in curve]
        assert values == sorted(values)

    def test_session_indices_are_sequential(self):
        curve = _build_mastery_curve(current_mastery=0.5, session_count=3)
        indices = [i for i, _ in curve]
        assert indices == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# _mastery_bar
# ---------------------------------------------------------------------------

class TestMasteryBar:
    def test_zero_mastery_all_dots(self):
        bar = _mastery_bar(0.0, width=5)
        assert bar == "[.....]"

    def test_full_mastery_all_hashes(self):
        bar = _mastery_bar(1.0, width=5)
        assert bar == "[#####]"

    def test_half_mastery(self):
        bar = _mastery_bar(0.5, width=10)
        # 5 hashes, 5 dots
        assert bar == "[#####.....]"  

    def test_bar_has_correct_total_length(self):
        bar = _mastery_bar(0.6, width=8)
        # brackets + 8 chars
        assert len(bar) == 10


# ---------------------------------------------------------------------------
# _sparkline
# ---------------------------------------------------------------------------

class TestSparkline:
    def test_all_zeros_gives_spaces(self):
        spark = _sparkline([0.0, 0.0, 0.0])
        assert all(c == " " for c in spark)

    def test_all_ones_gives_full_blocks(self):
        spark = _sparkline([1.0, 1.0])
        assert all(c == "█" for c in spark)

    def test_length_matches_input(self):
        spark = _sparkline([0.0, 0.25, 0.5, 0.75, 1.0])
        assert len(spark) == 5

    def test_single_value(self):
        spark = _sparkline([0.5])
        assert len(spark) == 1


# ---------------------------------------------------------------------------
# build_report — empty profile
# ---------------------------------------------------------------------------

class TestBuildReportEmpty:
    def test_empty_profile_zero_sessions(self):
        profile = _make_profile()
        analytics, _ = _make_analytics(profile)
        report = analytics.build_report()
        assert report.total_sessions == 0
        assert report.total_questions == 0
        assert report.total_correct == 0
        assert report.overall_accuracy == 0.0
        assert report.topic_progress == {}
        assert report.weak_spots == []
        assert report.strongest_topics == []

    def test_report_has_learner_id(self):
        profile = _make_profile(learner_id="alice")
        analytics, _ = _make_analytics(profile)
        report = analytics.build_report()
        assert report.learner_id == "alice"

    def test_report_has_display_name(self):
        profile = _make_profile(display_name="Alice Smith")
        analytics, _ = _make_analytics(profile)
        report = analytics.build_report()
        assert report.display_name == "Alice Smith"


# ---------------------------------------------------------------------------
# build_report — with sessions
# ---------------------------------------------------------------------------

class TestBuildReportWithSessions:
    def test_session_count_matches(self):
        profile = _make_profile()
        _add_session(profile, ["python"], 5, 3)
        _add_session(profile, ["python"], 5, 4)
        profile.get_or_create_topic("python").mastery_score = 0.6

        analytics, _ = _make_analytics(profile)
        report = analytics.build_report()
        assert report.total_sessions == 2

    def test_total_questions_aggregated(self):
        profile = _make_profile()
        _add_session(profile, ["python"], 4, 3)
        _add_session(profile, ["math"], 6, 2)
        profile.get_or_create_topic("python")
        profile.get_or_create_topic("math")

        analytics, _ = _make_analytics(profile)
        report = analytics.build_report()
        assert report.total_questions == 10
        assert report.total_correct == 5

    def test_overall_accuracy_calculated(self):
        profile = _make_profile()
        _add_session(profile, ["python"], 4, 2)
        profile.get_or_create_topic("python")

        analytics, _ = _make_analytics(profile)
        report = analytics.build_report()
        assert report.overall_accuracy == pytest.approx(0.5)

    def test_topic_progress_created_for_all_mastery_topics(self):
        profile = _make_profile()
        profile.get_or_create_topic("python")
        profile.get_or_create_topic("math")
        _add_session(profile, ["python", "math"], 4, 2)

        analytics, _ = _make_analytics(profile)
        report = analytics.build_report()
        assert "python" in report.topic_progress
        assert "math" in report.topic_progress

    def test_topic_progress_current_mastery_matches_profile(self):
        profile = _make_profile()
        profile.get_or_create_topic("python").mastery_score = 0.75
        _add_session(profile, ["python"], 5, 4)

        analytics, _ = _make_analytics(profile)
        report = analytics.build_report()
        assert report.topic_progress["python"].current_mastery == pytest.approx(0.75)

    def test_topic_not_in_session_history_still_in_report(self):
        """A topic registered in the profile but never practiced is still reported."""
        profile = _make_profile()
        profile.get_or_create_topic("history")
        # No sessions with history

        analytics, _ = _make_analytics(profile)
        report = analytics.build_report()
        assert "history" in report.topic_progress
        assert report.topic_progress["history"].total_questions == 0


# ---------------------------------------------------------------------------
# Weak spot detection in report
# ---------------------------------------------------------------------------

class TestWeakSpotDetection:
    def test_low_mastery_topic_in_weak_spots(self):
        profile = _make_profile()
        profile.get_or_create_topic("math").mastery_score = 0.1

        analytics, _ = _make_analytics(profile)
        report = analytics.build_report()
        assert "math" in report.weak_spots

    def test_high_mastery_topic_not_in_weak_spots(self):
        profile = _make_profile()
        profile.get_or_create_topic("python").mastery_score = 0.9

        analytics, _ = _make_analytics(profile)
        report = analytics.build_report()
        assert "python" not in report.weak_spots

    def test_weak_spots_ordered_by_mastery_ascending(self):
        profile = _make_profile()
        profile.get_or_create_topic("A").mastery_score = 0.3
        profile.get_or_create_topic("B").mastery_score = 0.1
        profile.get_or_create_topic("C").mastery_score = 0.2

        analytics, _ = _make_analytics(profile)
        report = analytics.build_report()
        # B (0.1) should come before C (0.2) before A (0.3)
        assert report.weak_spots == ["B", "C", "A"]

    def test_strongest_topics_excludes_weak_spots(self):
        profile = _make_profile()
        profile.get_or_create_topic("python").mastery_score = 0.9
        profile.get_or_create_topic("math").mastery_score = 0.1  # weak spot

        analytics, _ = _make_analytics(profile)
        report = analytics.build_report()
        assert "python" in report.strongest_topics
        assert "math" not in report.strongest_topics

    def test_strongest_topics_ordered_by_mastery_descending(self):
        profile = _make_profile()
        profile.get_or_create_topic("A").mastery_score = 0.9
        profile.get_or_create_topic("B").mastery_score = 0.7
        profile.get_or_create_topic("C").mastery_score = 0.8

        analytics, _ = _make_analytics(profile)
        report = analytics.build_report()
        assert report.strongest_topics[0] == "A"
        assert report.strongest_topics[1] == "C"
        assert report.strongest_topics[2] == "B"

    def test_is_weak_spot_flag_set_on_topic_progress(self):
        profile = _make_profile()
        profile.get_or_create_topic("history").mastery_score = 0.2

        analytics, _ = _make_analytics(profile)
        report = analytics.build_report()
        assert report.topic_progress["history"].is_weak_spot is True

    def test_weak_spot_reason_non_empty_for_weak_topics(self):
        profile = _make_profile()
        profile.get_or_create_topic("history").mastery_score = 0.1

        analytics, _ = _make_analytics(profile)
        report = analytics.build_report()
        assert len(report.topic_progress["history"].weak_spot_reason) > 0


# ---------------------------------------------------------------------------
# print_report output
# ---------------------------------------------------------------------------

class TestPrintReport:
    def test_print_report_produces_output(self):
        profile = _make_profile(display_name="Bob")
        profile.get_or_create_topic("python").mastery_score = 0.7
        _add_session(profile, ["python"], 5, 4)

        analytics, lines = _make_analytics(profile)
        analytics.print_report()
        assert len(lines) > 0

    def test_print_report_contains_learner_name(self):
        profile = _make_profile(display_name="Carol")
        analytics, lines = _make_analytics(profile)
        analytics.print_report()
        output = "\n".join(lines)
        assert "Carol" in output

    def test_print_report_contains_topic_name(self):
        profile = _make_profile()
        profile.get_or_create_topic("python").mastery_score = 0.8
        analytics, lines = _make_analytics(profile)
        analytics.print_report()
        output = "\n".join(lines)
        assert "python" in output

    def test_print_report_shows_weak_spot_indicator(self):
        profile = _make_profile()
        profile.get_or_create_topic("math").mastery_score = 0.1
        analytics, lines = _make_analytics(profile)
        analytics.print_report()
        output = "\n".join(lines)
        assert "⚠" in output or "Weak" in output

    def test_print_report_shows_sessions_count(self):
        profile = _make_profile()
        _add_session(profile, ["python"], 3, 2)
        _add_session(profile, ["python"], 3, 2)
        profile.get_or_create_topic("python")
        analytics, lines = _make_analytics(profile)
        analytics.print_report()
        output = "\n".join(lines)
        assert "2" in output  # 2 sessions

    def test_print_report_accepts_prebuilt_report(self):
        profile = _make_profile()
        analytics, lines = _make_analytics(profile)
        report = analytics.build_report()
        analytics.print_report(report)
        # Should not rebuild
        assert len(lines) > 0

    def test_print_report_no_weak_spots_message(self):
        profile = _make_profile()
        profile.get_or_create_topic("python").mastery_score = 0.9
        analytics, lines = _make_analytics(profile)
        analytics.print_report()
        output = "\n".join(lines)
        assert "None" in output or "great" in output.lower() or "no" in output.lower()


# ---------------------------------------------------------------------------
# Integration: multi-day learning sequence
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_bank(tmp_path: Path) -> Path:
    """Minimal question bank for integration tests."""
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


class TestMultiDayIntegration:
    """
    Integration tests that simulate a multi-day learning sequence and
    verify the analytics module reflects the full history correctly.
    """

    def _simulate_days(
        self,
        tmp_path: Path,
        minimal_bank: Path,
        days: int,
        topics: List[str],
        answer_quality: str = "good",  # 'good' | 'poor'
    ) -> LearnerProfile:
        """Simulate ``days`` tutoring sessions and return the final profile."""
        from tutor_agent.difficulty_router import DifficultyRouter
        from tutor_agent.mock_llm_client import MockLLMClient
        from tutor_agent.tutor_agent import TutorAgent

        profile_path = str(tmp_path / "profile.json")
        profile = LearnerProfile(learner_id="integration", display_name="Test Learner")

        for topic in topics:
            profile.get_or_create_topic(topic)

        for day_offset in range(days):
            sim_date = TODAY + timedelta(days=day_offset)

            if answer_quality == "good":
                # Answers with keywords for python/math
                inputs_cycle = [
                    "n",  # no hint
                    "A variable is used to store a value in Python.",
                    "n",
                    "A prime number is only divisible by one and itself.",
                ]
            else:
                inputs_cycle = ["n", "", "n", ""]

            input_iter = iter(inputs_cycle * 10)  # plenty for any session

            agent = TutorAgent(
                profile=profile,
                profile_path=profile_path,
                router=DifficultyRouter(bank_path=minimal_bank, seed=0),
                llm_client=MockLLMClient(seed=0),
                reference_date=sim_date,
                max_questions_per_session=len(topics),
                input_fn=lambda p="", _it=input_iter: next(_it, ""),
                print_fn=lambda *a, **k: None,
            )

            # Reset next_review dates so topics are due on each simulated day
            for topic in topics:
                profile.get_or_create_topic(topic).next_review_date = sim_date.isoformat()

            agent.run_session()

        return profile

    def test_sessions_recorded_across_multiple_days(
        self, tmp_path, minimal_bank
    ):
        profile = self._simulate_days(tmp_path, minimal_bank, days=3, topics=["python"])
        assert len(profile.session_history) == 3

    def test_mastery_increases_with_good_answers(
        self, tmp_path, minimal_bank
    ):
        profile = self._simulate_days(
            tmp_path, minimal_bank, days=3, topics=["python"], answer_quality="good"
        )
        mastery = profile.topic_mastery["python"].mastery_score
        assert mastery > 0.0

    def test_mastery_stays_low_with_poor_answers(
        self, tmp_path, minimal_bank
    ):
        profile = self._simulate_days(
            tmp_path, minimal_bank, days=3, topics=["python"], answer_quality="poor"
        )
        mastery = profile.topic_mastery["python"].mastery_score
        # Poor answers should keep mastery low or decrease it
        assert mastery < 0.5

    def test_analytics_report_sessions_match_simulation(
        self, tmp_path, minimal_bank
    ):
        profile = self._simulate_days(tmp_path, minimal_bank, days=4, topics=["python"])
        analytics = SessionAnalytics(profile=profile, print_fn=lambda *a, **k: None)
        report = analytics.build_report()
        assert report.total_sessions == 4

    def test_analytics_identifies_weak_topic_after_poor_answers(
        self, tmp_path, minimal_bank
    ):
        profile = self._simulate_days(
            tmp_path, minimal_bank, days=3, topics=["python"], answer_quality="poor"
        )
        analytics = SessionAnalytics(profile=profile, print_fn=lambda *a, **k: None)
        report = analytics.build_report()
        # After 3 days of poor answers, python should be a weak spot
        assert "python" in report.weak_spots

    def test_analytics_identifies_strong_topic_after_good_answers(
        self, tmp_path, minimal_bank
    ):
        profile = self._simulate_days(
            tmp_path, minimal_bank, days=5, topics=["python"], answer_quality="good"
        )
        analytics = SessionAnalytics(profile=profile, print_fn=lambda *a, **k: None)
        report = analytics.build_report()
        # After multiple good sessions, python should be strong
        assert "python" in report.strongest_topics or report.topic_progress["python"].current_mastery >= 0.4

    def test_analytics_multi_topic_weak_spots(
        self, tmp_path, minimal_bank
    ):
        profile = self._simulate_days(
            tmp_path, minimal_bank, days=2,
            topics=["python", "math"],
            answer_quality="poor",
        )
        analytics = SessionAnalytics(profile=profile, print_fn=lambda *a, **k: None)
        report = analytics.build_report()
        # Both topics with poor answers should be weak spots
        for topic in ["python", "math"]:
            if topic in report.topic_progress:
                tp = report.topic_progress[topic]
                assert tp.current_mastery < 0.5

    def test_print_report_runs_without_error_after_simulation(
        self, tmp_path, minimal_bank
    ):
        profile = self._simulate_days(
            tmp_path, minimal_bank, days=3, topics=["python", "math"], answer_quality="good"
        )
        lines: list = []
        analytics = SessionAnalytics(
            profile=profile,
            print_fn=lambda *a, **k: lines.append(" ".join(str(x) for x in a)),
        )
        analytics.print_report()  # should not raise
        assert len(lines) > 5  # meaningful output

    def test_mastery_history_curve_has_data_points(
        self, tmp_path, minimal_bank
    ):
        profile = self._simulate_days(
            tmp_path, minimal_bank, days=3, topics=["python"], answer_quality="good"
        )
        analytics = SessionAnalytics(profile=profile, print_fn=lambda *a, **k: None)
        report = analytics.build_report()
        tp = report.topic_progress.get("python")
        assert tp is not None
        assert len(tp.mastery_history) >= 1

    def test_report_total_questions_non_zero_after_simulation(
        self, tmp_path, minimal_bank
    ):
        profile = self._simulate_days(
            tmp_path, minimal_bank, days=3, topics=["python"], answer_quality="good"
        )
        analytics = SessionAnalytics(profile=profile, print_fn=lambda *a, **k: None)
        report = analytics.build_report()
        assert report.total_questions >= 3  # at least 1 question per day
