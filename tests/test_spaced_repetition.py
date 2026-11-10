"""Unit tests for SpacedRepetitionScheduler and SM-2 helpers."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from tutor_agent.learner_profile import LearnerProfile, TopicMastery
from tutor_agent.spaced_repetition import (
    SpacedRepetitionScheduler,
    _sm2_update,
    _quality_to_mastery_delta,
    compute_next_review_date,
    QUALITY_BLACKOUT,
    QUALITY_INCORRECT_HARD,
    QUALITY_INCORRECT_EASY,
    QUALITY_CORRECT_HARD,
    QUALITY_CORRECT_GOOD,
    QUALITY_CORRECT_EASY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TODAY = date(2024, 6, 15)  # fixed reference date for all tests


def _make_profile(*topics: str) -> LearnerProfile:
    """Create a LearnerProfile with the given topics pre-registered."""
    profile = LearnerProfile(learner_id="test_user")
    for topic in topics:
        profile.get_or_create_topic(topic)
    return profile


def _scheduler(profile: LearnerProfile, today: date = TODAY) -> SpacedRepetitionScheduler:
    return SpacedRepetitionScheduler(profile, reference_date=today)


# ---------------------------------------------------------------------------
# _sm2_update — core algorithm edge cases
# ---------------------------------------------------------------------------

class TestSM2Update:
    """Tests for the pure _sm2_update function."""

    # --- Initial state (repetitions=0) ---

    def test_first_correct_review_gives_interval_1(self):
        ef, interval, reps = _sm2_update(
            easiness_factor=2.5, interval_days=1, repetitions=0, quality=4
        )
        assert interval == 1
        assert reps == 1

    def test_first_incorrect_review_keeps_interval_1(self):
        ef, interval, reps = _sm2_update(
            easiness_factor=2.5, interval_days=1, repetitions=0, quality=1
        )
        assert interval == 1
        assert reps == 0

    # --- Second correct review (repetitions=1) ---

    def test_second_correct_review_gives_interval_6(self):
        ef, interval, reps = _sm2_update(
            easiness_factor=2.5, interval_days=1, repetitions=1, quality=4
        )
        assert interval == 6
        assert reps == 2

    # --- Subsequent reviews (repetitions>=2) ---

    def test_third_correct_review_uses_ef_multiplier(self):
        ef, interval, reps = _sm2_update(
            easiness_factor=2.5, interval_days=6, repetitions=2, quality=5
        )
        # EF update: 2.5 + 0.1 - 0*(...) = 2.6; interval = round(6 * 2.6) = 16
        expected_ef = max(1.3, 2.5 + 0.1 - 0 * (0.08 + 0 * 0.02))
        expected_interval = max(1, round(6 * expected_ef))
        assert interval == expected_interval
        assert reps == 3
        assert abs(ef - expected_ef) < 1e-9

    def test_long_running_interval_grows(self):
        """Verify interval keeps growing over multiple successful reviews."""
        ef = 2.5
        interval = 6
        reps = 2
        for _ in range(5):
            prev_interval = interval
            ef, interval, reps = _sm2_update(
                easiness_factor=ef, interval_days=interval,
                repetitions=reps, quality=5
            )
            assert interval > prev_interval

    # --- Incorrect responses reset state ---

    def test_incorrect_response_resets_reps_and_interval(self):
        for quality in (0, 1, 2):
            ef, interval, reps = _sm2_update(
                easiness_factor=2.5, interval_days=20, repetitions=5, quality=quality
            )
            assert reps == 0, f"quality={quality}: reps should reset"
            assert interval == 1, f"quality={quality}: interval should reset to 1"

    # --- Easiness factor floor ---

    def test_ef_never_goes_below_minimum(self):
        """Repeated failures must not push EF below 1.3."""
        ef = 1.3
        for _ in range(10):
            ef, interval, reps = _sm2_update(
                easiness_factor=ef, interval_days=1, repetitions=0, quality=0
            )
        assert ef >= 1.3

    def test_ef_increases_with_perfect_score(self):
        ef_before = 2.5
        new_ef, _, _ = _sm2_update(
            easiness_factor=ef_before, interval_days=1, repetitions=0, quality=5
        )
        assert new_ef > ef_before

    def test_ef_decreases_with_quality_3(self):
        """Quality 3 is 'barely correct' and should slightly lower EF."""
        ef_before = 2.5
        new_ef, _, _ = _sm2_update(
            easiness_factor=ef_before, interval_days=1, repetitions=1, quality=3
        )
        assert new_ef < ef_before

    def test_ef_unchanged_at_quality_4_approximately(self):
        """Quality 4 should leave EF almost unchanged (delta ≈ 0 for q=4 is not quite right;
        actual delta = 0.1 - 1*(0.08+1*0.02) = 0.1 - 0.10 = 0, so exactly unchanged."""
        ef_before = 2.5
        new_ef, _, _ = _sm2_update(
            easiness_factor=ef_before, interval_days=1, repetitions=0, quality=4
        )
        # delta = 0.1 - (5-4)*(0.08+(5-4)*0.02) = 0.1 - 1*(0.08+0.02) = 0.0
        assert abs(new_ef - ef_before) < 1e-9

    # --- Interval floor ---

    def test_interval_never_zero(self):
        """Interval must be at least 1 day for all quality values."""
        for quality in range(6):
            _, interval, _ = _sm2_update(
                easiness_factor=1.3, interval_days=1, repetitions=5, quality=quality
            )
            assert interval >= 1, f"quality={quality}: interval must be >= 1"


# ---------------------------------------------------------------------------
# _quality_to_mastery_delta
# ---------------------------------------------------------------------------

class TestQualityToMasteryDelta:
    def test_blackout_gives_largest_negative(self):
        assert _quality_to_mastery_delta(0) == -0.20

    def test_perfect_gives_largest_positive(self):
        assert _quality_to_mastery_delta(5) == 0.20

    def test_quality_3_gives_zero_delta(self):
        assert _quality_to_mastery_delta(3) == 0.0

    def test_monotonically_increasing(self):
        deltas = [_quality_to_mastery_delta(q) for q in range(6)]
        assert deltas == sorted(deltas)


# ---------------------------------------------------------------------------
# compute_next_review_date
# ---------------------------------------------------------------------------

class TestComputeNextReviewDate:
    def test_adds_interval_days(self):
        last = date(2024, 1, 1)
        result = compute_next_review_date(last, 7)
        assert result == date(2024, 1, 8)

    def test_one_day_interval(self):
        last = date(2024, 6, 15)
        result = compute_next_review_date(last, 1)
        assert result == date(2024, 6, 16)

    def test_zero_interval_same_day(self):
        last = date(2024, 6, 15)
        result = compute_next_review_date(last, 0)
        assert result == last


# ---------------------------------------------------------------------------
# SpacedRepetitionScheduler.due_topics
# ---------------------------------------------------------------------------

class TestDueTopics:
    def test_all_due_today_returned(self):
        profile = _make_profile("math", "history", "python")
        # All default to today
        scheduler = _scheduler(profile)
        due = scheduler.due_topics()
        assert set(due) == {"math", "history", "python"}

    def test_future_topic_excluded(self):
        profile = _make_profile("math")
        tm = profile.topic_mastery["math"]
        tm.next_review_date = (TODAY + timedelta(days=3)).isoformat()
        scheduler = _scheduler(profile)
        assert "math" not in scheduler.due_topics()

    def test_past_topic_included(self):
        profile = _make_profile("chemistry")
        tm = profile.topic_mastery["chemistry"]
        tm.next_review_date = (TODAY - timedelta(days=5)).isoformat()
        scheduler = _scheduler(profile)
        assert "chemistry" in scheduler.due_topics()

    def test_empty_profile_returns_empty_list(self):
        profile = LearnerProfile(learner_id="empty")
        scheduler = _scheduler(profile)
        assert scheduler.due_topics() == []

    def test_most_overdue_comes_first(self):
        profile = _make_profile("A", "B")
        profile.topic_mastery["A"].next_review_date = (TODAY - timedelta(days=1)).isoformat()
        profile.topic_mastery["B"].next_review_date = (TODAY - timedelta(days=10)).isoformat()
        scheduler = _scheduler(profile)
        due = scheduler.due_topics()
        assert due[0] == "B"  # 10 days overdue
        assert due[1] == "A"  # 1 day overdue

    def test_ties_broken_by_mastery_score(self):
        """When two topics are equally overdue, lowest mastery comes first."""
        profile = _make_profile("X", "Y")
        for name in ("X", "Y"):
            profile.topic_mastery[name].next_review_date = TODAY.isoformat()
        profile.topic_mastery["X"].mastery_score = 0.8
        profile.topic_mastery["Y"].mastery_score = 0.2
        scheduler = _scheduler(profile)
        due = scheduler.due_topics()
        assert due[0] == "Y"  # lower mastery
        assert due[1] == "X"

    def test_exactly_today_is_included(self):
        profile = _make_profile("algebra")
        profile.topic_mastery["algebra"].next_review_date = TODAY.isoformat()
        scheduler = _scheduler(profile)
        assert "algebra" in scheduler.due_topics()


# ---------------------------------------------------------------------------
# SpacedRepetitionScheduler.apply_review
# ---------------------------------------------------------------------------

class TestApplyReview:
    def test_apply_review_updates_profile_in_place(self):
        profile = _make_profile("python")
        scheduler = _scheduler(profile)
        result = scheduler.apply_review("python", quality=5)

        tm = profile.topic_mastery["python"]
        assert tm.interval_days == result.new_interval_days
        assert tm.easiness_factor == result.new_easiness_factor
        assert tm.repetitions == result.new_repetitions
        assert tm.next_review_date == result.next_review_date
        assert tm.last_reviewed_date == TODAY.isoformat()

    def test_apply_review_mastery_increases_on_correct(self):
        profile = _make_profile("math")
        profile.topic_mastery["math"].mastery_score = 0.5
        scheduler = _scheduler(profile)
        scheduler.apply_review("math", quality=5)
        assert profile.topic_mastery["math"].mastery_score > 0.5

    def test_apply_review_mastery_decreases_on_incorrect(self):
        profile = _make_profile("math")
        profile.topic_mastery["math"].mastery_score = 0.5
        scheduler = _scheduler(profile)
        scheduler.apply_review("math", quality=0)
        assert profile.topic_mastery["math"].mastery_score < 0.5

    def test_apply_review_mastery_clamped_at_1(self):
        profile = _make_profile("math")
        profile.topic_mastery["math"].mastery_score = 0.95
        scheduler = _scheduler(profile)
        scheduler.apply_review("math", quality=5)
        assert profile.topic_mastery["math"].mastery_score <= 1.0

    def test_apply_review_mastery_clamped_at_0(self):
        profile = _make_profile("math")
        profile.topic_mastery["math"].mastery_score = 0.05
        scheduler = _scheduler(profile)
        scheduler.apply_review("math", quality=0)
        assert profile.topic_mastery["math"].mastery_score >= 0.0

    def test_apply_review_invalid_quality_raises(self):
        profile = _make_profile("math")
        scheduler = _scheduler(profile)
        with pytest.raises(ValueError):
            scheduler.apply_review("math", quality=6)
        with pytest.raises(ValueError):
            scheduler.apply_review("math", quality=-1)

    def test_apply_review_creates_topic_if_absent(self):
        profile = LearnerProfile(learner_id="u1")
        scheduler = _scheduler(profile)
        result = scheduler.apply_review("new_topic", quality=4)
        assert "new_topic" in profile.topic_mastery
        assert result.topic == "new_topic"

    def test_apply_review_next_review_date_advances(self):
        profile = _make_profile("history")
        scheduler = _scheduler(profile)
        result = scheduler.apply_review("history", quality=4)
        next_date = date.fromisoformat(result.next_review_date)
        assert next_date > TODAY

    def test_apply_review_custom_reviewed_on_date(self):
        profile = _make_profile("science")
        custom_date = date(2024, 1, 10)
        scheduler = _scheduler(profile)
        result = scheduler.apply_review("science", quality=5, reviewed_on=custom_date)
        assert profile.topic_mastery["science"].last_reviewed_date == custom_date.isoformat()

    def test_failed_review_resets_interval_to_1(self):
        profile = _make_profile("chemistry")
        tm = profile.topic_mastery["chemistry"]
        tm.interval_days = 30
        tm.repetitions = 5
        scheduler = _scheduler(profile)
        result = scheduler.apply_review("chemistry", quality=1)
        assert result.new_interval_days == 1
        assert result.new_repetitions == 0

    def test_sequence_of_reviews_matches_sm2_progression(self):
        """Simulate a full successful learning sequence and verify interval growth."""
        profile = _make_profile("vocab")
        scheduler = _scheduler(profile)

        intervals = []
        for quality in [5, 5, 5, 5]:
            result = scheduler.apply_review("vocab", quality=quality)
            intervals.append(result.new_interval_days)

        # Intervals should grow: 1, 6, then increasing
        assert intervals[0] == 1
        assert intervals[1] == 6
        assert intervals[2] > intervals[1]
        assert intervals[3] > intervals[2]


# ---------------------------------------------------------------------------
# SpacedRepetitionScheduler.schedule_new_topic
# ---------------------------------------------------------------------------

class TestScheduleNewTopic:
    def test_new_topic_scheduled_for_today(self):
        profile = LearnerProfile(learner_id="u1")
        scheduler = _scheduler(profile)
        tm = scheduler.schedule_new_topic("physics")
        assert tm.next_review_date == TODAY.isoformat()
        assert "physics" in profile.topic_mastery

    def test_existing_topic_not_reset(self):
        profile = _make_profile("physics")
        future = (TODAY + timedelta(days=10)).isoformat()
        profile.topic_mastery["physics"].next_review_date = future
        profile.topic_mastery["physics"].repetitions = 3
        scheduler = _scheduler(profile)
        tm = scheduler.schedule_new_topic("physics")
        # Should not reset because repetitions > 0
        assert tm.next_review_date == future


# ---------------------------------------------------------------------------
# Integration: due_topics reflects apply_review side effects
# ---------------------------------------------------------------------------

class TestSchedulerIntegration:
    def test_reviewed_topic_no_longer_due_same_day(self):
        profile = _make_profile("algebra")
        scheduler = _scheduler(profile)
        assert "algebra" in scheduler.due_topics()

        scheduler.apply_review("algebra", quality=5)
        # Next review date should now be in the future
        assert "algebra" not in scheduler.due_topics()

    def test_failed_review_still_due_next_day(self):
        profile = _make_profile("geometry")
        scheduler_today = _scheduler(profile, today=TODAY)
        scheduler_today.apply_review("geometry", quality=0)
        # interval resets to 1 → due tomorrow
        scheduler_tomorrow = _scheduler(profile, today=TODAY + timedelta(days=1))
        assert "geometry" in scheduler_tomorrow.due_topics()

    def test_multi_topic_ordering_across_multiple_sessions(self):
        """Simulate two topics with different review histories and verify ordering."""
        profile = LearnerProfile(learner_id="learner")
        s = _scheduler(profile, today=TODAY)

        s.schedule_new_topic("topic_A")
        s.schedule_new_topic("topic_B")

        # topic_A reviewed successfully multiple times → higher mastery, further interval
        for _ in range(3):
            s.apply_review("topic_A", quality=5)

        # topic_B never reviewed → still due today
        due = _scheduler(profile, today=TODAY).due_topics()
        # topic_A should NOT be due today (interval has advanced)
        assert "topic_A" not in due
        assert "topic_B" in due
