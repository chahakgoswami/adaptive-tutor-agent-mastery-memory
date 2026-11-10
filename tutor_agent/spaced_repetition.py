"""SpacedRepetitionScheduler: SM-2-inspired algorithm for topic scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional

from tutor_agent.learner_profile import LearnerProfile, TopicMastery


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_EASINESS_FACTOR = 1.3
_DEFAULT_EASINESS_FACTOR = 2.5

# Quality scores (0-5 scale, SM-2 convention)
QUALITY_BLACKOUT = 0       # complete blackout
QUALITY_INCORRECT_HARD = 1 # incorrect; correct seemed easy
QUALITY_INCORRECT_EASY = 2 # incorrect; correct answer easy to recall
QUALITY_CORRECT_HARD = 3   # correct with serious difficulty
QUALITY_CORRECT_GOOD = 4   # correct after hesitation
QUALITY_CORRECT_EASY = 5   # perfect response


# ---------------------------------------------------------------------------
# Review result
# ---------------------------------------------------------------------------

@dataclass
class ReviewResult:
    """Output of a single SM-2 update step."""
    topic: str
    new_interval_days: int
    new_easiness_factor: float
    new_repetitions: int
    next_review_date: str   # ISO date string
    mastery_delta: float    # change applied to mastery_score


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class SpacedRepetitionScheduler:
    """
    SM-2-inspired spaced repetition scheduler.

    Reads a LearnerProfile, computes which topics are due for review today
    (or overdue), and can apply review results back to the profile.
    """

    def __init__(self, profile: LearnerProfile, reference_date: Optional[date] = None) -> None:
        """
        Parameters
        ----------
        profile:
            The learner profile to read from / write to.
        reference_date:
            The date to treat as "today".  Defaults to ``date.today()``.
            Injecting a fixed date makes the scheduler fully testable.
        """
        self.profile = profile
        self._today = reference_date or date.today()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def due_topics(self) -> List[str]:
        """
        Return topics that are due for practice today, ordered by urgency.

        Ordering rules (most urgent first):
        1. Overdue topics first, sorted by how many days they are overdue
           (most overdue = highest priority).
        2. Topics due exactly today next.
        3. Within each group, topics with lower mastery scores come first
           (weakest knowledge prioritised).

        Topics whose next_review_date is in the future are excluded.
        """
        due: List[tuple[int, float, str]] = []  # (days_overdue desc, mastery asc, topic)

        for topic, tm in self.profile.topic_mastery.items():
            next_review = _parse_date(tm.next_review_date)
            days_overdue = (self._today - next_review).days  # negative = future
            if days_overdue >= 0:
                # Negate days_overdue so that sorting ascending gives most overdue first
                due.append((-days_overdue, tm.mastery_score, topic))

        # Sort: most overdue first (smallest -days_overdue), then lowest mastery
        due.sort(key=lambda x: (x[0], x[1]))
        return [topic for _, _, topic in due]

    def apply_review(
        self,
        topic: str,
        quality: int,
        reviewed_on: Optional[date] = None,
    ) -> ReviewResult:
        """
        Apply an SM-2 review for *topic* with the given *quality* score (0–5).

        Updates the TopicMastery record inside ``self.profile`` in place and
        returns a :class:`ReviewResult` describing the new state.

        Parameters
        ----------
        topic:
            The topic that was reviewed.
        quality:
            SM-2 quality score in [0, 5].
            * 0–2  → incorrect response  → interval resets, repetitions reset
            * 3–5  → correct response    → interval advances per SM-2 formula
        reviewed_on:
            The date on which the review took place.  Defaults to
            ``self._today``.
        """
        if not 0 <= quality <= 5:
            raise ValueError(f"quality must be in [0, 5], got {quality}")

        reviewed_on = reviewed_on or self._today
        tm = self.profile.get_or_create_topic(topic)

        new_ef, new_interval, new_reps = _sm2_update(
            easiness_factor=tm.easiness_factor,
            interval_days=tm.interval_days,
            repetitions=tm.repetitions,
            quality=quality,
        )

        # Mastery delta: quality maps linearly to [-0.2, +0.2]
        mastery_delta = _quality_to_mastery_delta(quality)
        new_mastery = max(0.0, min(1.0, tm.mastery_score + mastery_delta))

        # Compute next review date
        next_review_date = reviewed_on + timedelta(days=new_interval)

        # Persist changes back to the profile
        tm.easiness_factor = new_ef
        tm.interval_days = new_interval
        tm.repetitions = new_reps
        tm.mastery_score = new_mastery
        tm.next_review_date = next_review_date.isoformat()
        tm.last_reviewed_date = reviewed_on.isoformat()

        return ReviewResult(
            topic=topic,
            new_interval_days=new_interval,
            new_easiness_factor=new_ef,
            new_repetitions=new_reps,
            next_review_date=next_review_date.isoformat(),
            mastery_delta=mastery_delta,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def schedule_new_topic(self, topic: str) -> TopicMastery:
        """
        Register *topic* in the profile with default SM-2 initial state,
        scheduling it for review starting today.

        If the topic already exists it is returned unchanged.
        """
        tm = self.profile.get_or_create_topic(topic)
        # Only reset schedule if it has never been reviewed
        if tm.repetitions == 0 and tm.last_reviewed_date is None:
            tm.next_review_date = self._today.isoformat()
        return tm


# ---------------------------------------------------------------------------
# SM-2 core algorithm (pure function — easy to unit-test)
# ---------------------------------------------------------------------------

def _sm2_update(
    easiness_factor: float,
    interval_days: int,
    repetitions: int,
    quality: int,
) -> tuple[float, int, int]:
    """
    Core SM-2 update function.

    Returns
    -------
    (new_easiness_factor, new_interval_days, new_repetitions)

    Algorithm
    ---------
    * If quality < 3 (incorrect): reset repetitions to 0 and interval to 1.
    * If quality >= 3 (correct):
        - repetitions == 0  → interval = 1
        - repetitions == 1  → interval = 6
        - repetitions >= 2  → interval = round(prev_interval * EF)
      Then increment repetitions.
    * Update EF:  EF' = EF + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
      Clamp EF' to [1.3, ∞).
    """
    # --- Easiness factor update (always applied, even on failure per SM-2) ---
    ef_delta = 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
    new_ef = max(_MIN_EASINESS_FACTOR, easiness_factor + ef_delta)

    # --- Interval and repetition update ---
    if quality < 3:
        # Incorrect response — reset
        new_reps = 0
        new_interval = 1
    else:
        # Correct response
        if repetitions == 0:
            new_interval = 1
        elif repetitions == 1:
            new_interval = 6
        else:
            new_interval = max(1, round(interval_days * new_ef))
        new_reps = repetitions + 1

    return new_ef, new_interval, new_reps


def compute_next_review_date(
    last_reviewed: date,
    interval_days: int,
) -> date:
    """Return the next review date given a last-reviewed date and interval."""
    return last_reviewed + timedelta(days=interval_days)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(date_str: str) -> date:
    """Parse an ISO date string (YYYY-MM-DD) into a ``date`` object."""
    return date.fromisoformat(date_str)


def _quality_to_mastery_delta(quality: int) -> float:
    """
    Map a quality score (0-5) to a mastery score delta.

    0 → -0.20  (complete blackout)
    1 → -0.15
    2 → -0.05
    3 →  0.00  (barely correct — no change)
    4 → +0.10
    5 → +0.20  (perfect)
    """
    mapping = {
        0: -0.20,
        1: -0.15,
        2: -0.05,
        3:  0.00,
        4:  0.10,
        5:  0.20,
    }
    return mapping[quality]
