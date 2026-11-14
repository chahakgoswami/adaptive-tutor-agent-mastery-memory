"""SessionAnalytics: progress curves, weak-spot detection, and text reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from tutor_agent.learner_profile import LearnerProfile, SessionRecord, TopicMastery


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TopicProgress:
    """
    Per-topic analytics derived from a learner profile.

    Attributes
    ----------
    topic:
        Topic name.
    mastery_history:
        List of (session_index, mastery_score) tuples representing snapshots
        of mastery over time (approximated from session history).
    current_mastery:
        The current mastery score stored in the profile.
    total_questions:
        Total questions answered for this topic across all sessions.
    total_correct:
        Total correct answers for this topic across all sessions.
    accuracy:
        Fraction of questions answered correctly (0.0 if none asked).
    is_weak_spot:
        True when this topic is flagged as a weak spot.
    weak_spot_reason:
        Human-readable explanation of why this topic is a weak spot.
    """
    topic: str
    mastery_history: List[Tuple[int, float]] = field(default_factory=list)
    current_mastery: float = 0.0
    total_questions: int = 0
    total_correct: int = 0
    accuracy: float = 0.0
    is_weak_spot: bool = False
    weak_spot_reason: str = ""


@dataclass
class AnalyticsReport:
    """
    Full analytics report for a learner profile.

    Attributes
    ----------
    learner_id:
        The learner's unique identifier.
    display_name:
        Human-readable learner name.
    total_sessions:
        Total number of sessions recorded.
    total_questions:
        Grand total questions asked across all sessions.
    total_correct:
        Grand total correct answers.
    overall_accuracy:
        Fraction correct across all sessions.
    topic_progress:
        Per-topic analytics, keyed by topic name.
    weak_spots:
        List of topic names flagged as weak spots (ordered worst first).
    strongest_topics:
        List of topic names with highest mastery (ordered best first).
    """
    learner_id: str
    display_name: str = ""
    total_sessions: int = 0
    total_questions: int = 0
    total_correct: int = 0
    overall_accuracy: float = 0.0
    topic_progress: Dict[str, TopicProgress] = field(default_factory=dict)
    weak_spots: List[str] = field(default_factory=list)
    strongest_topics: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Weak-spot detection thresholds
# ---------------------------------------------------------------------------

# Topics with mastery below this are always weak spots
_WEAK_MASTERY_THRESHOLD = 0.4

# Topics with accuracy below this (and enough questions) are weak spots
_WEAK_ACCURACY_THRESHOLD = 0.5

# Minimum questions needed before accuracy-based detection kicks in
_MIN_QUESTIONS_FOR_ACCURACY = 2

# Number of top/bottom topics to include in summary lists
_SUMMARY_LIST_SIZE = 5


# ---------------------------------------------------------------------------
# SessionAnalytics
# ---------------------------------------------------------------------------

class SessionAnalytics:
    """
    Reads a :class:`LearnerProfile` and derives progress analytics.

    Usage
    -----
    ::

        analytics = SessionAnalytics(profile)
        report = analytics.build_report()
        analytics.print_report(report)

    Parameters
    ----------
    profile:
        The learner profile to analyse.
    print_fn:
        Callable used to render output.  Defaults to ``print``.
    """

    def __init__(
        self,
        profile: LearnerProfile,
        print_fn=None,
    ) -> None:
        self.profile = profile
        self._print = print_fn or print

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_report(self) -> AnalyticsReport:
        """
        Compute and return a full :class:`AnalyticsReport` for the profile.

        Steps
        -----
        1. Aggregate session-level statistics (questions, correct answers).
        2. Build per-topic :class:`TopicProgress` records, including a
           mastery history curve reconstructed from session order.
        3. Detect weak spots.
        4. Rank topics by current mastery.
        """
        report = AnalyticsReport(
            learner_id=self.profile.learner_id,
            display_name=self.profile.display_name,
            total_sessions=len(self.profile.session_history),
        )

        # --- Aggregate session totals ---
        for session in self.profile.session_history:
            report.total_questions += session.questions_asked
            report.total_correct += session.correct_answers

        if report.total_questions > 0:
            report.overall_accuracy = report.total_correct / report.total_questions

        # --- Per-topic statistics from session history ---
        # topic -> (questions, correct) accumulated across sessions
        topic_stats: Dict[str, List[int]] = {}  # [total_q, total_correct]
        # topic -> ordered list of (session_idx, mastery_snapshot)
        # We approximate mastery snapshots from the final stored mastery and
        # the session's correct/incorrect ratio, working backward is imprecise,
        # so we use session index as x-axis and reconstruct a simple curve.
        topic_session_counts: Dict[str, int] = {}

        for session_idx, session in enumerate(self.profile.session_history):
            for topic in session.topics_practiced:
                if topic not in topic_stats:
                    topic_stats[topic] = [0, 0]
                # Distribute questions/correct proportionally across topics
                # (session record stores totals, not per-topic breakdown).
                # We credit each practiced topic equally for per-topic
                # accuracy approximation.
                n_topics = max(1, len(session.topics_practiced))
                q_share = session.questions_asked // n_topics
                c_share = session.correct_answers // n_topics
                topic_stats[topic][0] += q_share
                topic_stats[topic][1] += c_share
                topic_session_counts[topic] = (
                    topic_session_counts.get(topic, 0) + 1
                )

        # --- Build TopicProgress for every topic in the profile ---
        for topic, tm in self.profile.topic_mastery.items():
            stats = topic_stats.get(topic, [0, 0])
            total_q, total_c = stats
            accuracy = total_c / total_q if total_q > 0 else 0.0

            # Build a simple mastery history curve:
            # We only have the current mastery; we simulate a curve by using
            # session count as x-axis and linearly interpolating from 0.0.
            mastery_history = _build_mastery_curve(
                current_mastery=tm.mastery_score,
                session_count=topic_session_counts.get(topic, 0),
            )

            weak, reason = _detect_weak_spot(
                mastery=tm.mastery_score,
                accuracy=accuracy,
                total_questions=total_q,
            )

            tp = TopicProgress(
                topic=topic,
                mastery_history=mastery_history,
                current_mastery=tm.mastery_score,
                total_questions=total_q,
                total_correct=total_c,
                accuracy=accuracy,
                is_weak_spot=weak,
                weak_spot_reason=reason,
            )
            report.topic_progress[topic] = tp

        # --- Detect weak spots and rank topics ---
        report.weak_spots = [
            tp.topic
            for tp in sorted(
                report.topic_progress.values(),
                key=lambda x: x.current_mastery,
            )
            if tp.is_weak_spot
        ]

        report.strongest_topics = [
            tp.topic
            for tp in sorted(
                report.topic_progress.values(),
                key=lambda x: x.current_mastery,
                reverse=True,
            )
            if not tp.is_weak_spot
        ][:_SUMMARY_LIST_SIZE]

        return report

    def print_report(self, report: Optional[AnalyticsReport] = None) -> None:
        """
        Render a text-based progress report to the console.

        Parameters
        ----------
        report:
            Pre-built report.  If ``None``, :meth:`build_report` is called
            automatically.
        """
        if report is None:
            report = self.build_report()

        width = 62
        sep = "=" * width
        thin = "-" * width

        self._print(sep)
        self._print(f"  LEARNING PROGRESS REPORT")
        name = report.display_name or report.learner_id
        self._print(f"  Learner : {name}")
        self._print(sep)

        # Overall statistics
        self._print(f"\nOverall Statistics")
        self._print(thin)
        self._print(f"  Sessions completed : {report.total_sessions}")
        self._print(f"  Questions answered : {report.total_questions}")
        self._print(f"  Correct answers    : {report.total_correct}")
        self._print(f"  Overall accuracy   : {report.overall_accuracy:.0%}")

        # Per-topic progress
        if report.topic_progress:
            self._print(f"\nPer-Topic Progress")
            self._print(thin)
            for topic, tp in sorted(
                report.topic_progress.items(),
                key=lambda kv: kv[1].current_mastery,
                reverse=True,
            ):
                bar = _mastery_bar(tp.current_mastery)
                flag = " ⚠" if tp.is_weak_spot else ""
                self._print(
                    f"  {topic:<20} {bar}  {tp.current_mastery:.0%}{flag}"
                )
                if tp.total_questions > 0:
                    self._print(
                        f"  {'':20} Q:{tp.total_questions}  Correct:{tp.total_correct}  "
                        f"Accuracy:{tp.accuracy:.0%}"
                    )
                # Mastery curve (mini sparkline)
                if len(tp.mastery_history) > 1:
                    spark = _sparkline([m for _, m in tp.mastery_history])
                    self._print(f"  {'':20} Trend: {spark}")
                self._print("")

        # Weak spots
        self._print(f"Weak Spots Detected")
        self._print(thin)
        if report.weak_spots:
            for topic in report.weak_spots:
                tp = report.topic_progress[topic]
                self._print(f"  ⚠  {topic}")
                self._print(f"     Mastery : {tp.current_mastery:.0%}")
                if tp.weak_spot_reason:
                    self._print(f"     Reason  : {tp.weak_spot_reason}")
        else:
            self._print("  None — keep up the great work!")

        # Strongest topics
        self._print(f"\nStrongest Topics")
        self._print(thin)
        if report.strongest_topics:
            for topic in report.strongest_topics:
                tp = report.topic_progress[topic]
                self._print(f"  ★  {topic:<20} {tp.current_mastery:.0%}")
        else:
            self._print("  No topics above weak-spot threshold yet.")

        self._print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _detect_weak_spot(
    mastery: float,
    accuracy: float,
    total_questions: int,
) -> Tuple[bool, str]:
    """Return (is_weak_spot, reason) for a topic."""
    if mastery < _WEAK_MASTERY_THRESHOLD:
        return True, f"Mastery {mastery:.0%} is below the {_WEAK_MASTERY_THRESHOLD:.0%} threshold."
    if (
        total_questions >= _MIN_QUESTIONS_FOR_ACCURACY
        and accuracy < _WEAK_ACCURACY_THRESHOLD
    ):
        return (
            True,
            f"Accuracy {accuracy:.0%} is below {_WEAK_ACCURACY_THRESHOLD:.0%} "
            f"({total_questions} questions attempted).",
        )
    return False, ""


def _build_mastery_curve(
    current_mastery: float,
    session_count: int,
) -> List[Tuple[int, float]]:
    """
    Build a simple linear mastery history curve.

    We only have the current mastery, so we interpolate from 0.0 at
    session 0 to ``current_mastery`` at ``session_count``.
    """
    if session_count == 0:
        return [(0, current_mastery)]
    points: List[Tuple[int, float]] = []
    for i in range(session_count + 1):
        frac = i / session_count
        points.append((i, round(frac * current_mastery, 4)))
    return points


def _mastery_bar(mastery: float, width: int = 10) -> str:
    """Render a simple ASCII progress bar for a mastery score."""
    filled = round(mastery * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _sparkline(values: List[float]) -> str:
    """
    Render a tiny sparkline string from a list of float values (0.0–1.0).

    Uses block characters ▁▂▃▄▅▆▇█ to represent 8 levels.
    """
    bars = " ▁▂▃▄▅▆▇█"
    n = len(bars) - 1
    return "".join(bars[min(n, max(0, round(v * n)))] for v in values)
