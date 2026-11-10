"""Unit tests for the LearnerProfile dataclass and persistence layer."""

import json
import os
import tempfile
from datetime import date

import pytest

from tutor_agent.learner_profile import (
    LearnerProfile,
    SessionRecord,
    TopicMastery,
)


# ---------------------------------------------------------------------------
# TopicMastery tests
# ---------------------------------------------------------------------------

class TestTopicMastery:
    def test_default_values(self):
        tm = TopicMastery(topic="python")
        assert tm.mastery_score == 0.0
        assert tm.easiness_factor == 2.5
        assert tm.interval_days == 1
        assert tm.repetitions == 0
        assert tm.next_review_date == date.today().isoformat()
        assert tm.last_reviewed_date is None

    def test_roundtrip_serialisation(self):
        tm = TopicMastery(
            topic="algebra",
            mastery_score=0.75,
            easiness_factor=2.1,
            interval_days=4,
            repetitions=3,
            next_review_date="2024-06-10",
            last_reviewed_date="2024-06-06",
        )
        restored = TopicMastery.from_dict(tm.to_dict())
        assert restored == tm


# ---------------------------------------------------------------------------
# SessionRecord tests
# ---------------------------------------------------------------------------

class TestSessionRecord:
    def test_default_values(self):
        sr = SessionRecord(session_id="s1", started_at="2024-01-01T10:00:00")
        assert sr.ended_at is None
        assert sr.topics_practiced == []
        assert sr.questions_asked == 0
        assert sr.correct_answers == 0
        assert sr.notes == ""

    def test_roundtrip_serialisation(self):
        sr = SessionRecord(
            session_id="abc",
            started_at="2024-01-01T10:00:00",
            ended_at="2024-01-01T10:30:00",
            topics_practiced=["math", "physics"],
            questions_asked=5,
            correct_answers=4,
            notes="good session",
        )
        restored = SessionRecord.from_dict(sr.to_dict())
        assert restored == sr


# ---------------------------------------------------------------------------
# LearnerProfile tests
# ---------------------------------------------------------------------------

class TestLearnerProfile:
    def test_creation_defaults(self):
        profile = LearnerProfile(learner_id="u1", display_name="Alice")
        assert profile.learner_id == "u1"
        assert profile.display_name == "Alice"
        assert profile.topic_mastery == {}
        assert profile.session_history == []

    def test_get_or_create_topic_creates(self):
        profile = LearnerProfile(learner_id="u1")
        tm = profile.get_or_create_topic("python")
        assert tm.topic == "python"
        assert "python" in profile.topic_mastery

    def test_get_or_create_topic_returns_existing(self):
        profile = LearnerProfile(learner_id="u1")
        tm1 = profile.get_or_create_topic("python")
        tm1.mastery_score = 0.5
        tm2 = profile.get_or_create_topic("python")
        assert tm2.mastery_score == 0.5
        assert tm1 is tm2

    def test_update_mastery_clamps(self):
        profile = LearnerProfile(learner_id="u1")
        profile.update_mastery("math", 1.5)
        assert profile.topic_mastery["math"].mastery_score == 1.0
        profile.update_mastery("math", -0.3)
        assert profile.topic_mastery["math"].mastery_score == 0.0

    def test_start_and_end_session(self):
        profile = LearnerProfile(learner_id="u1")
        record = profile.start_session(session_id="s1")
        assert record.session_id == "s1"
        assert record.ended_at is None
        assert len(profile.session_history) == 1

        ended = profile.end_session("s1")
        assert ended is not None
        assert ended.ended_at is not None

    def test_end_session_unknown_id(self):
        profile = LearnerProfile(learner_id="u1")
        result = profile.end_session("nonexistent")
        assert result is None

    def test_roundtrip_serialisation(self):
        profile = LearnerProfile(learner_id="u2", display_name="Bob")
        profile.get_or_create_topic("history")
        profile.update_mastery("history", 0.6)
        profile.start_session(session_id="s1")

        restored = LearnerProfile.from_dict(profile.to_dict())
        assert restored.learner_id == profile.learner_id
        assert restored.display_name == profile.display_name
        assert "history" in restored.topic_mastery
        assert restored.topic_mastery["history"].mastery_score == 0.6
        assert len(restored.session_history) == 1


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load(self, tmp_path):
        path = str(tmp_path / "profile.json")
        profile = LearnerProfile(learner_id="u3", display_name="Carol")
        profile.update_mastery("physics", 0.4)
        profile.save(path)

        assert os.path.exists(path)
        loaded = LearnerProfile.load(path)
        assert loaded.learner_id == "u3"
        assert loaded.topic_mastery["physics"].mastery_score == 0.4

    def test_save_produces_valid_json(self, tmp_path):
        path = str(tmp_path / "profile.json")
        profile = LearnerProfile(learner_id="u4")
        profile.save(path)
        with open(path) as fh:
            data = json.load(fh)
        assert data["learner_id"] == "u4"

    def test_load_or_create_new(self, tmp_path):
        path = str(tmp_path / "new_profile.json")
        profile = LearnerProfile.load_or_create(path, learner_id="u5", display_name="Dave")
        assert profile.learner_id == "u5"
        assert os.path.exists(path)

    def test_load_or_create_existing(self, tmp_path):
        path = str(tmp_path / "existing.json")
        original = LearnerProfile(learner_id="u6", display_name="Eve")
        original.update_mastery("chemistry", 0.9)
        original.save(path)

        loaded = LearnerProfile.load_or_create(path, learner_id="u6")
        assert loaded.topic_mastery["chemistry"].mastery_score == 0.9

    def test_atomic_write_does_not_leave_tmp(self, tmp_path):
        path = str(tmp_path / "profile.json")
        profile = LearnerProfile(learner_id="u7")
        profile.save(path)
        assert not os.path.exists(path + ".tmp")
