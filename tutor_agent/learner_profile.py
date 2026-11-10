"""LearnerProfile dataclass with JSON persistence."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import Any, Dict, List, Optional


_DATE_FORMAT = "%Y-%m-%d"
_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"


@dataclass
class TopicMastery:
    """Tracks mastery and spaced-repetition state for a single topic."""

    topic: str
    mastery_score: float = 0.0          # 0.0 – 1.0
    easiness_factor: float = 2.5        # SM-2 E-factor
    interval_days: int = 1              # days until next review
    repetitions: int = 0               # number of successful reviews
    next_review_date: str = field(
        default_factory=lambda: date.today().isoformat()
    )                                   # ISO date string YYYY-MM-DD
    last_reviewed_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TopicMastery":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SessionRecord:
    """A record of a single learning session."""

    session_id: str
    started_at: str                     # ISO datetime string
    ended_at: Optional[str] = None
    topics_practiced: List[str] = field(default_factory=list)
    questions_asked: int = 0
    correct_answers: int = 0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class LearnerProfile:
    """Persistent learner profile storing mastery scores and session history."""

    learner_id: str
    display_name: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now().strftime(_DATETIME_FORMAT)
    )
    topic_mastery: Dict[str, TopicMastery] = field(default_factory=dict)
    session_history: List[SessionRecord] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Topic helpers
    # ------------------------------------------------------------------

    def get_or_create_topic(self, topic: str) -> TopicMastery:
        """Return existing TopicMastery for *topic*, creating it if absent."""
        if topic not in self.topic_mastery:
            self.topic_mastery[topic] = TopicMastery(topic=topic)
        return self.topic_mastery[topic]

    def update_mastery(self, topic: str, mastery_score: float) -> None:
        """Clamp and persist a new mastery score for *topic*."""
        tm = self.get_or_create_topic(topic)
        tm.mastery_score = max(0.0, min(1.0, mastery_score))

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    def start_session(self, session_id: Optional[str] = None) -> SessionRecord:
        """Create and register a new SessionRecord."""
        import uuid
        sid = session_id or str(uuid.uuid4())
        record = SessionRecord(
            session_id=sid,
            started_at=datetime.now().strftime(_DATETIME_FORMAT),
        )
        self.session_history.append(record)
        return record

    def end_session(self, session_id: str) -> Optional[SessionRecord]:
        """Mark a session as ended by its ID."""
        for record in self.session_history:
            if record.session_id == session_id and record.ended_at is None:
                record.ended_at = datetime.now().strftime(_DATETIME_FORMAT)
                return record
        return None

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learner_id": self.learner_id,
            "display_name": self.display_name,
            "created_at": self.created_at,
            "topic_mastery": {
                k: v.to_dict() for k, v in self.topic_mastery.items()
            },
            "session_history": [s.to_dict() for s in self.session_history],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearnerProfile":
        profile = cls(
            learner_id=data["learner_id"],
            display_name=data.get("display_name", ""),
            created_at=data.get("created_at", datetime.now().strftime(_DATETIME_FORMAT)),
        )
        for key, tm_data in data.get("topic_mastery", {}).items():
            profile.topic_mastery[key] = TopicMastery.from_dict(tm_data)
        for sr_data in data.get("session_history", []):
            profile.session_history.append(SessionRecord.from_dict(sr_data))
        return profile

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist profile to *path* as JSON (atomic write via temp file)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        os.replace(tmp_path, path)

    @classmethod
    def load(cls, path: str) -> "LearnerProfile":
        """Load a LearnerProfile from a JSON file at *path*."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    @classmethod
    def load_or_create(cls, path: str, learner_id: str, display_name: str = "") -> "LearnerProfile":
        """Load profile from *path* if it exists, otherwise create a new one."""
        if os.path.exists(path):
            return cls.load(path)
        profile = cls(learner_id=learner_id, display_name=display_name)
        profile.save(path)
        return profile
