"""Tests for the CLI entry point (tutor_agent.cli)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest
import yaml

from tutor_agent.cli import main, _build_parser, _resolve_profile_path, _env
from tutor_agent.learner_profile import LearnerProfile


# ---------------------------------------------------------------------------
# Fixtures
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


# ---------------------------------------------------------------------------
# _env helper
# ---------------------------------------------------------------------------

class TestEnvHelper:
    def test_returns_env_value_when_set(self, monkeypatch):
        monkeypatch.setenv("TUTOR_DATA_DIR", "/custom/path")
        assert _env("TUTOR_DATA_DIR", "default") == "/custom/path"

    def test_returns_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("TUTOR_DATA_DIR", raising=False)
        assert _env("TUTOR_DATA_DIR", "my_default") == "my_default"


# ---------------------------------------------------------------------------
# _resolve_profile_path
# ---------------------------------------------------------------------------

class TestResolveProfilePath:
    def test_combines_data_dir_and_learner_id(self):
        path = _resolve_profile_path("/data/profiles", "alice")
        assert path == str(Path("/data/profiles") / "alice.json")

    def test_uses_tmp_path(self, tmp_path):
        path = _resolve_profile_path(str(tmp_path), "bob")
        assert path.endswith("bob.json")
        assert str(tmp_path) in path


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------

class TestParser:
    def test_parser_builds_without_error(self):
        parser = _build_parser()
        assert parser is not None

    def test_session_subcommand_exists(self):
        parser = _build_parser()
        args = parser.parse_args(["session"])
        assert args.command == "session"

    def test_report_subcommand_exists(self):
        parser = _build_parser()
        args = parser.parse_args(["report"])
        assert args.command == "report"

    def test_add_topics_subcommand_exists(self):
        parser = _build_parser()
        args = parser.parse_args(["add-topics", "python", "math"])
        assert args.command == "add-topics"
        assert "python" in args.topics
        assert "math" in args.topics

    def test_reset_subcommand_exists(self):
        parser = _build_parser()
        args = parser.parse_args(["reset"])
        assert args.command == "reset"
        assert args.delete is False

    def test_reset_delete_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["reset", "--delete"])
        assert args.delete is True

    def test_global_data_dir_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--data-dir", "/tmp/profiles", "session"])
        assert args.data_dir == "/tmp/profiles"

    def test_global_learner_id_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--learner-id", "alice", "session"])
        assert args.learner_id == "alice"

    def test_global_model_endpoint_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--model-endpoint", "gpt-4", "session"])
        assert args.model_endpoint == "gpt-4"

    def test_session_max_questions_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["session", "--max-questions", "5"])
        assert args.max_questions == 5

    def test_session_topics_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["session", "--topics", "python", "math"])
        assert "python" in args.topics
        assert "math" in args.topics

    def test_env_vars_set_defaults(self, monkeypatch):
        monkeypatch.setenv("TUTOR_DATA_DIR", "/env/data")
        monkeypatch.setenv("TUTOR_LEARNER_ID", "env_learner")
        monkeypatch.setenv("TUTOR_MODEL_ENDPOINT", "env-model")
        monkeypatch.setenv("TUTOR_MAX_QUESTIONS", "7")
        parser = _build_parser()
        args = parser.parse_args(["session"])
        assert args.data_dir == "/env/data"
        assert args.learner_id == "env_learner"
        assert args.model_endpoint == "env-model"
        assert args.max_questions == 7


# ---------------------------------------------------------------------------
# add-topics command
# ---------------------------------------------------------------------------

class TestAddTopicsCommand:
    def test_add_topics_creates_profile(self, tmp_path, minimal_bank, monkeypatch):
        monkeypatch.setattr(
            "tutor_agent.cli.DifficultyRouter",
            lambda **kw: __import__(
                "tutor_agent.difficulty_router", fromlist=["DifficultyRouter"]
            ).DifficultyRouter(bank_path=minimal_bank),
        )
        data_dir = str(tmp_path / "profiles")
        rc = main([
            "--data-dir", data_dir,
            "--learner-id", "alice",
            "add-topics", "python", "math",
        ])
        assert rc == 0
        profile_path = str(tmp_path / "profiles" / "alice.json")
        assert Path(profile_path).exists()
        profile = LearnerProfile.load(profile_path)
        assert "python" in profile.topic_mastery
        assert "math" in profile.topic_mastery

    def test_add_topics_idempotent(self, tmp_path, minimal_bank, monkeypatch):
        monkeypatch.setattr(
            "tutor_agent.cli.DifficultyRouter",
            lambda **kw: __import__(
                "tutor_agent.difficulty_router", fromlist=["DifficultyRouter"]
            ).DifficultyRouter(bank_path=minimal_bank),
        )
        data_dir = str(tmp_path / "profiles")
        for _ in range(2):
            rc = main([
                "--data-dir", data_dir,
                "--learner-id", "bob",
                "add-topics", "python",
            ])
            assert rc == 0
        profile = LearnerProfile.load(str(tmp_path / "profiles" / "bob.json"))
        assert "python" in profile.topic_mastery


# ---------------------------------------------------------------------------
# report command
# ---------------------------------------------------------------------------

class TestReportCommand:
    def test_report_prints_output(self, tmp_path, capsys):
        # Create a profile first
        profile_path = str(tmp_path / "alice.json")
        profile = LearnerProfile(learner_id="alice", display_name="Alice")
        profile.get_or_create_topic("python").mastery_score = 0.8
        profile.save(profile_path)

        rc = main([
            "--data-dir", str(tmp_path),
            "--learner-id", "alice",
            "report",
        ])
        assert rc == 0
        captured = capsys.readouterr()
        assert "alice" in captured.out.lower() or "Alice" in captured.out

    def test_report_missing_profile_returns_error(self, tmp_path, capsys):
        rc = main([
            "--data-dir", str(tmp_path),
            "--learner-id", "nonexistent",
            "report",
        ])
        assert rc == 1
        captured = capsys.readouterr()
        assert "nonexistent" in captured.out or "Error" in captured.out


# ---------------------------------------------------------------------------
# reset command
# ---------------------------------------------------------------------------

class TestResetCommand:
    def test_reset_missing_profile_returns_error(self, tmp_path, capsys):
        rc = main([
            "--data-dir", str(tmp_path),
            "--learner-id", "ghost",
            "reset",
        ])
        assert rc == 1

    def test_reset_confirmed_clears_data(self, tmp_path, monkeypatch, capsys):
        profile_path = str(tmp_path / "charlie.json")
        profile = LearnerProfile(learner_id="charlie")
        profile.get_or_create_topic("python").mastery_score = 0.9
        profile.save(profile_path)

        # Simulate typing 'RESET'
        monkeypatch.setattr("builtins.input", lambda prompt="": "RESET")
        rc = main([
            "--data-dir", str(tmp_path),
            "--learner-id", "charlie",
            "reset",
        ])
        assert rc == 0
        reloaded = LearnerProfile.load(profile_path)
        assert reloaded.topic_mastery == {}

    def test_reset_cancelled_preserves_data(self, tmp_path, monkeypatch, capsys):
        profile_path = str(tmp_path / "dana.json")
        profile = LearnerProfile(learner_id="dana")
        profile.get_or_create_topic("math").mastery_score = 0.5
        profile.save(profile_path)

        monkeypatch.setattr("builtins.input", lambda prompt="": "no")
        rc = main([
            "--data-dir", str(tmp_path),
            "--learner-id", "dana",
            "reset",
        ])
        assert rc == 0
        reloaded = LearnerProfile.load(profile_path)
        assert pytest.approx(reloaded.topic_mastery["math"].mastery_score) == 0.5

    def test_delete_confirmed_removes_file(self, tmp_path, monkeypatch, capsys):
        profile_path = str(tmp_path / "edgar.json")
        profile = LearnerProfile(learner_id="edgar")
        profile.save(profile_path)
        assert Path(profile_path).exists()

        monkeypatch.setattr("builtins.input", lambda prompt="": "DELETE")
        rc = main([
            "--data-dir", str(tmp_path),
            "--learner-id", "edgar",
            "reset", "--delete",
        ])
        assert rc == 0
        assert not Path(profile_path).exists()


# ---------------------------------------------------------------------------
# session command (smoke test — no real stdin)
# ---------------------------------------------------------------------------

class TestSessionCommand:
    def test_session_with_no_due_topics_exits_cleanly(
        self, tmp_path, minimal_bank, monkeypatch, capsys
    ):
        """Session with no topics due should run and exit without error."""
        monkeypatch.setattr(
            "tutor_agent.cli.DifficultyRouter",
            lambda **kw: __import__(
                "tutor_agent.difficulty_router", fromlist=["DifficultyRouter"]
            ).DifficultyRouter(bank_path=minimal_bank),
        )
        # Create profile with topic far in the future (not due)
        profile_path = str(tmp_path / "fiona.json")
        profile = LearnerProfile(learner_id="fiona", display_name="Fiona")
        tm = profile.get_or_create_topic("python")
        tm.next_review_date = "2099-01-01"
        profile.save(profile_path)

        rc = main([
            "--data-dir", str(tmp_path),
            "--learner-id", "fiona",
            "session",
        ])
        assert rc == 0
        captured = capsys.readouterr()
        # Should mention no topics due
        assert "No topics due" in captured.out or "come back" in captured.out

    def test_session_registers_topics_flag(
        self, tmp_path, minimal_bank, monkeypatch, capsys
    ):
        """--topics flag should register topics before the session starts."""
        monkeypatch.setattr(
            "tutor_agent.cli.DifficultyRouter",
            lambda **kw: __import__(
                "tutor_agent.difficulty_router", fromlist=["DifficultyRouter"]
            ).DifficultyRouter(bank_path=minimal_bank),
        )
        # Provide inputs for the session: decline hint, give answer
        inputs = iter(["n", "A variable stores a value."])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs, ""))

        data_dir = str(tmp_path / "profiles")
        rc = main([
            "--data-dir", data_dir,
            "--learner-id", "grace",
            "session",
            "--topics", "python",
            "--max-questions", "1",
        ])
        assert rc == 0
        profile = LearnerProfile.load(str(tmp_path / "profiles" / "grace.json"))
        assert "python" in profile.topic_mastery


# ---------------------------------------------------------------------------
# Full offline pipeline smoke test
# ---------------------------------------------------------------------------

class TestOfflinePipeline:
    """Verifies the entire pipeline runs offline end-to-end."""

    def test_full_pipeline_add_session_report(
        self, tmp_path, minimal_bank, monkeypatch, capsys
    ):
        """
        1. Add topics via CLI.
        2. Run a session via CLI.
        3. Print a report via CLI.
        All offline, no external dependencies.
        """
        monkeypatch.setattr(
            "tutor_agent.cli.DifficultyRouter",
            lambda **kw: __import__(
                "tutor_agent.difficulty_router", fromlist=["DifficultyRouter"]
            ).DifficultyRouter(bank_path=minimal_bank),
        )

        data_dir = str(tmp_path / "profiles")
        learner_id = "pipeline_user"

        # Step 1: add topics
        rc = main([
            "--data-dir", data_dir,
            "--learner-id", learner_id,
            "--display-name", "Pipeline User",
            "add-topics", "python",
        ])
        assert rc == 0

        # Step 2: session (no topics due is fine — proves it runs)
        # Force topic due today by manipulating the profile directly
        profile_path = str(tmp_path / "profiles" / f"{learner_id}.json")
        profile = LearnerProfile.load(profile_path)
        from datetime import date
        profile.topic_mastery["python"].next_review_date = date.today().isoformat()
        profile.save(profile_path)

        inputs = iter(["n", "A variable stores a value."])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs, ""))

        rc = main([
            "--data-dir", data_dir,
            "--learner-id", learner_id,
            "session",
            "--max-questions", "1",
        ])
        assert rc == 0

        # Step 3: report
        rc = main([
            "--data-dir", data_dir,
            "--learner-id", learner_id,
            "report",
        ])
        assert rc == 0
        captured = capsys.readouterr()
        assert "python" in captured.out
