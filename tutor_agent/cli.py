"""CLI entry point for the Adaptive Tutor Agent.

Environment variables
---------------------
TUTOR_DATA_DIR
    Directory where learner profile JSON files are stored.
    Defaults to ``./data/profiles``.
TUTOR_MODEL_ENDPOINT
    Model endpoint string passed to MockLLMClient as the model name.
    Defaults to ``gpt-mock-4``.
TUTOR_LEARNER_ID
    Learner identifier used to load/create a profile.
    Defaults to ``default_learner``.
TUTOR_MAX_QUESTIONS
    Maximum questions per session (integer).
    Defaults to ``10``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from tutor_agent.learner_profile import LearnerProfile
from tutor_agent.mock_llm_client import MockLLMClient
from tutor_agent.difficulty_router import DifficultyRouter
from tutor_agent.session_analytics import SessionAnalytics
from tutor_agent.tutor_agent import TutorAgent


# ---------------------------------------------------------------------------
# Environment variable helpers
# ---------------------------------------------------------------------------

def _env(key: str, default: str) -> str:
    """Return the value of an environment variable, or *default* if unset."""
    return os.environ.get(key, default)


def _resolve_profile_path(data_dir: str, learner_id: str) -> str:
    """Construct the full path to a learner's JSON profile file."""
    return str(Path(data_dir) / f"{learner_id}.json")


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def cmd_session(args: argparse.Namespace) -> int:
    """Run a tutoring session."""
    data_dir = args.data_dir
    learner_id = args.learner_id
    model_endpoint = args.model_endpoint
    max_questions = args.max_questions

    profile_path = _resolve_profile_path(data_dir, learner_id)
    profile = LearnerProfile.load_or_create(
        profile_path,
        learner_id=learner_id,
        display_name=args.display_name or learner_id,
    )

    # Register any topics passed via --topics
    if args.topics:
        for topic in args.topics:
            profile.get_or_create_topic(topic)
        profile.save(profile_path)

    llm_client = MockLLMClient(model=model_endpoint)
    router = DifficultyRouter()

    agent = TutorAgent(
        profile=profile,
        profile_path=profile_path,
        router=router,
        llm_client=llm_client,
        max_questions_per_session=max_questions,
    )

    agent.run_session()
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Print a progress report for a learner."""
    data_dir = args.data_dir
    learner_id = args.learner_id
    profile_path = _resolve_profile_path(data_dir, learner_id)

    if not Path(profile_path).exists():
        print(f"Error: no profile found for learner '{learner_id}' at {profile_path}")
        return 1

    profile = LearnerProfile.load(profile_path)
    analytics = SessionAnalytics(profile=profile)
    analytics.print_report()
    return 0


def cmd_add_topics(args: argparse.Namespace) -> int:
    """Register topics in a learner profile."""
    data_dir = args.data_dir
    learner_id = args.learner_id
    profile_path = _resolve_profile_path(data_dir, learner_id)

    profile = LearnerProfile.load_or_create(
        profile_path,
        learner_id=learner_id,
        display_name=args.display_name or learner_id,
    )

    llm_client = MockLLMClient(model=args.model_endpoint)
    router = DifficultyRouter()

    agent = TutorAgent(
        profile=profile,
        profile_path=profile_path,
        router=router,
        llm_client=llm_client,
    )
    agent.add_topics(args.topics)
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    """Reset (or delete) a learner profile."""
    data_dir = args.data_dir
    learner_id = args.learner_id
    profile_path = _resolve_profile_path(data_dir, learner_id)

    if not Path(profile_path).exists():
        print(f"Error: no profile found for learner '{learner_id}' at {profile_path}")
        return 1

    profile = LearnerProfile.load(profile_path)
    llm_client = MockLLMClient(model=args.model_endpoint)
    router = DifficultyRouter()

    agent = TutorAgent(
        profile=profile,
        profile_path=profile_path,
        router=router,
        llm_client=llm_client,
    )

    if args.delete:
        agent.delete_profile()
    else:
        agent.reset_profile()
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with sub-commands."""
    # Read defaults from environment
    default_data_dir = _env("TUTOR_DATA_DIR", str(Path("data") / "profiles"))
    default_model = _env("TUTOR_MODEL_ENDPOINT", "gpt-mock-4")
    default_learner = _env("TUTOR_LEARNER_ID", "default_learner")
    default_max_q = int(_env("TUTOR_MAX_QUESTIONS", "10"))

    parser = argparse.ArgumentParser(
        prog="tutor-agent",
        description="Adaptive Tutor Agent — spaced repetition and mastery memory",
    )

    # Global options shared by all sub-commands
    parser.add_argument(
        "--data-dir",
        default=default_data_dir,
        help=(
            "Directory for learner profile files "
            f"(env: TUTOR_DATA_DIR, default: {default_data_dir})"
        ),
    )
    parser.add_argument(
        "--model-endpoint",
        default=default_model,
        help=(
            "Model name / endpoint string passed to the LLM client "
            f"(env: TUTOR_MODEL_ENDPOINT, default: {default_model})"
        ),
    )
    parser.add_argument(
        "--learner-id",
        default=default_learner,
        help=(
            "Learner identifier used to load/create a profile "
            f"(env: TUTOR_LEARNER_ID, default: {default_learner})"
        ),
    )
    parser.add_argument(
        "--display-name",
        default="",
        help="Human-readable learner name (used when creating a new profile).",
    )

    subparsers = parser.add_subparsers(dest="command", title="commands")
    subparsers.required = True

    # --- session ---
    session_p = subparsers.add_parser(
        "session",
        help="Run an interactive tutoring session.",
    )
    session_p.add_argument(
        "--max-questions",
        type=int,
        default=default_max_q,
        help=(
            "Maximum questions per session "
            f"(env: TUTOR_MAX_QUESTIONS, default: {default_max_q})"
        ),
    )
    session_p.add_argument(
        "--topics",
        nargs="+",
        default=[],
        metavar="TOPIC",
        help="Register these topics before starting the session.",
    )
    session_p.set_defaults(func=cmd_session)

    # --- report ---
    report_p = subparsers.add_parser(
        "report",
        help="Print a progress report for a learner.",
    )
    report_p.set_defaults(func=cmd_report)

    # --- add-topics ---
    add_p = subparsers.add_parser(
        "add-topics",
        help="Register one or more topics in a learner profile.",
    )
    add_p.add_argument(
        "topics",
        nargs="+",
        metavar="TOPIC",
        help="Topic names to register.",
    )
    add_p.set_defaults(func=cmd_add_topics)

    # --- reset ---
    reset_p = subparsers.add_parser(
        "reset",
        help="Reset or delete a learner profile (requires human confirmation).",
    )
    reset_p.add_argument(
        "--delete",
        action="store_true",
        default=False,
        help="Permanently delete the profile file instead of just clearing data.",
    )
    reset_p.set_defaults(func=cmd_reset)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """
    Main entry point for the ``tutor-agent`` CLI command.

    Parameters
    ----------
    argv:
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Exit code (0 = success).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
