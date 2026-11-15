# Adaptive Tutor Agent with Mastery Memory

Adaptive tutor agent with spaced repetition, difficulty routing, and persistent learner memory.

**Domain:** Agentic AI  
**Language:** Python 3.9+  
**Demonstrates:** Memory systems that personalise over time, SM-2 spaced repetition, offline LLM simulation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI  (tutor-agent)                       │
│                       tutor_agent/cli.py                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │ orchestrates
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                       TutorAgent                                │
│                   tutor_agent/tutor_agent.py                    │
│                                                                 │
│  ┌─────────────────────┐   ┌──────────────────────────────┐    │
│  │ SpacedRepetition     │   │      DifficultyRouter        │    │
│  │    Scheduler        │   │  (question_bank.yaml)        │    │
│  │  SM-2 algorithm     │   │  beginner/intermediate/      │    │
│  │  due_topics()       │   │  advanced tiers              │    │
│  └─────────┬───────────┘   └──────────────┬───────────────┘    │
│            │                              │                     │
│            └──────────────┬───────────────┘                     │
│                           ▼                                     │
│              ┌────────────────────────┐                         │
│              │    AnswerEvaluator     │                         │
│              │  keyword matching      │                         │
│              │  SM-2 quality score    │                         │
│              └───────────┬────────────┘                         │
│                          │                                      │
│                          ▼                                      │
│              ┌────────────────────────┐                         │
│              │    MockLLMClient       │                         │
│              │  offline canned resp.  │                         │
│              │  OpenAI-compatible API │                         │
│              └────────────────────────┘                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │ persists
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LearnerProfile                              │
│               data/profiles/<learner_id>.json                   │
│  TopicMastery: mastery_score, SM-2 state, next_review_date      │
│  SessionRecord: history of every session                        │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   SessionAnalytics                              │
│            progress curves · weak-spot detection               │
│               text-based console report                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start (single command, fully offline)

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install the package in editable mode with dev extras
pip install -e ".[dev]"

# 3. Add topics and start a session
tutor-agent --learner-id alice add-topics python math
tutor-agent --learner-id alice session

# 4. View your progress report
tutor-agent --learner-id alice report

# 5. Run the full test suite with coverage
pytest
```

---

## CLI Reference

```
tutor-agent [global options] <command> [command options]

Global options:
  --data-dir DIR          Profile storage directory
                          (env: TUTOR_DATA_DIR, default: data/profiles)
  --model-endpoint MODEL  LLM model name
                          (env: TUTOR_MODEL_ENDPOINT, default: gpt-mock-4)
  --learner-id ID         Learner identifier
                          (env: TUTOR_LEARNER_ID, default: default_learner)
  --display-name NAME     Human-readable learner name

Commands:
  session                 Run an interactive tutoring session
    --max-questions N     Max questions per session
                          (env: TUTOR_MAX_QUESTIONS, default: 10)
    --topics TOPIC ...    Register topics before starting

  report                  Print a progress report

  add-topics TOPIC ...    Register topics in a profile

  reset                   Reset profile data (requires confirmation)
    --delete              Permanently delete the profile file
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TUTOR_DATA_DIR` | `data/profiles` | Directory for learner profile JSON files |
| `TUTOR_MODEL_ENDPOINT` | `gpt-mock-4` | Model name reported by the mock LLM client |
| `TUTOR_LEARNER_ID` | `default_learner` | Default learner identifier |
| `TUTOR_MAX_QUESTIONS` | `10` | Maximum questions per tutoring session |

---

## Running Tests

```bash
# Full suite with coverage (HTML report in htmlcov/)
pytest

# Quick run without coverage
pytest --no-cov

# Specific test module
pytest tests/test_spaced_repetition.py -v

# Run with coverage and show missing lines
pytest --cov=tutor_agent --cov-report=term-missing
```

---

## Project Structure

```
adaptive-tutor-agent/
├── tutor_agent/
│   ├── __init__.py
│   ├── cli.py                  # CLI entry point
│   ├── tutor_agent.py          # TutorAgent orchestrator
│   ├── learner_profile.py      # LearnerProfile dataclass + JSON persistence
│   ├── spaced_repetition.py    # SM-2-inspired scheduler
│   ├── difficulty_router.py    # Mastery → tier → question selector
│   ├── answer_evaluator.py     # Keyword scoring + profile update
│   ├── mock_llm_client.py      # Offline OpenAI-compatible LLM mock
│   └── session_analytics.py   # Progress curves + weak-spot detection
├── data/
│   └── question_bank.yaml      # Question templates (python/math/history/science)
├── tests/
│   ├── test_cli.py
│   ├── test_tutor_agent.py
│   ├── test_spaced_repetition.py
│   ├── test_difficulty_router.py
│   ├── test_answer_evaluator.py
│   ├── test_mock_llm_client.py
│   ├── test_learner_profile.py
│   └── test_session_analytics.py
├── pyproject.toml
├── setup.cfg
├── requirements.txt
└── README.md
```

---

## 7-day build plan

- [x] Day 1: Scaffold project structure, `LearnerProfile` dataclass, JSON persistence.
- [x] Day 2: `SpacedRepetitionScheduler` with SM-2 algorithm and unit tests.
- [x] Day 3: `DifficultyRouter` with YAML question bank and tier routing.
- [x] Day 4: `MockLLMClient` (offline) + `AnswerEvaluator` with mastery updates.
- [x] Day 5: `TutorAgent` orchestrator with human-in-the-loop confirmation.
- [x] Day 6: `SessionAnalytics` with progress curves, weak-spot detection, text report.
- [x] Day 7: CLI entry point, environment variable config, pytest coverage, offline pipeline.
