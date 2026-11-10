# Adaptive Tutor Agent with Mastery Memory

Adaptive tutor agent with spaced repetition, difficulty routing, and persistent learner memory.

**Domain:** Agentic AI
**Language:** python
**Demonstrates:** You use memory systems that personalize over time.

## 7-day build plan

- [ ] Day 1: Scaffold the project structure with a virtual environment, dependency manifest (pyyaml, openai stub, pytest), and a core LearnerProfile dataclass that stores topic mastery scores, review intervals, and session history, persisted to a local JSON file.
- [ ] Day 2: Implement the SpacedRepetitionScheduler using an SM-2-inspired algorithm that reads LearnerProfile data, computes next-review dates per topic, and returns an ordered list of topics due for practice, with unit tests covering interval calculation edge cases.
- [ ] Day 3: Build the DifficultyRouter module that maps each topic's current mastery score to one of three difficulty tiers (beginner/intermediate/advanced), selects an appropriate question template from a local YAML question bank, and returns a structured Question object.
- [ ] Day 4: Create a MockLLMClient that simulates an OpenAI-compatible chat API entirely offline using canned responses and keyword-based answer evaluation, then wire it into an AnswerEvaluator that scores free-text learner responses and updates mastery scores in LearnerProfile.
- [ ] Day 5: Assemble the TutorAgent orchestrator that runs a full session loop — fetch due topics from the scheduler, route difficulty, present questions via the mock LLM, evaluate answers, update memory, and require human-in-the-loop confirmation before resetting or deleting any learner profile data.
- [ ] Day 6: Add a SessionAnalytics module that reads historical LearnerProfile snapshots, computes per-topic progress curves and weak-spot detection, and renders a text-based progress report to the console; write integration tests that simulate a multi-day learning sequence end-to-end.
- [ ] Day 7: Package the project as a runnable CLI entry point (tutor-agent command via pyproject.toml), add configurable environment variable support for model endpoint and data directory, wire up pytest with coverage reporting, and ensure the full pipeline runs offline with a single command.

_A comprehensive README with an architecture diagram is generated on Day 7._
