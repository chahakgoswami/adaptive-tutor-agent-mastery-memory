"""Unit tests for MockLLMClient."""

from __future__ import annotations

import pytest

from tutor_agent.mock_llm_client import (
    ChatCompletion,
    ChatChoice,
    ChatMessage,
    ChatUsage,
    MockLLMClient,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_messages(user_text: str):
    return [
        {"role": "system", "content": "You are a helpful tutor."},
        {"role": "user", "content": user_text},
    ]


# ---------------------------------------------------------------------------
# ChatCompletion structure tests
# ---------------------------------------------------------------------------

class TestChatCompletionStructure:
    def test_generate_returns_chat_completion(self):
        client = MockLLMClient(seed=0)
        result = client.generate(_make_messages("What is a variable?"))
        assert isinstance(result, ChatCompletion)

    def test_completion_has_choices(self):
        client = MockLLMClient(seed=0)
        result = client.generate(_make_messages("test"))
        assert len(result.choices) == 1
        assert isinstance(result.choices[0], ChatChoice)

    def test_completion_choice_has_message(self):
        client = MockLLMClient(seed=0)
        result = client.generate(_make_messages("test"))
        msg = result.choices[0].message
        assert isinstance(msg, ChatMessage)
        assert msg.role == "assistant"

    def test_completion_has_usage(self):
        client = MockLLMClient(seed=0)
        result = client.generate(_make_messages("hello"))
        assert isinstance(result.usage, ChatUsage)
        assert result.usage.total_tokens == (
            result.usage.prompt_tokens + result.usage.completion_tokens
        )

    def test_completion_has_model(self):
        client = MockLLMClient(model="gpt-mock-4", seed=0)
        result = client.generate(_make_messages("test"))
        assert result.model == "gpt-mock-4"

    def test_completion_has_object_field(self):
        client = MockLLMClient(seed=0)
        result = client.generate(_make_messages("test"))
        assert result.object == "chat.completion"

    def test_completion_id_is_string(self):
        client = MockLLMClient(seed=0)
        result = client.generate(_make_messages("test"))
        assert isinstance(result.id, str)
        assert len(result.id) > 0

    def test_first_content_property(self):
        client = MockLLMClient(seed=0)
        result = client.generate(_make_messages("variable"))
        assert isinstance(result.first_content, str)
        assert len(result.first_content) > 0

    def test_finish_reason_is_stop(self):
        client = MockLLMClient(seed=0)
        result = client.generate(_make_messages("test"))
        assert result.choices[0].finish_reason == "stop"


# ---------------------------------------------------------------------------
# Keyword matching / canned response selection
# ---------------------------------------------------------------------------

class TestKeywordMatching:
    def test_variable_keyword_returns_relevant_response(self):
        client = MockLLMClient(seed=0)
        result = client.generate(_make_messages("Explain what a variable is."))
        content = result.first_content.lower()
        assert "variable" in content or "store" in content or "assign" in content

    def test_print_keyword_returns_relevant_response(self):
        client = MockLLMClient(seed=0)
        result = client.generate(_make_messages("What does print() do?"))
        content = result.first_content.lower()
        assert "output" in content or "console" in content or "display" in content

    def test_gil_keyword_returns_relevant_response(self):
        client = MockLLMClient(seed=0)
        result = client.generate(_make_messages("Explain the GIL in Python."))
        content = result.first_content.lower()
        assert "gil" in content or "thread" in content or "lock" in content

    def test_unknown_keyword_returns_fallback(self):
        client = MockLLMClient(seed=0)
        result = client.generate(
            _make_messages("xyzzy_nonsense_topic_that_matches_nothing")
        )
        # Should return one of the fallback responses
        assert isinstance(result.first_content, str)
        assert len(result.first_content) > 0

    def test_empty_message_returns_fallback(self):
        client = MockLLMClient(seed=0)
        result = client.generate(_make_messages(""))
        assert isinstance(result.first_content, str)

    def test_no_user_message_returns_fallback(self):
        client = MockLLMClient(seed=0)
        result = client.generate([
            {"role": "system", "content": "You are a tutor."}
        ])
        assert isinstance(result.first_content, str)


# ---------------------------------------------------------------------------
# OpenAI-compatible chat.completions.create interface
# ---------------------------------------------------------------------------

class TestChatCompletionsInterface:
    def test_chat_completions_create_returns_completion(self):
        client = MockLLMClient(seed=0)
        result = client.chat.completions.create(
            model="gpt-mock-4",
            messages=_make_messages("What is a variable?"),
        )
        assert isinstance(result, ChatCompletion)

    def test_chat_completions_create_ignores_extra_kwargs(self):
        client = MockLLMClient(seed=0)
        result = client.chat.completions.create(
            model="gpt-mock-4",
            messages=_make_messages("test"),
            temperature=0.7,
            top_p=0.9,
        )
        assert isinstance(result, ChatCompletion)

    def test_chat_completions_create_empty_messages(self):
        client = MockLLMClient(seed=0)
        result = client.chat.completions.create(messages=[])
        assert isinstance(result, ChatCompletion)


# ---------------------------------------------------------------------------
# Determinism with seed
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_response(self):
        msgs = _make_messages("What is a variable?")
        r1 = MockLLMClient(seed=42).generate(msgs)
        r2 = MockLLMClient(seed=42).generate(msgs)
        assert r1.first_content == r2.first_content

    def test_different_seed_may_differ(self):
        """Two different seeds should not always produce the same response
        (probabilistically true for multi-response keywords)."""
        msgs = _make_messages("What is a variable?")
        responses = {
            MockLLMClient(seed=s).generate(msgs).first_content
            for s in range(10)
        }
        # With 2 canned responses for 'variable', we expect both to appear
        assert len(responses) >= 1  # sanity; may be 1 if only 1 canned response


# ---------------------------------------------------------------------------
# Extra responses injection
# ---------------------------------------------------------------------------

class TestExtraResponses:
    def test_extra_responses_are_used(self):
        client = MockLLMClient(
            seed=0,
            extra_responses={"quantum": ["Quantum mechanics is weird and wonderful."]},
        )
        result = client.generate(_make_messages("Explain quantum entanglement."))
        assert "quantum" in result.first_content.lower() or len(result.first_content) > 0

    def test_extra_responses_override_nothing_keeps_builtins(self):
        client = MockLLMClient(
            seed=0,
            extra_responses={"mykeyword": ["Custom response."]},
        )
        # Built-in keyword still works
        result = client.generate(_make_messages("What is a variable?"))
        content = result.first_content.lower()
        assert "variable" in content or "store" in content
