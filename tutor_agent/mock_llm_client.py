"""MockLLMClient: offline OpenAI-compatible chat API simulator."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Lightweight OpenAI-compatible response structures
# ---------------------------------------------------------------------------

@dataclass
class ChatMessage:
    """Mirrors openai.types.chat.ChatCompletionMessage (minimal subset)."""
    role: str
    content: str


@dataclass
class ChatChoice:
    """Mirrors openai.types.chat.Choice."""
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


@dataclass
class ChatUsage:
    """Mirrors openai.types.CompletionUsage."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ChatCompletion:
    """
    Mirrors openai.types.chat.ChatCompletion (minimal subset).

    Attributes
    ----------
    id:         Fake completion ID.
    model:      Model name used.
    choices:    List of generated choices.
    usage:      Token usage statistics.
    created:    Unix timestamp of creation.
    """
    id: str
    model: str
    choices: List[ChatChoice]
    usage: ChatUsage
    created: int = field(default_factory=lambda: int(time.time()))
    object: str = "chat.completion"

    # Convenience accessor matching the real SDK's pattern
    @property
    def first_content(self) -> str:
        """Return the text of the first choice's message."""
        if self.choices:
            return self.choices[0].message.content
        return ""


# ---------------------------------------------------------------------------
# Canned response bank
# ---------------------------------------------------------------------------

# Maps topic keywords → list of canned assistant responses.
# The MockLLMClient searches the user message for these keywords.
_CANNED_RESPONSES: Dict[str, List[str]] = {
    "variable": [
        "A variable in Python is a named reference to a value stored in memory. "
        "You assign a value using the = operator, e.g. x = 42.",
        "Variables store values and can be reassigned at any time. "
        "Python is dynamically typed, so you don't declare the type explicitly.",
    ],
    "print": [
        "The print() function outputs text to the standard console/screen. "
        "It converts its arguments to strings and displays them.",
        "print() displays output to the user. You can pass multiple arguments "
        "separated by commas and control the separator with the sep parameter.",
    ],
    "data type": [
        "Python has several basic data types: int (integer numbers), str (strings "
        "of text), float (decimal numbers), and bool (True/False).",
        "Common data types include int, str, float, bool, list, tuple, and dict.",
    ],
    "list comprehension": [
        "A list comprehension provides a concise way to create lists. The syntax is "
        "[expression for item in iterable if condition]. It iterates over the "
        "iterable and applies the expression to each item.",
        "List comprehensions let you create a new list by looping over an iterable "
        "and optionally filtering elements, all in a single readable expression.",
    ],
    "decorator": [
        "A decorator is a higher-order function that wraps another function to modify "
        "or extend its behaviour. Applied with the @ syntax, e.g. @my_decorator.",
        "Decorators wrap functions using the @ symbol. They receive the original "
        "function as an argument and return a modified version.",
    ],
    "GIL": [
        "The Global Interpreter Lock (GIL) is a mutex in CPython that allows only one "
        "thread to execute Python bytecode at a time. It simplifies memory management "
        "but limits true parallel execution of threads.",
    ],
    "generator": [
        "Generators are functions that use yield to lazily produce values one at a time, "
        "avoiding loading all data into memory. They implement the iterator protocol via "
        "__iter__ and __next__.",
    ],
    "prime": [
        "A prime number is a natural number greater than 1 that has no positive divisors "
        "other than 1 and itself. Examples: 2, 3, 5, 7, 11.",
    ],
    "derivative": [
        "A derivative measures the instantaneous rate of change of a function. "
        "Geometrically it represents the slope of the tangent line at a point.",
    ],
    "integral": [
        "An integral computes the area under a curve. The fundamental theorem of calculus "
        "shows that integration and differentiation are inverse operations.",
    ],
    "photosynthesis": [
        "Photosynthesis is the process by which plants use sunlight, water, and carbon "
        "dioxide to produce glucose and oxygen. It occurs in chloroplasts.",
    ],
    "newton": [
        "Newton's three laws of motion: 1) An object at rest stays at rest unless acted "
        "upon by a force (inertia). 2) F = ma. 3) Every action has an equal and opposite "
        "reaction.",
    ],
    "dna": [
        "DNA is a double helix made of nucleotides. Each nucleotide contains a sugar, a "
        "phosphate, and one of four bases: adenine, thymine, guanine, or cytosine. "
        "Base pairs are A-T and G-C.",
    ],
    "renaissance": [
        "The Renaissance was a cultural and intellectual rebirth in Europe spanning roughly "
        "the 14th to 17th centuries, marked by renewed interest in classical art, science, "
        "and humanism.",
    ],
    "world war": [
        "World War II ended in 1945 with the surrender of Germany in May and Japan in "
        "September following the atomic bombings of Hiroshima and Nagasaki.",
    ],
    "cold war": [
        "The Cold War was a period of geopolitical tension between the USA and the Soviet "
        "Union from roughly 1947 to 1991, characterised by ideological conflict between "
        "capitalism and communism without direct large-scale warfare.",
    ],
}

# Generic fallback responses when no keyword is matched
_FALLBACK_RESPONSES: List[str] = [
    "That's an interesting question. Let me think through it carefully. "
    "The key concepts here relate to the fundamentals of the topic at hand.",
    "Great question! This topic involves several important ideas that build on each other.",
    "I'd approach this by first identifying the core principles involved.",
]


# ---------------------------------------------------------------------------
# MockLLMClient
# ---------------------------------------------------------------------------

class MockLLMClient:
    """
    Offline OpenAI-compatible chat completion client.

    Uses keyword matching against a canned response bank to simulate an LLM.
    The public interface mirrors the ``openai.OpenAI`` client so it can be
    swapped in transparently.

    Parameters
    ----------
    model:
        Model name to report in responses (cosmetic only).
    seed:
        Optional random seed for deterministic response selection.
    extra_responses:
        Additional keyword → responses mapping merged with the built-in bank.
    """

    def __init__(
        self,
        model: str = "gpt-mock-4",
        seed: Optional[int] = None,
        extra_responses: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.model = model
        self._rng = random.Random(seed)
        # Build the response bank
        self._response_bank: Dict[str, List[str]] = dict(_CANNED_RESPONSES)
        if extra_responses:
            for kw, responses in extra_responses.items():
                self._response_bank.setdefault(kw, []).extend(responses)

        # Expose a ``chat.completions`` namespace mirroring the real SDK
        self.chat = _ChatNamespace(self)

    # ------------------------------------------------------------------
    # Core generation logic
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
    ) -> ChatCompletion:
        """
        Generate a canned chat completion for the given *messages*.

        Parameters
        ----------
        messages:
            List of ``{"role": ..., "content": ...}`` dicts (OpenAI format).
        max_tokens:
            Maximum response length hint (ignored in mock, present for
            API compatibility).

        Returns
        -------
        ChatCompletion
        """
        # Extract the last user message for keyword matching
        user_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_text = msg.get("content", "")
                break

        response_text = self._pick_response(user_text)

        # Build a fake completion ID
        fake_id = f"chatcmpl-mock-{self._rng.randint(100_000, 999_999)}"

        prompt_tokens = sum(len(m.get("content", "").split()) for m in messages)
        completion_tokens = len(response_text.split())

        return ChatCompletion(
            id=fake_id,
            model=self.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            usage=ChatUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    # ------------------------------------------------------------------
    # Keyword matching
    # ------------------------------------------------------------------

    def _pick_response(self, user_text: str) -> str:
        """Select the best canned response for *user_text* via keyword match."""
        text_lower = user_text.lower()
        matched: List[str] = []

        # Longer keywords are more specific — sort to prefer them
        for keyword in sorted(self._response_bank, key=len, reverse=True):
            if keyword.lower() in text_lower:
                matched.extend(self._response_bank[keyword])
                # Stop after first (most specific) keyword match
                break

        if matched:
            return self._rng.choice(matched)
        return self._rng.choice(_FALLBACK_RESPONSES)


# ---------------------------------------------------------------------------
# Namespace shim so callers can write client.chat.completions.create(...)
# ---------------------------------------------------------------------------

class _CompletionsNamespace:
    """Provides the ``completions.create`` method."""

    def __init__(self, client: MockLLMClient) -> None:
        self._client = client

    def create(
        self,
        model: str = "",
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 256,
        **kwargs: Any,
    ) -> ChatCompletion:
        """OpenAI-compatible create call (keyword args ignored)."""
        return self._client.generate(messages or [], max_tokens=max_tokens)


class _ChatNamespace:
    """Provides the ``chat.completions`` attribute."""

    def __init__(self, client: MockLLMClient) -> None:
        self.completions = _CompletionsNamespace(client)
