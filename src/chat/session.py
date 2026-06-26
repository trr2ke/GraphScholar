"""Chat session manager for GraphScholar.

Maintains conversation history and exposes it in the format expected by
BaseLLMProvider.complete() and AnswerGenerator.generate().
"""
from dataclasses import dataclass, field


@dataclass
class ChatSession:
    """Stateful conversation history with a bounded rolling window."""

    max_turns: int = 20   # each turn = one user + one assistant message

    _history: list[dict] = field(default_factory=list, repr=False)

    def add_turn(self, user_msg: str, assistant_msg: str) -> None:
        """Append a completed turn to the history.

        Trims the oldest turn when max_turns is exceeded so the context
        window stays bounded across long conversations.
        """
        self._history.append({"role": "user",      "content": user_msg})
        self._history.append({"role": "assistant",  "content": assistant_msg})

        # Each turn = 2 messages; trim pairs from the front
        max_messages = self.max_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    def get_history(self) -> list[dict]:
        """Return history as a list of {"role", "content"} dicts."""
        return list(self._history)

    def clear(self) -> None:
        """Reset conversation history."""
        self._history.clear()

    @property
    def turn_count(self) -> int:
        """Number of completed (user + assistant) turns."""
        return len(self._history) // 2
