from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol


class LLMError(RuntimeError):
    pass


@dataclass
class Message:
    role: str  # system | user | assistant
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class Provider(Protocol):
    name: str
    model: str

    def complete(self, messages: list[Message]) -> str:
        """Return a full completion."""

    def stream(self, messages: list[Message]) -> Iterator[str]:
        """Yield completion chunks as they arrive."""
