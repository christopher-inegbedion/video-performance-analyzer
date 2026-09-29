"""Pluggable LLM backends.

Any OpenAI-compatible chat-completions endpoint works, which covers OpenRouter,
OpenAI, Together, Groq, Fireworks, vLLM, LiteLLM and Ollama. Add a new backend by
subclassing Provider and registering it in `get_provider`.
"""

from __future__ import annotations

from .base import LLMError, Message, Provider
from .openai_compatible import OpenAICompatibleProvider


def get_provider(cfg) -> Provider:
    """Resolve the configured provider. Unknown names fall back to the generic
    OpenAI-compatible client, since most services expose that shape."""
    name = (cfg.llm.provider or "openrouter").lower()
    if name in {"openrouter", "openai", "openai_compatible", "ollama", "litellm", "custom"}:
        return OpenAICompatibleProvider(cfg)
    raise LLMError(
        f"Unknown provider '{name}'. Set llm.provider to one of: openrouter, openai, "
        "ollama, litellm, custom — or point llm.base_url at any OpenAI-compatible API."
    )


__all__ = ["LLMError", "Message", "OpenAICompatibleProvider", "Provider", "get_provider"]
