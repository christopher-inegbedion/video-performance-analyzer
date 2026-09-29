"""Generic OpenAI-compatible chat client. Default target is OpenRouter."""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx

from ..config import Config
from .base import LLMError, Message


class OpenAICompatibleProvider:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.name = cfg.llm.provider
        self.model = cfg.llm.model
        self.base_url = cfg.llm.base_url.rstrip("/")

    # ------------------------------------------------------------- internals

    def _headers(self) -> dict[str, str]:
        key = self.cfg.api_key()
        # Local servers (Ollama, vLLM) usually need no key at all.
        if not key and "localhost" not in self.base_url and "127.0.0.1" not in self.base_url:
            raise LLMError(
                f"No API key found. Set {self.cfg.llm.api_key_env} in your environment.\n"
                "  export OPENROUTER_API_KEY=sk-or-...\n"
                "Get one at https://openrouter.ai/keys — or point llm.base_url at a "
                "local model server and no key is needed."
            )
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        if "openrouter" in self.base_url:
            headers["HTTP-Referer"] = self.cfg.llm.referer
            headers["X-Title"] = self.cfg.llm.title
        return headers

    def _payload(self, messages: list[Message], stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": [m.as_dict() for m in messages],
            "max_tokens": self.cfg.llm.max_tokens,
            "temperature": self.cfg.llm.temperature,
            "stream": stream,
        }

    @staticmethod
    def _explain_http_error(exc: httpx.HTTPStatusError) -> str:
        code = exc.response.status_code
        hints = {
            401: "Authentication failed — check your API key.",
            402: "Payment required — your provider account is out of credit.",
            404: "Model not found — check llm.model is a valid id for this provider.",
            429: "Rate limited — wait a moment, or switch to a less busy model.",
        }
        detail = ""
        try:
            detail = exc.response.json().get("error", {}).get("message", "")
        except Exception:
            detail = exc.response.text[:300]
        return f"HTTP {code}. {hints.get(code, '')} {detail}".strip()

    # ---------------------------------------------------------------- public

    def complete(self, messages: list[Message]) -> str:
        try:
            with httpx.Client(timeout=self.cfg.llm.timeout_s) as client:
                resp = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=self._payload(messages, stream=False),
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise LLMError(self._explain_http_error(exc)) from exc
        except httpx.RequestError as exc:
            raise LLMError(f"Could not reach {self.base_url}: {exc}") from exc
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected response shape: {json.dumps(data)[:400]}") from exc

    def stream(self, messages: list[Message]) -> Iterator[str]:
        try:
            with httpx.Client(timeout=self.cfg.llm.timeout_s) as client, client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=self._payload(messages, stream=True),
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    chunk = line[6:]
                    if chunk.strip() == "[DONE]":
                        break
                    try:
                        delta = json.loads(chunk)["choices"][0].get("delta", {})
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    if piece := delta.get("content"):
                        yield piece
        except httpx.HTTPStatusError as exc:
            raise LLMError(self._explain_http_error(exc)) from exc
        except httpx.RequestError as exc:
            raise LLMError(f"Could not reach {self.base_url}: {exc}") from exc
