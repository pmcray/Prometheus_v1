"""Thin client for the vLLM OpenAI-compatible endpoint.

Wraps the `openai` package (which must be installed at runtime, not a declared
dependency of the core library).  All network I/O is in this module; the rest
of the harness is network-free and fully unit-testable.

Usage::

    client = ModelClient(base_url="http://localhost:8000/v1", model_id="Qwen/Qwen3-8B")
    reply = client.chat([{"role": "user", "content": "Hello"}])
"""

from __future__ import annotations

import json
from typing import Any

# openai is a runtime dependency installed on Colab; import lazily so tests
# can mock it without the package being present.
_openai_module: Any = None


def _openai():
    global _openai_module
    if _openai_module is None:
        try:
            import openai as _mod
            _openai_module = _mod
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required to use ModelClient. "
                "Install it with:  pip install openai>=1.0"
            ) from exc
    return _openai_module


class ModelClientError(RuntimeError):
    """Raised when the model server returns an error or an unparseable reply."""


class ModelClient:
    """A minimal wrapper around the OpenAI chat-completions API as exposed by vLLM.

    Args:
        base_url:    The root of the vLLM OpenAI-compatible server,
                     e.g. ``"http://localhost:8000/v1"``.
        model_id:    The Hugging Face model identifier that vLLM was started
                     with, e.g. ``"Qwen/Qwen3-8B"``.
        api_key:     Passed verbatim to the openai client.  vLLM accepts any
                     non-empty string (default ``"EMPTY"``).
        temperature: Sampling temperature (default 0.0 for deterministic runs).
        max_tokens:  Maximum tokens to generate per call.
        timeout:     HTTP timeout in seconds.
        extra_body:  Additional vLLM-specific parameters forwarded via
                     ``extra_body`` (e.g. ``{"enable_thinking": False}``).
    """

    def __init__(
        self,
        *,
        base_url: str,
        model_id: str,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = 120.0,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_body = extra_body or {}
        self._api_key = api_key

    def _client(self):
        openai = _openai()
        return openai.OpenAI(
            base_url=self.base_url,
            api_key=self._api_key,
            timeout=self.timeout,
        )

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Send *messages* to the model and return the assistant reply text.

        Args:
            messages: OpenAI-format message list, e.g.
                      ``[{"role": "system", "content": "..."}, ...]``.
            **kwargs: Override instance defaults for this call only
                      (``temperature``, ``max_tokens``, ``extra_body``).

        Returns:
            The assistant message content as a plain string.

        Raises:
            ModelClientError: On API errors or unexpected response shapes.
        """
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        extra_body = {**self.extra_body, **kwargs.get("extra_body", {})}

        try:
            resp = self._client().chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **({"extra_body": extra_body} if extra_body else {}),
            )
        except Exception as exc:
            raise ModelClientError(f"vLLM API call failed: {exc}") from exc

        try:
            content = resp.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise ModelClientError(f"Unexpected response shape: {exc}") from exc

        if content is None:
            raise ModelClientError("Model returned null content (possible refusal or error)")
        return content

    def chat_json(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        """Like :meth:`chat` but parse the reply as JSON.

        Strips markdown code fences (```json … ```) before parsing.

        Raises:
            ModelClientError: If the reply is not valid JSON.
        """
        raw = self.chat(messages, **kwargs)
        text = raw.strip()
        # Strip optional markdown code block
        if text.startswith("```"):
            lines = text.splitlines()
            # drop first line (```json or ```) and last (```)
            inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            text = "\n".join(inner).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModelClientError(
                f"Model reply is not valid JSON: {exc}\n--- raw reply ---\n{raw[:500]}"
            ) from exc

    def health_check(self) -> bool:
        """Return True if the server responds to a trivial request."""
        try:
            self.chat([{"role": "user", "content": "ping"}], max_tokens=4)
            return True
        except ModelClientError:
            return False
