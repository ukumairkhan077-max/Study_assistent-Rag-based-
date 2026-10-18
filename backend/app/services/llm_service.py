"""LLM access through any OpenAI-compatible chat-completions endpoint.

Message roles: the *system* message carries the assistant's rules (highest
priority, written by us); *user* messages carry the question plus the retrieved
excerpts. Retrieved text is data - it is wrapped in delimiters and the system
prompt tells the model never to follow instructions found inside it.

Temperature controls sampling randomness (0 = near-deterministic, good for
evaluation; higher = more varied wording). Generation tokens (what the model
reads/writes, billed and limited by the context window) are unrelated to
embedding tokens (consumed by the local embedding model, free here).
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import openai

from ..core.config import Settings

log = logging.getLogger("llm")


class LLMError(Exception):
    """Base error; message is safe to show to the user (never contains the key)."""


class LLMNotConfiguredError(LLMError):
    pass


class LLMUnavailableError(LLMError):
    """Rate limit / timeout / outage that persisted after retries."""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMRequestError(LLMError):
    """Non-recoverable: bad key, unknown model, invalid parameters. Never retried."""


class LLMService:
    def __init__(self, settings: Settings):
        self.s = settings
        self._client: Optional[openai.OpenAI] = None

    @property
    def configured(self) -> bool:
        return self.s.llm_configured

    def _get_client(self) -> openai.OpenAI:
        if not self.configured:
            raise LLMNotConfiguredError(
                "LLM is not configured. Set LLM_API_KEY and LLM_MODEL (and LLM_BASE_URL) in backend/.env."
            )
        if self._client is None:
            kwargs = {"api_key": self.s.llm_api_key, "timeout": self.s.llm_timeout_seconds, "max_retries": 0}
            if self.s.llm_base_url:
                kwargs["base_url"] = self.s.llm_base_url
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def chat(self, messages: list[dict], *, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        client = self._get_client()
        params: dict = {"model": self.s.llm_model, "messages": messages}
        if self.s.llm_send_temperature:
            params["temperature"] = self.s.llm_temperature if temperature is None else temperature
        params[self.s.llm_max_tokens_param] = max_tokens or self.s.llm_max_tokens

        transient = (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError)
        permanent = (
            openai.AuthenticationError,
            openai.PermissionDeniedError,
            openai.NotFoundError,
            openai.BadRequestError,
        )
        last: Optional[Exception] = None
        for attempt in range(1, self.s.llm_max_attempts + 1):
            try:
                resp = client.chat.completions.create(**params)
                text = (resp.choices[0].message.content or "").strip() if resp.choices else ""
                if not text:
                    raise LLMUnavailableError("The model returned an empty response.")
                return text
            except permanent as exc:
                # Only the error *type/status* is logged - never headers or the key.
                log.error("LLM request rejected (%s)", type(exc).__name__)
                raise LLMRequestError(f"LLM request rejected by provider ({type(exc).__name__}). Check model name, key and parameters.") from exc
            except transient as exc:
                last = exc
                wait = min(2 ** attempt, 10)
                log.warning("LLM transient error %s (attempt %d/%d)", type(exc).__name__, attempt, self.s.llm_max_attempts)
                if attempt < self.s.llm_max_attempts:
                    time.sleep(wait)
            except openai.APIStatusError as exc:  # other HTTP errors: do not loop
                raise LLMRequestError(f"LLM provider returned HTTP {exc.status_code}.") from exc
        raise LLMUnavailableError(
            f"LLM provider unavailable or rate-limited ({type(last).__name__}). Try again shortly."
        )

    def complete(self, system: str, user: str) -> str:
        """Convenience wrapper used by query expansion."""
        return self.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=300)
