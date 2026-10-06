# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Core abstractions for llm_bridge — a vendor-neutral LLM client interface.

A single small contract, :class:`LLMClient`, that every backend implements.
Application code depends only on this interface, never on a specific vendor
SDK or cloud provider.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from llm_bridge.errors import LLMInvalidOutputError, LLMRateLimitError, LLMTimeoutError
from llm_bridge.structured import JSONResult, Validator, extract_json, schema_instruction

logger = logging.getLogger("llm_bridge.base")

# A chat message is a simple dict: {"role": "system|user|assistant", "content": "..."}
Message = Dict[str, str]


@dataclass
class LLMResponse:
    """A normalised response returned by every provider."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    raw: Any = field(default=None, repr=False)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    cost_usd: Optional[float] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class UsageEvent:
    """One successful model call, for usage and cost logging."""

    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: Optional[float] = None


def split_system_messages(
    messages: List[Message],
) -> Tuple[Optional[str], List[Message]]:
    """Split a message list into ``(system_text, turns)``.

    ``system_text`` concatenates all ``system`` messages (joined by blank
    lines); ``turns`` keeps the ``user``/``assistant`` messages in order.
    Useful for providers (Bedrock, Gemini) that take the system prompt as a
    separate field rather than inline in the message list.
    """
    system_parts: List[str] = []
    turns: List[Message] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            if content:
                system_parts.append(content)
        else:
            turns.append({"role": role, "content": content})
    system_text = "\n\n".join(system_parts) if system_parts else None
    return system_text, turns


class LLMClient(ABC):
    """Vendor-neutral chat LLM client.

    Implementations only need to provide :meth:`chat` and the :attr:`model`
    property. :meth:`complete` is a convenience wrapper for single-turn calls.
    """

    @property
    @abstractmethod
    def model(self) -> str:
        """The model identifier this client talks to."""

    @property
    def provider(self) -> str:
        """Short provider name (e.g. ``"openai"``)."""
        return "unknown"

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a list of chat messages and return a normalised response."""

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        """Single-turn convenience: build messages from ``prompt``/``system``."""
        messages: List[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)

    # ------------------------------------------------------------------ #
    # Usage logging, retries and structured output (provider independent)
    # ------------------------------------------------------------------ #
    _usage_hook: Optional[Callable[[UsageEvent], None]] = None

    def set_usage_hook(self, hook: Optional[Callable[[UsageEvent], None]]) -> None:
        """Register a callable that receives a :class:`UsageEvent` after each successful call."""
        self._usage_hook = hook

    def _emit_usage(self, resp: LLMResponse) -> None:
        if self._usage_hook is None:
            return
        try:
            self._usage_hook(
                UsageEvent(
                    provider=self.provider,
                    model=resp.model or self.model,
                    prompt_tokens=resp.prompt_tokens,
                    completion_tokens=resp.completion_tokens,
                    latency_ms=resp.latency_ms,
                    cost_usd=resp.cost_usd,
                )
            )
        except Exception:  # a broken logging hook must never break a model call
            logger.exception("usage hook failed")

    def chat_with_retries(
        self,
        messages: List[Message],
        *,
        retries: int = 2,
        backoff: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        **kwargs: Any,
    ) -> LLMResponse:
        """Call :meth:`chat`, retrying transient errors (timeout, rate limit) with exponential backoff."""
        attempt = 0
        while True:
            try:
                resp = self.chat(messages, **kwargs)
                self._emit_usage(resp)
                return resp
            except (LLMTimeoutError, LLMRateLimitError):
                if attempt >= retries:
                    raise
                sleep(backoff * (2**attempt))
                attempt += 1

    # Providers override these two hooks to use native structured output.
    def _json_kwargs(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Extra ``chat`` keyword arguments that request structured output natively (default: none)."""
        return {}

    def _json_instruction(self, schema: Dict[str, Any]) -> Optional[str]:
        """System text that asks for JSON in prompt mode (``None`` when the backend enforces the schema)."""
        return schema_instruction(schema)

    def chat_json(
        self,
        messages: List[Message],
        schema: Dict[str, Any],
        *,
        validator: Optional[Validator] = None,
        max_repairs: int = 1,
        retries: int = 2,
        backoff: float = 1.0,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        sleep: Callable[[float], None] = time.sleep,
        **kwargs: Any,
    ) -> JSONResult:
        """Ask for a JSON object that follows ``schema``.

        The answer is parsed and passed to ``validator`` (a callable returning a list of error strings, empty when
        valid). On a parse or validation error the model is asked again, up to ``max_repairs`` times, with the errors in
        the message; the original messages are never modified. Raises :class:`LLMInvalidOutputError` (with the raw text)
        when the repairs are exhausted.
        """
        convo: List[Message] = list(messages)
        instruction = self._json_instruction(schema)
        if instruction:
            convo = [{"role": "system", "content": instruction}] + convo
        native = self._json_kwargs(schema)
        repairs = 0
        while True:
            resp = self.chat_with_retries(
                convo,
                retries=retries,
                backoff=backoff,
                sleep=sleep,
                temperature=temperature,
                max_tokens=max_tokens,
                **native,
                **kwargs,
            )
            errors: List[str] = []
            data: Optional[Dict[str, Any]] = None
            try:
                data = extract_json(resp.content)
                if validator is not None:
                    errors = list(validator(data))
            except LLMInvalidOutputError as exc:
                errors = [str(exc)]
            if not errors and data is not None:
                return JSONResult(data=data, response=resp, repairs=repairs, raw_text=resp.content)
            if repairs >= max_repairs:
                raise LLMInvalidOutputError("; ".join(errors) or "invalid output", raw=resp.content)
            repairs += 1
            convo = convo + [
                {"role": "assistant", "content": resp.content},
                {
                    "role": "user",
                    "content": "Your answer was not valid: "
                    + "; ".join(errors)
                    + ". Reply again with only the corrected JSON object.",
                },
            ]
