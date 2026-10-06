# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""OpenAI provider using the official ``openai`` SDK.

Optional dependency — requires ``pip install llm-bridge[openai]``.
Also works with OpenAI-compatible servers by passing a ``base_url``.

Configuration (config keys, with environment fallbacks):
    model      (required)
    api_key    OPENAI_API_KEY
    base_url   OPENAI_BASE_URL   (optional; for Azure/vLLM/Ollama/OpenRouter/etc.)
    timeout    seconds (optional)
    json_mode  "prompt" (default: schema in the prompt, answer parsed and validated), "schema"
               (response_format json_schema) or "object" (response_format json_object)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from llm_bridge.base import LLMClient, LLMResponse, Message
from llm_bridge.errors import LLMBridgeError, LLMProviderError, LLMRateLimitError, LLMTimeoutError


class OpenAIClient(LLMClient):
    """Chat client backed by the official OpenAI Python SDK."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        json_mode: str = "prompt",
    ):
        if not model:
            raise ValueError("openai provider requires 'model'.")
        self._model = model
        self._json_mode = json_mode

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'openai' provider requires the openai SDK. "
                "Install it with: pip install llm-bridge[openai]"
            ) from exc

        client_kwargs: Dict[str, Any] = {
            "api_key": api_key or os.environ.get("OPENAI_API_KEY"),
            "base_url": base_url or os.environ.get("OPENAI_BASE_URL"),
        }
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        self._client = OpenAI(**client_kwargs)

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "openai"

    def chat(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        start = time.perf_counter() * 1000
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        except LLMBridgeError:
            raise
        except Exception as exc:  # map SDK errors by class name, the SDK is an optional dependency
            name = type(exc).__name__
            if "Timeout" in name:
                raise LLMTimeoutError(str(exc)[:200]) from exc
            if "RateLimit" in name:
                raise LLMRateLimitError(str(exc)[:200]) from exc
            raise LLMProviderError(f"{name}: {str(exc)[:200]}") from exc
        latency = time.perf_counter() * 1000 - start

        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        tool_calls: List[Dict[str, Any]] = []
        for call in getattr(choice.message, "tool_calls", None) or []:
            raw_args = getattr(call.function, "arguments", "") or ""
            try:
                parsed = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                parsed = {"_raw": raw_args}
            tool_calls.append(
                {"id": getattr(call, "id", ""), "name": call.function.name, "arguments": parsed}
            )
        return LLMResponse(
            tool_calls=tool_calls,
            content=choice.message.content or "",
            model=getattr(resp, "model", self._model),
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            finish_reason=choice.finish_reason or "stop",
            latency_ms=latency,
            raw=resp,
        )

    def _json_kwargs(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        if self._json_mode == "schema":
            return {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "response", "schema": schema, "strict": False},
                }
            }
        if self._json_mode == "object":
            return {"response_format": {"type": "json_object"}}
        return {}

    def _json_instruction(self, schema: Dict[str, Any]) -> Optional[str]:
        return None if self._json_mode == "schema" else super()._json_instruction(schema)


def build(config: Dict[str, Any]) -> OpenAIClient:
    """Build an :class:`OpenAIClient` from a config dict."""
    model = config.get("model")
    if not model:
        raise ValueError("openai provider requires 'model'.")
    return OpenAIClient(
        model=model,
        api_key=config.get("api_key"),
        base_url=config.get("base_url"),
        timeout=config.get("timeout"),
        json_mode=config.get("json_mode", "prompt"),
    )
