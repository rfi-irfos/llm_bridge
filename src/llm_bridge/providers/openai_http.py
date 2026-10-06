# Copyright (c) 2026 Santander Group (original), RFI-IRFOS fork additions
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible chat provider on the standard library only (``urllib``).

Same wire format as the ``openai`` provider (OpenRouter, NVIDIA NIM, Gemini compatibility endpoint, vLLM, Ollama) without the SDK.
Configuration: ``model`` (required), ``api_key`` (never logged), ``base_url``, ``timeout`` seconds, ``json_mode`` as for ``openai``.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from llm_bridge.base import LLMResponse, Message
from llm_bridge.errors import LLMProviderError, LLMRateLimitError, LLMTimeoutError
from llm_bridge.providers.openai_sdk import OpenAIClient

Opener = Callable[[urllib.request.Request, float], Any]


def _default_opener(request: urllib.request.Request, timeout: float) -> Any:
    return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310 (scheme is fixed by base_url validation)


class OpenAIHTTPClient(OpenAIClient):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        json_mode: str = "prompt",
        opener: Optional[Opener] = None,
    ):
        if not model:
            raise ValueError("openai_http provider requires 'model'.")
        if not base_url or not base_url.startswith(
            ("https://", "http://localhost", "http://127.0.0.1")
        ):
            raise ValueError("openai_http requires an https base_url (or a loopback http URL).")
        self._model = model
        self._json_mode = json_mode
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = float(timeout or 120)
        self._opener = opener or _default_opener

    @property
    def provider(self) -> str:
        return "openai_http"

    def chat(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        body: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        body.update(kwargs)
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        start = time.perf_counter() * 1000
        try:
            with self._opener(request, self._timeout) as handle:
                data = json.loads(handle.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:200] if hasattr(exc, "read") else ""
            if exc.code == 429:
                raise LLMRateLimitError(f"HTTP 429: {detail}") from exc
            raise LLMProviderError(f"HTTP {exc.code}: {detail}") from exc
        except (socket.timeout, TimeoutError) as exc:
            raise LLMTimeoutError(str(exc)[:200]) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                raise LLMTimeoutError(str(exc.reason)[:200]) from exc
            raise LLMProviderError(f"URLError: {str(exc.reason)[:200]}") from exc
        except json.JSONDecodeError as exc:
            raise LLMProviderError("the endpoint did not return JSON") from exc
        latency = time.perf_counter() * 1000 - start
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError("the response has no choices") from exc
        usage = data.get("usage") or {}
        tool_calls: List[Dict[str, Any]] = []
        for call in message.get("tool_calls") or []:
            raw = (call.get("function") or {}).get("arguments") or ""
            try:
                parsed = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                parsed = {"_raw": raw}
            tool_calls.append(
                {
                    "id": call.get("id", ""),
                    "name": (call.get("function") or {}).get("name", ""),
                    "arguments": parsed,
                }
            )
        return LLMResponse(
            content=message.get("content") or "",
            model=data.get("model", self._model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason") or "stop",
            latency_ms=latency,
            raw=data,
            tool_calls=tool_calls,
        )


def build(config: Dict[str, Any]) -> OpenAIHTTPClient:
    return OpenAIHTTPClient(
        model=config.get("model", ""),
        api_key=config.get("api_key"),
        base_url=config.get("base_url"),
        timeout=config.get("timeout"),
        json_mode=config.get("json_mode", "prompt"),
    )
