# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Tool-call proposals: the OpenAI adapter maps tool calls into LLMResponse and never executes them."""

from __future__ import annotations

import contextlib
import sys
import types
from typing import Any

import pytest

from llm_bridge import create_llm


@pytest.fixture
def fake_openai(monkeypatch):
    captured: dict[str, Any] = {}

    class _Fn:
        name = "artefact.grep"
        arguments = '{"pattern": "initOrder"}'

    class _Call:
        id = "call_1"
        function = _Fn()

    class _Msg:
        content = None
        tool_calls = [_Call()]

    class _Choice:
        message = _Msg()
        finish_reason = "tool_calls"

    class _Usage:
        prompt_tokens, completion_tokens = 11, 7

    class _Resp:
        choices = [_Choice()]
        usage = _Usage()
        model = "gpt-test"

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Resp()

    class _OpenAI:
        def __init__(self, **kwargs):
            captured["init"] = kwargs
            self.chat = types.SimpleNamespace(completions=_Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_OpenAI))
    return captured


def test_tools_are_passed_through_and_tool_calls_are_returned_as_proposals(fake_openai):
    llm = create_llm(
        {
            "provider": "openai",
            "model": "gpt-test",
            "api_key": "k",
            "base_url": "http://x/v1",
            "timeout": 30,
        }
    )
    tools = [
        {
            "type": "function",
            "function": {"name": "artefact.grep", "parameters": {"type": "object"}},
        }
    ]
    resp = llm.chat([{"role": "user", "content": "go"}], tools=tools, tool_choice="auto")
    assert fake_openai["tools"] == tools and fake_openai["tool_choice"] == "auto"
    assert resp.content == "" and resp.tool_calls == [
        {"id": "call_1", "name": "artefact.grep", "arguments": {"pattern": "initOrder"}}
    ]
    assert fake_openai["init"]["timeout"] == 30


def test_json_mode_schema_sends_a_response_format_and_prompt_mode_does_not(fake_openai):
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    llm = create_llm({"provider": "openai", "model": "m", "api_key": "k", "json_mode": "schema"})
    with contextlib.suppress(
        Exception
    ):  # the fake returns no content; only the request matters here
        llm.chat_json([{"role": "user", "content": "x"}], schema, max_repairs=0)
    assert fake_openai["response_format"]["type"] == "json_schema"
    fake_openai.clear()
    llm = create_llm({"provider": "openai", "model": "m", "api_key": "k"})
    with contextlib.suppress(Exception):
        llm.chat_json([{"role": "user", "content": "x"}], schema, max_repairs=0)
    assert "response_format" not in fake_openai


def test_ollama_alias_points_the_openai_client_at_the_local_server(fake_openai):
    llm = create_llm({"provider": "ollama", "model": "qwen2.5:3b"})
    assert (
        llm.provider == "openai"
        and fake_openai["init"]["base_url"] == "http://localhost:11434/v1"
        and fake_openai["init"]["api_key"] == "ollama"
    )


def test_sdk_errors_are_mapped_to_typed_errors(monkeypatch):
    from llm_bridge import LLMProviderError, LLMRateLimitError, LLMTimeoutError

    class APITimeoutError(Exception):
        pass

    class RateLimitError(Exception):
        pass

    class _Completions:
        def __init__(self, exc):
            self.exc = exc

        def create(self, **kwargs):
            raise self.exc

    def make(exc):
        class _OpenAI:
            def __init__(self, **kw):
                self.chat = types.SimpleNamespace(completions=_Completions(exc))

        monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_OpenAI))
        return create_llm({"provider": "openai", "model": "m", "api_key": "k"})

    msgs = [{"role": "user", "content": "x"}]
    for exc, expected in (
        (APITimeoutError("t"), LLMTimeoutError),
        (RateLimitError("r"), LLMRateLimitError),
        (RuntimeError("boom"), LLMProviderError),
    ):
        with pytest.raises(expected):
            make(exc).chat(msgs)
