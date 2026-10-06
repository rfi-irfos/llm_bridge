# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Retries with backoff for transient errors, and the usage hook."""

from __future__ import annotations

import pytest

from llm_bridge import (
    LLMInvalidOutputError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    UsageEvent,
    create_llm,
)


def flaky(errors, final="ok"):
    state = {"n": 0}

    def responder(messages):
        state["n"] += 1
        if state["n"] <= len(errors):
            raise errors[state["n"] - 1]
        return final

    return responder, state


def test_transient_errors_are_retried_with_exponential_backoff():
    responder, state = flaky([LLMTimeoutError("t"), LLMRateLimitError("r")])
    sleeps = []
    llm = create_llm({"provider": "mock", "responder": responder})
    resp = llm.chat_with_retries(
        [{"role": "user", "content": "x"}], retries=3, backoff=0.5, sleep=sleeps.append
    )
    assert resp.content == "ok" and state["n"] == 3 and sleeps == [0.5, 1.0]


def test_retries_are_bounded():
    responder, state = flaky([LLMTimeoutError("t")] * 5)
    llm = create_llm({"provider": "mock", "responder": responder})
    with pytest.raises(LLMTimeoutError):
        llm.chat_with_retries(
            [{"role": "user", "content": "x"}], retries=2, backoff=0, sleep=lambda s: None
        )
    assert state["n"] == 3


@pytest.mark.parametrize(
    "error", [LLMProviderError("boom"), LLMInvalidOutputError("bad", raw="r"), ValueError("v")]
)
def test_non_transient_errors_are_not_retried(error):
    responder, state = flaky([error])
    llm = create_llm({"provider": "mock", "responder": responder})
    with pytest.raises(type(error)):
        llm.chat_with_retries(
            [{"role": "user", "content": "x"}], retries=3, backoff=0, sleep=lambda s: None
        )
    assert state["n"] == 1


def test_usage_hook_receives_one_event_per_successful_call():
    events = []
    llm = create_llm({"provider": "mock", "response": '{"ok": true}', "model": "m1"})
    llm.set_usage_hook(events.append)
    llm.chat_with_retries([{"role": "user", "content": "x"}])
    llm.chat_json([{"role": "user", "content": "y"}], {"type": "object"})
    assert len(events) == 2 and all(
        isinstance(e, UsageEvent) and e.model == "m1" and e.provider == "mock" for e in events
    )
    assert events[0].prompt_tokens >= 1 and events[0].latency_ms >= 0


def test_a_failing_usage_hook_never_breaks_the_call():
    llm = create_llm({"provider": "mock", "response": "fine"})
    llm.set_usage_hook(lambda e: 1 / 0)
    assert llm.chat_with_retries([{"role": "user", "content": "x"}]).content == "fine"
