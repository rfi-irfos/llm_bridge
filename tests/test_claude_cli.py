# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Claude Code CLI provider (print mode, no tools, isolated settings), tested with an injected runner."""

from __future__ import annotations

import json

import pytest

from llm_bridge import LLMProviderError, LLMRateLimitError, LLMTimeoutError, create_llm
from llm_bridge.providers.claude_cli import ClaudeCLIClient


class Runner:
    def __init__(self, stdout, returncode=0, stderr="", raises=None):
        self.stdout, self.returncode, self.stderr, self.raises, self.calls = (
            stdout,
            returncode,
            stderr,
            raises,
            [],
        )

    def __call__(self, cmd, input_text, timeout, cwd):
        self.calls.append({"cmd": cmd, "input": input_text, "timeout": timeout, "cwd": cwd})
        if self.raises:
            raise self.raises
        return self.returncode, self.stdout, self.stderr


def payload(**over):
    base = {
        "type": "result",
        "is_error": False,
        "result": "hello",
        "total_cost_usd": 0.0123,
        "duration_ms": 900,
        "usage": {
            "input_tokens": 2,
            "cache_creation_input_tokens": 1000,
            "cache_read_input_tokens": 50,
            "output_tokens": 40,
        },
        "modelUsage": {"claude-opus-5-5": {"inputTokens": 2}},
    }
    base.update(over)
    return json.dumps(base)


def client(runner, **kw):
    return ClaudeCLIClient(runner=runner, **kw)


def test_command_is_isolated_and_has_no_tools_and_no_session():
    r = Runner(payload())
    client(r, model="opus").chat(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    )
    cmd = r.calls[0]["cmd"]
    assert cmd[0] == "claude" and "-p" in cmd and cmd[cmd.index("--output-format") + 1] == "json"
    assert (
        cmd[cmd.index("--tools") + 1] == ""
        and "--no-session-persistence" in cmd
        and "--strict-mcp-config" in cmd
        and "--disable-slash-commands" in cmd
    )
    assert (
        cmd[cmd.index("--setting-sources") + 1] == "project"
        and cmd[cmd.index("--model") + 1] == "opus"
    )
    assert "--dangerously-skip-permissions" not in cmd and "--allowedTools" not in cmd


def test_system_prompt_goes_through_a_file_and_the_user_text_through_stdin(tmp_path):
    r = Runner(payload())
    client(r).chat(
        [{"role": "system", "content": "SYSTEM TEXT"}, {"role": "user", "content": "USER TEXT"}]
    )
    call = r.calls[0]
    assert "SYSTEM TEXT" not in " ".join(call["cmd"]) and "--system-prompt-file" in call["cmd"]
    assert "USER TEXT" in call["input"]


def test_response_fields_are_mapped_including_cost_and_tokens():
    resp = client(Runner(payload())).chat([{"role": "user", "content": "hi"}])
    assert (
        resp.content == "hello"
        and resp.prompt_tokens == 1052
        and resp.completion_tokens == 40
        and resp.cost_usd == 0.0123
    )
    assert resp.model == "claude-opus-5-5" and resp.latency_ms == 900


def test_json_schema_is_passed_and_structured_output_is_returned_as_json_text():
    r = Runner(payload(result="", structured_output={"ok": True}))
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    result = client(r).chat_json([{"role": "user", "content": "go"}], schema)
    cmd = r.calls[0]["cmd"]
    assert json.loads(cmd[cmd.index("--json-schema") + 1]) == schema and result.data == {"ok": True}


def test_errors_are_mapped():
    with pytest.raises(LLMProviderError):
        client(Runner("", returncode=1, stderr="boom")).chat([{"role": "user", "content": "x"}])
    with pytest.raises(LLMProviderError):
        client(Runner(payload(is_error=True, result="invalid request"))).chat(
            [{"role": "user", "content": "x"}]
        )
    with pytest.raises(LLMRateLimitError):
        client(Runner(payload(is_error=True, result="rate limit exceeded"))).chat(
            [{"role": "user", "content": "x"}]
        )
    with pytest.raises(LLMProviderError):
        client(Runner("not json")).chat([{"role": "user", "content": "x"}])
    with pytest.raises(LLMTimeoutError):
        client(Runner("", raises=TimeoutError("t"))).chat([{"role": "user", "content": "x"}])


def test_budget_limit_and_extra_args_are_forwarded_and_the_registry_builds_the_provider():
    r = Runner(payload())
    ClaudeCLIClient(runner=r, max_budget_usd=0.5, extra_args=["--effort", "low"]).chat(
        [{"role": "user", "content": "x"}]
    )
    cmd = r.calls[0]["cmd"]
    assert cmd[cmd.index("--max-budget-usd") + 1] == "0.5" and "--effort" in cmd
    assert create_llm({"provider": "claude_cli", "model": "opus"}).provider == "claude_cli"
