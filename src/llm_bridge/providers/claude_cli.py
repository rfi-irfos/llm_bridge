# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Claude through the Claude Code CLI in print mode (``claude -p``). Stdlib only.

The CLI runs isolated: no tools, no session persistence, no MCP servers, no slash commands, no user settings, an empty working directory.
The model can only answer in text (or JSON with ``json_schema``), it cannot read files or run commands.

Configuration (config keys):
    model            optional model alias passed as ``--model``
    binary           CLI name or path (default ``claude``)
    timeout          seconds (default 300)
    max_budget_usd   optional spending cap for one call
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from llm_bridge.base import LLMClient, LLMResponse, Message, split_system_messages
from llm_bridge.errors import LLMProviderError, LLMRateLimitError, LLMTimeoutError

Runner = Callable[[List[str], str, float, Optional[str]], Tuple[int, str, str]]


def _subprocess_runner(
    cmd: List[str], input_text: str, timeout: float, cwd: Optional[str]
) -> Tuple[int, str, str]:
    proc = subprocess.run(
        cmd, input=input_text, capture_output=True, text=True, timeout=timeout, cwd=cwd, check=False
    )
    return proc.returncode, proc.stdout, proc.stderr


class ClaudeCLIClient(LLMClient):
    def __init__(
        self,
        model: Optional[str] = None,
        binary: str = "claude",
        timeout: float = 300.0,
        max_budget_usd: Optional[float] = None,
        extra_args: Optional[Sequence[str]] = None,
        runner: Optional[Runner] = None,
        cwd: Optional[str] = None,
    ):
        self._model = model
        self._binary = binary
        self._timeout = timeout
        self._max_budget = max_budget_usd
        self._extra = list(extra_args or [])
        self._runner = runner or _subprocess_runner
        self._cwd = cwd

    @property
    def model(self) -> str:
        return self._model or "claude-cli"

    @property
    def provider(self) -> str:
        return "claude_cli"

    def _json_kwargs(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        return {"json_schema": schema}

    def _json_instruction(self, schema: Dict[str, Any]) -> Optional[str]:
        return None  # the CLI enforces the schema itself

    def chat(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        system, turns = split_system_messages(messages)
        if len(turns) == 1 and turns[0]["role"] == "user":
            text = turns[0]["content"]
        else:
            text = "\n\n".join(f"{t['role'].upper()}: {t['content']}" for t in turns)
        schema = kwargs.get("json_schema")
        with tempfile.TemporaryDirectory(prefix="llm_bridge_cli_") as work:
            cmd = [
                self._binary,
                "-p",
                "--output-format",
                "json",
                "--no-session-persistence",
                "--tools",
                "",
                "--setting-sources",
                "project",
                "--strict-mcp-config",
                "--disable-slash-commands",
            ]
            if system:
                path = os.path.join(work, "system.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(system)
                cmd += ["--system-prompt-file", path]
            if self._model:
                cmd += ["--model", self._model]
            if schema is not None:
                cmd += ["--json-schema", json.dumps(schema, sort_keys=True)]
            if self._max_budget is not None:
                cmd += ["--max-budget-usd", str(self._max_budget)]
            cmd += self._extra
            start = time.perf_counter() * 1000
            try:
                code, out, err = self._runner(cmd, text, self._timeout, self._cwd or work)
            except (TimeoutError, subprocess.TimeoutExpired) as exc:
                raise LLMTimeoutError(f"claude did not answer within {self._timeout} s") from exc
            wall = time.perf_counter() * 1000 - start
        if code != 0:
            raise LLMProviderError(f"claude exited with status {code}: {(err or out)[:200]}")
        try:
            data = json.loads(out)
        except json.JSONDecodeError as exc:
            raise LLMProviderError("claude returned output that is not JSON") from exc
        if not isinstance(data, dict):
            raise LLMProviderError("claude returned an unexpected JSON shape")
        if data.get("is_error"):
            message = str(data.get("result", ""))[:200]
            if "rate limit" in message.lower():
                raise LLMRateLimitError(message)
            raise LLMProviderError(message)
        usage = data.get("usage") or {}
        content = (
            json.dumps(data["structured_output"])
            if data.get("structured_output") is not None
            else str(data.get("result", ""))
        )
        models = list((data.get("modelUsage") or {}).keys())
        return LLMResponse(
            content=content,
            model=models[0] if models else self.model,
            prompt_tokens=int(usage.get("input_tokens", 0))
            + int(usage.get("cache_creation_input_tokens", 0))
            + int(usage.get("cache_read_input_tokens", 0)),
            completion_tokens=int(usage.get("output_tokens", 0)),
            finish_reason=str(data.get("terminal_reason", "stop")),
            latency_ms=float(data.get("duration_ms", wall)),
            raw=data,
            cost_usd=data.get("total_cost_usd"),
        )


def build(config: Dict[str, Any]) -> ClaudeCLIClient:
    """Build a :class:`ClaudeCLIClient` from a config dict."""
    return ClaudeCLIClient(
        model=config.get("model"),
        binary=config.get("binary", "claude"),
        timeout=float(config.get("timeout", 300)),
        max_budget_usd=config.get("max_budget_usd"),
        extra_args=config.get("extra_args"),
    )
