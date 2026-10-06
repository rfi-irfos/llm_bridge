# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Structured (JSON) output helpers. Pure stdlib: validation is pluggable, so no schema library is required."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from llm_bridge.errors import LLMInvalidOutputError

Validator = Callable[[Dict[str, Any]], List[str]]
_FENCE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Dict[str, Any]:
    """Return the first JSON object found in ``text`` (plain, fenced, or surrounded by prose)."""
    candidate = (text or "").strip()
    fenced = _FENCE.match(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    decoder = json.JSONDecoder()
    for start in [i for i, ch in enumerate(candidate) if ch == "{"]:
        try:
            value, _ = decoder.raw_decode(candidate[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise LLMInvalidOutputError("no JSON object found in the model output", raw=text or "")


def schema_instruction(schema: Dict[str, Any]) -> str:
    return (
        "Respond with a single JSON object and nothing else (no prose, no code fences). "
        "It must conform to this JSON Schema: " + json.dumps(schema, sort_keys=True)
    )


@dataclass
class JSONResult:
    """Parsed structured answer, the last raw response, and how many repair rounds were needed."""

    data: Dict[str, Any]
    response: Any
    repairs: int = 0
    raw_text: Optional[str] = None
