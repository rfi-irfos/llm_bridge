# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Structured output: JSON extraction, schema instruction, validation and bounded repair."""

from __future__ import annotations

import json

import pytest

from llm_bridge import LLMInvalidOutputError, create_llm
from llm_bridge.structured import extract_json

SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


def test_extract_json_handles_plain_fenced_and_surrounded_text():
    assert extract_json('{"ok": true}') == {"ok": True}
    assert extract_json('```json\n{"ok": true}\n```') == {"ok": True}
    assert extract_json('Sure, here you go: {"ok": true} hope that helps') == {"ok": True}
    assert extract_json('noise {"a": {"b": 1}} trailing {"c": 2}') == {"a": {"b": 1}}


@pytest.mark.parametrize("text", ["", "no json here", "[1, 2, 3]", '{"unterminated": ', "null"])
def test_extract_json_rejects_non_objects(text):
    with pytest.raises(LLMInvalidOutputError):
        extract_json(text)


def test_chat_json_returns_parsed_data_and_the_response():
    llm = create_llm({"provider": "mock", "response": '{"ok": true}'})
    result = llm.chat_json([{"role": "user", "content": "go"}], SCHEMA)
    assert (
        result.data == {"ok": True}
        and result.repairs == 0
        and result.response.content == '{"ok": true}'
    )


def test_prompt_mode_adds_the_schema_instruction_as_a_system_message():
    seen = []

    def responder(messages):
        seen.append(messages)
        return '{"ok": true}'

    llm = create_llm({"provider": "mock", "responder": responder})
    llm.chat_json([{"role": "user", "content": "go"}], SCHEMA)
    system = [m for m in seen[0] if m["role"] == "system"]
    assert (
        system
        and "JSON Schema" in system[0]["content"]
        and json.dumps(SCHEMA, sort_keys=True) in system[0]["content"]
    )


def test_invalid_output_is_repaired_once_with_the_validator_errors_in_the_message():
    replies = iter(['{"ok": "yes"}', '{"ok": true}'])
    prompts = []

    def responder(messages):
        prompts.append(messages)
        return next(replies)

    def validator(data):
        return [] if isinstance(data.get("ok"), bool) else ["ok must be a boolean"]

    llm = create_llm({"provider": "mock", "responder": responder})
    result = llm.chat_json(
        [{"role": "user", "content": "go"}], SCHEMA, validator=validator, max_repairs=1
    )
    assert result.data == {"ok": True} and result.repairs == 1
    assert (
        "ok must be a boolean" in prompts[1][-1]["content"]
        and prompts[1][-2]["role"] == "assistant"
    )


def test_repairs_are_bounded_and_the_failure_keeps_the_raw_text():
    llm = create_llm({"provider": "mock", "response": "I refuse to produce JSON"})
    with pytest.raises(LLMInvalidOutputError) as err:
        llm.chat_json([{"role": "user", "content": "go"}], SCHEMA, max_repairs=2)
    assert err.value.raw == "I refuse to produce JSON"


def test_validator_failure_after_all_repairs_raises():
    llm = create_llm({"provider": "mock", "response": '{"ok": 1}'})
    with pytest.raises(LLMInvalidOutputError):
        llm.chat_json(
            [{"role": "user", "content": "go"}], SCHEMA, validator=lambda d: ["bad"], max_repairs=1
        )
