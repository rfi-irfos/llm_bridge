import io
import json
import urllib.error

import pytest

from llm_bridge import create_llm
from llm_bridge.errors import LLMProviderError, LLMRateLimitError, LLMTimeoutError
from llm_bridge.providers.openai_http import OpenAIHTTPClient


class Reply(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def client(handler):
    return OpenAIHTTPClient(
        "m", api_key="SECRET-KEY", base_url="https://example.invalid/v1", opener=handler
    )


def test_posts_to_chat_completions_with_bearer_and_parses_usage_and_tool_calls():
    seen = {}

    def handler(request, timeout):
        seen["url"], seen["auth"], seen["body"], seen["timeout"] = (
            request.full_url,
            request.get_header("Authorization"),
            json.loads(request.data),
            timeout,
        )
        return Reply(
            json.dumps(
                {
                    "model": "m2",
                    "choices": [
                        {
                            "message": {
                                "content": "hi",
                                "tool_calls": [
                                    {"id": "1", "function": {"name": "t", "arguments": '{"a": 1}'}}
                                ],
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 4},
                }
            ).encode()
        )

    r = client(handler).chat([{"role": "user", "content": "x"}], max_tokens=5)
    assert (
        seen["url"] == "https://example.invalid/v1/chat/completions"
        and seen["auth"] == "Bearer SECRET-KEY"
        and seen["body"]["max_tokens"] == 5
    )
    assert (r.content, r.prompt_tokens, r.completion_tokens, r.model) == ("hi", 3, 4, "m2")
    assert r.tool_calls == [{"id": "1", "name": "t", "arguments": {"a": 1}}]


def test_http_429_maps_to_rate_limit_and_other_errors_to_provider_error():
    def h(code):
        def handler(request, timeout):
            raise urllib.error.HTTPError(request.full_url, code, "x", {}, io.BytesIO(b"slow down"))

        return handler

    with pytest.raises(LLMRateLimitError):
        client(h(429)).chat([{"role": "user", "content": "x"}])
    with pytest.raises(LLMProviderError) as exc:
        client(h(401)).chat([{"role": "user", "content": "x"}])
    assert "SECRET-KEY" not in str(exc.value)


def test_timeout_maps_to_timeout_error():
    def handler(request, timeout):
        raise TimeoutError("late")

    with pytest.raises(LLMTimeoutError):
        client(handler).chat([{"role": "user", "content": "x"}])


def test_plain_http_to_a_remote_host_is_refused_and_registry_builds_it():
    with pytest.raises(ValueError):
        OpenAIHTTPClient("m", base_url="http://example.com/v1")
    assert (
        create_llm(
            {"provider": "openai_http", "model": "m", "base_url": "https://example.invalid/v1"}
        ).provider
        == "openai_http"
    )
