# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Typed errors, so callers can tell a transient failure from a bad answer."""

from __future__ import annotations


class LLMBridgeError(Exception):
    """Base class of every error raised by llm_bridge."""


class LLMTimeoutError(LLMBridgeError):
    """The backend did not answer in time (transient, retried)."""


class LLMRateLimitError(LLMBridgeError):
    """The backend asked to slow down (transient, retried)."""


class LLMProviderError(LLMBridgeError):
    """The backend failed in a way that retrying does not help."""


class LLMInvalidOutputError(LLMBridgeError):
    """The model answered, but not with the requested structure. ``raw`` keeps the text for audit."""

    def __init__(self, message: str, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw
