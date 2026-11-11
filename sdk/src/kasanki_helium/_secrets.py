# Copyright © 2026 Kasanki Labs.
# SPDX-License-Identifier: AGPL-3.0-only
# Licensed under the GNU AGPL v3 with an attribution term: see LICENSE and CREDIT.md.

"""Stripping secrets before anything is recorded (trace-format spec §10, D11).

This always happens, whatever the recorder profile says:

- The value under any key that names a secret (``api_key``, ``password``,
  ``Authorization``, ``access_token``, ``cookie``…) is replaced.
- Well-known secret formats are replaced wherever they appear in text:
  provider API keys, bearer tokens, cloud and code-hosting keys, JSON web
  tokens, private key blocks, and passwords inside URLs.

Replay of a stripped value is not exact, by design.
"""

from __future__ import annotations

import functools
import re
import threading
from typing import Any

REDACTED = "[redacted]"

# A key names a secret if, lowercased with separators removed, it ends with
# one of these ("max_tokens" does not: it ends with "tokens").
_SECRET_KEY_ENDINGS = (
    "password",
    "passwd",
    "passphrase",
    "secret",
    "token",
    "apikey",
    "accesskey",
    "privatekey",
    "authorization",
    "cookie",
    "sessionid",
    "credential",
    "credentials",
)
# Short names that are only secrets on their own.
_SECRET_KEYS = {"pwd", "pass", "auth"}

# Each pattern, with the text (lowercase) a match must contain: a model call
# carries hundreds of KiB of text, and a substring check is far cheaper than
# a regular expression, so most text is never searched.
_PATTERNS = [
    # Private key blocks (PEM).
    (
        ("private key-----",),
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
        REDACTED,
    ),
    # Passwords inside URLs: scheme://user:password@host
    (
        ("://",),
        re.compile(r"(\b[a-zA-Z][a-zA-Z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@"),
        r"\1" + REDACTED + "@",
    ),
    # Bearer tokens.
    (
        ("bearer",),
        re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9\-._~+/]+=*"),
        r"\1" + REDACTED,
    ),
    # OpenAI / Anthropic style keys: sk-..., sk-ant-...
    (("sk-",), re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"), REDACTED),
    # NVIDIA keys.
    (("nvapi-",), re.compile(r"\bnvapi-[A-Za-z0-9_\-]{20,}"), REDACTED),
    # Stripe keys.
    (
        ("k_live_", "k_test_"),
        re.compile(r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{16,}"),
        REDACTED,
    ),
    # AWS access key IDs.
    (("akia", "asia"), re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), REDACTED),
    # GitHub tokens.
    (
        ("ghp_", "gho_", "ghu_", "ghs_", "ghr_"),
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}"),
        REDACTED,
    ),
    (("github_pat_",), re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}"), REDACTED),
    # Slack tokens.
    (("xox",), re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"), REDACTED),
    # Google API keys.
    (("aiza",), re.compile(r"\bAIza[0-9A-Za-z_\-]{35}"), REDACTED),
    # JSON web tokens.
    (
        ("eyj",),
        re.compile(
            r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"
        ),
        REDACTED,
    ),
]


# Payloads repeat the same few keys, so each is checked once.
@functools.lru_cache(maxsize=4096)
def is_secret_key(key: str) -> bool:
    normalized = "".join(c for c in key.lower() if c.isalnum())
    return normalized in _SECRET_KEYS or normalized.endswith(_SECRET_KEY_ENDINGS)


#: Long texts already stripped, and the result: a conversation sends every
#: earlier message again on each model call. Bounded by their total length.
_CACHE_FROM = 512
_CACHE_CHARS = 16 * 1024 * 1024
_cache: dict[str, str] = {}
_cache_chars = 0
_cache_lock = threading.Lock()


def redact_text(text: str) -> str:
    if len(text) >= _CACHE_FROM:
        done = _cache.get(text)
        if done is not None:
            return done
    lowered = text.lower()
    out = text
    for needles, pattern, replacement in _PATTERNS:
        if any(needle in lowered for needle in needles):
            out = pattern.sub(replacement, out)
    if len(text) >= _CACHE_FROM:
        _remember(text, out)
    return out


def _remember(text: str, out: str) -> None:
    global _cache_chars
    with _cache_lock:
        if text in _cache:
            return
        _cache[text] = out
        _cache_chars += len(text) + len(out)
        # Oldest first: dicts keep insertion order.
        while _cache_chars > _CACHE_CHARS and _cache:
            old, old_out = next(iter(_cache.items()))
            del _cache[old]
            _cache_chars -= len(old) + len(old_out)


def redact(value: Any) -> Any:
    """A copy of JSON data with secrets replaced by ``"[redacted]"``."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, dict):
        return {
            k: (REDACTED if is_secret_key(k) and v is not None else redact(v))
            for k, v in value.items()
        }
    return value
