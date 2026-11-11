# Copyright © 2026 Kasanki Labs.
# SPDX-License-Identifier: AGPL-3.0-only
# Licensed under the GNU AGPL v3 with an attribution term: see LICENSE and CREDIT.md.

"""Turning Python values into what an event may hold (trace-format spec §5.3)."""

from __future__ import annotations

import calendar
import math
import os
import time
import uuid
from typing import Any

#: Larger whole numbers are recorded as strings (spec §5.3 rule 1).
MAX_SAFE_INTEGER = 2**53 - 1

#: How deep recorded values may nest. An event may nest 64 levels; the event
#: and its payload use some of them.
MAX_DEPTH = 56


def to_json(value: Any, inexact: set[str], depth: int = 0) -> Any:
    """Converts `value` to JSON data. Anything that cannot be recorded exactly
    is described in `inexact`, so the caller can warn that replay may differ."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if type(value) is not int:
            inexact.add(type(value).__qualname__)
        if -MAX_SAFE_INTEGER <= value <= MAX_SAFE_INTEGER:
            return int(value)
        inexact.add("int beyond ±(2^53 − 1), recorded as a string")
        return str(int(value))
    if isinstance(value, float):
        if math.isfinite(value):
            return float(value)
        inexact.add("float NaN or Infinity")
        return repr(value)
    if isinstance(value, str):
        try:
            value.encode("utf-8")
            return str(value)
        except UnicodeEncodeError:
            inexact.add("str with lone surrogates")
            return value.encode("utf-8", "backslashreplace").decode("utf-8")
    if isinstance(value, (bytes, bytearray, memoryview)):
        # Bytes never go inside an event (spec §5.3 rule 4); blobs come later.
        inexact.add("bytes")
        return f"[{len(value)} bytes, not recorded]"
    if depth >= MAX_DEPTH:
        inexact.add(f"values nested deeper than {MAX_DEPTH} levels")
        return repr(value)
    if isinstance(value, list):
        return [to_json(v, inexact, depth + 1) for v in value]
    if isinstance(value, tuple):
        inexact.add("tuple, recorded as a list")
        return [to_json(v, inexact, depth + 1) for v in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            name = key if isinstance(key, str) else str(key)
            if not isinstance(key, str):
                inexact.add("dict keys that are not strings")
            if name in out:
                inexact.add("dict keys that collide as strings")
                n = 2
                while f"{name} ({n})" in out:
                    n += 1
                name = f"{name} ({n})"
            out[to_json(name, inexact)] = to_json(item, inexact, depth + 1)
        return out
    inexact.add(type(value).__qualname__)
    return repr(value)


def key_order(value: Any) -> dict[str, list[str]] | None:
    """Where JSON data's keys aren't sorted: each such object's path and its keys,
    in order. Events are stored with sorted keys (spec §5.3), and an agent that
    writes a tool's answer into a prompt with ``json.dumps`` needs them back to ask
    for the same thing in a replay. `None` if there's nothing to keep."""
    found: dict[str, list[str]] = {}

    def walk(item: Any, path: str) -> None:
        if isinstance(item, dict):
            keys = list(item)
            if keys != sorted(keys):
                found[path] = keys
            for key, child in item.items():
                walk(child, f"{path}/{key.replace('~', '~0').replace('/', '~1')}")
        elif isinstance(item, list):
            for i, child in enumerate(item):
                walk(child, f"{path}/{i}")

    walk(value, "")
    return found or None


def in_order(value: Any, order: dict[str, list[str]] | None) -> Any:
    """JSON data with its keys back in the order `key_order` kept."""
    if not order:
        return value

    def walk(item: Any, path: str) -> Any:
        if isinstance(item, dict):
            keys = order.get(path)
            names = [k for k in keys if k in item] if keys else []
            names += [k for k in item if k not in names]
            return {
                k: walk(item[k], f"{path}/{k.replace('~', '~0').replace('/', '~1')}")
                for k in names
            }
        if isinstance(item, list):
            return [walk(child, f"{path}/{i}") for i, child in enumerate(item)]
        return item

    return walk(value, "")


def wall_time(nanos: int) -> str:
    """`YYYY-MM-DDTHH:MM:SS.nnnnnnnnnZ` (UTC, 9 decimals), as events need."""
    seconds, fraction = divmod(nanos, 1_000_000_000)
    return (
        time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(seconds)) + f".{fraction:09d}Z"
    )


def wall_time_nanos(text: str) -> int:
    """Nanoseconds since 1970 from a `wall_time` string."""
    seconds = calendar.timegm(time.strptime(text[:19], "%Y-%m-%dT%H:%M:%S"))
    return seconds * 1_000_000_000 + int(text[20:29])


def uuid7() -> str:
    """A UUID version 7 (RFC 9562): 48-bit Unix milliseconds, then random bits."""
    millis = time.time_ns() // 1_000_000
    rand = int.from_bytes(os.urandom(10), "big")
    rand_a = (rand >> 62) & 0xFFF
    rand_b = rand & ((1 << 62) - 1)
    value = (
        (millis & ((1 << 48) - 1)) << 80
        | 0x7 << 76
        | rand_a << 64
        | 0b10 << 62
        | rand_b
    )
    return str(uuid.UUID(int=value))
