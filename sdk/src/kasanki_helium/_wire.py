# Copyright © 2026 Kasanki Labs.
# SPDX-License-Identifier: AGPL-3.0-only
# Licensed under the GNU AGPL v3 with an attribution term: see LICENSE and CREDIT.md.

"""The engine protocol (trace-format spec §7.5): 4-byte big-endian length + JSON."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Any

PROTOCOL_VERSION = 0

#: The largest frame either side accepts.
MAX_FRAME = 2 * 1024 * 1024

#: An event is at most 1 MiB once canonical (spec §5.3). The SDK keeps a
#: margin for the chain fields the engine adds and for spelling differences.
MAX_BODY = 1024 * 1024 - 1024

#: A blob's bytes per ``blob`` part: 1 MiB once in base64.
BLOB_PART = 768 * 1024

#: The largest blob.
MAX_BLOB = 256 * 1024 * 1024


def dumps(value: Any) -> bytes:
    """Compact JSON, UTF-8, no NaN or Infinity."""
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def frame(payload: bytes) -> bytes:
    if not 0 < len(payload) <= MAX_FRAME:
        raise ValueError(f"frame of {len(payload)} bytes is outside 1..{MAX_FRAME}")
    return len(payload).to_bytes(4, "big") + payload


@dataclass(frozen=True, eq=False)
class Blob:
    """Content stored once, by its hash, and sent before the events that
    name it."""

    hash: str
    content: bytes

    @classmethod
    def of(cls, content: bytes) -> Blob:
        hasher = hashlib.sha256(b"helium/blob/v0\x00")
        hasher.update(content)
        return cls(f"sha256:{hasher.hexdigest()}", content)

    def messages(self) -> list[bytes]:
        """Its ``blob`` messages, in order, without frames."""
        size = len(self.content)
        parts = max(1, -(-size // BLOB_PART))
        view = memoryview(self.content)
        return [
            dumps(
                {
                    "type": "blob",
                    "hash": self.hash,
                    "size": str(size),
                    "part": str(part),
                    "parts": str(parts),
                    "data": base64.b64encode(
                        view[part * BLOB_PART : (part + 1) * BLOB_PART]
                    ).decode("ascii"),
                }
            )
            for part in range(parts)
        ]


def hello(client: str) -> bytes:
    return frame(
        dumps({"type": "hello", "protocol": PROTOCOL_VERSION, "client": client})
    )


def event_message(run_id: str, seq: int, body_json: bytes) -> bytes:
    """An event message around an already-encoded body, without its frame."""
    head = dumps({"type": "event", "run_id": run_id, "seq": str(seq)})
    # Splice the body in as the last field: {"type":...,"seq":"N","body":{...}}
    return head[:-1] + b',"body":' + body_json + b"}"


def event(run_id: str, seq: int, body_json: bytes) -> bytes:
    """An event message around an already-encoded body, framed."""
    return frame(event_message(run_id, seq, body_json))


class FrameReader:
    """Splits the incoming byte stream into decoded messages."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes) -> list[dict[str, Any]]:
        self._buffer += data
        messages = []
        while len(self._buffer) >= 4:
            length = int.from_bytes(self._buffer[:4], "big")
            if not 0 < length <= MAX_FRAME:
                raise ValueError(f"frame length {length} is outside 1..{MAX_FRAME}")
            if len(self._buffer) < 4 + length:
                break
            payload = bytes(self._buffer[4 : 4 + length])
            del self._buffer[: 4 + length]
            message = json.loads(payload)
            if not isinstance(message, dict) or not isinstance(
                message.get("type"), str
            ):
                raise TypeError("a message must be a JSON object with a type")
            messages.append(message)
        return messages
