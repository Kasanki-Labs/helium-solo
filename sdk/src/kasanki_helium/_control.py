# Copyright © 2026 Kasanki Labs.
# SPDX-License-Identifier: AGPL-3.0-only
# Licensed under the GNU AGPL v3 with an attribution term: see LICENSE and CREDIT.md.

"""A synchronous connection for control requests: one
request at a time, each waiting for its answer. The replay harness asks the
engine for every recorded value this way, apart from
the recording connection, which only sends."""

from __future__ import annotations

import base64
import errno
import json
import socket
import sys
import threading
import time
from typing import Any

from . import _wire


class ControlError(Exception):
    """The engine refused a request, or could not be reached."""


#: How long to keep trying a pipe whose instances are all in use.
PIPE_BUSY_WAIT = 5.0
#: How long to keep trying a pipe that isn't there yet: the engine makes the
#: next instance just after handing one out, so a stopped engine is still
#: reported almost at once.
PIPE_MISSING_WAIT = 0.3


def _open_pipe(endpoint: str) -> Any:
    """Opens the engine's named pipe. Several connections opened at the same
    moment can find every instance in use, which Windows reports as a busy
    pipe or as an invalid argument (STATUS §4): those are tried again."""
    start = time.monotonic()
    wait = 0.02
    while True:
        try:
            return open(endpoint, "r+b", buffering=0)
        except OSError as e:
            busy = getattr(e, "winerror", None) == 231 or e.errno == errno.EINVAL
            missing = e.errno == errno.ENOENT
            waited = time.monotonic() - start
            if not (busy and waited < PIPE_BUSY_WAIT) and not (
                missing and waited < PIPE_MISSING_WAIT
            ):
                raise
        time.sleep(wait)
        wait = min(wait * 2, 0.25)


class Control:
    def __init__(self, endpoint: str, client_name: str, timeout: float = 30.0) -> None:
        self._lock = threading.Lock()
        self._next_id = 0
        self._reader = _wire.FrameReader()
        self._pending: list[dict[str, Any]] = []
        try:
            if sys.platform == "win32":
                # A named pipe opened as a file: blocking reads and writes.
                self._pipe: Any = _open_pipe(endpoint)
                self._sock: socket.socket | None = None
            else:
                self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self._sock.settimeout(timeout)
                self._sock.connect(endpoint)
                self._pipe = None
        except OSError as e:
            raise ControlError(
                f"the Kasanki Helium engine is not reachable at {endpoint} ({e}); "
                "start it with `helium engine run`"
            ) from e
        self._write(_wire.hello(client_name))
        welcome = self._receive()
        if welcome.get("type") != "welcome":
            self.close()
            raise ControlError(f"the engine refused the connection: {welcome}")
        # The timeout is for connecting. A model's answer can take minutes.
        if self._sock is not None:
            self._sock.settimeout(None)

    def call(self, method: str, params: dict[str, Any]) -> Any:
        """The result of `method`, or `ControlError` with the engine's reason."""
        with self._lock:
            self._next_id += 1
            request_id = str(self._next_id)
            self._write(
                _wire.frame(
                    _wire.dumps(
                        {
                            "type": "request",
                            "id": request_id,
                            "method": method,
                            "params": params,
                        }
                    )
                )
            )
            while True:
                message = self._receive()
                if (
                    message.get("type") == "response"
                    and message.get("id") == request_id
                ):
                    if message.get("error") is not None:
                        raise ControlError(str(message["error"]))
                    return message.get("result")
                if message.get("type") == "error":
                    raise ControlError(str(message.get("reason")))

    def blob(self, hash: str) -> bytes:
        """A stored blob's content, fetched in parts."""
        parts: list[bytes] = []
        part, count = 0, 1
        while part < count:
            got = self.call("blob.get", {"hash": hash, "part": part})
            parts.append(base64.b64decode(got["data"]))
            count = int(got["parts"])
            part += 1
        return b"".join(parts)

    def close(self) -> None:
        with self._lock:
            if self._pipe is not None:
                self._pipe.close()
            if self._sock is not None:
                self._sock.close()

    def _write(self, data: bytes) -> None:
        try:
            if self._pipe is not None:
                self._pipe.write(data)
            else:
                assert self._sock is not None
                self._sock.sendall(data)
        except OSError as e:
            raise ControlError(f"the connection to the engine broke ({e})") from e

    def _receive(self) -> dict[str, Any]:
        while not self._pending:
            try:
                if self._pipe is not None:
                    data = self._pipe.read(_wire.MAX_FRAME + 4)
                else:
                    assert self._sock is not None
                    data = self._sock.recv(256 * 1024)
            except OSError as e:
                raise ControlError(f"the connection to the engine broke ({e})") from e
            if not data:
                raise ControlError("the engine closed the connection")
            try:
                self._pending.extend(self._reader.feed(data))
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                raise ControlError(f"the engine sent something unreadable ({e})") from e
        return self._pending.pop(0)
