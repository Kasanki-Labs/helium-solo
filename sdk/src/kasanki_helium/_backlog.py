# Copyright © 2026 Kasanki Labs.
# SPDX-License-Identifier: AGPL-3.0-only
# Licensed under the GNU AGPL v3 with an attribution term: see LICENSE and CREDIT.md.

"""Backlog files: events the SDK hands to the backlog monitor instead of
sending them itself.

Each process writes its own file, ``<data_dir>/backlog/<uuid v7>.jsonl``, of
JSON lines: a header ``{"type": "backlog", "format": 0, …}``, events exactly
as the SDK would have sent them (``{"type": "event", "run_id", "seq",
"body"}``), each after the ``blob`` lines it needs that the file does not
hold yet, and ``{"type": "closed"}`` when the process
ends normally. While
the file is open the process holds a lock on ``<uuid v7>.lock`` beside it, so
the monitor knows its writer is alive. A new file starts after 64 MiB.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from ._values import uuid7, wall_time
from ._wire import Blob

#: An event line and the blobs it names.
Line = tuple[bytes, Sequence[Blob]]

#: A new file starts once the next lines would take a file past this.
MAX_FILE_BYTES = 64 * 1024 * 1024

FORMAT = 0

CLOSED = b'{"type":"closed"}\n'

_BINARY = getattr(os, "O_BINARY", 0)


def header(client: str) -> bytes:
    """A file's first line, in canonical JSON."""
    fields = {
        "type": "backlog",
        "format": FORMAT,
        "pid": os.getpid(),
        "client": client,
        "created": wall_time(time.time_ns()),
    }
    text = json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return text.encode("utf-8") + b"\n"


def _lock(fd: int) -> None:
    """Takes the lock the monitor tries (Rust's ``File::try_lock``)."""
    if sys.platform == "win32":
        import msvcrt

        # The first byte; the monitor asks for the whole file, so they clash.
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


def _remove(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


class BacklogFile:
    """This process's backlog file in `folder`, opened on the first write.
    Not thread-safe: the client writes from one thread at a time."""

    def __init__(self, folder: Path, client: str) -> None:
        self.folder = folder
        self._client = client
        self._fd: int | None = None
        self._lock_fd: int | None = None
        self._lock_path: Path | None = None
        self._size = 0
        #: The size of the header: a file that size holds no events.
        self._empty_size = 0
        self._last_name = ""
        #: Blobs the current file holds.
        self._blobs: set[str] = set()
        #: Written since the last fsync.
        self.dirty = False
        #: Every file this process started.
        self.paths: list[Path] = []

    def write(self, lines: Sequence[bytes | Line]) -> tuple[int, OSError | None]:
        """Appends whole lines (each ending in a newline). An event line that
        names blobs comes after the blob lines this file does not hold yet,
        in one piece with them. Returns how many lines were written, and the
        error that stopped the rest, if any."""
        written = 0
        try:
            while written < len(lines):
                if self._fd is None:
                    self._open()
                # As many lines as fit in this file; at least one in an
                # empty file, so a large line cannot stall.
                end, size = written, self._size
                pieces: list[bytes] = []
                added: set[str] = set()
                while end < len(lines):
                    piece, new = self._piece(lines[end], added)
                    if size + len(piece) > MAX_FILE_BYTES and not (
                        end == written and self._size == self._empty_size
                    ):
                        break
                    pieces.append(piece)
                    added |= new
                    size += len(piece)
                    end += 1
                if end == written:
                    self._finish()
                    continue
                self._append(b"".join(pieces))
                self._blobs |= added
                written = end
        except OSError as e:
            return written, e
        return written, None

    def _piece(self, line: bytes | Line, added: set[str]) -> tuple[bytes, set[str]]:
        """A line with the blob lines it needs before it, and their hashes."""
        if isinstance(line, bytes):
            return line, set()
        event, blobs = line
        new: set[str] = set()
        out: list[bytes] = []
        for blob in blobs:
            if blob.hash in self._blobs or blob.hash in added or blob.hash in new:
                continue
            new.add(blob.hash)
            out.extend(message + b"\n" for message in blob.messages())
        out.append(event)
        return b"".join(out), new

    def sync(self) -> None:
        """Makes what was written durable."""
        if self._fd is not None and self.dirty:
            os.fsync(self._fd)
        self.dirty = False

    def close(self) -> None:
        """Ends the current file with ``closed`` and lets it go."""
        if self._fd is not None:
            self._finish()

    def _open(self) -> None:
        # Only the backlog folder itself: a data folder that does not exist
        # is not this SDK's to create. Owner-only, inside the data folder.
        try:
            self.folder.mkdir(mode=0o700)
        except FileExistsError:
            pass
        # The monitor reads files in name order, so a run continued in a new
        # file must sort after the last one (UUIDs of one millisecond need not).
        name = uuid7()
        while name <= self._last_name:
            time.sleep(0.001)
            name = uuid7()
        self._last_name = name
        # The lock comes first: the monitor never sees a file whose writer
        # does not yet hold it, which would look like a writer that is gone.
        lock_path = self.folder / f"{name}.lock"
        lock_fd = os.open(
            lock_path, os.O_RDWR | os.O_CREAT | os.O_EXCL | _BINARY, 0o640
        )
        path = self.folder / f"{name}.jsonl"
        try:
            _lock(lock_fd)
            fd = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | _BINARY,
                0o640,
            )
        except OSError:
            os.close(lock_fd)
            _remove(lock_path)
            raise
        self._fd, self._lock_fd, self._lock_path = fd, lock_fd, lock_path
        self._size = 0
        self._blobs = set()
        self.paths.append(path)
        try:
            self._append(header(self._client))
        except OSError:
            self._let_go()
            _remove(path)
            raise
        self._empty_size = self._size

    def _append(self, data: bytes) -> None:
        assert self._fd is not None
        start = self._size
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(self._fd, view) :]
        except OSError:
            # Cut back to the last whole line, so the next write starts a
            # line of its own. If even that fails, leave the file: once its
            # lock is free, the monitor moves the half line aside.
            try:
                os.ftruncate(self._fd, start)
            except OSError:
                self._let_go()
            raise
        self._size += len(data)
        self.dirty = True

    def _finish(self) -> None:
        try:
            self._append(CLOSED)
            self.sync()
        finally:
            self._let_go()

    def _let_go(self) -> None:
        """Closes the file and releases its lock."""
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self.dirty = False
        if self._lock_fd is not None:
            try:
                _unlock(self._lock_fd)
            except OSError:
                pass
            os.close(self._lock_fd)
            self._lock_fd = None
        if self._lock_path is not None:
            # The monitor removes it too; whichever comes first.
            _remove(self._lock_path)
            self._lock_path = None
