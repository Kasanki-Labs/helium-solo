# Copyright © 2026 Kasanki Labs.
# SPDX-License-Identifier: AGPL-3.0-only
# Licensed under the GNU AGPL v3 with an attribution term: see LICENSE and CREDIT.md.

"""The connection to the engine, run on a background thread.

The agent never waits on it: recording an event only encodes it and appends
it to a queue. The background thread sends queued events, and keeps each one
until the engine acknowledges it as saved on disk.

When the engine cannot take events, runs spill to a backlog file for the
backlog monitor to deliver: once the engine has
been unreachable for 2 seconds; when more than 64 MiB or 100,000 events wait
in memory (the run with the oldest one first); and at exit, for whatever the
engine has not confirmed after 5 seconds. From its first spilled event on,
the rest of a run goes to the backlog too (M2). A second background thread
writes backlog files. If they cannot be written, events wait in memory up to
the same limits; beyond them they are dropped, and each dropped range is
recorded as a ``gap`` event once the backlog can be written again (M3).

Blobs an event names go before it, once per connection:
a blob is sent again only on a new connection, and only if no event naming
it has been acknowledged yet (the engine stores a blob before the event).
"""

from __future__ import annotations

import asyncio
import atexit
import errno
import logging
import sys
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import _wire
from ._backlog import BacklogFile, Line
from ._wire import Blob

log = logging.getLogger("kasanki_helium")

#: How long the SDK waits at exit for unsaved events.
EXIT_WAIT_SECONDS = 5.0

#: How long the SDK waits before sending again what the engine could not
#: save yet (``refused`` with ``retry: true``).
RETRY_SECONDS = 1.0

#: How long the engine may be unreachable before runs spill (M6).
UNREACHABLE_SECONDS = 2.0

#: Events waiting in memory, for all runs together, before runs spill (M6).
MEMORY_BYTES = 64 * 1024 * 1024
MEMORY_EVENTS = 100_000

#: Backlog lines are made durable at most this often (§4.2).
SYNC_SECONDS = 0.1

#: How long to wait before trying again when the backlog cannot be written.
BACKLOG_RETRY_SECONDS = 1.0

#: Blob hashes remembered as sent on a connection, or as stored by the
#: engine. Past this the oldest are forgotten, and sent again if needed.
KNOWN_BLOBS = 100_000

_HELLO_TIMEOUT = 5.0
_FIRST_RECONNECT_DELAY = 0.05
_MAX_RECONNECT_DELAY = 2.0


class _Known:
    """Blob hashes, the oldest forgotten past ``KNOWN_BLOBS``."""

    def __init__(self) -> None:
        self._hashes: OrderedDict[str, None] = OrderedDict()

    def __contains__(self, hash: object) -> bool:
        return hash in self._hashes

    def add(self, hash: str) -> None:
        self._hashes[hash] = None
        self._hashes.move_to_end(hash)
        if len(self._hashes) > KNOWN_BLOBS:
            self._hashes.popitem(last=False)


class Event:
    """An event waiting to be saved: its framed message and the blobs it
    names. A model call's is built when first needed, off the agent's
    thread: stripping secrets from hundreds of KiB, splitting and encoding
    them takes milliseconds."""

    __slots__ = ("_build", "_built", "_lock", "size")

    def __init__(
        self,
        frame: bytes = b"",
        blobs: Sequence[Blob] = (),
        *,
        build: Callable[[], tuple[bytes, Sequence[Blob]]] | None = None,
        size: int = 0,
    ) -> None:
        self._build = build
        self._built: tuple[bytes, tuple[Blob, ...]] | None = None
        if build is None:
            self._built = (frame, tuple(blobs))
            size = len(frame) + sum(len(b.content) for b in blobs)
        #: Bytes counted against the memory limits (an estimate until built).
        self.size = size
        self._lock = threading.Lock()

    def get(self) -> tuple[bytes, tuple[Blob, ...]]:
        """The framed message and its blobs, built on first use."""
        built = self._built
        if built is not None:
            return built
        with self._lock:
            if self._built is None:
                assert self._build is not None
                frame, blobs = self._build()
                self._built = (frame, tuple(blobs))
                self._build = None
            return self._built


@dataclass
class _Gap:
    """Steps `first` to `last` of a run, dropped because nothing could hold
    them; written to the backlog as one ``gap`` event at seq `first`."""

    first: int
    last: int
    #: The first dropped event's times, which the gap event takes.
    wall_time: str
    mono_time: str
    reason: str


@dataclass
class _RunQueue:
    """One run's events that are not yet saved."""

    #: Sent to the engine and not yet acknowledged: seq -> (when, event).
    frames: dict[int, tuple[float, Event]] = field(default_factory=dict)
    ended: bool = False
    #: Recording stopped for good after a final refusal.
    abandoned: bool = False
    retry_scheduled: bool = False
    #: The rest of the run goes through the backlog (M2).
    spilled: bool = False
    #: Spilled events and dropped ranges not yet written, in seq order.
    backlog: deque[tuple[int, Event] | _Gap] = field(default_factory=deque)


def _gap_reason(error: OSError | None) -> str:
    if error is not None and error.errno == errno.ENOSPC:
        return "disk full: events dropped"
    cause = error.strerror if error is not None and error.strerror else error
    return f"the backlog could not be written ({cause}): events dropped"


class _Protocol(asyncio.Protocol):
    def __init__(self, client: Client) -> None:
        self._client = client
        self._reader = _wire.FrameReader()
        self.welcome: asyncio.Future[dict[str, Any]] = (
            asyncio.get_running_loop().create_future()
        )
        self.transport: asyncio.WriteTransport | None = None
        self.lost = False
        #: Blobs sent on this connection.
        self.blobs_sent = _Known()

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        assert isinstance(transport, asyncio.WriteTransport)
        self.transport = transport

    def data_received(self, data: bytes) -> None:
        try:
            messages = self._reader.feed(data)
        except (ValueError, TypeError) as e:
            log.warning("Kasanki Helium: unreadable message from the engine: %s", e)
            if self.transport is not None:
                self.transport.close()
            return
        for message in messages:
            self._client._handle(message, self)

    def connection_lost(self, exc: Exception | None) -> None:
        self.lost = True
        if not self.welcome.done():
            self.welcome.set_exception(ConnectionError("closed during the handshake"))
        self._client._wake_up()


class Client:
    """Sends events to one engine. One per endpoint per process."""

    def __init__(self, endpoint: str, client_name: str, backlog_dir: Path) -> None:
        self.endpoint = endpoint
        self._client_name = client_name
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._runs: dict[str, _RunQueue] = {}
        #: (run_id, event) not yet written to the connection.
        self._outbox: list[tuple[str, Event]] = []
        #: Blobs named by acknowledged events: the engine has them. Used by
        #: the connection thread only.
        self._engine_has = _Known()
        self._signalled = False
        self._closed = False
        #: The engine has been unreachable long enough that runs spill.
        self._unreachable = False
        self._told_unreachable = False
        #: Bytes and events held in memory: unacknowledged and unwritten.
        self._held_bytes = 0
        self._held_events = 0
        #: Events taken out to be written to the backlog right now.
        self._writing = 0
        #: Why the backlog could not be written last time, if it could not.
        self._backlog_error: OSError | None = None
        self._dropping = False
        #: Events written to backlog files, and dropped steps recorded there.
        self.handed = 0
        self.lost = 0
        self.backlog = BacklogFile(backlog_dir, client_name)
        self._write_lock = threading.Lock()
        self._last_sync = 0.0
        self._writer_wake = threading.Event()
        self._writer: threading.Thread | None = None
        self._loop = asyncio.new_event_loop()
        self._wake = asyncio.Event()
        self._thread = threading.Thread(
            target=self._loop.run_until_complete,
            args=(self._main(),),
            name="kasanki-helium",
            daemon=True,
        )
        self._thread.start()

    # Called from agent threads. Kept short: the agent never waits on storage.

    def submit(
        self,
        run_id: str,
        seq: int,
        event: Event,
        times: tuple[str, str],
        *,
        ends_run: bool = False,
    ) -> None:
        """Queues one event. `times` is its (wall_time, mono_time), for a gap
        event standing in for it if it has to be dropped."""
        wake_sender = wake_writer = False
        with self._lock:
            queue = self._runs.setdefault(run_id, _RunQueue())
            if queue.abandoned:
                return
            queue.ended = queue.ended or ends_run
            if self._unreachable and not queue.spilled:
                self._spill_unreachable(run_id, queue)
            full = self._full()
            # A gap may not stand at seq 0 or 1, so those are always kept.
            if full and self._backlog_error is not None and seq > 1:
                if not queue.spilled:
                    self._spill(run_id, queue)
                self._drop(queue, seq, times)
                wake_writer = True
            elif queue.spilled:
                queue.backlog.append((seq, event))
                self._hold(event)
                wake_writer = True
            else:
                queue.frames[seq] = (time.monotonic(), event)
                self._hold(event)
                self._outbox.append((run_id, event))
                wake_writer = full
                if not self._signalled:
                    self._signalled = wake_sender = True
            if wake_writer:
                self._start_writer()
        if wake_sender:
            self._loop.call_soon_threadsafe(self._wake_up)
        if wake_writer:
            self._writer_wake.set()

    def flush(self, timeout: float) -> bool:
        """Waits until every event is acknowledged or written to the
        backlog, or `timeout` seconds."""
        with self._changed:
            return self._changed.wait_for(
                lambda: (
                    not self._writing
                    and not any(q.frames or q.backlog for q in self._runs.values())
                ),
                timeout,
            )

    def unsaved(self) -> dict[str, list[int]]:
        """Seqs neither acknowledged nor written to the backlog, per run."""
        with self._lock:
            found: dict[str, list[int]] = {}
            for run_id, queue in self._runs.items():
                seqs = list(queue.frames)
                for item in queue.backlog:
                    if isinstance(item, _Gap):
                        seqs.extend(range(item.first, item.last + 1))
                    else:
                        seqs.append(item[0])
                if seqs:
                    found[run_id] = sorted(seqs)
            return found

    def hand_over(self) -> None:
        """At exit: every event not yet acknowledged goes to the backlog
        (M6), and the backlog file is closed."""
        with self._lock:
            for run_id, queue in self._runs.items():
                if queue.frames:
                    self._spill(run_id, queue)
        self._write_backlog()
        with self._write_lock:
            try:
                self.backlog.close()
            except OSError as e:
                log.warning("Kasanki Helium: cannot close the backlog file: %s", e)

    def close(self) -> None:
        """Stops the background thread (events not yet saved are dropped)."""
        self._closed = True
        self._loop.call_soon_threadsafe(self._wake_up)
        self._thread.join(timeout=5)

    # Memory and spilling; called with self._lock held.

    def _hold(self, event: Event) -> None:
        # A blob named by several events counts once per event: an
        # overestimate, so memory never holds more than the limit.
        self._held_bytes += event.size
        self._held_events += 1

    def _release(self, event: Event) -> None:
        self._held_bytes -= event.size
        self._held_events -= 1

    def _full(self) -> bool:
        return self._held_bytes > MEMORY_BYTES or self._held_events > MEMORY_EVENTS

    def _spill(self, run_id: str, queue: _RunQueue) -> None:
        """From now on the run goes through the backlog, starting with its
        first unacknowledged event. It is no longer sent here (M2)."""
        queue.spilled = True
        if queue.frames:
            queue.backlog.extend((s, e) for s, (_, e) in sorted(queue.frames.items()))
            queue.frames.clear()
            self._outbox = [item for item in self._outbox if item[0] != run_id]

    def _spill_unreachable(self, run_id: str, queue: _RunQueue) -> None:
        if not self._told_unreachable:
            self._told_unreachable = True
            log.warning(
                "Kasanki Helium: the engine has not been reachable for %g seconds. "
                "Events go to a backlog file in %s, for the backlog monitor to deliver.",
                UNREACHABLE_SECONDS,
                self.backlog.folder,
            )
        self._spill(run_id, queue)
        self._start_writer()

    def _spill_all(self) -> None:
        self._unreachable = True
        for run_id, queue in self._runs.items():
            if queue.frames:
                self._spill_unreachable(run_id, queue)

    def _spill_oldest(self) -> None:
        """Spills the run holding the oldest unacknowledged event."""
        oldest = min(
            (
                (next(iter(q.frames.values()))[0], run_id)
                for run_id, q in self._runs.items()
                if q.frames
            ),
            default=None,
        )
        if oldest is None:
            return
        if not any(q.spilled for q in self._runs.values()):
            log.warning(
                "Kasanki Helium: more than %d MiB or %d events wait for the engine. "
                "Runs go to a backlog file in %s, for the backlog monitor to deliver.",
                MEMORY_BYTES // (1024 * 1024),
                MEMORY_EVENTS,
                self.backlog.folder,
            )
        self._spill(oldest[1], self._runs[oldest[1]])

    def _drop(self, queue: _RunQueue, seq: int, times: tuple[str, str]) -> None:
        last = queue.backlog[-1] if queue.backlog else None
        if isinstance(last, _Gap) and last.last == seq - 1:
            last.last = seq
        else:
            reason = _gap_reason(self._backlog_error)
            queue.backlog.append(_Gap(seq, seq, times[0], times[1], reason))
        if not self._dropping:
            self._dropping = True
            log.warning(
                "Kasanki Helium: the backlog cannot be written and memory is full; "
                "events are dropped, and each run records what it lost as a gap "
                "event once the backlog can be written again."
            )

    def _start_writer(self) -> None:
        if self._writer is not None:
            return
        try:
            thread = threading.Thread(
                target=self._write_loop, name="kasanki-helium-backlog", daemon=True
            )
            thread.start()
        except RuntimeError:
            # The interpreter is shutting down; the exit hand-over writes.
            return
        self._writer = thread

    # The backlog thread.

    def _write_loop(self) -> None:
        while True:
            with self._lock:
                if self._backlog_error is not None:
                    wait: float | None = BACKLOG_RETRY_SECONDS
                else:
                    wait = SYNC_SECONDS if self.backlog.dirty else None
            self._writer_wake.wait(wait)
            self._writer_wake.clear()
            try:
                self._write_backlog()
            except Exception:
                log.exception("Kasanki Helium: unexpected error writing the backlog")

    def _write_backlog(self) -> None:
        """Writes everything waiting for the backlog, spilling runs while
        memory is full, then makes it durable if it is time."""
        with self._write_lock:
            while True:
                with self._lock:
                    if self._full():
                        self._spill_oldest()
                    batch = [
                        (run_id, queue.backlog.popleft())
                        for run_id, queue in self._runs.items()
                        for _ in range(len(queue.backlog))
                    ]
                    self._writing += len(batch)
                if not batch:
                    break
                lines = [self._line(run_id, item) for run_id, item in batch]
                written, error = self.backlog.write(lines)
                with self._changed:
                    self._settle(batch, written, error)
                if error is not None:
                    break
                self._sync_if_due()
            self._sync_if_due()

    def _sync_if_due(self) -> None:
        """fsync at most every SYNC_SECONDS; the backlog thread comes back
        within that time while anything is unsynced."""
        now = time.monotonic()
        if not self.backlog.dirty or now - self._last_sync < SYNC_SECONDS:
            return
        self._last_sync = now
        try:
            self.backlog.sync()
        except OSError as e:
            log.warning("Kasanki Helium: cannot make the backlog durable: %s", e)

    @staticmethod
    def _line(run_id: str, item: tuple[int, Event] | _Gap) -> bytes | Line:
        if isinstance(item, _Gap):
            body = {
                "format_version": 0,
                "kind": "gap",
                "wall_time": item.wall_time,
                "mono_time": item.mono_time,
                "links": [],
                "payload": {
                    "from_seq": str(item.first),
                    "to_seq": str(item.last),
                    "reason": item.reason,
                    "source": "sdk",
                },
                "blobs": [],
            }
            message = _wire.event_message(run_id, item.first, _wire.dumps(body))
            return message + b"\n"
        # The message as it would have been sent, without its frame header.
        frame, blobs = item[1].get()
        return (frame[4:] + b"\n", blobs) if blobs else frame[4:] + b"\n"

    def _settle(
        self,
        batch: list[tuple[str, tuple[int, Event] | _Gap]],
        written: int,
        error: OSError | None,
    ) -> None:
        """Called with self._lock held, after writing `batch`."""
        self._writing -= len(batch)
        for _, item in batch[:written]:
            if isinstance(item, _Gap):
                self.lost += item.last - item.first + 1
            else:
                self._release(item[1])
                self.handed += 1
        # What was not written goes back in front of anything newer.
        for run_id, item in reversed(batch[written:]):
            self._runs[run_id].backlog.appendleft(item)
        if error is None:
            if self._backlog_error is not None:
                log.warning(
                    "Kasanki Helium: the backlog in %s can be written again.",
                    self.backlog.folder,
                )
            self._backlog_error = None
            self._dropping = False
        else:
            if self._backlog_error is None:
                log.warning(
                    "Kasanki Helium: cannot write the backlog in %s (%s). Events wait "
                    "in memory and are written when it can be.",
                    self.backlog.folder,
                    error,
                )
            self._backlog_error = error
        for run_id in {run_id for run_id, _ in batch[:written]}:
            queue = self._runs.get(run_id)
            if queue is not None and queue.ended and not queue.backlog:
                del self._runs[run_id]
        self._changed.notify_all()

    # The connection thread.

    def _wake_up(self) -> None:
        self._wake.set()

    async def _connect(self) -> _Protocol:
        loop = asyncio.get_running_loop()
        transport: asyncio.BaseTransport
        protocol: _Protocol
        if sys.platform == "win32":
            proactor = loop
            assert isinstance(proactor, asyncio.ProactorEventLoop)
            # typeshed types this for stream protocols only.
            transport, protocol = await proactor.create_pipe_connection(  # type: ignore[assignment]
                lambda: _Protocol(self),  # type: ignore[arg-type,return-value]
                self.endpoint,
            )
        else:
            transport, protocol = await loop.create_unix_connection(
                lambda: _Protocol(self), self.endpoint
            )
        # On Windows, connection_made may not have run yet.
        assert isinstance(transport, asyncio.WriteTransport)
        protocol.transport = transport
        transport.write(_wire.hello(self._client_name))
        try:
            await asyncio.wait_for(asyncio.shield(protocol.welcome), _HELLO_TIMEOUT)
        except BaseException:
            transport.close()
            raise
        return protocol

    async def _main(self) -> None:
        delay = _FIRST_RECONNECT_DELAY
        unreachable = False
        #: When the engine was last found unreachable, if it still is.
        down_since: float | None = None
        while not self._closed:
            try:
                protocol = await self._connect()
            except Exception as e:
                # Whatever goes wrong, keep trying: this thread is the only
                # way events reach the engine.
                if not isinstance(e, (OSError, asyncio.TimeoutError, ConnectionError)):
                    log.exception(
                        "Kasanki Helium: unexpected error connecting to the engine"
                    )
                if not unreachable:
                    log.warning(
                        "Kasanki Helium: the engine is not reachable at %s (%s). "
                        "Events wait in memory, and go to a backlog file if it "
                        "stays unreachable for %g seconds.",
                        self.endpoint,
                        e,
                        UNREACHABLE_SECONDS,
                    )
                    unreachable = True
                now = time.monotonic()
                if down_since is None:
                    down_since = now
                wait = delay
                if now - down_since >= UNREACHABLE_SECONDS:
                    with self._lock:
                        self._spill_all()
                    self._writer_wake.set()
                else:
                    # Wake up in time to spill.
                    wait = min(delay, down_since + UNREACHABLE_SECONDS - now)
                await asyncio.sleep(wait)
                delay = min(delay * 2, _MAX_RECONNECT_DELAY)
                continue
            if unreachable:
                log.warning(
                    "Kasanki Helium: connected to the engine at %s.", self.endpoint
                )
                unreachable = False
            delay = _FIRST_RECONNECT_DELAY
            down_since = None
            with self._lock:
                self._unreachable = self._told_unreachable = False
                # Everything not yet acknowledged, in order: the engine saves
                # each event once and acknowledges repeats.
                self._outbox = [
                    (run_id, e)
                    for run_id, q in self._runs.items()
                    if not q.abandoned
                    for _, e in q.frames.values()
                ]
            try:
                await self._pump(protocol)
            except Exception:
                log.exception("Kasanki Helium: unexpected error sending to the engine")
                if protocol.transport is not None:
                    protocol.transport.close()
                await asyncio.sleep(delay)
            down_since = time.monotonic()

    async def _pump(self, protocol: _Protocol) -> None:
        """Writes queued events until the connection is lost."""
        assert protocol.transport is not None
        self._wake.set()
        while True:
            await self._wake.wait()
            self._wake.clear()
            if protocol.lost or self._closed:
                if self._closed:
                    protocol.transport.close()
                return
            with self._lock:
                frames, self._outbox = self._outbox, []
                self._signalled = False
            if frames:
                protocol.transport.write(b"".join(self._with_blobs(frames, protocol)))

    def _with_blobs(
        self, events: list[tuple[str, Event]], protocol: _Protocol
    ) -> list[bytes]:
        """The events' frames, each after the blobs it names that the engine
        may not have yet. Events not built yet are built here."""
        out: list[bytes] = []
        for _, event in events:
            frame, blobs = event.get()
            for blob in blobs:
                if blob.hash in protocol.blobs_sent or blob.hash in self._engine_has:
                    continue
                out.extend(_wire.frame(m) for m in blob.messages())
                protocol.blobs_sent.add(blob.hash)
            out.append(frame)
        return out

    def _handle(self, message: dict[str, Any], protocol: _Protocol) -> None:
        kind = message.get("type")
        if kind == "welcome":
            if not protocol.welcome.done():
                protocol.welcome.set_result(message)
            return
        if kind == "error":
            log.warning(
                "Kasanki Helium: the engine closed the connection: %s",
                message.get("reason"),
            )
            return
        run_id = str(message.get("run_id"))
        try:
            seq = int(message["seq"])
        except (KeyError, TypeError, ValueError):
            return
        reason = message.get("reason")
        with self._changed:
            queue = self._runs.get(run_id)
            if queue is None:
                return
            if kind == "replaced":
                log.warning(
                    "Kasanki Helium: event %d of run %s broke a recording rule and was "
                    "saved as an error event instead: %s",
                    seq,
                    run_id,
                    reason,
                )
            elif queue.spilled:
                # The monitor delivers the run now; answers to what was sent
                # here before it spilled no longer matter.
                return
            elif kind == "ack":
                for s in [s for s in queue.frames if s <= seq]:
                    _, event = queue.frames.pop(s)
                    self._release(event)
                    for blob in event.get()[1]:
                        self._engine_has.add(blob.hash)
            elif kind == "refused" and message.get("retry") is True:
                if not queue.retry_scheduled:
                    queue.retry_scheduled = True
                    self._loop.call_later(RETRY_SECONDS, self._send_again, run_id)
            elif kind == "refused":
                if not queue.abandoned:
                    log.warning(
                        "Kasanki Helium: recording of run %s stopped: event %d was refused: "
                        "%s. The agent keeps running; later events of this run are not "
                        "recorded.",
                        run_id,
                        seq,
                        reason,
                    )
                queue.abandoned = True
                for _, e in queue.frames.values():
                    self._release(e)
                queue.frames.clear()
            if queue.ended and not queue.spilled and not queue.frames:
                del self._runs[run_id]
            self._changed.notify_all()

    def _send_again(self, run_id: str) -> None:
        with self._lock:
            queue = self._runs.get(run_id)
            if queue is None:
                return
            queue.retry_scheduled = False
            self._outbox.extend((run_id, e) for _, e in queue.frames.values())
        self._wake_up()


_clients: dict[str, Client] = {}
_clients_lock = threading.Lock()


def client_for(endpoint: str, client_name: str, data_dir: Path) -> Client:
    with _clients_lock:
        client = _clients.get(endpoint)
        if client is None:
            if not _clients:
                atexit.register(_at_exit)
            client = _clients[endpoint] = Client(
                endpoint, client_name, data_dir / "backlog"
            )
        return client


def flush(timeout: float) -> bool:
    """Waits until every recorded event is saved or handed to the backlog,
    or `timeout` seconds."""
    with _clients_lock:
        clients = list(_clients.values())
    deadline = time.monotonic() + timeout
    return all(c.flush(max(0.0, deadline - time.monotonic())) for c in clients)


def ranges(seqs: list[int]) -> str:
    """``[0, 1, 2, 5]`` -> ``"0-2, 5"``."""
    spans: list[list[int]] = []
    for s in seqs:
        if spans and s == spans[-1][1] + 1:
            spans[-1][1] = s
        else:
            spans.append([s, s])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in spans)


def _at_exit() -> None:
    flush(EXIT_WAIT_SECONDS)
    with _clients_lock:
        clients = list(_clients.values())
    # Whatever the engine has not confirmed goes to the backlog instead of
    # being reported as not saved (M6).
    for client in clients:
        client.hand_over()
    out: list[str] = []
    handed = sum(c.handed for c in clients)
    if handed:
        folders = ", ".join(str(c.backlog.folder) for c in clients if c.handed)
        out.append(
            f"kasanki-helium: {handed} recorded event(s) were handed to the backlog "
            f"monitor in {folders} (the engine did not confirm them in time)."
        )
    lost = sum(c.lost for c in clients)
    if lost:
        out.append(
            f"kasanki-helium: {lost} recorded event(s) were lost while nothing could "
            "hold them; their runs record each lost range as a gap event."
        )
    lines = []
    total = 0
    for client in clients:
        for run_id, seqs in client.unsaved().items():
            total += len(seqs)
            lines.append(f"  run {run_id}: seq {ranges(seqs)} ({client.endpoint})")
    if total:
        out.append(
            f"kasanki-helium: {total} recorded event(s) were not saved "
            "(the engine did not confirm them in time, and the backlog could not "
            "be written):"
        )
        out.extend(lines)
    if out:
        print(*out, sep="\n", file=sys.stderr)
