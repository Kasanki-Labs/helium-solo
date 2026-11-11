# Copyright © 2026 Kasanki Labs.
# SPDX-License-Identifier: AGPL-3.0-only
# Licensed under the GNU AGPL v3 with an attribution term: see LICENSE and CREDIT.md.

"""Recording one run: the ``with helium.run(...)`` block and its helpers.

Every helper returns the real value to the agent and records it, so replay
can serve the same value later (trace-format spec §4). Payload shapes are the
D21 rules in the spec, §5.5.
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import json
import logging
import os
import platform
import random as _random
import re
import subprocess
import sys
import threading
import time
import uuid
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, TypeVar

from . import _client, _endpoint, _llm, _project, _session, _wire
from ._secrets import redact
from ._values import in_order, key_order, to_json, uuid7, wall_time, wall_time_nanos
from ._warnings import HeliumWarning, first_time
from ._wire import Blob

if TYPE_CHECKING:
    # Only for type checkers: the SDK has no dependencies at run time.
    from typing_extensions import Self

log = logging.getLogger("kasanki_helium")

T = TypeVar("T")

VERSION = "0.1.0"
CLIENT_NAME = f"kasanki-helium/{VERSION}"

#: The recorder settings this SDK actually applies, recorded in every run's
#: ``profile`` event with the run's ``streaming`` setting (D43). Only what is
#: built is listed, so the record never claims protection that is not there.
PROFILE: dict[str, Any] = {"template": "standard", "checkpoint_interval_s": 10}

_CUSTOM_KIND = re.compile(r"custom\.[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*")

_LOCKFILES = ("uv.lock", "poetry.lock", "pdm.lock", "Pipfile.lock", "requirements.txt")


def _warn_inexact(inexact: set[str]) -> None:
    new = first_time(inexact)
    if new:
        warnings.warn(
            "Kasanki Helium recorded these values as text, so replay may not be "
            f"exact: {', '.join(new)}",
            HeliumWarning,
            stacklevel=5,
        )


def _git_commit(folder: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=folder,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = out.stdout.strip()
    if out.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
        return commit
    return None


def _lockfile(folder: str) -> dict[str, str] | None:
    for name in _LOCKFILES:
        path = Path(folder) / name
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
        return {"path": name, "hash": f"sha256:{digest}"}
    return None


def _run_context(folder: str) -> dict[str, Any]:
    """What a run records of where it ran. The project is read each time, since a
    board can give the folder its identity while this process is running."""
    project = _project.find(folder)
    return {
        **_context(folder),
        # Which project this is, whatever folder it's in today.
        "project": None
        if project is None
        else {"id": project.id, "name": project.name},
    }


@functools.lru_cache(maxsize=8)
def _context(folder: str) -> dict[str, Any]:
    """The run context (spec §4.1), worked out once per working folder."""
    return {
        "sdk": CLIENT_NAME,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "code_version": _git_commit(folder),
        "lockfile": _lockfile(folder),
        # What `helium replay` starts again (replay-harness R13).
        "interpreter": sys.executable,
        "cwd": folder,
    }


_TOOL_CLASSES = (None, "pure", "read_only", "stateful")
#: helium.memory's tools: what they answer is the memory, not outside text.
_MEMORY_TOOLS = frozenset({"remember", "recall", "forget"})


def _error(e: BaseException) -> dict[str, str]:
    return {"type": type(e).__qualname__, "message": str(e)}


_event_time_lock = threading.Lock()
_last_event_time = 0


def _event_time() -> int:
    """An event's wall time: never twice the same in a process. A clock can tick
    coarsely (16 ms on Windows), and order across agents' runs comes from these."""
    global _last_event_time
    with _event_time_lock:
        _last_event_time = max(time.time_ns(), _last_event_time + 1)
        return _last_event_time


#: Marks the header of a handoff between processes.
_HANDOFF_HEADER = re.compile(r"helium-01-([0-9a-f-]{36})-(\d+)")


@dataclass(frozen=True)
class Handoff:
    """Where one agent handed work to another: a run and its step. Within a process it travels by itself;
    between processes, send `header` with the work, as you would a
    ``traceparent``, and give it to the other side::

        handoff = run.send("refunds", ticket)
        requests.post(url, headers={"x-helium-handoff": handoff.header})
        ...
        with helium.run("refunds", input=ticket, caused_by=header_value) as run:
    """

    run_id: str
    seq: int

    @property
    def header(self) -> str:
        return f"helium-01-{self.run_id}-{self.seq}"

    @classmethod
    def of(cls, value: Handoff | str) -> Handoff:
        if isinstance(value, Handoff):
            return value
        match = _HANDOFF_HEADER.fullmatch(value)
        if match is None:
            raise ValueError(f"not a Helium handoff header: {value!r}")
        return cls(match.group(1), int(match.group(2)))

    def link(self, rel: str) -> dict[str, str]:
        return {"rel": rel, "run_id": self.run_id, "seq": str(self.seq)}


#: The run open in this context, which a run started inside it is a child of.
_open_run: contextvars.ContextVar[Run | None] = contextvars.ContextVar(
    "kasanki_helium_open_run", default=None
)
#: The last handoff in this context, which the next run started picks up.
_last_handoff: contextvars.ContextVar[Handoff | None] = contextvars.ContextVar(
    "kasanki_helium_handoff", default=None
)
#: How many runs have started in this process. The auto-recorder reads it to
#: tell a launch that recorded nothing from one that recorded something (W7).
_opened_runs = 0


class Run:
    """One recorded run. Use it through ``with helium.run(...) as run:``."""

    def __init__(
        self,
        name: str,
        input: Any = None,
        version: str | None = None,
        streaming: str = _llm.STREAMING[0],
        caused_by: Handoff | str | None = None,
        entry: str | None = None,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("a run needs a non-empty name")
        if streaming not in _llm.STREAMING:
            raise ValueError(
                f"streaming must be one of {', '.join(_llm.STREAMING)}, got {streaming!r}"
            )
        self._streaming = streaming
        #: Streamed model calls not yet recorded (W7): recorded at the run's
        #: end if nobody finished reading them.
        self._open_calls: set[Any] = set()
        self._calls_lock = threading.Lock()
        self.run_id = uuid7()
        self.name = name
        self._input = input
        #: What this run's tools answered, for labelling memory facts taken
        #: from them.
        self._answers: list[str] = []
        self._version = version
        self._lock = threading.Lock()
        self._seq = 0
        self._mono_start = 0
        self._started = False
        self._ended = False
        self._output: Any = None
        self._random = _random.Random(int.from_bytes(os.urandom(32), "big"))
        self._client: _client.Client | None = None
        #: The replay session this run replays, if the replay harness
        #: started it.
        self._replay: _session.Session | None = None
        self._caused_by = None if caused_by is None else Handoff.of(caused_by)
        self._entry = entry
        self._open_token: contextvars.Token[Run | None] | None = None

    # The run itself.

    def __enter__(self) -> Self:
        if self._started:
            raise RuntimeError("a run can only be started once")
        self._started = True
        global _opened_runs
        _opened_runs += 1
        parent = _open_run.get()
        session = self._replay = _session.claim(
            self.name,
            self._input,
            None if parent is None else parent._replay,
            None if self._caused_by is None else self._caused_by.run_id,
        )
        links: list[dict[str, str]] = self._multiplayer_links()
        self._open_token = _open_run.set(self)
        if session is not None:
            if not session.records:
                # Replayed: nothing is recorded; the engine serves and checks.
                self.run_id = session.run_id
                self._mono_start = time.monotonic_ns()
                return self
            # A fork or a rehearsal is recorded as a new run, a rollout (§6).
            self.run_id = session.rollout_run_id or self.run_id
            if session.branched:
                branched = {"rel": "branched_from", "run_id": session.run_id}
                if session.fork_at is not None:
                    # A rehearsal branches where it first differs, not at a step.
                    branched["seq"] = str(session.fork_at)
                links = [branched, *links]
        self._connect()
        self._mono_start = time.monotonic_ns()
        folder = os.getcwd()
        entry = (
            _session.agent_entry(folder, self.name)
            or self._entry
            or _session.caller_entry(sys._getframe(1))
        )
        context = {**_run_context(folder), "entry": entry}
        if not _llm.at_surface(self):
            # Model calls are recorded at the HTTP layer.
            context["model_calls"] = _llm.MODEL_CALLS
        start = {
            "agent": {"name": self.name, "version": self._version},
            "input": self._input,
            "context": context,
        }
        if session is not None and session.fresh:
            # A test case started from its input is no rollout, so it has no
            # `policy` event; its run says what it is here.
            start["context"]["case"] = {"from": "input", "policy": session.policy}
        if session is not None and session.rehearsal_of is not None:
            # Which request it rehearses, so it's never taken for one.
            start["context"]["rehearsal"] = session.rehearsal_of
        order = key_order(to_json(self._input, set()))
        if order is not None:
            start["input_order"] = order
        self._record("run.start", start, links=links)
        self._record("profile", {**PROFILE, "streaming": self._streaming})
        if session is not None and session.branched:
            self._record(
                "policy",
                {
                    "policy": session.policy,
                    "change": None
                    if session.change is None
                    else session.change.to_json(),
                },
            )
        return self

    def _multiplayer_links(self) -> list[dict[str, str]]:
        """How this run joins other agents' runs: the run it was
        started inside, and the handoff that caused it."""
        caused = self._caused_by
        if caused is None:
            caused = _last_handoff.get()
            if caused is not None:
                # One handoff starts one run.
                _last_handoff.set(None)
        links = []
        parent = _open_run.get()
        if parent is not None:
            links.append({"rel": "parent", "run_id": parent.run_id})
        elif caused is not None:
            links.append({"rel": "parent", "run_id": caused.run_id})
        if caused is not None:
            links.append(caused.link("caused_by"))
        return links

    def _connect(self) -> None:
        try:
            data = _endpoint.data_dir()
            self._client = _client.client_for(
                _endpoint.endpoint(data), CLIENT_NAME, data
            )
        except Exception:
            log.exception(
                "Kasanki Helium: cannot start recording; the agent keeps running"
            )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        with self._calls_lock:
            unfinished = list(self._open_calls)
        for call in unfinished:
            call.abandon()
        if self._open_token is not None:
            try:
                _open_run.reset(self._open_token)
            except ValueError:
                # Ended in another context than it started in: nothing to undo here.
                pass
        error = None if exc is None else _error(exc)
        if self._replay is None or self._replay.records:
            self._record(
                "run.end",
                {
                    "status": "ok" if exc is None else "error",
                    "output": self._output,
                    "error": error,
                },
                ends_run=True,
            )
        else:
            self._ended = True
        if self._replay is not None:
            # The engine compares the end with the recording, and stores the
            # replay record.
            self._replay.end(_session._plain(self._output), error)
        # Returning None lets an exception from the block carry on.

    def set_output(self, value: Any) -> None:
        """The run's final output, recorded in ``run.end``."""
        self._output = value

    @property
    def input(self) -> Any:
        """The run's input, as given to ``helium.run``."""
        return self._input

    # Helpers: each returns the real value and records it.

    def _served(self, kind: str, request: dict[str, Any]) -> Any:
        """In a replay: the recorded value, or `None` after a fork point
        (the helper then does it for real)."""
        return self._served_at(kind, request)[0]

    def _served_at(self, kind: str, request: dict[str, Any]) -> tuple[Any, int]:
        """`_served`, and the step the value was recorded at."""
        if self._replay is None:
            return None, 0
        answer = self._replay.ask(kind, request)
        if answer["answer"] == "serve":
            return answer["value"], int(answer["seq"])
        return None, 0

    def now(self) -> datetime:
        """The current UTC time (``clock.read``)."""
        served = self._served("clock.read", {"clock": "wall"})
        nanos = time.time_ns() if served is None else wall_time_nanos(served["value"])
        self._record("clock.read", {"clock": "wall", "value": wall_time(nanos)})
        seconds, rest = divmod(nanos, 1_000_000_000)
        return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(
            microsecond=rest // 1000
        )

    def monotonic_ns(self) -> int:
        """A monotonic clock in nanoseconds (``clock.read``)."""
        served = self._served("clock.read", {"clock": "monotonic"})
        value = time.monotonic_ns() if served is None else int(served["value"])
        self._record("clock.read", {"clock": "monotonic", "value": str(value)})
        return value

    def random(self) -> float:
        """A float in [0, 1) (``random.draw``)."""
        served = self._served("random.draw", {"method": "random", "args": []})
        value = self._random.random() if served is None else float(served["value"])
        self._record("random.draw", {"method": "random", "args": [], "value": value})
        return value

    def randint(self, a: int, b: int) -> int:
        """A whole number from `a` to `b`, both included (``random.draw``)."""
        served = self._served("random.draw", {"method": "randint", "args": [a, b]})
        value = self._random.randint(a, b) if served is None else int(served["value"])
        self._record(
            "random.draw", {"method": "randint", "args": [a, b], "value": value}
        )
        return value

    def choice(self, items: Sequence[T]) -> T:
        """One of `items` (``random.draw``). The recorded value is the chosen
        position, so replay is exact whatever the items are."""
        served = self._served("random.draw", {"method": "choice", "args": [len(items)]})
        index = (
            self._random.randrange(len(items))
            if served is None
            else int(served["value"])
        )
        self._record(
            "random.draw", {"method": "choice", "args": [len(items)], "value": index}
        )
        return items[index]

    def uuid4(self) -> uuid.UUID:
        """A random UUID (``id.generate``)."""
        served = self._served("id.generate", {"method": "uuid4"})
        value = uuid.uuid4() if served is None else uuid.UUID(served["value"])
        self._record("id.generate", {"method": "uuid4", "value": str(value)})
        return value

    def call_tool(
        self,
        name: str,
        fn: Callable[..., T],
        /,
        *args: Any,
        tool_class: str | None = None,
        **kwargs: Any,
    ) -> T:
        """Calls ``fn(*args, **kwargs)`` and records the call (``tool.call``).
        An exception is recorded and then raised as usual.

        `tool_class` says what the tool does, for forks (replay-harness R7):
        ``"pure"`` (no effects; runs again after a fork), ``"read_only"`` or
        ``"stateful"``. Unclassified tools are served from the recording
        after a fork, or blocked."""
        if tool_class not in _TOOL_CLASSES:
            raise ValueError(
                f"tool_class must be pure, read_only or stateful, got {tool_class!r}"
            )
        started = time.time_ns()
        begin = time.monotonic_ns()
        try:
            served, seq = self._served_at(
                "tool.call",
                {
                    **_session._plain(
                        {"name": name, "input": {"args": list(args), "kwargs": kwargs}}
                    ),
                    "tool_class": tool_class,
                },
            )
        except _session.ReplayBlocked as e:
            self._tool_call(
                name, args, kwargs, None, _error(e), started, begin, tool_class
            )
            raise
        if served is not None:
            if (
                self._replay is not None
                and self._replay.rehearsal
                and self._replay.sandbox is not None
            ):
                # A rehearsal: the tool runs in the sandbox too, and from the
                # first result that differs, the agent gets the sandbox's.
                own = _session.hooks().rehearse(self._replay, name, args, kwargs, seq)
                if own is not None:
                    result, raised = own
                    self._tool_call(
                        name,
                        args,
                        kwargs,
                        None if raised is not None else result,
                        None if raised is None else _error(raised),
                        started,
                        begin,
                        tool_class,
                    )
                    if raised is not None:
                        raise raised
                    return result  # type: ignore[no-any-return]
            elif self._replay is not None and self._replay.sandbox is not None:
                _session.hooks().advance(self._replay, name, args, kwargs, served, seq)
            error = served.get("error")
            output: T = in_order(served.get("output"), served.get("key_order"))
            self._tool_call(
                name,
                args,
                kwargs,
                output,
                error,
                started,
                begin,
                tool_class,
            )
            if error is not None:
                raise _session.rebuilt_error(error)
            return output
        if self._replay is not None and self._replay.sandbox is not None:
            # A helium test case: the sandbox's tool of that name runs instead.
            fn = _session.hooks().tool(self._replay.sandbox, name)
        try:
            result = fn(*args, **kwargs)
        except BaseException as e:
            self._tool_call(
                name, args, kwargs, None, _error(e), started, begin, tool_class
            )
            raise
        self._tool_call(name, args, kwargs, result, None, started, begin, tool_class)
        return result

    def wrap(self, client: T) -> T:
        """A model client that records its calls into this run as
        ``llm.call`` events, and otherwise behaves like `client`::

            llm = run.wrap(client)
            reply = llm.chat.completions.create(model=..., messages=...)

        Wraps OpenAI clients (``OpenAI``, ``AsyncOpenAI`` and those built on
        them) and Anthropic clients (every client of the ``anthropic``
        package), and returns the package's own client. Its calls are recorded
        as they cross the wire, streamed or not, keeping
        the client's proxies, timeouts and limits; a call that is not a model
        call is recorded as a plain HTTP exchange, with a warning once.

        Given a LangChain model (``ChatOpenAI``, ``ChatAnthropic``,
        ``OpenAIEmbeddings``, or one of these with ``bind_tools``), returns a
        copy whose clients are wrapped; the original is left as it was. Build
        the agent with the copy inside the run::

            agent = create_agent(run.wrap(model), tools)"""
        wrapped: T = _llm.wrap(self, client)
        return wrapped

    def http_client(self, **kwargs: Any) -> Any:
        """An HTTP client that records the model calls made through it into
        this run, to build a client with yourself::

            client = OpenAI(http_client=run.http_client())
            client = Anthropic(http_client=run.http_client(proxy=proxy_url))

        It is a client of the HTTP library the provider packages are built on
        (``httpx2``, else ``httpx``), with `kwargs` passed to it and the
        packages' own defaults otherwise. It relies on nothing private in
        either package. Make it inside the run, like a wrapped client."""
        from . import _http

        return _http.http_client(self, False, kwargs)

    def async_http_client(self, **kwargs: Any) -> Any:
        """`http_client` for an async client: ``AsyncOpenAI(http_client=...)``."""
        from . import _http

        return _http.http_client(self, True, kwargs)

    def event(self, kind: str, payload: Mapping[str, Any]) -> None:
        """Records an event of your own. `kind` is ``custom.<name>``, where a
        name is lowercase dot-separated parts."""
        if not isinstance(kind, str) or not _CUSTOM_KIND.fullmatch(kind):
            raise ValueError(
                f"kind must be custom.<name> (lowercase parts like custom.escalated), got {kind!r}"
            )
        # In a replay, checked against the recording.
        self._served(kind, _session._plain(dict(payload)))
        self._record(kind, dict(payload))

    # Working with other agents.

    def send(self, to: str, message: Any, *, run_id: str | None = None) -> Handoff:
        """Records a message to another agent and returns the handoff. The next
        run started in this context is linked to it by itself; for another
        process or thread, pass the handoff (or its `header`) as ``caused_by``.
        `run_id`: the other agent's run, when it is already running."""
        payload = {"direction": "sent", "to": to, "content": message}
        self._served("message", _session._plain(payload))
        links = [] if run_id is None else [{"rel": "message_to", "run_id": run_id}]
        handoff = Handoff(self.run_id, self._record("message", payload, links=links))
        _last_handoff.set(handoff)
        return handoff

    def receive(
        self, handoff: Handoff | str, message: Any, *, sender: str | None = None
    ) -> Any:
        """Records a message from another agent and returns it, for an agent
        that is already running when the message arrives."""
        given = Handoff.of(handoff)
        payload = {"direction": "received", "from": sender, "content": message}
        self._served("message", _session._plain(payload))
        self._record("message", payload, links=[given.link("message_from")])
        return message

    def shared_read(self, key: str, value: T) -> T:
        """Records that this agent read `key` from state it shares with others,
        and returns the value: ``run.shared_read("case", store["case"])``."""
        self.event("custom.shared.read", {"key": key, "value": value})
        return value

    def shared_write(self, key: str, value: Any) -> None:
        """Records that this agent wrote `key` to state it shares with others."""
        self.event("custom.shared.write", {"key": key, "value": value})

    # Recording.

    def _open_call(self, call: Any) -> None:
        with self._calls_lock:
            self._open_calls.add(call)

    def _close_call(self, call: Any) -> None:
        with self._calls_lock:
            self._open_calls.discard(call)

    def _tool_call(
        self,
        name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
        error: dict[str, str] | None,
        started: int,
        begin: int,
        tool_class: str | None = None,
    ) -> None:
        payload = {
            "name": name,
            "input": {"args": list(args), "kwargs": kwargs},
            "output": output,
            "error": error,
            "started": wall_time(started),
            "ended": wall_time(time.time_ns()),
            "duration_ns": str(time.monotonic_ns() - begin),
        }
        if tool_class is not None:
            payload["tool_class"] = tool_class
        order = key_order(to_json(output, set()))
        if order is not None:
            payload["key_order"] = order
        if error is None and output is not None and name not in _MEMORY_TOOLS:
            try:
                text = (
                    output
                    if isinstance(output, str)
                    else json.dumps(_session._plain(output), default=str)
                )
            except (TypeError, ValueError):
                text = str(output)
            self._answers = [*self._answers[-49:], text]
        self._record("tool.call", payload)

    def _record_tool(
        self,
        name: str,
        fields: dict[str, Any],
        output: Any,
        error: dict[str, str] | None,
    ) -> None:
        """Records a board's answer: a tool call that always says what the engine
        said, in a sandbox or not."""
        now = time.time_ns()
        self._tool_call(
            name, (), fields, output, error, now, time.monotonic_ns(), "pure"
        )

    def _record(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        ends_run: bool = False,
        blobs: Sequence[tuple[Blob, str, str]] = (),
        split: Split | None = None,
        links: Sequence[dict[str, str]] = (),
        size: int = 0,
    ) -> int:
        """Records one event and returns its step. `blobs`: (blob, media_type,
        role) for each blob the payload names. `split`
        moves parts of the payload into blobs, after secrets are stripped, and
        may fill it in from what it holds itself, `size` bytes more."""
        with self._lock:
            if not self._started:
                raise RuntimeError(
                    "use the run in a with block: with helium.run(...) as run:"
                )
            if self._ended:
                raise RuntimeError("this run has ended")
            self._ended = ends_run
            seq = self._seq
            if self._client is None:
                return seq
            try:
                self._send(kind, payload, ends_run, blobs, split, links, size)
            except Exception:
                # Recording must never break the agent.
                log.exception("Kasanki Helium: could not record a %s event", kind)
            return seq

    def _send(
        self,
        kind: str,
        payload: dict[str, Any],
        ends_run: bool,
        blobs: Sequence[tuple[Blob, str, str]],
        split: Split | None,
        links: Sequence[dict[str, str]] = (),
        size: int = 0,
    ) -> None:
        assert self._client is not None
        seq = self._seq
        self._seq += 1
        mono = str(0 if seq == 0 else time.monotonic_ns() - self._mono_start)
        wall = wall_time(_event_time())
        if split is None:
            # Copied here, so later changes by the agent are not recorded.
            data = _to_json(payload)
            event = _client.Event(
                *_encode(self.run_id, seq, kind, wall, mono, data, blobs, None, links)
            )
        else:
            # A model call: the wrapper's payload is already its own copy, so
            # it is converted and built off the agent's thread.
            build = functools.partial(
                _encode, self.run_id, seq, kind, wall, mono, payload, blobs, split
            )
            event = _client.Event(build=build, size=_estimate(payload) + size)
        self._client.submit(self.run_id, seq, event, (wall, mono), ends_run=ends_run)


#: Moves parts of a payload into blobs.
Split = Callable[[dict[str, Any]], list[tuple[Blob, str, str]]]


def _encode(
    run_id: str,
    seq: int,
    kind: str,
    wall: str,
    mono: str,
    data: dict[str, Any],
    blobs: Sequence[tuple[Blob, str, str]],
    split: Split | None,
    links: Sequence[dict[str, str]] = (),
) -> tuple[bytes, list[Blob]]:
    """An event's framed message and the blobs it names: secrets stripped,
    large parts moved into blobs, and at most 1 MiB. With `split`, `data` is
    converted to JSON data here."""
    try:
        if split is not None:
            data = _to_json(data)
        data = redact(data)
        if split is not None:
            blobs = [*blobs, *split(data)]
        reason = None
    except Exception as e:
        log.exception("Kasanki Helium: could not record a %s event", kind)
        reason = f"the {kind} event could not be recorded: {type(e).__qualname__}: {e}"
    # Each blob listed once.
    named: dict[str, tuple[Blob, str, str]] = {}
    for blob in blobs:
        named.setdefault(blob[0].hash, blob)
    body: dict[str, Any] = {
        "format_version": 0,
        "kind": kind,
        "wall_time": wall,
        "mono_time": mono,
        "links": list(links),
        "payload": data,
        "blobs": [
            {
                "hash": blob.hash,
                "size": str(len(blob.content)),
                "media_type": media_type,
                "role": role,
            }
            for blob, media_type, role in named.values()
        ],
    }
    encoded = b"" if reason else _wire.dumps(body)
    if reason is None and len(encoded) > _wire.MAX_BODY:
        reason = (
            f"the {kind} event is {len(encoded)} bytes; events are limited to 1 MiB "
            "(large values will go in blobs)"
        )
        log.warning("Kasanki Helium: %s. An error event was recorded instead.", reason)
    if reason is not None:
        body["kind"] = "error"
        body["payload"] = {"source": "sdk", "rejected_kind": kind, "reason": reason}
        body["blobs"] = []
        named.clear()
        encoded = _wire.dumps(body)
    frame = _wire.event(run_id, seq, encoded)
    return frame, [blob for blob, _, _ in named.values()]


def _to_json(payload: dict[str, Any]) -> Any:
    inexact: set[str] = set()
    data = to_json(payload, inexact, depth=2)
    if inexact:
        _warn_inexact(inexact)
    return data


def _estimate(value: Any) -> int:
    """Roughly the bytes `value` takes as JSON, for the memory limits."""
    if isinstance(value, str):
        return len(value) + 2
    if isinstance(value, dict):
        return sum(len(k) + 4 + _estimate(v) for k, v in value.items()) + 2
    if isinstance(value, list):
        return sum(_estimate(v) + 1 for v in value) + 2
    return 8


def run(
    name: str,
    *,
    input: Any = None,
    version: str | None = None,
    streaming: str = _llm.STREAMING[0],
    caused_by: Handoff | str | None = None,
    entry: str | None = None,
) -> Run:
    """Records one run of an agent::

        with helium.run("support-agent", input=ticket) as run:
            ...

    `name` names the agent; `version` is its version, if it has one; `input`
    is the task it was given. `streaming` sets how streamed model answers
    are recorded: ``"final_and_timings"`` (the final answer and when each
    piece arrived), ``"every_chunk"`` (also every piece) or ``"final_only"``.
    `caused_by`: the handoff that started this run, from another agent in
    another thread or process (`Run.send`). Within one context it is found by
    itself, and so is the run this one was started inside.
    """
    return Run(
        name,
        input=input,
        version=version,
        streaming=streaming,
        caused_by=caused_by,
        entry=entry,
    )


def rollout(
    agent: Mapping[str, Any],
    input: Any,
    source: str,
    at: int,
    policy: Mapping[str, Any],
    *,
    training: bool = False,
) -> Run:
    """A rollout of run `source` at step `at`, recorded without an agent
    replaying it: an environment's episode. Its first
    events are ``run.start`` (linked ``branched_from``), ``profile`` and
    ``policy``; the caller records the rest with ``_record`` and ends it."""
    r = Run(
        str(agent.get("name") or "agent"), input=input, version=agent.get("version")
    )
    r._started = True
    r._connect()
    r._mono_start = time.monotonic_ns()
    r._record(
        "run.start",
        {
            "agent": {"name": r.name, "version": r._version},
            "input": input,
            "context": {**_run_context(os.getcwd()), "entry": None},
        },
        links=[{"rel": "branched_from", "run_id": source, "seq": str(at)}],
    )
    r._record("profile", {**PROFILE, "streaming": r._streaming})
    r._record("policy", {"policy": dict(policy), "change": None, "training": training})
    return r


def flush(timeout: float = 5.0) -> bool:
    """Waits until everything recorded so far is saved by the engine, or
    written to a backlog file for the backlog monitor to deliver, for at most
    `timeout` seconds. Returns whether it all was. Agents never need to call
    this; it suits tests and short scripts."""
    return _client.flush(timeout)
