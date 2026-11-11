# Copyright © 2026 Kasanki Labs.
# SPDX-License-Identifier: AGPL-3.0-only
# Licensed under the GNU AGPL v3 with an attribution term: see LICENSE and CREDIT.md.

"""The replay session: what lets a
recorded run's answers be served to the agent in place of live ones.

Inside a replay, ``helium.run(...)`` claims the session the replay opened
(`claim`), and each helper asks the engine for the recorded value
(`Session.ask`). The engine matches every request and serves the answer; this
module only relays it. Starting a replay or a fork, and running tools in a
sandbox, belong to Helium's tools, which register the sandbox hooks here
(`sandbox_hooks`).

It also finds the entry point a run is recorded with, so a run can be
replayed later.
"""

from __future__ import annotations

import builtins
import contextvars
import functools
import inspect
import json
import os
import re
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any

from . import _control, _endpoint
from ._secrets import redact
from ._values import to_json

#: Sentinel: no tool result change.
_UNSET: Any = object()


class HeliumReplayDiverged(Exception):
    """Raised inside the agent when it asks for something the recorded run
    did not do (replay-harness §5.3). `divergence` says where and how."""

    def __init__(self, divergence: dict[str, Any]) -> None:
        self.divergence = divergence
        step = divergence.get("seq")
        where = f"step {step}" if step is not None else "the end"
        super().__init__(
            f"replay diverged at {where} ({divergence.get('kind')}): "
            f"{divergence.get('reason')}"
        )


class ReplayBlocked(Exception):
    """A tool the fork's access policy does not let run (replay-harness R7)."""


class ReplayedError(Exception):
    """A recorded tool error whose type cannot be rebuilt; `type_name` is
    the recorded type."""

    def __init__(self, type_name: str, message: str) -> None:
        self.type_name = type_name
        super().__init__(f"{type_name}: {message}")


class ReplayError(Exception):
    """A replay could not run (the engine refused it, or the entry point
    never started a run)."""


@dataclass(frozen=True)
class Change:
    """What a fork changes (replay-harness R6). Model call changes apply to
    the model call at the fork step and every later one; a tool result
    change is served for the tool call at the fork step::

        helium.Change(system_prompt=open("prompt-v2.txt").read())
        helium.Change(model="gpt-5-mini", params={"temperature": 0.2})
        helium.Change(tool_descriptions={"search": "Searches open orders only."})
        helium.Change(tool_output="no results")
        helium.Change(tool_error=("TimeoutError", "search timed out"))
        helium.Change(model_answer=edited_response)  # the model call at the fork step
    """

    system_prompt: str | None = None
    model: str | None = None
    params: Mapping[str, Any] | None = None
    tool_output: Any = _UNSET
    tool_error: tuple[str, str] | None = None
    #: New descriptions of tools the agent offers its model, by tool name.
    tool_descriptions: Mapping[str, str] | None = None
    #: The provider's response to serve for the model call at the fork step,
    #: in the shape the recording holds it. A person's
    #: words, not the model's: such a run is never exported for fine-tuning.
    model_answer: Any = None

    @classmethod
    def from_json(cls, found: Mapping[str, Any]) -> Change:
        """The change `to_json` described, for an agent in another process."""
        result: dict[str, Any] = {}
        tool_result = found.get("tool_result")
        if tool_result is not None:
            if tool_result["error"] is not None:
                error = tool_result["error"]
                result["tool_error"] = (error["type"], error["message"])
            else:
                result["tool_output"] = tool_result["output"]
        return cls(
            system_prompt=found.get("system_prompt"),
            model=found.get("model"),
            params=found.get("params"),
            tool_descriptions=found.get("tool_descriptions"),
            model_answer=found.get("model_answer"),
            **result,
        )

    @property
    def changes_model_calls(self) -> bool:
        return (
            self.system_prompt is not None
            or self.model is not None
            or bool(self.params)
            or bool(self.tool_descriptions)
        )

    def to_json(self) -> dict[str, Any]:
        tool_result = None
        if self.tool_error is not None:
            kind, message = self.tool_error
            tool_result = {"output": None, "error": {"type": kind, "message": message}}
        elif self.tool_output is not _UNSET:
            tool_result = {"output": _plain(self.tool_output), "error": None}
        return {
            "system_prompt": self.system_prompt,
            "model": self.model,
            "params": _plain(dict(self.params)) if self.params else None,
            "tool_descriptions": dict(self.tool_descriptions)
            if self.tool_descriptions
            else None,
            "tool_result": tool_result,
            "model_answer": None
            if self.model_answer is None
            else _plain(self.model_answer),
        }


@dataclass
class Session:
    """One replay session in the engine."""

    control: _control.Control
    id: str
    run_id: str
    mode: str
    input: Any
    agent: dict[str, Any]
    context: dict[str, Any]
    warnings: list[str]
    rollout_run_id: str | None = None
    fork_at: int | None = None
    change: Change | None = None
    policy: dict[str, Any] = field(default_factory=dict)
    claimed: bool = False
    finished: dict[str, Any] | None = None
    #: What the entry point returned.
    output: Any = None
    #: Blob contents fetched so far.
    blobs: dict[str, bytes] = field(default_factory=dict)
    #: In a `helium test` case: the sandbox the agent's tools run in, by
    #: name, instead of the functions the agent passes.
    sandbox: Any = None
    #: Whether this run branches from a recorded one. An agent started past a
    #: fork that the recording never had is an ordinary new run.
    branched: bool = True
    #: A test case started from its input, with no recording:
    #: whichever agent the entry point starts is the case's.
    fresh: bool = False
    #: The sessions of runs started inside this replay, on its first session.
    joined: list[Session] = field(default_factory=list)
    #: The first session of this replay; `None` on the first itself.
    root: Session | None = None
    #: A fork asked of a run started inside another: the first session holds the
    #: change, and the engine says which run it belongs to.
    asked_change: Change | None = None
    #: What the agent's code is now, as `replay.open` was told.
    code: dict[str, Any] = field(default_factory=dict)
    #: The first run's end: the replay finishes when the entry point returns,
    #: since another agent can start after the first has ended.
    ended: tuple[Any, dict[str, str] | None] | None = None
    #: On the first session, for each board: how far the recording got, and the
    #: board a fork carries on with.
    boards: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: On the first session: what differs without being a divergence.
    notes: list[str] = field(default_factory=list)
    #: A rehearsal: a new version whose tools run in the sandbox
    #: even while it is served, recorded as a new run.
    rehearsal: bool = False
    #: What a rehearsal's run says it is, in its context: ``{"of", "depth"}``.
    rehearsal_of: dict[str, Any] | None = None

    @property
    def records(self) -> bool:
        """Whether its runs are recorded as new ones: a fork's, or a rehearsal's."""
        return self.mode == "fork" or self.rehearsal

    def ask(self, kind: str, request: Any) -> dict[str, Any]:
        """The engine's answer to the agent's request: ``serve`` (with
        ``value``), ``checked`` or ``live``. Raises `HeliumReplayDiverged` or
        `ReplayBlocked`."""
        try:
            answer: dict[str, Any] = self.control.call(
                "replay.next",
                {"session_id": self.id, "kind": kind, "request": request},
            )
        except _control.ControlError as e:
            raise ReplayError(str(e)) from e
        if answer["answer"] == "diverged":
            raise HeliumReplayDiverged(answer["divergence"])
        if answer["answer"] == "blocked":
            raise ReplayBlocked(answer["reason"])
        return answer

    def blob(self, hash: str) -> bytes:
        if hash not in self.blobs:
            self.blobs[hash] = self.control.blob(hash)
        return self.blobs[hash]

    def end(self, output: Any, error: dict[str, str] | None) -> None:
        """The run ended. A run started inside another finishes now; the first
        when the entry point returns."""
        if self.root is None:
            self.ended = (output, error)
        else:
            _withdraw(self)
            self.finish(output, error)

    def finish(self, output: Any, error: dict[str, str] | None) -> None:
        try:
            self.finished = self.control.call(
                "replay.finish",
                {"session_id": self.id, "output": output, "error": error},
            )
        except _control.ControlError as e:
            raise ReplayError(str(e)) from e


#: The session the next `helium.run(...)` in this context replays.
_pending: contextvars.ContextVar[Session | None] = contextvars.ContextVar(
    "kasanki_helium_replay", default=None
)


#: Runs being replayed, by recorded and new run ID: a thread that isn't given the
#: context still names one when it passes `caused_by`.
_by_run: dict[str, Session] = {}
_by_run_lock = threading.Lock()


#: Tells whatever starts during a replay which replay it's part of, so an agent in
#: another process or a thread with no context is served from the recording like any
#: other, and never does anything real. A child process inherits it.
REPLAY_VARIABLE = "KASANKI_HELIUM_REPLAY"

#: The sessions running in this process, newest last; the variable names the newest.
_announced: list[Session] = []
_announced_lock = threading.Lock()
#: The replay this process inherited, once it has connected to it.
_inherited_session: tuple[str, Session] | None = None


def _announce(session: Session) -> None:
    with _announced_lock:
        _announced.append(session)
        _name_newest()


def _withdraw(session: Session) -> None:
    with _announced_lock:
        _announced[:] = [s for s in _announced if s is not session]
        _name_newest()


def _name_newest() -> None:
    if not _announced:
        if _inherited_session is not None:
            os.environ[REPLAY_VARIABLE] = _inherited_session[0]
        else:
            os.environ.pop(REPLAY_VARIABLE, None)
        return
    session = _announced[-1]
    change = (session.root or session).asked_change or session.change
    os.environ[REPLAY_VARIABLE] = json.dumps(
        {
            "session_id": session.id,
            "mode": session.mode,
            "policy": session.policy,
            "sandboxed": session.sandbox is not None,
            "code": session.code,
            # A board's first answer depends on it, even in another process.
            "knows_project": "project" in session.context,
            "change": None if change is None else change.to_json(),
        }
    )


def _inherited() -> Session | None:
    """The session a run started with nothing to go on belongs to: the one running in
    this process, or the one the process that started this one was running. Refuses
    what can't be served, since the only other choice is to do it for real."""
    global _inherited_session
    raw = os.environ.get(REPLAY_VARIABLE)
    if not raw:
        return None
    with _announced_lock:
        if _announced:
            return _announced[-1]
        if _inherited_session is not None and _inherited_session[0] == raw:
            return _inherited_session[1]
    found = json.loads(raw)
    if found["sandboxed"]:
        raise ReplayError(
            "an agent in another process can't reach this fork's sandbox, so it "
            "isn't run; start it in the same process, or fork without a sandbox"
        )
    from ._run import CLIENT_NAME

    try:
        control = _control.Control(
            _endpoint.endpoint(_endpoint.data_dir()), CLIENT_NAME
        )
    except _control.ControlError as e:
        raise ReplayError(str(e)) from e
    session = Session(
        control=control,
        id=found["session_id"],
        run_id="",
        mode=found["mode"],
        input=None,
        agent={},
        context={"project": None} if found.get("knows_project") else {},
        warnings=[],
        policy=found["policy"],
        claimed=True,
        code=found["code"],
        asked_change=None
        if found["change"] is None
        else Change.from_json(found["change"]),
    )
    with _announced_lock:
        _inherited_session = (raw, session)
    return session


def remember(session: Session) -> None:
    with _by_run_lock:
        for run_id in (session.run_id, session.rollout_run_id):
            if run_id:
                _by_run[run_id] = session


def forget(session: Session) -> None:
    with _by_run_lock:
        for run_id in (session.run_id, session.rollout_run_id):
            if run_id:
                _by_run.pop(run_id, None)


def claim(
    name: str,
    input: Any,
    inside: Session | None = None,
    caused_by_run: str | None = None,
) -> Session | None:
    """For `helium.run(...)`: the session it replays, if any. `inside` is the
    session of the run it was started inside; `caused_by_run`, the run that handed
    work to it, which finds the replay from another thread."""
    session = _pending.get()
    if session is None and inside is None and caused_by_run is not None:
        with _by_run_lock:
            inside = _by_run.get(caused_by_run)
    if inside is not None:
        return _join(inside, name, input)
    if session is None:
        inside = _inherited()
        return None if inside is None else _join(inside, name, input)
    if session.claimed:
        # The first run has ended and another starts: it's still this replay.
        return _join(session, name, input)
    session.claimed = True
    remember(session)
    if session.fresh:
        session.agent = {"name": name}
    recorded_name = (session.agent or {}).get("name")
    if name != recorded_name:
        raise HeliumReplayDiverged(
            {
                "seq": "0",
                "kind": "run.start",
                "reason": f"the run was started as {name!r}, recorded as {recorded_name!r}",
                "expected": recorded_name,
                "actual": name,
                "diff": [],
            }
        )
    actual = _plain(input)
    if actual != session.input:
        raise HeliumReplayDiverged(
            {
                "seq": "0",
                "kind": "run.start",
                "reason": "the run was started with another input",
                "expected": session.input,
                "actual": actual,
                "diff": [],
            }
        )
    return session


def _join(parent: Session, name: str, input: Any) -> Session:
    """A run started inside a replayed one: another
    agent's recording, never a real run with real tools."""
    try:
        joined = parent.control.call(
            "replay.join",
            {
                "session_id": parent.id,
                "name": name,
                "input": _plain(input),
                "context": parent.code,
            },
        )
    except _control.ControlError as e:
        raise ReplayError(str(e)) from e
    if joined.get("answer") == "diverged":
        raise HeliumReplayDiverged(joined["divergence"])
    fork_at = joined.get("fork_at")
    run_id = joined.get("run_id")
    session = Session(
        control=parent.control,
        id=joined["session_id"],
        run_id=run_id or "",
        mode=parent.mode,
        input=joined.get("input"),
        agent=joined.get("agent") or {},
        context=joined.get("context") or {},
        warnings=list(joined.get("warnings") or []),
        rollout_run_id=joined.get("rollout_run_id"),
        fork_at=int(fork_at) if fork_at is not None else None,
        change=parent_change(parent) if joined.get("changed") else None,
        policy=parent.policy,
        claimed=True,
        blobs=parent.blobs,
        sandbox=parent.sandbox,
        branched=bool(run_id),
        code=parent.code,
        rehearsal=parent.rehearsal,
        rehearsal_of=None
        if parent.rehearsal_of is None
        else {**parent.rehearsal_of, "of": run_id},
    )
    session.root = parent.root or parent
    session.root.joined.append(session)
    remember(session)
    _announce(session)
    return session


def parent_change(parent: Session) -> Change | None:
    return (parent.root or parent).asked_change


def _plain(value: Any) -> Any:
    """`value` as it is recorded: JSON, secrets stripped."""
    return redact(to_json(value, set()))


# Finding the entry point when recording (R13).


_TOOL_SECTION = re.compile(r"^\[tool\.kasanki-helium\]\s*$")
_AGENT_SECTION = re.compile(r"^\[tool\.kasanki-helium\.agents\.([A-Za-z0-9_-]+)\]\s*$")
_SETTING_LINE = re.compile(r"""^([a-z_]+)\s*=\s*(["'])([^"']+)\2\s*(#.*)?$""")


@functools.lru_cache(maxsize=32)
def pyproject_setting(folder: str, name: str) -> str | None:
    """``[tool.kasanki-helium] <name>`` in the folder's pyproject.toml: the
    entry point, the graders module or the sandbox."""
    try:
        text = (Path(folder) / "pyproject.toml").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    inside = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("["):
            inside = bool(_TOOL_SECTION.match(line))
            continue
        if inside:
            match = _SETTING_LINE.match(line)
            if match and match.group(1) == name:
                return match.group(3)
    return None


def pyproject_entry(folder: str) -> str | None:
    """``[tool.kasanki-helium] entry`` in the folder's pyproject.toml: the project's
    one agent, or its default one."""
    return pyproject_setting(folder, "entry")


@functools.lru_cache(maxsize=32)
def pyproject_agents(folder: str) -> dict[str, str]:
    """The project's agents by name: ``[tool.kasanki-helium.agents.<name>] entry``."""
    try:
        text = (Path(folder) / "pyproject.toml").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    agents: dict[str, str] = {}
    agent = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("["):
            found = _AGENT_SECTION.match(line)
            agent = found.group(1) if found else None
            continue
        if agent is not None:
            match = _SETTING_LINE.match(line)
            if match and match.group(1) == "entry":
                agents[agent] = match.group(3)
    return agents


def agent_entry(folder: str, name: str) -> str | None:
    """The entry point of the agent this run is named after, else the default one."""
    return pyproject_agents(folder).get(name) or pyproject_entry(folder)


def caller_entry(frame: FrameType | None) -> str | None:
    """The function running in `frame`, as ``module:function``, when it is
    a module-level function taking the task input as its only required
    argument; otherwise `None`."""
    if frame is None:
        return None
    return _entry_for(frame.f_code, frame.f_globals.get("__name__"), _main_file(frame))


def _main_file(frame: FrameType) -> str | None:
    if frame.f_globals.get("__name__") != "__main__":
        return None
    spec = frame.f_globals.get("__spec__")
    if spec is not None and getattr(spec, "name", None):
        return None
    return frame.f_code.co_filename


@functools.lru_cache(maxsize=256)
def _entry_for(code: Any, module: str | None, main_file: str | None) -> str | None:
    if module is None or code.co_name.startswith("<"):
        return None
    try:
        mod = sys.modules[module]
    except KeyError:
        return None
    function: Any = getattr(mod, code.co_name, None)
    # Only a module-level function whose code is the one running.
    if function is None or id(getattr(function, "__code__", None)) != id(code):
        return None
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return None
    positional = [
        p for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    required = [
        p
        for p in parameters
        if p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    if not positional or len(required) != 1 or required[0] is not positional[0]:
        return None
    if module == "__main__":
        if main_file is None:
            spec = getattr(mod, "__spec__", None)
            name = getattr(spec, "name", None)
            return f"{name}:{code.co_name}" if name else None
        try:
            path = Path(main_file).resolve().relative_to(Path.cwd().resolve())
        except ValueError:
            path = Path(main_file).resolve()
        return f"{path.as_posix()}:{code.co_name}"
    return f"{module}:{code.co_name}"


#: Exception types a recorded tool error is rebuilt as, by recorded name, beyond
#: the built-in ones. The module that defines one registers it: an MCP tool's
#: error text reaches the model, so a replay must show the same type.
REBUILT: dict[str, type[Exception]] = {}


def rebuilt_error(error: Mapping[str, Any]) -> BaseException:
    """A recorded tool error as an exception the agent can catch: the
    built-in exception of that name if there is one."""
    name = str(error.get("type") or "Exception")
    message = str(error.get("message") or "")
    kind = REBUILT.get(name) or getattr(builtins, name, None)
    if isinstance(kind, type) and issubclass(kind, Exception):
        try:
            return kind(message)
        except TypeError:
            # One that needs other arguments (UnicodeDecodeError…).
            pass
    return ReplayedError(name, message)


#: Running tools in a sandbox, registered by Helium's tools: ``advance(session,
#: name, args, kwargs, served, seq)``, ``rehearse(session, name, args, kwargs,
#: seq)`` and ``tool(sandbox, name)``. A session only has a sandbox when they
#: opened it.
sandbox_hooks: Any = None


def hooks() -> Any:
    """The sandbox hooks, for a session that has a sandbox."""
    if sandbox_hooks is None:
        raise ReplayError("this run has a sandbox, which needs Helium's tools")
    return sandbox_hooks
