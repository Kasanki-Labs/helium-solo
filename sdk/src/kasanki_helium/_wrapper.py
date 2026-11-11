# Copyright © 2026 Kasanki Labs.
# SPDX-License-Identifier: AGPL-3.0-only
# Licensed under the GNU AGPL v3 with an attribution term: see LICENSE and CREDIT.md.

"""Wrapped model clients: what every provider's wrapper shares.

Model calls are recorded at the client's HTTP layer (``_http``). What
follows here is the path the SDK took before that: the calls a `Provider`
lists became ``llm.call`` events at the package's surface, everything
else passed through with a warning, and replay rebuilt the answers.
It stays for runs recorded that way: their replays and forks take it,
and its `Provider` describes each package for both paths.

The SDK does not depend on the providers' packages: a client is recognised
by its class, and a package's own helpers are imported only once one of its
clients exists.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import re
import sys
import threading
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urlsplit

from . import _llm
from ._values import wall_time
from ._warnings import warn_once
from ._wire import Blob

if TYPE_CHECKING:
    # Only for type checkers: the SDK has no dependencies at run time.
    from typing_extensions import Self

    from ._run import Run
    from ._session import Change

log = logging.getLogger("kasanki_helium")

#: The client's own HTTP methods: calls, but not recorded.
_RAW_HTTP = frozenset({"get", "post", "put", "patch", "delete", "request"})

#: A resource of the client whose calls return the raw HTTP response; its
#: recorded calls are recorded as without it (llm-wrappers W11).
_RAW_RESPONSE = "with_raw_response"

#: Client methods that return a new client, wrapped in turn.
_NEW_CLIENT = frozenset({"with_options", "copy"})

#: Keyword arguments that shape how the call is made, not what is asked.
_OPTIONS = ("timeout",)


class StreamState:
    """Builds a streamed call's response from its chunks."""

    def attach(self, stream: Any) -> None:
        """Called with the package's stream, before its first chunk."""

    def handle(self, chunk: Any) -> None:
        raise NotImplementedError

    def response(self, complete: bool) -> Any:
        """The response so far, as the package's object, or `None`."""
        raise NotImplementedError


class Provider:
    """What differs between providers' packages."""

    #: The package's name, and the payload's ``provider``.
    name = ""
    #: The module whose __version__ is the package's version.
    module = ""

    def module_name(self) -> str:
        return self.module or self.name

    #: The calls recorded, by their path on the client, and their ``api``.
    recorded: ClassVar[dict[tuple[str, ...], str]] = {}
    #: Calls that return a stream manager (`with ... as stream:`); not
    #: awaited, even on an async client.
    managers: frozenset[str] = frozenset()
    #: Arguments that take a type, recorded as the JSON schema sent for it.
    schema_keys: frozenset[str] = frozenset()
    #: The package's major version this wrapper was built and tested for.
    known_major = 0
    #: Older major versions also tested, and recorded without a warning.
    also_tested: tuple[int, ...] = ()
    #: Calls at the HTTP layer (W18): a POST whose path matches, and its
    #: ``api``; the first that matches wins.
    paths: ClassVar[tuple[tuple[re.Pattern[str], str], ...]] = ()

    def recognises(self, client: Any) -> bool:
        raise NotImplementedError

    def api_for(
        self, method: str, path: str, query: str, headers: Mapping[str, str]
    ) -> str | None:
        """The ``api`` of an HTTP request, or `None` for one not in the table.
        `headers`: those kept, lowercase."""
        if method.upper() != "POST":
            return None
        for pattern, api in self.paths:
            if pattern.search(path):
                return api
        return None

    def assemble(self, api: str, events: list[Any]) -> Any:
        """A streamed answer's final response, as JSON, from the events that
        came over the wire; leniently, since any field may be new."""
        return None

    def is_async(self, client: Any) -> bool:
        raise NotImplementedError

    def schema(self, model: Any, key: str) -> Any:
        """The JSON the package sends for a type passed to ``.parse``."""
        return model.model_json_schema()

    def response(self, response: Any) -> Any:
        """The answer with only the fields the provider sent, as JSON."""
        return to_dict(response)

    def summary(self, api: str, response: Any) -> dict[str, Any]:
        raise NotImplementedError

    def stream_state(self, api: str, kwargs: dict[str, Any]) -> StreamState | None:
        return None

    def manager(self, manager: Any, call: Call) -> Any:
        """A stream manager that records its stream into `call`."""
        if hasattr(manager, "__aenter__"):
            return _AsyncManager(manager, call)
        return _Manager(manager, call)

    # Replay (replay-harness R14): the recorded answer as the server sent it.

    def events(self, api: str, response: Any) -> list[Any]:
        """A streamed answer's events, rebuilt from its final response when
        the chunks were not recorded (W7: the same content, other pieces)."""
        raise NotImplementedError

    def sse(self, events: list[Any]) -> str:
        """Events as the server sends them (server-sent events)."""
        raise NotImplementedError

    def error_body(self, error: dict[str, Any]) -> Any:
        """The HTTP body of a recorded error."""
        return error.get("body")

    def set_system(self, api: str, kwargs: dict[str, Any], prompt: str) -> None:
        """Replaces the system prompt in a call's arguments (a fork's change)."""
        raise NotImplementedError

    def change_request(self, body: dict[str, Any], path: str, change: Change) -> str:
        """A fork's changes to the request body and path. Returns the updated path."""
        if change.model is not None:
            body["model"] = change.model
        if change.params:
            body.update(change.params)
        if change.system_prompt is not None:
            api = self.api_for("POST", path, "", {}) or ""
            self.set_system(api, body, change.system_prompt)
        if change.tool_descriptions:
            body["tools"] = described(body.get("tools"), change.tool_descriptions)
        return path


def described(tools: Any, descriptions: Mapping[str, str]) -> Any:
    """Tool definitions with new descriptions, by tool name: OpenAI's chat
    shape keeps them under ``function``, the others at the top."""
    out = []
    for tool in tools or []:
        if not isinstance(tool, Mapping):
            out.append(tool)
            continue
        function = tool.get("function")
        if isinstance(function, Mapping) and function.get("name") in descriptions:
            name = function["name"]
            out.append(
                {**tool, "function": {**function, "description": descriptions[name]}}
            )
        elif tool.get("name") in descriptions:
            out.append({**tool, "description": descriptions[tool["name"]]})
        else:
            out.append(tool)
    return out


def wrap(run: Run, client: Any, provider: Provider) -> Any:
    check_version(provider)
    return _Proxy(client, _Calls.of(run, client, provider), ())


def check_version(provider: Provider) -> None:
    version = getattr(sys.modules.get(provider.module_name()), "__version__", "")
    major = version.split(".")[0]
    if major not in {str(m) for m in (provider.known_major, *provider.also_tested)}:
        warn_once(
            f"{provider.name} {major}",
            f"Kasanki Helium was built for {provider.name} {provider.known_major}.x, "
            f"not {version}; calls are recorded, but replay may not be exact.",
        )


def classes(client: Any) -> list[tuple[str, str]]:
    """(top-level package, class name) of each class the client is."""
    return [(c.__module__.split(".")[0], c.__name__) for c in type(client).__mro__]


def _server(client: Any) -> str | None:
    """The host (and port) that answers, never credentials."""
    try:
        return urlsplit(str(client.base_url)).netloc.rpartition("@")[2] or None
    except (AttributeError, ValueError):
        return None


def _is_resource(value: Any, package: str) -> bool:
    """One of the client's resources (``chat``, ``files``, ``with_raw_response``
    …), as opposed to a plain value or a model."""
    kind = type(value)
    return (
        not isinstance(value, type)
        and kind.__module__.split(".")[0] == package
        and not hasattr(value, "model_dump")
        and not callable(value)
    )


class _Proxy:
    """The client, or one of its resources, wrapped."""

    __slots__ = ("_calls", "_path", "_target")

    def __init__(self, target: Any, calls: _Calls, path: tuple[str, ...]) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_calls", calls)
        object.__setattr__(self, "_path", path)

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._target, name)
        path = (*self._path, name)
        provider = self._calls.provider
        raw = _RAW_RESPONSE in path
        api = provider.recorded.get(
            tuple(p for p in path if p != _RAW_RESPONSE) if raw else path
        )
        if api is not None and not (raw and api in provider.managers):
            return self._calls.recorded(value, api, raw=raw, path=path)
        if _is_resource(value, provider.name):
            return _Proxy(value, self._calls, path)
        if not self._path and name in _NEW_CLIENT:
            run = self._calls.run

            @functools.wraps(value)
            def new_client(*args: Any, **kwargs: Any) -> Any:
                made = value(*args, **kwargs)
                return _Proxy(made, _Calls.of(run, made, provider), ())

            return new_client
        if callable(value) and (self._path or name in _RAW_HTTP):
            return self._calls.unrecorded(value, ".".join(path))
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._target, name, value)

    def __dir__(self) -> list[str]:
        return dir(self._target)

    def __repr__(self) -> str:
        return f"<recorded by Kasanki Helium: {self._target!r}>"

    def __enter__(self) -> Self:
        self._target.__enter__()
        return self

    def __exit__(self, *exc: object) -> Any:
        return self._target.__exit__(*exc)

    async def __aenter__(self) -> Self:
        await self._target.__aenter__()
        return self

    async def __aexit__(self, *exc: object) -> Any:
        return await self._target.__aexit__(*exc)


class _Calls:
    """Makes the recording versions of one client's methods."""

    def __init__(
        self, run: Run, provider: Provider, is_async: bool, server: str | None
    ) -> None:
        self.run = run
        self.provider = provider
        self.is_async = is_async
        self.server = server
        #: The client itself: replay makes copies of it (R14).
        self.client: Any = None

    @classmethod
    def of(cls, run: Run, client: Any, provider: Provider) -> _Calls:
        calls = cls(run, provider, provider.is_async(client), _server(client))
        calls.client = client
        return calls

    def recorded(
        self,
        method: Callable[..., Any],
        api: str,
        *,
        raw: bool = False,
        path: tuple[str, ...] = (),
    ) -> Callable[..., Any]:
        """`raw`: called through ``with_raw_response``. `path`: where the
        method is on the client."""
        if api in self.provider.managers:

            @functools.wraps(method)
            def stream_manager(*args: Any, **kwargs: Any) -> Any:
                target, kwargs = self.replayed(method, path, api, args, kwargs, True)
                call = Call.start(self, api, args, kwargs, streamed=True)
                try:
                    manager = target(*args, **kwargs)
                except BaseException as e:
                    if call is not None:
                        call.failed(e)
                    raise
                if call is None:
                    return manager
                return self.provider.manager(manager, call)

            return stream_manager

        if self.is_async:

            @functools.wraps(method)
            async def call_async(*args: Any, **kwargs: Any) -> Any:
                streamed = kwargs.get("stream") is True
                target, kwargs = self.replayed(
                    method, path, api, args, kwargs, streamed
                )
                call = Call.start(self, api, args, kwargs, streamed=streamed)
                try:
                    result = await target(*args, **kwargs)
                except BaseException as e:
                    if call is not None:
                        call.failed(e)
                    raise
                if call is None:
                    return result
                if raw:
                    return await _raw_async(result, call, streamed)
                if streamed:
                    return AsyncStream(result, call)
                call.answered(result)
                return result

            return call_async

        @functools.wraps(method)
        def call(*args: Any, **kwargs: Any) -> Any:
            streamed = kwargs.get("stream") is True
            target, kwargs = self.replayed(method, path, api, args, kwargs, streamed)
            call = Call.start(self, api, args, kwargs, streamed=streamed)
            try:
                result = target(*args, **kwargs)
            except BaseException as e:
                if call is not None:
                    call.failed(e)
                raise
            if call is None:
                return result
            if raw:
                return _raw(result, call, streamed)
            if streamed:
                return Stream(result, call)
            call.answered(result)
            return result

        return call

    def replayed(
        self,
        method: Callable[..., Any],
        path: tuple[str, ...],
        api: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        streamed: bool,
    ) -> tuple[Callable[..., Any], dict[str, Any]]:
        """In a replay (R14): the same method on a copy of the client whose
        HTTP layer answers from the recording. After a fork point: the real
        method, with the fork's change applied. Raises when the replay
        diverges."""
        session = getattr(self.run, "_replay", None)
        if session is None or args or self.client is None:
            return method, kwargs
        answer = session.ask("llm.call", match_request(self.provider, api, kwargs))
        if answer["answer"] != "serve":
            change = session.change
            if change is not None and change.changes_model_calls:
                kwargs = dict(kwargs)
                if change.model is not None:
                    kwargs["model"] = change.model
                if change.params:
                    kwargs.update(change.params)
                if change.system_prompt is not None:
                    self.provider.set_system(api, kwargs, change.system_prompt)
                if change.tool_descriptions:
                    kwargs["tools"] = described(
                        kwargs.get("tools"), change.tool_descriptions
                    )
            return method, kwargs
        client = replay_client(
            self.client, self.provider, api, answer["value"], session, streamed
        )
        target = client
        for name in path:
            target = getattr(target, name)
        return target, kwargs

    def unrecorded(self, method: Callable[..., Any], name: str) -> Callable[..., Any]:
        package = self.provider.name

        @functools.wraps(method)
        def call(*args: Any, **kwargs: Any) -> Any:
            warn_once(
                f"{package} {name}",
                f"Kasanki Helium does not record {name} calls, so replay of this "
                "run may not be exact.",
            )
            return method(*args, **kwargs)

        return call


class Call:
    """One model call, from its start to its ``llm.call`` event."""

    def __init__(self, calls: _Calls, api: str, kwargs: dict[str, Any]) -> None:
        self.run = calls.run
        self.provider = calls.provider
        self.api = api
        self.server = calls.server
        self.started = time.time_ns()
        # Fine-grained: the monotonic clock ticks every ~16 ms on Windows.
        self.begin = time.perf_counter_ns()
        # Copied now: the agent may change its message list while a
        # streamed answer arrives.
        self.request = {
            k: plain(v, self.provider, k)
            for k, v in kwargs.items()
            if k not in _OPTIONS and not absent(v)
        }
        options = {
            k: _options(kwargs[k])
            for k in _OPTIONS
            if k in kwargs and not absent(kwargs[k])
        }
        self.options = options or None
        self.capture = calls.run._streaming
        self.chunk_times: list[str] = []
        self.chunks: list[Any] = []
        self.count = 0
        self.stream_state: StreamState | None = None
        self._lock = threading.Lock()
        self._done = False

    @classmethod
    def start(
        cls,
        calls: _Calls,
        api: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        streamed: bool,
    ) -> Call | None:
        """`None` if this call cannot be recorded; it still goes ahead."""
        if args:
            # The client takes keyword arguments only; the call will say so.
            return None
        try:
            call = cls(calls, api, kwargs)
            if streamed:
                try:
                    call.stream_state = calls.provider.stream_state(api, kwargs)
                except Exception:
                    log.exception("Kasanki Helium: cannot assemble streamed answers")
        except Exception:
            log.exception("Kasanki Helium: cannot record a model call")
            return None
        if streamed:
            calls.run._open_call(call)
        return call

    # How the call ends.

    def answered(self, response: Any) -> None:
        self.record(response=self._response(response), error=None, stream=None)

    def failed(self, error: BaseException) -> None:
        self.record(response=None, error=_error(error), stream=None)

    def chunk(self, item: Any) -> None:
        try:
            self.count += 1
            if self.capture != "final_only":
                self.chunk_times.append(str(time.perf_counter_ns() - self.begin))
            if self.capture == "every_chunk":
                self.chunks.append(to_dict(item))
            if self.stream_state is not None:
                try:
                    self.stream_state.handle(item)
                except Exception:
                    log.exception("Kasanki Helium: cannot assemble a streamed answer")
                    self.stream_state = None
        except Exception:
            log.exception("Kasanki Helium: cannot record a streamed chunk")

    def stream_ended(self, complete: bool, error: BaseException | None = None) -> None:
        """The stream ran out (`complete`), or was closed, abandoned or broke."""
        with self._lock:
            if self._done:
                return
        response = None
        try:
            if self.stream_state is not None and self.count:
                response = self._response(self.stream_state.response(complete))
        except Exception:
            log.exception("Kasanki Helium: cannot assemble a streamed answer")
        stream = {
            "capture": self.capture,
            "chunks": self.count,
            "chunk_times_ns": None
            if self.capture == "final_only"
            else self.chunk_times,
            "complete": complete,
        }
        self.record(
            response=response,
            error=None if error is None else _error(error),
            stream=stream,
        )

    def abandon(self) -> None:
        """At the run's end: a stream nobody finished reading."""
        self.stream_ended(complete=False)

    def _response(self, response: Any) -> Any:
        return None if response is None else self.provider.response(response)

    def record(self, response: Any, error: Any, stream: Any) -> None:
        with self._lock:
            if self._done:
                return
            self._done = True
        self.run._close_call(self)
        try:
            blobs: list[tuple[Blob, str, str]] = []
            if self.chunks:
                lines = b"".join(_llm.canonical(c) + b"\n" for c in self.chunks)
                blobs.append((Blob.of(lines), "application/x-ndjson", "chunk"))
            payload = {
                "provider": self.provider.name,
                "api": self.api,
                "server": self.server,
                "request": self.request,
                "options": self.options,
                "response": response,
                "stream": stream,
                "summary": self.provider.summary(self.api, response),
                "error": error,
                "blobbed": {},
                "started": wall_time(self.started),
                "ended": wall_time(time.time_ns()),
                "duration_ns": str(time.perf_counter_ns() - self.begin),
            }
            self.run._record("llm.call", payload, blobs=blobs, split=_llm.split)
        except RuntimeError as e:
            warn_once(
                "llm.call after run end",
                f"Kasanki Helium did not record a model call: {e}.",
                stacklevel=5,
            )
        except Exception:
            log.exception("Kasanki Helium: cannot record a model call")


def _attach(stream: Any, call: Call) -> None:
    if call.stream_state is not None:
        try:
            call.stream_state.attach(stream)
        except Exception:
            log.exception("Kasanki Helium: cannot assemble a streamed answer")
            call.stream_state = None


class Stream:
    """A streamed answer, recorded as the agent reads it."""

    def __init__(self, stream: Any, call: Call) -> None:
        self._stream = stream
        self._call = call
        _attach(stream, call)

    def __iter__(self) -> Stream:
        return self

    def __next__(self) -> Any:
        try:
            item = next(self._stream)
        except StopIteration:
            self._call.stream_ended(complete=True)
            raise
        except BaseException as e:
            self._call.stream_ended(complete=False, error=e)
            raise
        self._call.chunk(item)
        return item

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._stream.close()
        finally:
            self._call.stream_ended(complete=False)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class AsyncStream:
    """A streamed answer of an async client, recorded as the agent reads it."""

    def __init__(self, stream: Any, call: Call) -> None:
        self._stream = stream
        self._call = call
        _attach(stream, call)

    def __aiter__(self) -> AsyncStream:
        return self

    async def __anext__(self) -> Any:
        try:
            item = await self._stream.__anext__()
        except StopAsyncIteration:
            self._call.stream_ended(complete=True)
            raise
        except BaseException as e:
            self._call.stream_ended(complete=False, error=e)
            raise
        self._call.chunk(item)
        return item

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        try:
            await self._stream.close()
        finally:
            self._call.stream_ended(complete=False)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _raw(response: Any, call: Call, streamed: bool) -> Any:
    """A ``with_raw_response`` call: the agent gets the raw response as it
    came, and what its ``parse()`` gives is recorded. The packages keep what
    ``parse()`` returned, so parsing it here costs the agent nothing."""
    if streamed:
        return RawResponse(response, call, Stream)
    try:
        call.answered(response.parse())
    except Exception as e:
        # The agent's own parse() fails the same way.
        log.debug("the answer cannot be parsed", exc_info=True)
        call.failed(e)
    return response


async def _raw_async(response: Any, call: Call, streamed: bool) -> Any:
    """`_raw` for an async client, whose ``parse()`` may be a coroutine."""
    if streamed:
        return RawResponse(response, call, AsyncStream)
    try:
        parsed = response.parse()
        if inspect.isawaitable(parsed):
            parsed = await parsed
        call.answered(parsed)
    except Exception as e:
        # The agent's own parse() fails the same way.
        log.debug("the answer cannot be parsed", exc_info=True)
        call.failed(e)
    return response


class RawResponse:
    """The raw response of a streamed call: its ``parse()`` gives the stream,
    recorded as the agent reads it. Never parsed, the call is recorded at the
    run's end as a stream with no chunks."""

    def __init__(
        self, response: Any, call: Call, wrapper: type[Stream | AsyncStream]
    ) -> None:
        self._response = response
        self._call = call
        self._wrapper = wrapper
        self._stream: Stream | AsyncStream | None = None

    def parse(self, **kwargs: Any) -> Any:
        if kwargs:
            # Parsed into a type of the agent's choosing: not the stream.
            return self._response.parse(**kwargs)
        parsed = self._response.parse()
        if inspect.isawaitable(parsed):
            return self._later(parsed)
        return self._recorded(parsed)

    async def _later(self, parsed: Any) -> Any:
        return self._recorded(await parsed)

    def _recorded(self, stream: Any) -> Stream | AsyncStream:
        if self._stream is None:
            self._stream = self._wrapper(stream, self._call)
        return self._stream

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)


# Stream helpers: `with client.messages.stream(...) as stream:`.


def _watch(stream: Any, call: Call, wrapper: type[Stream | AsyncStream]) -> None:
    """Records the events the package's stream helper reads, as it reads
    them: every way of reading the helper (its events, ``text_stream``,
    ``get_final_message``...) goes through its raw stream."""
    raw = getattr(stream, "_raw_stream", None)
    if raw is None:
        warn_once(
            f"{call.provider.name} stream helper",
            f"Kasanki Helium cannot read this {call.provider.name} version's "
            "stream helpers; their answers are not recorded.",
        )
        return
    stream._raw_stream = wrapper(raw, call)


class _Manager:
    """A stream helper: the request is made when the block starts."""

    def __init__(self, manager: Any, call: Call) -> None:
        self._manager = manager
        self._call = call

    def __enter__(self) -> Any:
        try:
            stream = self._manager.__enter__()
        except BaseException as e:
            self._call.failed(e)
            raise
        _watch(stream, self._call, Stream)
        return stream

    def __exit__(self, *exc: object) -> Any:
        try:
            return self._manager.__exit__(*exc)
        finally:
            # Left before the stream ran out: what arrived is recorded.
            self._call.stream_ended(complete=False)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager, name)


class _AsyncManager:
    """A stream helper of an async client."""

    def __init__(self, manager: Any, call: Call) -> None:
        self._manager = manager
        self._call = call

    async def __aenter__(self) -> Any:
        try:
            stream = await self._manager.__aenter__()
        except BaseException as e:
            self._call.failed(e)
            raise
        _watch(stream, self._call, AsyncStream)
        return stream

    async def __aexit__(self, *exc: object) -> Any:
        try:
            return await self._manager.__aexit__(*exc)
        finally:
            self._call.stream_ended(complete=False)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager, name)


# Replay (replay-harness R14, R15).


def match_request(
    provider: Provider, api: str, kwargs: dict[str, Any]
) -> dict[str, Any]:
    """What the engine matches a model call on (R15), made exactly as the
    call is recorded: converted, secrets stripped, large parts as blobs.

    A request so large that it was stored as one blob when recorded (its
    parts did not make it small enough, next to an answer under 16 KiB) is
    split in parts here, so it does not match; such requests are rare."""
    from . import _llm
    from ._secrets import redact
    from ._values import to_json

    request = {
        k: plain(v, provider, k)
        for k, v in kwargs.items()
        if k not in _OPTIONS and not absent(v)
    }
    payload = {"provider": provider.name, "api": api, "request": request, "blobbed": {}}
    data = redact(to_json(payload, set(), depth=2))
    _llm.split(data)
    return {
        "provider": data["provider"],
        "api": data["api"],
        "request": data["request"],
        "blobbed": data["blobbed"],
    }


def _http_module(client: Any) -> Any:
    """The HTTP library a provider's client is built on. A stock client's HTTP
    client is the provider package's own subclass, so this looks through its
    bases for the library itself."""
    for kind in type(client._client).__mro__:
        module = sys.modules.get(kind.__module__.split(".")[0])
        if module is not None and hasattr(module, "MockTransport"):
            return module
    raise RuntimeError(
        f"{type(client._client).__qualname__} is not built on an HTTP library "
        "a replay can answer through"
    )


def replay_client(
    client: Any,
    provider: Provider,
    api: str,
    recorded: dict[str, Any],
    session: Any,
    streamed: bool,
) -> Any:
    """A copy of `client` whose HTTP layer answers with the recorded
    response, stream or error, and never reaches the network."""
    payload = _filled(recorded, session)
    http = _http_module(client)

    def answer(request: Any) -> Any:
        error = payload.get("error")
        if error is not None:
            status = error.get("status")
            if status is None:
                # No answer came: a timeout or a broken connection.
                kind = (
                    http.ReadTimeout
                    if "Timeout" in str(error.get("type"))
                    else http.ConnectError
                )
                raise kind(str(error.get("message") or ""), request=request)
            return http.Response(status, json=provider.error_body(error))
        if streamed:
            events = payload.get("chunks")
            if events is None:
                events = provider.events(api, payload.get("response"))
            return http.Response(
                200,
                text=provider.sse(events),
                headers={"content-type": "text/event-stream"},
            )
        return http.Response(200, json=payload.get("response"))

    transport = http.MockTransport(answer)
    if provider.is_async(client):
        http_client = http.AsyncClient(transport=transport)
    else:
        http_client = http.Client(transport=transport)
    return client.with_options(http_client=http_client, max_retries=0)


def _filled(recorded: dict[str, Any], session: Any) -> dict[str, Any]:
    """The recorded payload with its blobbed answer put back, and the
    recorded chunks (``every_chunk``) as ``chunks``."""
    data: dict[str, Any] = json.loads(json.dumps(recorded))
    for pointer, hash in (data.get("blobbed") or {}).items():
        if pointer == "/response" or pointer.startswith("/response/"):
            _put(data, pointer, json.loads(session.blob(hash)))
    data["chunks"] = None
    for ref in data.get("blobs") or []:
        if ref.get("role") == "chunk":
            lines = session.blob(ref["hash"]).splitlines()
            data["chunks"] = [json.loads(line) for line in lines if line.strip()]
    return data


def _put(data: Any, pointer: str, value: Any) -> None:
    *path, last = [
        part.replace("~1", "/").replace("~0", "~") for part in pointer.split("/")[1:]
    ]
    at = data
    for part in path:
        at = at[int(part)] if isinstance(at, list) else at[part]
    if isinstance(at, list):
        at[int(last)] = value
    else:
        at[last] = value


# Turning the call into JSON.


def absent(value: Any) -> bool:
    """The packages' markers for an argument not given (``NOT_GIVEN``, ``omit``)."""
    kind = type(value)
    return kind.__name__ in ("NotGiven", "Omit") and kind.__module__.split(".")[0] in (
        "openai",
        "anthropic",
    )


def plain(value: Any, provider: Provider, key: str = "") -> Any:
    """A copy as plain data: models as the JSON the package sends, response
    format types as their JSON schema (§4.1)."""
    if isinstance(value, str) or value is None:
        return value
    if isinstance(value, dict):
        return {k: plain(v, provider, k) for k, v in value.items() if not absent(v)}
    if isinstance(value, (list, tuple)):
        return [plain(v, provider) for v in value]
    if (isinstance(value, type) and hasattr(value, "model_json_schema")) or (
        key in provider.schema_keys and not isinstance(value, (list, tuple))
    ):
        return provider.schema(value, key)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json", by_alias=True, exclude_unset=True, warnings=False)
    return value


def to_dict(value: Any) -> Any:
    """A response or chunk with only the fields the provider sent."""
    convert = getattr(value, "to_dict", None)
    if callable(convert):
        return convert(mode="json", warnings=False)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json", by_alias=True, exclude_unset=True, warnings=False)
    return value


def _options(value: Any) -> Any:
    if hasattr(value, "connect") and hasattr(value, "read"):
        return {
            "connect": value.connect,
            "read": value.read,
            "write": getattr(value, "write", None),
            "pool": getattr(value, "pool", None),
        }
    return value


def _error(e: BaseException) -> dict[str, Any]:
    """Enough to raise the same error in replay (§4.1)."""
    body = getattr(e, "body", None)
    return {
        "type": type(e).__qualname__,
        "message": str(e),
        "status": getattr(e, "status_code", None),
        "body": body if isinstance(body, (dict, list, str)) or body is None else None,
    }


def get(data: Any, *path: Any) -> Any:
    """``data[path[0]][path[1]]…``, or `None` where a step is missing."""
    for key in path:
        if isinstance(key, int):
            data = data[key] if isinstance(data, list) and len(data) > key else None
        else:
            data = data.get(key) if isinstance(data, dict) else None
    return data
