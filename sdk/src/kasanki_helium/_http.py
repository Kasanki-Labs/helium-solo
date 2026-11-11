# Copyright © 2026 Kasanki Labs.
# SPDX-License-Identifier: AGPL-3.0-only
# Licensed under the GNU AGPL v3 with an attribution term: see LICENSE and CREDIT.md.

"""Recording model calls below the SDK, at the client's HTTP layer.

``run.wrap(client)`` gives a client a copy of its own HTTP client whose
transports record each exchange, so the team's proxies, timeouts and limits
are kept; ``run.http_client()`` makes one to pass in. Either way the provider
package builds every request and parses every answer itself:

- the request is recorded as it goes out (method, path, query, the headers
  that change the answer, the body), the response as it comes in (status,
  headers, the body, and a stream's bytes with their arrival times, W7);
- in a replay the recorded bytes are served back through the same
  transport, and whatever version of the package is installed parses them
  (W15), never reaching the network;
- calls are told apart by path, in one table per provider (W18).

The HTTP library is the one the client is built on (``httpx2`` for openai 3
and anthropic 2, or ``httpx``), never one imported here by name.
"""

from __future__ import annotations

import functools
import importlib
import json
import logging
import re
import sys
import threading
import time
import zlib
from collections.abc import AsyncIterator, Iterator
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any

from . import _llm
from ._secrets import is_secret_key, redact, redact_text
from ._values import to_json, wall_time
from ._warnings import warn_once
from ._wire import MAX_BLOB, Blob

if TYPE_CHECKING:
    from ._run import Run
    from ._session import Change, Session
    from ._wrapper import Provider

log = logging.getLogger("kasanki_helium")

#: Request headers never kept: who sent the request (the package's identity,
#: W16), per-call IDs and retry counters (W4, W16), framing, and signatures.
#: Secrets go by name besides (trace-format Â§10).
_LEFT_OUT = frozenset(
    {
        "user-agent",
        "host",
        "connection",
        "keep-alive",
        "content-length",
        "transfer-encoding",
        "te",
        "accept-encoding",
        "idempotency-key",
        "x-request-id",
        "x-amz-date",
        "x-amz-content-sha256",
        "x-amz-user-agent",
        "traceparent",
        "tracestate",
        "x-goog-api-client",
    }
)
_LEFT_OUT_PREFIXES = ("x-stainless-",)
_SECRET_HEADERS = frozenset(
    {"authorization", "proxy-authorization", "ocp-apim-subscription-key"}
)

#: Response headers not kept: framing, cookies, and the body's length, since
#: the body is kept decoded.
_RESPONSE_LEFT_OUT = frozenset(
    {
        "connection",
        "keep-alive",
        "content-length",
        "transfer-encoding",
        "te",
        "trailer",
        "upgrade",
        "set-cookie",
    }
)

#: Response types whose body is a stream of events.
_STREAMS = ("text/event-stream", "application/vnd.amazon.eventstream")

#: A multipart body's boundary is random on every request; it is recorded as
#: this, so the same upload matches.
_BOUNDARY = "kasanki-helium-boundary"
_BOUNDARY_PARAM = re.compile(r'boundary="?([^";\s]+)"?', re.IGNORECASE)

_MEDIA_TYPE = re.compile(r"^[a-z0-9!#$&^_.+-]+/[a-z0-9!#$&^_.+-]+$")


#: The ``api`` of a call whose path is in no table: a plain HTTP exchange.
PLAIN = "http"


def _no_summary() -> dict[str, Any]:
    """The summary of a call that is not one the provider's table knows."""
    return {
        "model": None,
        "input_tokens": None,
        "output_tokens": None,
        "stop_reason": None,
        "tool_calls": [],
    }


# Getting in (W17).


def wrap(run: Run, client: Any, provider: Provider) -> Any:
    """A copy of `client` whose HTTP client records into `run`. The copy is
    the package's own client, so it behaves like one in every way."""
    from ._wrapper import _http_module

    http = _http_module(client)
    recorder = Recorder(run, http, provider)
    made = recording_copy(client._client, recorder, provider.is_async(client))
    return client.with_options(http_client=made)


def http_client(run: Run, is_async: bool, kwargs: dict[str, Any]) -> Any:
    """An HTTP client that records into `run`, of the library the provider
    packages are built on, with their defaults where `kwargs` sets none."""
    http = _library()
    kwargs.setdefault("follow_redirects", True)
    kwargs.setdefault("timeout", http.Timeout(600.0, connect=5.0))
    kwargs.setdefault(
        "limits", http.Limits(max_connections=1000, max_keepalive_connections=100)
    )
    made = (http.AsyncClient if is_async else http.Client)(**kwargs)
    _record_through(made, Recorder(run, http, None), is_async, owns=True)
    return made


def _library() -> ModuleType:
    for name in ("httpx2", "httpx"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise RuntimeError(
        "run.http_client() needs httpx2 or httpx, which the openai and anthropic "
        "packages install"
    )


def recording_copy(original: Any, recorder: Recorder, is_async: bool) -> Any:
    """A shallow copy of an HTTP client whose transports record. The client's
    own transports are kept inside, with their proxies, pools and limits, and
    still belong to it: closing the copy leaves them open."""
    import copy

    made = copy.copy(original)
    _record_through(made, recorder, is_async, owns=False)
    return made


def _record_through(
    client: Any, recorder: Recorder, is_async: bool, owns: bool
) -> None:
    kinds = _kinds(recorder.http)
    kind = kinds.AsyncRecording if is_async else kinds.Recording
    try:
        client._transport = kind(client._transport, recorder, owns)
        client._mounts = {
            pattern: None if transport is None else kind(transport, recorder, owns)
            for pattern, transport in client._mounts.items()
        }
    except AttributeError:
        raise TypeError(
            f"Kasanki Helium cannot record through a {type(client).__qualname__}: "
            "its HTTP library has no transports to wrap"
        ) from None


@functools.cache
def _kinds(http: ModuleType) -> SimpleNamespace:
    """The library's transport and stream classes, recording (made once per
    library, since each library's own base classes are needed)."""

    class Recording(http.BaseTransport):  # type: ignore[misc, name-defined]
        def __init__(self, inner: Any, recorder: Recorder, owns: bool) -> None:
            self._inner = inner
            self._recorder = recorder
            self._owns = owns

        def handle_request(self, request: Any) -> Any:
            return self._recorder.handle(request, self._inner)

        def close(self) -> None:
            # A copy's transports belong to the client they came from.
            if self._owns:
                self._inner.close()

    class AsyncRecording(http.AsyncBaseTransport):  # type: ignore[misc, name-defined]
        def __init__(self, inner: Any, recorder: Recorder, owns: bool) -> None:
            self._inner = inner
            self._recorder = recorder
            self._owns = owns

        async def handle_async_request(self, request: Any) -> Any:
            return await self._recorder.handle_async(request, self._inner)

        async def aclose(self) -> None:
            if self._owns:
                await self._inner.aclose()

    class Capture(http.SyncByteStream):  # type: ignore[misc, name-defined]
        """The response's bytes, recorded as the package reads them."""

        def __init__(self, inner: Any, exchange: Exchange) -> None:
            self._inner = inner
            self._exchange = exchange

        def __iter__(self) -> Iterator[bytes]:
            try:
                for piece in self._inner:
                    self._exchange.piece(piece)
                    yield piece
            except GeneratorExit:
                self._exchange.ended(complete=False)
                raise
            except BaseException as e:
                self._exchange.ended(complete=False, error=e)
                raise
            self._exchange.ended(complete=True)

        def close(self) -> None:
            try:
                close = getattr(self._inner, "close", None)
                if callable(close):
                    close()
            finally:
                # Closed before it ran out: what arrived is recorded.
                self._exchange.ended(complete=False)

    class AsyncCapture(http.AsyncByteStream):  # type: ignore[misc, name-defined]
        def __init__(self, inner: Any, exchange: Exchange) -> None:
            self._inner = inner
            self._exchange = exchange

        async def __aiter__(self) -> AsyncIterator[bytes]:
            try:
                async for piece in self._inner:
                    self._exchange.piece(piece)
                    yield piece
            except GeneratorExit:
                self._exchange.ended(complete=False)
                raise
            except BaseException as e:
                self._exchange.ended(complete=False, error=e)
                raise
            self._exchange.ended(complete=True)

        async def aclose(self) -> None:
            try:
                close = getattr(self._inner, "aclose", None)
                if callable(close):
                    await close()
            finally:
                self._exchange.ended(complete=False)

    class Replayed(http.SyncByteStream):  # type: ignore[misc, name-defined]
        """Recorded bytes, in the pieces they arrived in when those were kept."""

        def __init__(self, pieces: list[bytes], broke: BaseException | None) -> None:
            self._pieces = pieces
            self._broke = broke

        def __iter__(self) -> Iterator[bytes]:
            yield from self._pieces
            if self._broke is not None:
                # It broke while it came, and breaks again at the same place.
                raise self._broke

    class AsyncReplayed(http.AsyncByteStream):  # type: ignore[misc, name-defined]
        def __init__(self, pieces: list[bytes], broke: BaseException | None) -> None:
            self._pieces = pieces
            self._broke = broke

        async def __aiter__(self) -> AsyncIterator[bytes]:
            for piece in self._pieces:
                yield piece
            if self._broke is not None:
                raise self._broke

    return SimpleNamespace(
        Recording=Recording,
        AsyncRecording=AsyncRecording,
        Capture=Capture,
        AsyncCapture=AsyncCapture,
        Replayed=Replayed,
        AsyncReplayed=AsyncReplayed,
    )


# Recording and serving each exchange.


class Recorder:
    """Records one client's HTTP exchanges into a run, and in a replay
    answers them from the recording. `provider`: the client's, or `None` to
    tell it from each request (``run.http_client()``)."""

    def __init__(self, run: Run, http: ModuleType, provider: Provider | None) -> None:
        self.run = run
        self.http = http
        self.provider = provider
        self.kinds = _kinds(http)

    def handle(self, request: Any, inner: Any) -> Any:
        exchange = self._begin(request, request.read(), is_async=False)
        if exchange is None:
            return inner.handle_request(request)
        if exchange.served is not None:
            return exchange.answered(exchange.served, self.kinds.Capture)
        try:
            response = inner.handle_request(exchange.request)
        except BaseException as e:
            exchange.failed(e)
            raise
        return exchange.answered(response, self.kinds.Capture)

    async def handle_async(self, request: Any, inner: Any) -> Any:
        exchange = self._begin(request, await request.aread(), is_async=True)
        if exchange is None:
            return await inner.handle_async_request(request)
        if exchange.served is not None:
            return exchange.answered(exchange.served, self.kinds.AsyncCapture)
        try:
            response = await inner.handle_async_request(exchange.request)
        except BaseException as e:
            exchange.failed(e)
            raise
        return exchange.answered(response, self.kinds.AsyncCapture)

    def _begin(self, request: Any, body: bytes, is_async: bool) -> Exchange | None:
        """The exchange, served from the recording in a replay. `None` when
        it cannot be recorded; it still goes ahead. Raises when the replay
        diverges."""
        session = None if self.run._ended else self.run._replay
        try:
            provider = self.provider or detect(request)
            exchange = Exchange(self, request, body, provider)
        except Exception:
            if session is not None:
                raise
            log.exception("Kasanki Helium: cannot record a model call")
            return None
        if session is None:
            return exchange
        answer = session.ask("llm.call", exchange.key())
        if answer["answer"] == "serve":
            exchange.served = self._serve(exchange, answer["value"], session, is_async)
        else:
            # After the fork point: the real call, with the fork's change.
            exchange.change(session.change)
        return exchange

    def _serve(
        self,
        exchange: Exchange,
        recorded: dict[str, Any],
        session: Session,
        is_async: bool,
    ) -> Any:
        """The recorded response, as the transport would have given it."""
        from ._wrapper import _filled

        payload = _filled(recorded, session)
        wire = payload.get("http") or {}
        error = payload.get("error")
        status = wire.get("status")
        _note_version(session, wire.get("package"), exchange.package)
        if status is None:
            # No answer came: a timeout or a broken connection.
            raise _transport_error(self.http, error, exchange.request)
        pieces: list[bytes]
        if wire.get("body"):
            content = session.blob(wire["body"])
            pieces = _cut(content, wire.get("pieces"))
        elif payload.get("stream") is not None and exchange.provider is not None:
            # A fork's edited answer: the stream rebuilt
            # from it, as the package would have read it.
            events = exchange.provider.events(exchange.api, payload.get("response"))
            pieces = [exchange.provider.sse(events).encode("utf-8")]
        elif status >= 400:
            body = (error or {}).get("body")
            pieces = [_encoded(body)]
        else:
            pieces = [_encoded(payload.get("response"))]
        broke = None
        if error is not None and status < 400:
            broke = _transport_error(self.http, error, exchange.request)
        stream = (self.kinds.AsyncReplayed if is_async else self.kinds.Replayed)(
            [p for p in pieces if p], broke
        )
        return self.http.Response(
            status,
            headers=[(k, v) for k, v in wire.get("response_headers") or []],
            stream=stream,
            request=exchange.request,
        )


def detect(request: Any) -> Provider | None:
    """The provider a request is for, when the client did not say: by the
    headers each package sends, then by path."""
    from ._anthropic import Anthropic
    from ._google import Google
    from ._openai import OpenAI

    headers = request.headers
    agent = headers.get("user-agent", "")
    goog_client = headers.get("x-goog-api-client", "")
    if "anthropic-version" in headers or agent.startswith("Anthropic/"):
        return Anthropic()
    if agent.startswith("OpenAI/"):
        return OpenAI()
    if goog_client.startswith("google-genai-sdk/") or agent.startswith(
        "google-genai-sdk/"
    ):
        return Google()
    for provider in (OpenAI(), Anthropic(), Google()):
        if provider.api_for(request.method, request.url.path, "", {}) is not None:
            return provider
    return None


class Exchange:
    """One HTTP exchange, from its request to its ``llm.call`` event."""

    def __init__(
        self, recorder: Recorder, request: Any, body: bytes, provider: Provider | None
    ) -> None:
        self.run = recorder.run
        self.http = recorder.http
        self.provider = provider
        self.started = time.time_ns()
        # Fine-grained: the monotonic clock ticks every ~16 ms on Windows.
        self.begin = time.perf_counter_ns()
        self.request = request
        self.body = body
        url = request.url
        self.method = str(request.method)
        self.path = str(url.path)
        query = url.query
        self.query = query.decode("ascii") if isinstance(query, bytes) else str(query)
        port = url.port
        self.server = str(url.host) + ("" if port is None else f":{port}")
        self.headers = request_headers(request.headers)
        self.content_type = request.headers.get("content-type", "")
        found = (
            None
            if provider is None
            else provider.api_for(self.method, self.path, self.query, self.headers)
        )
        if found is None:
            where = "" if provider is None else f"{provider.name} "
            warn_once(
                f"{where}plain http",
                f"Kasanki Helium records {where}{self.method} {self.path} as a plain "
                "HTTP exchange: it is replayed, but not summarised as a model call.",
                stacklevel=6,
            )
        self.api = found or PLAIN
        self.package = _package(provider, request)
        self.capture = self.run._streaming
        self.pieces: list[bytes] = []
        self.times: list[int] = []
        self.status: int | None = None
        self.response_headers: list[tuple[str, str]] = []
        #: In a replay, the recorded response to hand back.
        self.served: Any = None
        self._lock = threading.Lock()
        self._done = False

    # What the call is matched on (W16, R15).

    def _head(self) -> dict[str, Any]:
        return {
            "provider": None if self.provider is None else self.provider.name,
            "api": self.api,
            "server": self.server,
            "request": None,
            "options": None,
            "response": None,
            "stream": None,
            "summary": _no_summary(),
            "error": None,
            "blobbed": {},
            "started": wall_time(self.started),
            "ended": None,
            "duration_ns": None,
            "http": {
                "method": self.method,
                "path": self.path,
                "query": self.query,
                "headers": self.headers,
                "request_body": None,
                "status": None,
                "response_headers": None,
                "body": None,
                "pieces": None,
                "package": self.package,
            },
        }

    def key(self) -> dict[str, Any]:
        """What the engine matches this call on, made exactly as it is
        recorded: secrets stripped, large parts as blobs."""
        data = redact(to_json(self._head(), set(), depth=2))
        _request_part(data, self.body, self.content_type)
        _llm.split(data)
        wire = data["http"]
        return {
            "provider": data["provider"],
            "api": data["api"],
            "request": data["request"],
            "blobbed": data["blobbed"],
            "http": {
                k: wire[k]
                for k in (
                    "method",
                    "path",
                    "query",
                    "headers",
                    "request_body",
                    "package",
                )
            },
        }

    def change(self, change: Change | None) -> None:
        """A fork's change to model calls, made to the request body."""
        if (
            change is None
            or not change.changes_model_calls
            or self.provider is None
            or self.api == PLAIN
        ):
            return

        try:
            body = json.loads(self.body)
        except ValueError:
            return
        if not isinstance(body, dict):
            return
        new_path = self.provider.change_request(body, self.path, change)
        content = json.dumps(body).encode("utf-8")
        headers = self.request.headers.copy()
        del headers["content-length"]
        url = self.request.url
        if new_path != self.path:
            url = url.copy_with(path=new_path)
            self.path = new_path
        self.request = self.http.Request(
            self.request.method,
            url,
            headers=headers,
            content=content,
            extensions=self.request.extensions,
        )
        self.body = content

    # How the exchange goes.

    def answered(self, response: Any, capture: Any) -> Any:
        """The response the package gets: the same, its body recorded as it
        is read."""
        self.status = response.status_code
        self.response_headers = list(response.headers.multi_items())
        self.run._open_call(self)
        return self.http.Response(
            response.status_code,
            headers=response.headers,
            stream=capture(response.stream, self),
            request=self.request,
            extensions=response.extensions,
        )

    def failed(self, error: BaseException) -> None:
        self.ended(complete=False, error=error)

    def piece(self, data: bytes) -> None:
        self.pieces.append(data)
        self.times.append(time.perf_counter_ns() - self.begin)

    def abandon(self) -> None:
        """At the run's end: a response nobody finished reading."""
        self.ended(complete=False)

    def ended(self, complete: bool, error: BaseException | None = None) -> None:
        with self._lock:
            if self._done:
                return
            self._done = True
        self.run._close_call(self)
        try:
            payload = self._head()
            payload["ended"] = wall_time(time.time_ns())
            payload["duration_ns"] = str(time.perf_counter_ns() - self.begin)
            wire = payload["http"]
            wire["status"] = self.status
            wire["response_headers"] = [
                [k.lower(), v]
                for k, v in self.response_headers
                if k.lower() not in _RESPONSE_LEFT_OUT and not _secret_header(k)
            ]
            if error is not None:
                payload["error"] = {
                    "type": type(error).__qualname__,
                    "message": str(error),
                    "status": None,
                    "body": None,
                }
            raw = b"".join(self.pieces)
            finish = functools.partial(
                _finish,
                provider=self.provider,
                http=self.http.__name__,
                body=self.body,
                content_type=self.content_type,
                raw=raw,
                pieces=[len(p) for p in self.pieces],
                times=list(self.times),
                complete=complete,
                capture=self.capture,
            )
            self.run._record(
                "llm.call", payload, split=finish, size=len(raw) + len(self.body)
            )
        except RuntimeError as e:
            warn_once(
                "llm.call after run end",
                f"Kasanki Helium did not record a model call: {e}.",
                stacklevel=5,
            )
        except Exception:
            log.exception("Kasanki Helium: cannot record a model call")


# Turning the exchange into an event, on the SDK's own thread.


def _finish(
    data: dict[str, Any],
    *,
    provider: Provider | None,
    http: str,
    body: bytes,
    content_type: str,
    raw: bytes,
    pieces: list[int],
    times: list[int],
    complete: bool,
    capture: str,
) -> list[tuple[Blob, str, str]]:
    """Fills in an ``llm.call`` payload, already stripped of secrets, from
    the bytes that crossed the wire, and moves its large parts into blobs.
    Read leniently: a field nobody knows never stops the recording (W14)."""
    blobs = _request_part(data, body, content_type)
    wire = data["http"]
    api = data["api"]
    headers = {k: v for k, v in wire["response_headers"] or []}
    kind = headers.get("content-type", "").split(";")[0].strip().lower()
    decoded = _decoded(raw, headers.get("content-encoding", ""), http)
    if decoded is None:
        # An encoding this SDK cannot undo: kept as it came, and said so.
        decoded = raw
    else:
        wire["response_headers"] = [
            [k, v] for k, v in wire["response_headers"] if k != "content-encoding"
        ]
        if headers.get("content-encoding", "identity") not in ("", "identity"):
            pieces = []
    status = wire["status"]
    response: Any = None
    if kind in _STREAMS:
        content = decoded
        if kind == "text/event-stream":
            text = decoded.decode("utf-8", errors="replace")
            stripped = redact_text(text)
            if stripped != text:
                # Secrets came in the stream: what is kept differs from what
                # came, so its pieces no longer fit it.
                content, pieces = stripped.encode("utf-8"), []
            events = sse_events(stripped)
            # The package closes a stream once its last event came, before
            # the bytes run out: that is the end too.
            complete = complete or _ended(stripped, events)
            if provider is not None and api != PLAIN:
                response = _lenient(provider.assemble, api, events)
        found = _blob(content, kind)
        if found is not None:
            blobs.append(found)
            wire["body"] = found[0].hash
        timed = capture != "final_only"
        wire["pieces"] = pieces if timed and pieces else None
        data["stream"] = {
            "capture": capture,
            "chunks": len(times),
            "chunk_times_ns": [str(t) for t in times] if timed else None,
            "complete": complete,
        }
    else:
        parsed = _json(decoded)
        if parsed is _NOT_JSON:
            # Not JSON (a file, audio, a gateway's page): kept as bytes.
            readable = _text(decoded)
            kept = decoded if readable is None else redact_text(readable).encode()
            found = _blob(kept, kind or "application/octet-stream")
            if found is not None:
                blobs.append(found)
                wire["body"] = found[0].hash
            parsed = None
        else:
            parsed = redact(parsed)
            if status is not None and status < 400:
                response = parsed
        if status is not None and status >= 400:
            data["error"] = {
                "type": "HTTPStatusError",
                "message": _error_message(parsed, status),
                "status": status,
                "body": parsed,
            }
    data["response"] = response
    if provider is not None and api != PLAIN:
        summary = _lenient(provider.summary, api, response) or _no_summary()
        if (
            provider.name == "google"
            and summary.get("model") is None
            and hasattr(provider, "model_of")
        ):
            summary["model"] = provider.model_of(wire.get("path", ""))
        data["summary"] = summary
    blobs.extend(_llm.split(data))
    return blobs


def _request_part(
    data: dict[str, Any], body: bytes, content_type: str
) -> list[tuple[Blob, str, str]]:
    """The request body: its JSON as ``request`` (secrets stripped), or the
    bytes as a blob named in ``http.request_body``."""
    if not body:
        return []
    parsed = _json(body)
    if parsed is not _NOT_JSON:
        data["request"] = redact(parsed)
        return []
    match = _BOUNDARY_PARAM.search(content_type)
    if match is not None:
        body = body.replace(match.group(1).encode("latin-1"), _BOUNDARY.encode())
    kind = content_type.split(";")[0].strip().lower() or "application/octet-stream"
    found = _blob(body, kind, role="request")
    if found is None:
        return []
    data["http"]["request_body"] = found[0].hash
    return [found]


def _blob(
    content: bytes, kind: str, role: str = "response"
) -> tuple[Blob, str, str] | None:
    if not content:
        return None
    if len(content) > MAX_BLOB:
        log.warning(
            "Kasanki Helium: a model call's body of %d bytes is over the blob limit "
            "and was not recorded",
            len(content),
        )
        return None
    media = kind if _MEDIA_TYPE.match(kind) else "application/octet-stream"
    return (Blob.of(content), media, role)


_NOT_JSON: Any = object()


def _json(content: bytes) -> Any:
    if not content or content[:1] not in b"{[":
        return _NOT_JSON
    try:
        return json.loads(content)
    except ValueError:
        return _NOT_JSON


def _text(content: bytes) -> str | None:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _decoded(raw: bytes, encoding: str, http: str) -> bytes | None:
    """`raw` with its content encodings undone, by the HTTP library's own
    decoders where it has them; `None` for an encoding none can undo."""
    codings = [c.strip().lower() for c in encoding.split(",") if c.strip()]
    codings = [c for c in codings if c != "identity"]
    if not codings:
        return raw
    try:
        decoders = importlib.import_module(f"{http}._decoders").SUPPORTED_DECODERS
    except (ImportError, AttributeError):
        decoders = {}
    out = raw
    for coding in reversed(codings):
        try:
            if coding in decoders:
                decoder = decoders[coding]()
                out = b"".join(decoder.decode(out)) + b"".join(decoder.flush())
            elif coding in ("gzip", "x-gzip"):
                out = zlib.decompress(out, zlib.MAX_WBITS | 16)
            elif coding == "deflate":
                out = zlib.decompress(out)
            else:
                return None
        except Exception:
            log.debug("cannot undo the %s encoding", coding, exc_info=True)
            return None
    return out


def _lenient(read: Any, api: str, value: Any) -> Any:
    try:
        return read(api, value)
    except Exception:
        log.debug("cannot read the model call's %s answer", api, exc_info=True)
        return None


def _error_message(body: Any, status: int) -> str:
    error = body.get("error") if isinstance(body, dict) else None
    for found in (error, body):
        if isinstance(found, dict) and isinstance(found.get("message"), str):
            return str(found["message"])
    if isinstance(error, str):
        return error
    return f"HTTP {status}"


#: The events a stream ends with.
_LAST_EVENTS = frozenset(
    {"message_stop", "response.completed", "response.incomplete", "response.failed"}
)


def _ended(text: str, events: list[Any]) -> bool:
    """Whether a stream's last event came: OpenAI's ``[DONE]``, or an event
    that ends a message or a response."""
    if text.rstrip().endswith("data: [DONE]"):
        return True
    last = events[-1] if events else None
    return isinstance(last, dict) and last.get("type") in _LAST_EVENTS


def sse_events(text: str) -> list[Any]:
    """The JSON of each server-sent event, leniently: what is not JSON (an
    OpenAI stream's ``[DONE]``, a comment) is skipped."""
    events: list[Any] = []
    data: list[str] = []

    def done() -> None:
        if data:
            joined = "\n".join(data)
            data.clear()
            if joined.startswith(("{", "[")):
                try:
                    events.append(json.loads(joined))
                except ValueError:
                    pass

    for line in text.splitlines():
        if not line:
            done()
        elif line.startswith("data:"):
            value = line[5:]
            data.append(value.removeprefix(" "))
    done()
    return events


# Headers.


def _secret_header(name: str) -> bool:
    return name.lower() in _SECRET_HEADERS or is_secret_key(name)


def request_headers(headers: Any) -> dict[str, str]:
    """The request headers that can change the answer: all but the
    package's identity, per-call IDs, retry counters, framing and secrets."""
    kept: dict[str, str] = {}
    for name, value in headers.multi_items():
        name = name.lower()
        if (
            name in _LEFT_OUT
            or name.startswith(_LEFT_OUT_PREFIXES)
            or _secret_header(name)
        ):
            continue
        if name == "content-type":
            value = _BOUNDARY_PARAM.sub(f"boundary={_BOUNDARY}", value)
        kept[name] = f"{kept[name]}, {value}" if name in kept else value
    return dict(sorted(kept.items()))


def _package(provider: Provider | None, request: Any) -> dict[str, str] | None:
    """The provider package that sent the request, and its version."""
    if provider is None:
        return None
    version = getattr(sys.modules.get(provider.module_name()), "__version__", None)
    if not version:
        # "OpenAI/Python 3.37.0"
        agent = str(request.headers.get("user-agent", ""))
        version = agent.rpartition(" ")[2] or None
    return {"name": provider.name, "version": str(version)}


# Replay (W15, W16).


def _cut(content: bytes, pieces: Any) -> list[bytes]:
    """`content` in the recorded pieces, or in one."""
    if not isinstance(pieces, list) or not pieces:
        return [content]
    out = []
    at = 0
    for size in pieces:
        out.append(content[at : at + int(size)])
        at += int(size)
    if at < len(content):
        out.append(content[at:])
    return out


def _encoded(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, str):
        return value.encode("utf-8")
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


def _transport_error(http: ModuleType, error: Any, request: Any) -> BaseException:
    """The recorded transport error, as the library's own exception."""
    error = error or {}
    name = str(error.get("type") or "")
    kind = getattr(http, name, None) if name.isidentifier() else None
    if not (isinstance(kind, type) and issubclass(kind, http.TransportError)):
        kind = http.ReadTimeout if "Timeout" in name else http.ConnectError
    made: BaseException = kind(str(error.get("message") or ""), request=request)
    return made


def _note_version(session: Session, recorded: Any, now: Any) -> None:
    """W16: a package version that differs from the recording's is a note of
    its own, never a divergence by itself."""
    if not isinstance(recorded, dict) or not isinstance(now, dict):
        return
    if recorded.get("version") == now.get("version"):
        return
    note = (
        f"{recorded.get('name')} {recorded.get('version')} made the recording and "
        f"{now.get('version')} replays it; each call still matched on what it sent"
    )
    notes = (session.root or session).notes
    if note not in notes:
        notes.append(note)
