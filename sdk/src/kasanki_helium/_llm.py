# Copyright © 2026 Kasanki Labs.
# SPDX-License-Identifier: AGPL-3.0-only
# Licensed under the GNU AGPL v3 with an attribution term: see LICENSE and CREDIT.md.

"""Wrapped model clients: what every provider shares.

``run.wrap(client)`` picks the provider by the client's class, and records the
client's model calls at its HTTP layer (``_http``). Each call becomes
one ``llm.call`` event, whose large parts are stored as blobs (§5): a long
conversation repeats every earlier message on each call, so each message is
stored once.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import TYPE_CHECKING, Any

from ._wire import MAX_BLOB, MAX_BODY, Blob, dumps

if TYPE_CHECKING:
    from ._run import Run

log = logging.getLogger("kasanki_helium")

#: Request parts at least this long (canonical JSON) become blobs (W5).
REQUEST_BLOB_BYTES = 1024

#: A response at least this long becomes a blob.
RESPONSE_BLOB_BYTES = 16 * 1024

#: Request fields whose elements, or whose value, become blobs.
_LISTS = ("messages", "input", "tools", "contents")
_VALUES = ("system", "instructions", "systemInstruction")

#: Room left for the chain fields the engine adds to an event.
_EVENT_ROOM = MAX_BODY - 4096

#: How streamed calls are recorded (D1, W7). The first is the default.
STREAMING = ("final_and_timings", "every_chunk", "final_only")

_JSON = "application/json"


#: Records model calls at the provider package's surface, as the SDK did
#: before the HTTP recording layer was added. Only for tests of runs recorded that way: a
#: replay or fork of one takes that path by itself.
SURFACE = False

#: What a run's context says when its model calls are recorded at the HTTP
#: layer, so a replay knows which way to serve them.
MODEL_CALLS = "http"


def at_surface(run: Run) -> bool:
    """Whether `run` records and replays model calls at the package's
    surface: a replay of a run recorded before §13 does."""
    session = run._replay
    if session is not None and session.run_id:
        return session.context.get("model_calls") != MODEL_CALLS
    return SURFACE


def wrap(run: Run, client: Any) -> Any:
    from . import _anthropic, _google, _http, _langchain, _openai, _wrapper

    if _google.Google().recognises(client):
        return _google.wrap(run, client)

    for provider in (_openai.OpenAI(), _anthropic.Anthropic()):
        if provider.recognises(client):
            if at_surface(run):
                return _wrapper.wrap(run, client, provider)
            _wrapper.check_version(provider)
            return _http.wrap(run, client, provider)
    copy = _langchain.wrap(run, client)
    if copy is not None:
        return copy
    raise TypeError(
        "Kasanki Helium can wrap OpenAI clients (OpenAI, AsyncOpenAI and those "
        "built on them), Anthropic clients (Anthropic, AsyncAnthropic, and the "
        "Bedrock, Vertex, Foundry and AWS ones), Google clients (genai.Client "
        "and its .aio) and LangChain models (ChatOpenAI and those built on it, "
        "ChatAnthropic, OpenAIEmbeddings, and these with arguments bound); got "
        f"{type(client).__module__}.{type(client).__qualname__}"
    )


def canonical(value: Any) -> bytes:
    """A blob's content: JSON with sorted keys, so equal values are equal
    bytes and share one blob."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


#: Messages already made into blobs, by their fields: a conversation sends
#: every earlier message again on each call. Bounded by their total size.
_SEEN_BYTES = 16 * 1024 * 1024
_seen: dict[tuple[Any, ...], Blob] = {}
_seen_bytes = 0
_seen_lock = threading.Lock()


def _seen_key(value: Any) -> tuple[Any, ...] | None:
    """A key for a flat message (text fields only), or `None`."""
    if not isinstance(value, dict):
        return None
    items = tuple(value.items())
    for _, v in items:
        if not (v is None or isinstance(v, str)):
            return None
    return items


def _remember(key: tuple[Any, ...], blob: Blob) -> None:
    global _seen_bytes
    with _seen_lock:
        if key in _seen:
            return
        _seen[key] = blob
        _seen_bytes += len(blob.content)
        # Oldest first: dicts keep insertion order.
        while _seen_bytes > _SEEN_BYTES and _seen:
            old = next(iter(_seen))
            _seen_bytes -= len(_seen.pop(old).content)


def split(payload: dict[str, Any]) -> list[tuple[Blob, str, str]]:
    """Moves an ``llm.call`` payload's large parts into blobs (W5): each
    replaced by ``null``, its JSON pointer mapped to the blob's hash in
    ``blobbed``. Returns the blobs, as (blob, media_type, role)."""
    blobs: dict[str, tuple[Blob, str, str]] = {}
    blobbed: dict[str, str] = payload["blobbed"]
    # (container, key, original): to put a request back whole.
    moved: list[tuple[Any, Any, Any]] = []

    def move(container: Any, key: Any, pointer: str, role: str, least: int) -> None:
        value = container[key]
        if value is None:
            return
        seen = _seen_key(value)
        blob = _seen.get(seen) if seen is not None else None
        if blob is None:
            content = canonical(value)
            if len(content) < least:
                return
            if len(content) > MAX_BLOB:
                container[key] = f"[{len(content)} bytes, not recorded]"
                log.warning(
                    "Kasanki Helium: a model call's %s is over the blob limit and "
                    "was not recorded",
                    pointer,
                )
                return
            blob = Blob.of(content)
            if seen is not None:
                _remember(seen, blob)
        blobs.setdefault(blob.hash, (blob, _JSON, role))
        blobbed[pointer] = blob.hash
        if role == "request":
            moved.append((container, key, value))
        container[key] = None

    request = payload.get("request")
    if isinstance(request, dict):
        for name in _LISTS:
            items = request.get(name)
            if isinstance(items, list):
                for i in range(len(items)):
                    move(
                        items, i, f"/request/{name}/{i}", "request", REQUEST_BLOB_BYTES
                    )
        for name in _VALUES:
            if name in request:
                move(request, name, f"/request/{name}", "request", REQUEST_BLOB_BYTES)
    if "response" in payload:
        move(payload, "response", "/response", "response", RESPONSE_BLOB_BYTES)

    if isinstance(request, dict) and len(dumps(payload)) > _EVENT_ROOM:
        # Still too big (a long list of small parts): the request is one blob.
        for container, key, value in moved:
            container[key] = value
        for pointer in [p for p in blobbed if p.startswith("/request/")]:
            del blobbed[pointer]
        move(payload, "request", "/request", "request", 0)
        kept = set(blobbed.values())
        blobs = {h: b for h, b in blobs.items() if h in kept}
    return list(blobs.values())
