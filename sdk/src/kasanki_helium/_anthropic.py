# Copyright © 2026 Kasanki Labs.
# SPDX-License-Identifier: AGPL-3.0-only
# Licensed under the GNU AGPL v3 with an attribution term: see LICENSE and CREDIT.md.

"""Recording Anthropic model calls: ``run.wrap(client)``.

`Anthropic`, `AsyncAnthropic`, and the package's other clients (Bedrock,
Vertex, Foundry, AWS, Google Cloud): every one is built on the package's
``SyncAPIClient`` or ``AsyncAPIClient``. What all wrappers share is in
``_wrapper``.
"""

from __future__ import annotations

import importlib
import json
import logging
import re
from collections.abc import Mapping
from typing import Any, ClassVar

from ._wrapper import Provider, StreamState, classes, get, to_dict

log = logging.getLogger("kasanki_helium")

_STOP = {
    "end_turn": "end",
    "stop_sequence": "end",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_use",
    "refusal": "refusal",
}

#: Usage fields that together are everything the model read.
_INPUT = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


class Anthropic(Provider):
    name = "anthropic"
    recorded: ClassVar[dict[tuple[str, ...], str]] = {
        ("messages", "create"): "messages",
        ("messages", "parse"): "messages.parse",
        ("messages", "stream"): "messages.stream",
        ("messages", "count_tokens"): "messages.count_tokens",
        ("beta", "messages", "create"): "beta.messages",
        ("beta", "messages", "parse"): "beta.messages.parse",
        ("beta", "messages", "stream"): "beta.messages.stream",
        ("beta", "messages", "count_tokens"): "beta.messages.count_tokens",
    }
    managers = frozenset({"messages.stream", "beta.messages.stream"})
    schema_keys = frozenset({"output_format"})
    known_major = 2
    paths: ClassVar[tuple[tuple[re.Pattern[str], str], ...]] = (
        (re.compile(r"/messages/count_tokens$"), "messages.count_tokens"),
        (re.compile(r"/messages$"), "messages"),
        # Bedrock, and Google Cloud's workspaces.
        (re.compile(r"/invoke(-with-response-stream)?$"), "messages"),
        # Vertex.
        (re.compile(r"/count-tokens:rawPredict$"), "messages.count_tokens"),
        (re.compile(r":(stream)?[rR]awPredict$"), "messages"),
    )

    def recognises(self, client: Any) -> bool:
        found = classes(client)
        return ("anthropic", "SyncAPIClient") in found or self.is_async(client)

    def api_for(
        self, method: str, path: str, query: str, headers: Mapping[str, str]
    ) -> str | None:
        api = super().api_for(method, path, query, headers)
        # The beta API: the same path, asked for by query or by header.
        if api is not None and (
            "beta=true" in query.split("&") or "anthropic-beta" in headers
        ):
            return f"beta.{api}"
        return api

    def assemble(self, api: str, events: list[Any]) -> Any:
        if api.endswith("count_tokens"):
            return None
        return _message_answer(events)

    def is_async(self, client: Any) -> bool:
        return ("anthropic", "AsyncAPIClient") in classes(client)

    def schema(self, model: Any, key: str) -> Any:
        """``output_format``: the format the package puts in ``output_config``."""
        try:
            pydantic = importlib.import_module("pydantic")
            transform = importlib.import_module("anthropic.lib._parse._transform")
            schema = pydantic.TypeAdapter(model).json_schema()
            return {"type": "json_schema", "schema": transform.transform_schema(schema)}
        except Exception:
            log.debug("the package's schema helper failed", exc_info=True)
            return {"type": "json_schema", "schema": None, "name": repr(model)}

    def response(self, response: Any) -> Any:
        """``parsed_output`` values are the package's own work, rebuilt from
        the content, so they are left out."""
        data = to_dict(response)
        if isinstance(data, dict):
            data.pop("parsed_output", None)
            for block in data.get("content") or []:
                if isinstance(block, dict):
                    block.pop("parsed_output", None)
        return data

    def stream_state(self, api: str, kwargs: dict[str, Any]) -> StreamState | None:
        if api.endswith("count_tokens"):
            return None
        return _MessageState(beta=api.startswith("beta."))

    # Replay (replay-harness R14).

    def events(self, api: str, response: Any) -> list[Any]:
        if not isinstance(response, dict):
            return []
        usage = response.get("usage") or {}
        events: list[dict[str, Any]] = [
            {
                "type": "message_start",
                "message": {
                    **response,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {**usage, "output_tokens": 0},
                },
            }
        ]
        for i, block in enumerate(response.get("content") or []):
            kind = block.get("type")
            deltas: list[dict[str, Any]] = []
            if kind == "text":
                start = {**block, "text": ""}
                deltas.append({"type": "text_delta", "text": block.get("text", "")})
            elif kind in ("tool_use", "server_tool_use", "mcp_tool_use"):
                start = {**block, "input": {}}
                deltas.append(
                    {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(block.get("input") or {}),
                    }
                )
            elif kind == "thinking":
                start = {**block, "thinking": "", "signature": ""}
                deltas.append(
                    {"type": "thinking_delta", "thinking": block.get("thinking", "")}
                )
                deltas.append(
                    {"type": "signature_delta", "signature": block.get("signature", "")}
                )
            else:
                start = block
            events.append(
                {"type": "content_block_start", "index": i, "content_block": start}
            )
            events.extend(
                {"type": "content_block_delta", "index": i, "delta": d} for d in deltas
            )
            events.append({"type": "content_block_stop", "index": i})
        events.append(
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": response.get("stop_reason"),
                    "stop_sequence": response.get("stop_sequence"),
                },
                "usage": {"output_tokens": usage.get("output_tokens", 0)},
            }
        )
        events.append({"type": "message_stop"})
        return events

    def sse(self, events: list[Any]) -> str:
        return "".join(
            f"event: {e.get('type')}\ndata: {json.dumps(e)}\n\n" for e in events
        )

    def error_body(self, error: dict[str, Any]) -> Any:
        body = error.get("body")
        if body is None:
            return {
                "type": "error",
                "error": {"type": "api_error", "message": error.get("message")},
            }
        return body

    def set_system(self, api: str, kwargs: dict[str, Any], prompt: str) -> None:
        kwargs["system"] = prompt

    def summary(self, api: str, response: Any) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "model": get(response, "model"),
            "input_tokens": None,
            "output_tokens": None,
            "stop_reason": None,
            "tool_calls": [],
        }
        if api.endswith("count_tokens"):
            summary["input_tokens"] = get(response, "input_tokens")
            return summary
        usage = get(response, "usage")
        if get(usage, "input_tokens") is not None:
            summary["input_tokens"] = sum(get(usage, k) or 0 for k in _INPUT)
        summary["output_tokens"] = get(usage, "output_tokens")
        # Read from the cache: inside input_tokens, as Anthropic counts it apart.
        if get(usage, "cache_read_input_tokens") is not None:
            summary["cached_input_tokens"] = get(usage, "cache_read_input_tokens")
        summary["stop_reason"] = _STOP.get(get(response, "stop_reason"))
        for block in get(response, "content") or []:
            # Tools the agent runs (tool_use) and those the server runs
            # (server_tool_use, mcp_tool_use).
            if (get(block, "type") or "").endswith("tool_use") and get(block, "name"):
                summary["tool_calls"].append(get(block, "name"))
        return summary


def _message_answer(events: list[Any]) -> Any:
    """A streamed message's events put together as the message the same call
    gives unstreamed, so far as it came."""
    message: dict[str, Any] | None = None
    json_parts: dict[int, str] = {}
    blocks: dict[int, dict[str, Any]] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind == "message_start" and isinstance(event.get("message"), dict):
            message = dict(event["message"])
            continue
        if message is None:
            continue
        index = event.get("index")
        if kind == "content_block_start" and isinstance(index, int):
            blocks[index] = dict(event.get("content_block") or {})
        elif kind == "content_block_delta" and index in blocks:
            block, delta = blocks[index], event.get("delta") or {}
            step = delta.get("type")
            if step == "text_delta":
                block["text"] = (block.get("text") or "") + str(delta.get("text", ""))
            elif step == "input_json_delta":
                json_parts[index] = json_parts.get(index, "") + str(
                    delta.get("partial_json", "")
                )
            elif step == "thinking_delta":
                block["thinking"] = (block.get("thinking") or "") + str(
                    delta.get("thinking", "")
                )
            elif step == "signature_delta":
                block["signature"] = delta.get("signature")
            elif step == "citations_delta" and delta.get("citation") is not None:
                block.setdefault("citations", []).append(delta["citation"])
        elif kind == "message_delta":
            for key, value in (event.get("delta") or {}).items():
                message[key] = value
            usage = event.get("usage")
            if isinstance(usage, dict):
                message["usage"] = {
                    **(message.get("usage") or {}),
                    **{k: v for k, v in usage.items() if v is not None},
                }
    if message is None:
        return None
    for index, text in json_parts.items():
        try:
            blocks[index]["input"] = json.loads(text) if text else {}
        except ValueError:
            # Cut off inside the JSON: kept as it came.
            blocks[index]["input"] = text
    message["content"] = [blocks[i] for i in sorted(blocks)]
    return message


class _MessageState(StreamState):
    """The package's own accumulator for streamed messages."""

    def __init__(self, beta: bool) -> None:
        module = "_beta_messages" if beta else "_messages"
        streaming = importlib.import_module(f"anthropic.lib.streaming.{module}")
        self.accumulate = streaming.accumulate_event
        self.beta = beta
        self.snapshot: Any = None
        self.json_bufs: dict[int, bytes] = {}
        self.headers: Any = None

    def attach(self, stream: Any) -> None:
        if self.beta:
            self.headers = stream.response.request.headers

    def handle(self, chunk: Any) -> None:
        if self.snapshot is None and getattr(chunk, "type", None) != "message_start":
            return
        given: dict[str, Any] = {
            "event": chunk,
            "current_snapshot": self.snapshot,
            "json_bufs": self.json_bufs,
        }
        if self.beta:
            given["request_headers"] = self.headers
        self.snapshot = self.accumulate(**given)

    def response(self, complete: bool) -> Any:
        return self.snapshot
