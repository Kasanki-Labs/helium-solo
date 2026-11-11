# Copyright © 2026 Kasanki Labs.
# SPDX-License-Identifier: AGPL-3.0-only
# Licensed under the GNU AGPL v3 with an attribution term: see LICENSE and CREDIT.md.

"""Recording OpenAI model calls: ``run.wrap(client)``.

`OpenAI`, `AsyncOpenAI`, and the package's clients built on them (Azure,
Bedrock). What all wrappers share is in ``_wrapper``.
"""

from __future__ import annotations

import importlib
import json
import logging
import re
from typing import Any, ClassVar

from ._wrapper import Provider, StreamState, absent, classes, get, to_dict

log = logging.getLogger("kasanki_helium")

_STOP = {
    "stop": "end",
    "length": "length",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "content_filter",
}


class OpenAI(Provider):
    name = "openai"
    recorded: ClassVar[dict[tuple[str, ...], str]] = {
        ("chat", "completions", "create"): "chat.completions",
        ("chat", "completions", "parse"): "chat.completions.parse",
        ("chat", "completions", "stream"): "chat.completions.stream",
        ("beta", "chat", "completions", "stream"): "chat.completions.stream",
        ("responses", "create"): "responses",
        ("responses", "parse"): "responses.parse",
        ("responses", "stream"): "responses.stream",
        ("embeddings", "create"): "embeddings",
    }
    managers = frozenset({"chat.completions.stream", "responses.stream"})
    known_major = 3
    # By the end of the path: OpenAI's own, Azure's deployment paths and
    # OpenAI-compatible servers alike.
    paths: ClassVar[tuple[tuple[re.Pattern[str], str], ...]] = (
        (re.compile(r"/chat/completions$"), "chat.completions"),
        (re.compile(r"/responses$"), "responses"),
        (re.compile(r"/embeddings$"), "embeddings"),
    )

    def recognises(self, client: Any) -> bool:
        found = classes(client)
        return ("openai", "OpenAI") in found or ("openai", "AsyncOpenAI") in found

    def is_async(self, client: Any) -> bool:
        return ("openai", "AsyncOpenAI") in classes(client)

    def schema(self, model: type, key: str) -> Any:
        try:
            if key == "text_format":
                responses = importlib.import_module("openai.lib._parsing._responses")
                return responses.type_to_text_format_param(model)
            completions = importlib.import_module("openai.lib._parsing._completions")
            return completions.type_to_response_format_param(model)
        except Exception:
            log.debug("the package's schema helper failed", exc_info=True)
            schema: Any = model.model_json_schema()  # type: ignore[attr-defined]
            return {"type": "json_schema", "name": model.__name__, "schema": schema}

    def response(self, response: Any) -> Any:
        """``parsed`` values are the package's own work, rebuilt from the
        content, so they are left out."""
        data = to_dict(response)
        if isinstance(data, dict):
            for choice in data.get("choices") or []:
                if isinstance(choice, dict) and isinstance(choice.get("message"), dict):
                    choice["message"].pop("parsed", None)
            for item in data.get("output") or []:
                for part in item.get("content") or [] if isinstance(item, dict) else []:
                    if isinstance(part, dict):
                        part.pop("parsed", None)
        return data

    def stream_state(self, api: str, kwargs: dict[str, Any]) -> StreamState | None:
        if api in ("chat.completions", "chat.completions.stream"):
            module = importlib.import_module("openai.lib.streaming.chat")
            given = {
                "input_tools": kwargs.get("tools"),
                "response_format": kwargs.get("response_format"),
            }
            return _ChatState(
                module.ChatCompletionStreamState(
                    **{
                        k: v
                        for k, v in given.items()
                        if v is not None and not absent(v)
                    }
                )
            )
        if api in ("responses", "responses.stream"):
            return _ResponsesState()
        return None

    def assemble(self, api: str, events: list[Any]) -> Any:
        if api.startswith("responses"):
            # Its last event that carries the response (completed, failed…).
            last = None
            for event in events:
                if isinstance(event, dict) and isinstance(event.get("response"), dict):
                    last = event["response"]
            return last
        if api.startswith("chat.completions"):
            return _chat_answer(events)
        return None

    # Replay (replay-harness R14).

    def events(self, api: str, response: Any) -> list[Any]:
        if not isinstance(response, dict):
            return []
        if api.startswith("responses"):
            events: list[dict[str, Any]] = [
                {
                    "type": "response.created",
                    "response": {**response, "status": "in_progress", "output": []},
                }
            ]
            for i, item in enumerate(response.get("output") or []):
                events.append(
                    {
                        "type": "response.output_item.added",
                        "output_index": i,
                        "item": item,
                    }
                )
                events.append(
                    {
                        "type": "response.output_item.done",
                        "output_index": i,
                        "item": item,
                    }
                )
            events.append({"type": "response.completed", "response": response})
            for number, event in enumerate(events):
                event["sequence_number"] = number
            return events
        choices = []
        for choice in response.get("choices") or []:
            message = choice.get("message") or {}
            delta: dict[str, Any] = {"role": message.get("role", "assistant")}
            for key in ("content", "refusal"):
                if message.get(key) is not None:
                    delta[key] = message[key]
            if message.get("tool_calls"):
                delta["tool_calls"] = [
                    {"index": i, **tool} for i, tool in enumerate(message["tool_calls"])
                ]
            choices.append(
                {
                    "index": choice.get("index", 0),
                    "delta": delta,
                    "finish_reason": choice.get("finish_reason"),
                    "logprobs": choice.get("logprobs"),
                }
            )
        head = {
            k: response[k]
            for k in ("id", "created", "model", "system_fingerprint", "service_tier")
            if k in response
        }
        chunks: list[Any] = [
            {**head, "object": "chat.completion.chunk", "choices": choices}
        ]
        if response.get("usage") is not None:
            chunks.append(
                {
                    **head,
                    "object": "chat.completion.chunk",
                    "choices": [],
                    "usage": response["usage"],
                }
            )
        return chunks

    def sse(self, events: list[Any]) -> str:
        return (
            "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
        )

    def error_body(self, error: dict[str, Any]) -> Any:
        # The package keeps the body's inner `error` object.
        body = error.get("body")
        if isinstance(body, dict) and "error" not in body:
            return {"error": body}
        if body is None:
            return {
                "error": {"message": error.get("message"), "type": None, "code": None}
            }
        return body

    def set_system(self, api: str, kwargs: dict[str, Any], prompt: str) -> None:
        if api.startswith("responses"):
            kwargs["instructions"] = prompt
            return
        messages = list(kwargs.get("messages") or [])
        for i, message in enumerate(messages):
            if isinstance(message, dict) and message.get("role") in (
                "system",
                "developer",
            ):
                messages[i] = {**message, "content": prompt}
                break
        else:
            messages.insert(0, {"role": "system", "content": prompt})
        kwargs["messages"] = messages

    def summary(self, api: str, response: Any) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "model": get(response, "model"),
            "input_tokens": None,
            "output_tokens": None,
            "stop_reason": None,
            "tool_calls": [],
        }
        if api.startswith("chat.completions"):
            message = get(response, "choices", 0, "message")
            summary["input_tokens"] = get(response, "usage", "prompt_tokens")
            summary["output_tokens"] = get(response, "usage", "completion_tokens")
            summary["stop_reason"] = _STOP.get(
                get(response, "choices", 0, "finish_reason")
            )
            if get(message, "refusal"):
                summary["stop_reason"] = "refusal"
            for tool in get(message, "tool_calls") or []:
                name = get(tool, "function", "name") or get(tool, "custom", "name")
                if name:
                    summary["tool_calls"].append(name)
        elif api.startswith("responses"):
            summary["input_tokens"] = get(response, "usage", "input_tokens")
            summary["output_tokens"] = get(response, "usage", "output_tokens")
            refused = False
            for item in get(response, "output") or []:
                kind = get(item, "type") or ""
                if kind in ("function_call", "custom_tool_call"):
                    summary["tool_calls"].append(get(item, "name"))
                elif kind.endswith("_call") and kind != "function_call_output":
                    summary["tool_calls"].append(
                        get(item, "name") or kind[: -len("_call")]
                    )
                for part in get(item, "content") or []:
                    refused = refused or get(part, "type") == "refusal"
            status = get(response, "status")
            if refused:
                summary["stop_reason"] = "refusal"
            elif status == "completed":
                summary["stop_reason"] = "tool_use" if summary["tool_calls"] else "end"
            elif status == "incomplete":
                reason = get(response, "incomplete_details", "reason")
                summary["stop_reason"] = {
                    "max_output_tokens": "length",
                    "content_filter": "content_filter",
                }.get(reason)
        elif api == "embeddings":
            summary["input_tokens"] = get(response, "usage", "prompt_tokens")
        # Of the input, how much the provider served from its cache; counted
        # inside input_tokens, and left out when the provider doesn't say.
        cached = get(response, "usage", "prompt_tokens_details", "cached_tokens")
        if cached is None:
            cached = get(response, "usage", "input_tokens_details", "cached_tokens")
        if cached is not None:
            summary["cached_input_tokens"] = cached
        return summary


def _chat_answer(chunks: list[Any]) -> Any:
    """A streamed chat completion's chunks put together as the answer the
    same call gives unstreamed."""
    answer: dict[str, Any] | None = None
    choices: dict[int, dict[str, Any]] = {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        if answer is None:
            answer = {"id": chunk.get("id"), "object": "chat.completion"}
        for key in ("id", "created", "model", "system_fingerprint", "service_tier"):
            if chunk.get(key) is not None:
                answer[key] = chunk[key]
        if chunk.get("usage") is not None:
            answer["usage"] = chunk["usage"]
        for part in chunk.get("choices") or []:
            if not isinstance(part, dict):
                continue
            index = part.get("index") or 0
            choice = choices.setdefault(
                index,
                {
                    "index": index,
                    "message": {"role": "assistant", "content": None, "refusal": None},
                    "finish_reason": None,
                    "logprobs": None,
                },
            )
            message = choice["message"]
            delta = part.get("delta") or {}
            if delta.get("role"):
                message["role"] = delta["role"]
            for key in ("content", "refusal"):
                if isinstance(delta.get(key), str):
                    message[key] = (message[key] or "") + delta[key]
            for call in delta.get("tool_calls") or []:
                calls = message.setdefault("tool_calls", [])
                at = call.get("index", len(calls))
                while len(calls) <= at:
                    calls.append(
                        {
                            "id": None,
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    )
                slot = calls[at]
                for key in ("id", "type"):
                    if call.get(key):
                        slot[key] = call[key]
                function = call.get("function") or {}
                for key in ("name", "arguments"):
                    if isinstance(function.get(key), str):
                        slot["function"][key] += function[key]
            if part.get("finish_reason") is not None:
                choice["finish_reason"] = part["finish_reason"]
    if answer is None:
        return None
    answer["choices"] = [choices[i] for i in sorted(choices)]
    return answer


class _ChatState(StreamState):
    """The package's own accumulator for streamed chat completions."""

    def __init__(self, state: Any) -> None:
        self.state = state

    def handle(self, chunk: Any) -> None:
        self.state.handle_chunk(chunk)

    def response(self, complete: bool) -> Any:
        if complete:
            try:
                return self.state.get_final_completion()
            except Exception:
                # The package refuses some answers (cut off while parsing);
                # what arrived is still recorded.
                log.debug("final answer not assembled", exc_info=True)
        return self.state.current_completion_snapshot


class _ResponsesState(StreamState):
    """A streamed Responses call: its last event that carries the response."""

    def __init__(self) -> None:
        self.last: Any = None

    def handle(self, chunk: Any) -> None:
        if getattr(chunk, "response", None) is not None:
            self.last = chunk.response

    def response(self, complete: bool) -> Any:
        return self.last
