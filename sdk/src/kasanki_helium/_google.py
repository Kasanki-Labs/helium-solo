# Copyright © 2026 Kasanki Labs.
# SPDX-License-Identifier: AGPL-3.0-only
# Licensed under the GNU AGPL v3 with an attribution term: see LICENSE and CREDIT.md.

"""Recording Google model calls: ``run.wrap(client)``."""

from __future__ import annotations

import copy
import json
import re
import sys
from typing import TYPE_CHECKING, Any, ClassVar

from ._wrapper import Provider

if TYPE_CHECKING:
    from ._run import Run
    from ._session import Change


def _http_module_for(client: Any) -> Any:
    for kind in type(client).__mro__:
        module = sys.modules.get(kind.__module__.split(".")[0])
        if module is not None and hasattr(module, "MockTransport"):
            return module
    raise RuntimeError(
        f"{type(client).__qualname__} is not built on an HTTP library "
        "a replay can answer through"
    )


class Google(Provider):
    name = "google"
    module = "google.genai"
    known_major = 3
    # 3.x needs Python 3.11; on 3.10 the newest is 2.x, tested from 2.18.
    also_tested = (2,)

    paths: ClassVar[tuple[tuple[re.Pattern[str], str], ...]] = (
        (re.compile(r":streamGenerateContent$"), "generate_content"),
        (re.compile(r":generateContent$"), "generate_content"),
        (re.compile(r":countTokens$"), "count_tokens"),
        (re.compile(r":computeTokens$"), "compute_tokens"),
        (re.compile(r":batchEmbedContents$"), "embed_content"),
        (re.compile(r":embedContent$"), "embed_content"),
        (re.compile(r":predict$"), "predict"),
    )

    def recognises(self, client: Any) -> bool:
        return any(
            cls.__module__ == "google.genai.client"
            and cls.__name__ in ("Client", "AsyncClient")
            for cls in type(client).__mro__
        )

    def is_async(self, client: Any) -> bool:
        return any(
            cls.__module__ == "google.genai.client" and cls.__name__ == "AsyncClient"
            for cls in type(client).__mro__
        )

    def model_of(self, path: str) -> str | None:
        m = re.search(r"models/([^:]+):", path)
        return m.group(1) if m else None

    def summary(self, api: str, response: Any) -> dict[str, Any]:
        if not isinstance(response, dict):
            return {
                "model": None,
                "input_tokens": None,
                "output_tokens": None,
                "stop_reason": None,
                "tool_calls": [],
            }

        model = response.get("modelVersion")
        if api == "generate_content":
            usage = response.get("usageMetadata") or {}
            input_tokens = usage.get("promptTokenCount")
            cand_tokens = usage.get("candidatesTokenCount")
            thought_tokens = usage.get("thoughtsTokenCount")
            if cand_tokens is None and thought_tokens is None:
                output_tokens = None
            else:
                output_tokens = (cand_tokens or 0) + (thought_tokens or 0)
            cached_input_tokens = usage.get("cachedContentTokenCount")

            candidates = response.get("candidates") or []
            first_cand = (
                candidates[0] if candidates and isinstance(candidates[0], dict) else {}
            )
            stop_reason = first_cand.get("finishReason")

            tool_calls: list[str] = []
            content = first_cand.get("content") or {}
            for part in content.get("parts") or []:
                if isinstance(part, dict):
                    fc = part.get("functionCall")
                    if isinstance(fc, dict) and fc.get("name"):
                        tool_calls.append(str(fc["name"]))

            summary: dict[str, Any] = {
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "stop_reason": stop_reason,
                "tool_calls": tool_calls,
            }
            if cached_input_tokens is not None:
                summary["cached_input_tokens"] = cached_input_tokens
            return summary

        if api == "count_tokens":
            return {
                "model": model,
                "input_tokens": response.get("totalTokens"),
                "output_tokens": None,
                "stop_reason": None,
                "tool_calls": [],
            }

        return {
            "model": model,
            "input_tokens": None,
            "output_tokens": None,
            "stop_reason": None,
            "tool_calls": [],
        }

    def assemble(self, api: str, events: list[Any]) -> Any:
        if not events:
            return None
        first_model_version: str | None = None
        first_response_id: str | None = None
        last_usage_metadata: dict[str, Any] | None = None

        cand_parts: dict[int, list[dict[str, Any]]] = {}
        cand_roles: dict[int, Any] = {}
        cand_finish_reason: dict[int, str | None] = {}

        for event in events:
            if not isinstance(event, dict):
                continue
            if first_model_version is None and event.get("modelVersion") is not None:
                first_model_version = event.get("modelVersion")
            if first_response_id is None and event.get("responseId") is not None:
                first_response_id = event.get("responseId")
            if event.get("usageMetadata") is not None:
                last_usage_metadata = event.get("usageMetadata")

            for cand in event.get("candidates") or []:
                if not isinstance(cand, dict):
                    continue
                idx = cand.get("index", 0)
                if cand.get("finishReason") is not None:
                    cand_finish_reason[idx] = cand.get("finishReason")

                content = cand.get("content") or {}
                if content.get("role"):
                    cand_roles[idx] = content.get("role")

                parts = content.get("parts") or []
                if idx not in cand_parts:
                    cand_parts[idx] = []

                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    if "text" in part:
                        if cand_parts[idx] and "text" in cand_parts[idx][-1]:
                            cand_parts[idx][-1]["text"] += str(part.get("text", ""))
                        else:
                            cand_parts[idx].append(dict(part))
                    else:
                        cand_parts[idx].append(dict(part))

        candidates_list: list[dict[str, Any]] = []
        indices = sorted(cand_parts.keys()) if cand_parts else [0]
        for idx in indices:
            c_dict: dict[str, Any] = {
                "index": idx,
                "content": {
                    "parts": cand_parts.get(idx, []),
                    "role": cand_roles.get(idx, "model"),
                },
            }
            if idx in cand_finish_reason and cand_finish_reason[idx] is not None:
                c_dict["finishReason"] = cand_finish_reason[idx]
            candidates_list.append(c_dict)

        result: dict[str, Any] = {"candidates": candidates_list}
        if first_model_version is not None:
            result["modelVersion"] = first_model_version
        if first_response_id is not None:
            result["responseId"] = first_response_id
        if last_usage_metadata is not None:
            result["usageMetadata"] = last_usage_metadata
        return result

    def events(self, api: str, response: Any) -> list[Any]:
        return [response]

    def sse(self, events: list[Any]) -> str:
        return "".join(
            f"data: {json.dumps(e, separators=(',', ':'))}\n\n" for e in events
        )

    def error_body(self, error: dict[str, Any]) -> Any:
        body = error.get("body")
        if body is None:
            return {
                "type": "error",
                "error": {"type": "api_error", "message": error.get("message")},
            }
        return body

    def set_system(self, api: str, body: dict[str, Any], prompt: str) -> None:
        body["systemInstruction"] = {"parts": [{"text": prompt}]}

    def change_request(self, body: dict[str, Any], path: str, change: Change) -> str:
        _top_level = {
            "contents",
            "tools",
            "toolConfig",
            "safetySettings",
            "systemInstruction",
            "cachedContent",
            "labels",
        }
        if change.model is not None:
            path = re.sub(r"models/[^:]+:", f"models/{change.model}:", path)
        if change.params:
            for k, v in change.params.items():
                if k in _top_level:
                    body[k] = v
                else:
                    gen_cfg = body.get("generationConfig")
                    if not isinstance(gen_cfg, dict):
                        gen_cfg = {}
                        body["generationConfig"] = gen_cfg
                    gen_cfg[k] = v
        if change.system_prompt is not None:
            self.set_system("", body, change.system_prompt)
        if change.tool_descriptions:
            for tool in body.get("tools") or []:
                if isinstance(tool, dict):
                    for decl in tool.get("functionDeclarations") or []:
                        if (
                            isinstance(decl, dict)
                            and decl.get("name") in change.tool_descriptions
                        ):
                            decl["description"] = change.tool_descriptions[decl["name"]]
        return path


def wrap(run: Run, client: Any) -> Any:
    from . import _http, _wrapper

    provider = Google()
    _wrapper.check_version(provider)
    api = client._api_client
    is_async = provider.is_async(client)
    if api._async_httpx_client is None and is_async:
        raise TypeError(
            "Kasanki Helium can't record this Google client: its async calls go "
            "through google-auth's own session (mutual TLS)"
        )

    wapi = copy.copy(api)
    http_sync = _http_module_for(api._httpx_client)
    recorder_sync = _http.Recorder(run, http_sync, provider)
    wapi._httpx_client = _http.recording_copy(api._httpx_client, recorder_sync, False)
    if api._async_httpx_client is not None:
        http_async = _http_module_for(api._async_httpx_client)
        recorder_async = _http.Recorder(run, http_async, provider)
        wapi._async_httpx_client = _http.recording_copy(
            api._async_httpx_client, recorder_async, True
        )
    wapi._http_options = api._http_options.model_copy(
        update={
            "httpx_client": wapi._httpx_client,
            "httpx_async_client": wapi._async_httpx_client,
        }
    )

    wrapped_client = copy.copy(client)
    wrapped_client._api_client = wapi
    for k, v in list(wrapped_client.__dict__.items()):
        if getattr(v, "_api_client", None) is api:
            sub = copy.copy(v)
            sub._api_client = wapi
            setattr(wrapped_client, k, sub)

    if not is_async:
        # The package's AsyncClient, taken from the client so nothing here
        # imports google.genai (some versions ship no type information).
        wrapped_client._aio = type(client._aio)(wapi)

    return wrapped_client
