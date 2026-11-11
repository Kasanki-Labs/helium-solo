# Copyright © 2026 Kasanki Labs.
# SPDX-License-Identifier: AGPL-3.0-only
# Licensed under the GNU AGPL v3 with an attribution term: see LICENSE and CREDIT.md.

"""Recording LangChain models' calls: ``run.wrap(model)``.

LangChain checks that a model is one of its own, so the model is not wrapped
itself: ``run.wrap`` returns a copy holding wrapped provider clients, and its
calls are recorded exactly like direct ones. The original is left as it was.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from ._warnings import warn_once
from ._wrapper import classes

if TYPE_CHECKING:
    from ._run import Run

#: The integrations' major version this was built and tested for.
_KNOWN_MAJOR = "1"


def wrap(run: Run, model: Any) -> Any | None:
    """A copy of `model` that records into `run`, or `None` if it is not a
    LangChain model this SDK knows."""
    found = classes(model)
    if ("langchain_core", "RunnableBinding") in found:
        # `model.bind_tools(...)`: the same binding around a wrapped copy.
        bound = wrap(run, model.bound)
        return None if bound is None else model.model_copy(update={"bound": bound})
    if ("langchain_openai", "BaseChatOpenAI") in found:
        _check_version("langchain_openai")
        update = {}
        for root, resource in (
            ("root_client", "client"),
            ("root_async_client", "async_client"),
        ):
            client = getattr(model, root, None)
            if client is not None:
                wrapped = _client(run, client, model)
                update[root] = wrapped
                update[resource] = wrapped.chat.completions
        return _copy(model, update)
    if ("langchain_openai", "OpenAIEmbeddings") in found:
        _check_version("langchain_openai")
        update = {}
        for resource in ("client", "async_client"):
            embeddings = getattr(model, resource, None)
            if embeddings is not None:
                # The package's resources keep the client they belong to.
                client = getattr(embeddings, "_client", None)
                update[resource] = _client(run, client, model).embeddings
        return _copy(model, update)
    if ("langchain_anthropic", "ChatAnthropic") in found:
        _check_version("langchain_anthropic")
        copy = model.model_copy()
        # Cached properties, made on first use: set on the copy only.
        for name in ("_client", "_async_client"):
            copy.__dict__[name] = _client(run, getattr(model, name), model)
        return copy
    return None


def _client(run: Run, client: Any, model: Any) -> Any:
    from . import _llm

    try:
        return _llm.wrap(run, client)
    except TypeError:
        raise TypeError(
            f"Kasanki Helium cannot record {type(model).__qualname__}: its client "
            f"is a {type(client).__module__}.{type(client).__qualname__}, not an "
            "OpenAI or Anthropic client"
        ) from None


def _copy(model: Any, update: dict[str, Any]) -> Any:
    if not update:
        raise TypeError(
            f"Kasanki Helium cannot record {type(model).__qualname__}: it has no "
            "client yet"
        )
    return model.model_copy(update=update)


def _check_version(package: str) -> None:
    version = getattr(sys.modules.get(package), "__version__", "")
    if version.split(".")[0] != _KNOWN_MAJOR:
        warn_once(
            f"{package} {version}",
            f"Kasanki Helium was built for {package} {_KNOWN_MAJOR}.x, not "
            f"{version}; calls are recorded, but replay may not be exact.",
        )
