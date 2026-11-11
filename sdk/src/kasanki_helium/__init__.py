# Copyright © 2026 Kasanki Labs.
# SPDX-License-Identifier: AGPL-3.0-only
# Licensed under the GNU AGPL v3 with an attribution term: see LICENSE and CREDIT.md.

"""Kasanki Helium Python SDK: records AI agent runs for the Helium engine.

::

    import kasanki_helium as helium

    with helium.run("support-agent", input=ticket) as run:
        now = run.now()
        hits = run.call_tool("search", search, query="refund policy")
        run.set_output(answer)

    result = helium.replay(run_id, entry=handle_ticket)   # the same again?

Recording is this package. Replay, tests, Autofix and the rest are Helium's
tools, loaded the first time one of their names is used,
so recording never loads them.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from ._auto import agent
from ._run import VERSION, Handoff, Run, flush, run
from ._session import (
    Change,
    HeliumReplayDiverged,
    ReplayBlocked,
    ReplayedError,
    ReplayError,
)
from ._warnings import HeliumWarning

if TYPE_CHECKING:
    from kasanki_helium_tools import _mcp as mcp
    from kasanki_helium_tools import _memory as memory
    from kasanki_helium_tools import _search as search
    from kasanki_helium_tools._earned import canary
    from kasanki_helium_tools._fixing import HeliumFixError, run_fix
    from kasanki_helium_tools._grading import (
        Grade,
        GradedRun,
        Grader,
        GradingError,
        ModelCall,
        RunGrade,
        ToolCall,
        grade,
        grader,
        judge,
        load_graders,
    )
    from kasanki_helium_tools._loops import loop_limits
    from kasanki_helium_tools._mcp import HeliumMcpError, McpToolError
    from kasanki_helium_tools._memory import HeliumMemoryError
    from kasanki_helium_tools._pack import HeliumPackError
    from kasanki_helium_tools._planted import check_graders
    from kasanki_helium_tools._replay import ForkResult, ReplayResult, fork, replay
    from kasanki_helium_tools._search import HeliumSearchError
    from kasanki_helium_tools._testing import (
        HeliumTestError,
        is_provider_error,
        run_test,
    )

__version__ = VERSION

#: Where Helium's tools are.
_TOOLS_PACKAGE = "kasanki_helium_tools"

#: Helium's tools, by public name: the module that holds it, and its name there
#: (`None`: the module itself).
_TOOLS: dict[str, tuple[str, str | None]] = {
    "mcp": ("_mcp", None),
    "memory": ("_memory", None),
    "search": ("_search", None),
    **{
        name: (module, name)
        for module, names in {
            "_earned": ("canary",),
            "_fixing": ("HeliumFixError", "run_fix"),
            "_grading": (
                "Grade",
                "GradedRun",
                "Grader",
                "GradingError",
                "ModelCall",
                "RunGrade",
                "ToolCall",
                "grade",
                "grader",
                "judge",
                "load_graders",
            ),
            "_loops": ("loop_limits",),
            "_mcp": ("HeliumMcpError", "McpToolError"),
            "_memory": ("HeliumMemoryError",),
            "_pack": ("HeliumPackError",),
            "_planted": ("check_graders",),
            "_replay": ("ForkResult", "ReplayResult", "fork", "replay"),
            "_search": ("HeliumSearchError",),
            "_testing": ("HeliumTestError", "is_provider_error", "run_test"),
        }.items()
        for name in names
    },
}


def _tools_missing() -> bool:
    """Whether Helium's tools aren't installed: then a missing module is
    theirs, not a fault inside them."""
    try:
        importlib.import_module(_TOOLS_PACKAGE)
    except ModuleNotFoundError as e:
        return e.name == _TOOLS_PACKAGE
    return False


def _paid_name(name: str) -> tuple[str, str | None] | None:
    """Where a name of the paid tools is, when they are installed."""
    if name.startswith("_"):
        return None
    try:
        paid = importlib.import_module(f"{_TOOLS_PACKAGE}._paid_names")
    except ModuleNotFoundError:
        return None
    found: tuple[str, str | None] | None = paid.NAMES.get(name)
    return found


def __getattr__(name: str) -> Any:
    found = _TOOLS.get(name) or _paid_name(name)
    if found is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = found
    try:
        module = importlib.import_module(f"{_TOOLS_PACKAGE}.{module_name}")
    except ModuleNotFoundError as e:
        if not _tools_missing():
            if e.name == f"{_TOOLS_PACKAGE}.{module_name}":
                raise AttributeError(
                    f"helium.{name} is not part of Helium Solo",
                    name=name,
                    obj=None,
                ) from None
            raise
        raise AttributeError(
            f"helium.{name} is one of Helium's tools, which aren't installed: "
            "pip install helium-solo",
            name=name,
            obj=None,
        ) from None
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    paid_names: set[str] = set()
    try:
        paid = importlib.import_module(f"{_TOOLS_PACKAGE}._paid_names")
        paid_names = set(paid.NAMES)
    except ModuleNotFoundError:
        pass
    return sorted(set(globals()) | set(_TOOLS) | paid_names)


__all__ = [
    "Change",
    "ForkResult",
    "Grade",
    "GradedRun",
    "Grader",
    "GradingError",
    "Handoff",
    "HeliumFixError",
    "HeliumMcpError",
    "HeliumMemoryError",
    "HeliumPackError",
    "HeliumReplayDiverged",
    "HeliumSearchError",
    "HeliumTestError",
    "HeliumWarning",
    "McpToolError",
    "ModelCall",
    "ReplayBlocked",
    "ReplayError",
    "ReplayResult",
    "ReplayedError",
    "Run",
    "RunGrade",
    "ToolCall",
    "__version__",
    "agent",
    "canary",
    "check_graders",
    "flush",
    "fork",
    "grade",
    "grader",
    "is_provider_error",
    "judge",
    "load_graders",
    "loop_limits",
    "mcp",
    "memory",
    "replay",
    "run",
    "run_fix",
    "run_test",
    "search",
]
