# Copyright © 2026 Kasanki Labs.
# SPDX-License-Identifier: AGPL-3.0-only
# Licensed under the GNU AGPL v3 with an attribution term: see LICENSE and CREDIT.md.

"""Automatic recording of model calls for Helium.

Turns on auto-recording for OpenAI, Anthropic, Google, and LangChain models during an
open run without requiring client code changes. Also provides the @agent()
decorator, script execution via `python -m kasanki_helium._auto`, and AST-based
code inspection and diff generation for adding recording.
"""

from __future__ import annotations

import ast
import contextvars
import difflib
import functools
import hashlib
import importlib
import inspect
import json
import os
import re
import runpy
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, overload

F = TypeVar("F", bound=Callable[..., Any])

from . import _http, _llm, _run
from ._http import Recorder, detect
from ._run import Run, _open_run

_ENABLED = False
_active_sync_transport: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "helium_active_sync_transport", default=None
)
_active_async_transport: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "helium_active_async_transport", default=None
)
_launch_hook: Callable[[Any], None] | None = None
_patched_libraries: set[str] = set()


def set_launch_hook(hook: Callable[[Any], None] | None) -> None:
    """Sets a hook called when a model call occurs with no open run."""
    global _launch_hook
    _launch_hook = hook


_orig_use_aiohttp: Callable[..., bool] | None = None


def _patch_google_genai() -> None:
    global _orig_use_aiohttp
    mod = sys.modules.get("google.genai._api_client")
    if mod is None:
        try:
            mod = importlib.import_module("google.genai._api_client")
        except ImportError:
            return
    if hasattr(mod, "BaseApiClient") and _orig_use_aiohttp is None:
        _orig_use_aiohttp = mod.BaseApiClient._use_aiohttp
        mod.BaseApiClient._use_aiohttp = lambda self: False


def _restore_google_genai() -> None:
    global _orig_use_aiohttp
    if _orig_use_aiohttp is not None:
        mod = sys.modules.get("google.genai._api_client")
        if mod is not None and hasattr(mod, "BaseApiClient"):
            mod.BaseApiClient._use_aiohttp = _orig_use_aiohttp
        _orig_use_aiohttp = None


def enable() -> None:
    """Turns on auto recording for this process. Idempotent."""
    global _ENABLED
    _ENABLED = True
    _patch_http_libraries()
    _patch_google_genai()


def disable() -> None:
    """Turns off auto recording for this process. Idempotent."""
    global _ENABLED
    _ENABLED = False
    set_launch_hook(None)
    _restore_google_genai()


def _patch_http_libraries() -> None:
    for name in ("httpx2", "httpx"):
        if name in _patched_libraries:
            continue
        try:
            mod = sys.modules.get(name) or importlib.import_module(name)
        except ImportError:
            continue
        _patch_module(mod)
        _patched_libraries.add(name)


def _patch_module(mod: Any) -> None:
    # 1. Sync Client
    orig_transport_for_url = mod.Client._transport_for_url
    orig_send_single_request = mod.Client._send_single_request

    def _patched_transport_for_url(self: Any, url: Any) -> Any:
        active = _active_sync_transport.get()
        if active is not None:
            return active
        return orig_transport_for_url(self, url)

    def _patched_send_single_request(self: Any, request: Any) -> Any:
        if _ENABLED and not getattr(_llm, "SURFACE", False):
            transport = orig_transport_for_url(self, request.url)
            if getattr(transport, "_recorder", None) is None:
                run = _open_run.get()
                if run is None and _launch_hook is not None:
                    _launch_hook(request)
                    run = _open_run.get()
                if run is not None and detect(request) is not None:
                    recorder = Recorder(run, mod, None)
                    kinds = _http._kinds(mod)
                    wrapped = kinds.Recording(transport, recorder, owns=False)
                    token = _active_sync_transport.set(wrapped)
                    try:
                        return orig_send_single_request(self, request)
                    finally:
                        _active_sync_transport.reset(token)
        return orig_send_single_request(self, request)

    mod.Client._transport_for_url = _patched_transport_for_url
    mod.Client._send_single_request = _patched_send_single_request

    # 2. Async Client
    orig_async_transport_for_url = mod.AsyncClient._transport_for_url
    orig_async_send_single_request = mod.AsyncClient._send_single_request

    def _patched_async_transport_for_url(self: Any, url: Any) -> Any:
        active = _active_async_transport.get()
        if active is not None:
            return active
        return orig_async_transport_for_url(self, url)

    async def _patched_async_send_single_request(self: Any, request: Any) -> Any:
        if _ENABLED and not getattr(_llm, "SURFACE", False):
            transport = orig_async_transport_for_url(self, request.url)
            if getattr(transport, "_recorder", None) is None:
                run = _open_run.get()
                if run is None and _launch_hook is not None:
                    _launch_hook(request)
                    run = _open_run.get()
                if run is not None and detect(request) is not None:
                    recorder = Recorder(run, mod, None)
                    kinds = _http._kinds(mod)
                    wrapped = kinds.AsyncRecording(transport, recorder, owns=False)
                    token = _active_async_transport.set(wrapped)
                    try:
                        return await orig_async_send_single_request(self, request)
                    finally:
                        _active_async_transport.reset(token)
        return await orig_async_send_single_request(self, request)

    mod.AsyncClient._transport_for_url = _patched_async_transport_for_url
    mod.AsyncClient._send_single_request = _patched_async_send_single_request


def _project_name(start_folder: Path | None = None) -> str | None:
    here = (start_folder or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        pyproj = folder / "pyproject.toml"
        if pyproj.is_file():
            try:
                text = pyproj.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                return None
            in_project = False
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("["):
                    in_project = stripped == "[project]"
                    continue
                if in_project:
                    m = re.match(r'^name\s*=\s*"([^"]+)"', stripped)
                    if m:
                        return m.group(1)
            return None
    return None


#: Script names that say nothing about the agent, so its folder names it instead.
_GENERIC_STEMS = frozenset(
    {"main", "app", "run", "agent", "__main__", "cli", "start", "index", "bot"}
)


def _slug(name: str) -> str:
    """Lower-cases ``name``, turns each run of characters other than ``a-z0-9``
    into one hyphen, and trims hyphens from both ends. Empty when nothing is left."""
    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")


def _fallback_name(path: Path) -> str:
    """The name for a script with nothing else to name the agent: a generic
    script name (``main``, ``app``, ``run``, ...) takes its folder's slug, any
    other name keeps its stem."""
    if path.stem.lower() in _GENERIC_STEMS:
        return _slug(path.resolve().parent.name) or path.stem
    return path.stem


def _message(name: str | None) -> str:
    """The message printed when a script finished without calling a model."""
    script = name or "agent"
    return (
        "Helium: nothing was recorded. "
        f"{script} finished without calling a model "
        "Helium records (OpenAI, Anthropic, Google Gemini, LangChain, "
        "or an OpenAI-compatible server). If it needs an API key, "
        "put it in .env in this folder."
    )


def _is_one_input_param(func: Callable[..., Any]) -> tuple[bool, str | None]:
    try:
        parameters = list(inspect.signature(func).parameters.values())
    except (TypeError, ValueError):
        return False, None
    positional = [
        p for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    required = [
        p
        for p in parameters
        if p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    if positional and len(required) == 1 and required[0] is positional[0]:
        return True, positional[0].name
    return False, None


def _function_entry(func: Callable[..., Any]) -> str:
    module = getattr(func, "__module__", "")
    fn_name = getattr(func, "__name__", "")
    if module == "__main__":
        try:
            fn_file = inspect.getfile(func)
            try:
                rel = Path(fn_file).resolve().relative_to(Path.cwd().resolve())
            except ValueError:
                rel = Path(fn_file).resolve()
            return f"{rel.as_posix()}:{fn_name}"
        except (TypeError, OSError):
            pass
    return f"{module}:{fn_name}"


@overload
def agent(name: F, *, version: str | None = None) -> F: ...


@overload
def agent(
    name: str | None = None, *, version: str | None = None
) -> Callable[[F], F]: ...


def agent(
    name: str | Callable[..., Any] | None = None,
    *,
    version: str | None = None,
) -> Any:
    """Decorator to record each call of an agent function as its own run."""
    enable()

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        name_str = name if isinstance(name, str) else None
        agent_name = name_str or _project_name()
        if agent_name is None:
            try:
                agent_name = Path(inspect.getfile(func)).stem
            except (TypeError, OSError):
                agent_name = func.__name__

        entry_str = _function_entry(func)
        one_input, param_name = _is_one_input_param(func)

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_agent_wrapper(*args: Any, **kwargs: Any) -> Any:
                enable()
                if _open_run.get() is not None:
                    return await func(*args, **kwargs)
                sig = inspect.signature(func)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                run_input = (
                    bound.arguments[param_name]
                    if (one_input and param_name)
                    else dict(bound.arguments)
                )
                with _run.run(
                    agent_name, input=run_input, version=version, entry=entry_str
                ) as r:
                    res = await func(*args, **kwargs)
                    r.set_output(res)
                    return res

            return async_agent_wrapper
        else:

            @functools.wraps(func)
            def sync_agent_wrapper(*args: Any, **kwargs: Any) -> Any:
                enable()
                if _open_run.get() is not None:
                    return func(*args, **kwargs)
                sig = inspect.signature(func)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                run_input = (
                    bound.arguments[param_name]
                    if (one_input and param_name)
                    else dict(bound.arguments)
                )
                with _run.run(
                    agent_name, input=run_input, version=version, entry=entry_str
                ) as r:
                    res = func(*args, **kwargs)
                    r.set_output(res)
                    return res

            return sync_agent_wrapper

    if callable(name):
        actual_func = name
        name = None
        return decorator(actual_func)

    return decorator


def inspect_functions(file_path: str) -> dict[str, Any]:
    """Inspects module-level functions in file_path with ast."""
    path = Path(file_path)
    raw_bytes = path.read_bytes()
    sha = hashlib.sha256(raw_bytes).hexdigest()
    has_bom = raw_bytes.startswith(b"\xef\xbb\xbf")
    try:
        text = raw_bytes.decode("utf-8-sig" if has_bom else "utf-8")
    except UnicodeDecodeError:
        return {"file": file_path, "sha256": sha, "functions": []}

    try:
        tree = ast.parse(text, filename=file_path)
    except SyntaxError:
        return {"file": file_path, "sha256": sha, "functions": []}

    funcs_list = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            recorded = False
            for dec in node.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                if isinstance(target, ast.Attribute) and target.attr == "agent":
                    recorded = True
                    break
                if isinstance(target, ast.Name) and target.id == "agent":
                    recorded = True
                    break

            posonlyargs = [a.arg for a in node.args.posonlyargs]
            args = [a.arg for a in node.args.args]
            kwonlyargs = [a.arg for a in node.args.kwonlyargs]
            vararg = [node.args.vararg.arg] if node.args.vararg else []
            kwarg = [node.args.kwarg.arg] if node.args.kwarg else []
            param_names = posonlyargs + args + vararg + kwonlyargs + kwarg

            total_pos = len(posonlyargs) + len(args)
            num_defaults = len(node.args.defaults)
            req_pos = total_pos - num_defaults
            num_kw_req = sum(1 for d in node.args.kw_defaults if d is None)
            one_input = total_pos >= 1 and req_pos == 1 and num_kw_req == 0

            funcs_list.append(
                {
                    "name": node.name,
                    "line": node.lineno,
                    "async": isinstance(node, ast.AsyncFunctionDef),
                    "params": param_names,
                    "one_input": one_input,
                    "recorded": recorded,
                }
            )

    return {
        "file": file_path,
        "sha256": sha,
        "functions": funcs_list,
    }


def patch_function(
    file_path: str,
    target_func: str,
    *,
    apply: bool = False,
    expect_sha256: str | None = None,
) -> dict[str, Any]:
    """Generates (and optionally applies) a patch adding @agent() to target_func."""
    path = Path(file_path)
    raw_bytes = path.read_bytes()
    sha = hashlib.sha256(raw_bytes).hexdigest()
    has_bom = raw_bytes.startswith(b"\xef\xbb\xbf")
    try:
        text = raw_bytes.decode("utf-8-sig" if has_bom else "utf-8")
    except UnicodeDecodeError:
        return {
            "file": file_path,
            "sha256": sha,
            "applies": False,
            "reason": f"{file_path} isn't UTF-8 text.",
            "diff": "",
        }

    try:
        tree = ast.parse(text, filename=file_path)
    except SyntaxError as e:
        return {
            "file": file_path,
            "sha256": sha,
            "applies": False,
            "reason": f"{file_path} has a syntax error at line {e.lineno}.",
            "diff": "",
        }

    target_node = None
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == target_func
        ):
            target_node = node
            break

    if target_node is None:
        return {
            "file": file_path,
            "sha256": sha,
            "applies": False,
            "reason": f"No function {target_func} at the top level of {file_path}.",
            "diff": "",
        }

    for dec in target_node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if (isinstance(target, ast.Attribute) and target.attr == "agent") or (
            isinstance(target, ast.Name) and target.id == "agent"
        ):
            return {
                "file": file_path,
                "sha256": sha,
                "applies": False,
                "reason": f"{target_func} is already recorded.",
                "diff": "",
            }

    if apply and expect_sha256 is not None and sha != expect_sha256:
        return {
            "file": file_path,
            "sha256": sha,
            "applies": False,
            "reason": f"{file_path} changed since the diff was made.",
            "diff": "",
        }

    alias = "helium"
    needs_import = True
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            for n in stmt.names:
                if n.name == "kasanki_helium":
                    alias = n.asname or "kasanki_helium"
                    needs_import = False
                    break
        elif isinstance(stmt, ast.ImportFrom) and stmt.module == "kasanki_helium":
            alias = "kasanki_helium"
            needs_import = False
            break

    eol = "\r\n" if "\r\n" in text else "\n"
    orig_lines = text.splitlines(keepends=True)

    import_insert_line = 0
    body_idx = 0
    if tree.body:
        first_stmt = tree.body[0]
        if (
            isinstance(first_stmt, ast.Expr)
            and isinstance(first_stmt.value, ast.Constant)
            and isinstance(first_stmt.value.value, str)
        ):
            import_insert_line = first_stmt.end_lineno or 1
            body_idx = 1
        while body_idx < len(tree.body):
            stmt = tree.body[body_idx]
            if isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
                import_insert_line = stmt.end_lineno or (import_insert_line + 1)
                body_idx += 1
            else:
                break

    if target_node.decorator_list:
        func_first_line = target_node.decorator_list[0].lineno
    else:
        func_first_line = target_node.lineno

    dec_line_idx = func_first_line - 1
    indent = ""
    if 0 <= dec_line_idx < len(orig_lines):
        m = re.match(r"^\s*", orig_lines[dec_line_idx])
        indent = m.group(0) if m else ""

    dec_line = f"{indent}@{alias}.agent(){eol}"
    import_line = f"import kasanki_helium as helium{eol}"

    new_lines = list(orig_lines)
    new_lines.insert(dec_line_idx, dec_line)
    if needs_import:
        new_lines.insert(import_insert_line, import_line)

    file_name = path.name
    diff_lines = list(
        difflib.unified_diff(
            orig_lines,
            new_lines,
            fromfile=f"a/{file_name}",
            tofile=f"b/{file_name}",
            n=3,
        )
    )
    diff_str = "".join(diff_lines)

    if apply:
        new_content = "".join(new_lines)
        new_bytes = (b"\xef\xbb\xbf" if has_bom else b"") + new_content.encode("utf-8")
        path.write_bytes(new_bytes)
        new_sha = hashlib.sha256(new_bytes).hexdigest()
        return {
            "file": file_path,
            "sha256": new_sha,
            "applies": True,
            "reason": None,
            "diff": diff_str,
        }

    return {
        "file": file_path,
        "sha256": sha,
        "applies": True,
        "reason": None,
        "diff": diff_str,
    }


def run_script(script: str, args: list[str], *, name: str | None = None) -> None:
    """Runs a script with auto-recording enabled and a lazy launch run."""
    path = Path(script)
    if not path.is_file():
        print(f"No such file: {script}", file=sys.stderr)
        sys.exit(2)

    enable()

    try:
        rel = path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        rel = path.resolve()
    entry_str = f"script:{rel.as_posix()}"

    agent_name = name or _project_name(path.resolve().parent) or _fallback_name(path)
    # Runs opened while the script runs -- its own ``helium.run`` calls and the
    # launch run -- are counted, so a launch that recorded nothing is known.
    opened_before = _run._opened_runs

    launch_run: Run | None = None

    def _on_model_call(request: Any) -> None:
        nonlocal launch_run
        if launch_run is None and _open_run.get() is None:
            launch_run = _run.run(
                agent_name,
                input={"argv": list(args)},
                entry=entry_str,
            )
            launch_run.__enter__()

    set_launch_hook(_on_model_call)

    script_folder = str(path.resolve().parent)
    if script_folder not in sys.path:
        sys.path.insert(0, script_folder)
    sys.argv = [script, *args]

    exit_code = 0
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        if e.code is None:
            exit_code = 0
        elif isinstance(e.code, int):
            exit_code = e.code
        else:
            print(str(e.code), file=sys.stderr)
            exit_code = 1
        if launch_run is not None:
            launch_run.__exit__(None, None, None)
            launch_run = None
        if _run._opened_runs == opened_before:
            print(_message(path.name), file=sys.stderr)
    except KeyboardInterrupt:
        if launch_run is not None:
            launch_run.__exit__(*sys.exc_info())
            launch_run = None
        exit_code = 130
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        if launch_run is not None:
            launch_run.__exit__(*sys.exc_info())
            launch_run = None
        exit_code = 1
    else:
        if launch_run is not None:
            launch_run.__exit__(None, None, None)
            launch_run = None
        if _run._opened_runs == opened_before:
            print(_message(path.name), file=sys.stderr)
    finally:
        set_launch_hook(None)
        _run.flush(15.0)

    sys.exit(exit_code)


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        sys.exit(2)

    if argv[0] == "--functions":
        if len(argv) < 2:
            print(
                "Usage: python -m kasanki_helium._auto --functions <file>",
                file=sys.stderr,
            )
            sys.exit(2)
        file_path = argv[1]
        if not os.path.exists(file_path):
            print(f"No such file: {file_path}", file=sys.stderr)
            sys.exit(2)
        res = inspect_functions(file_path)
        print(json.dumps(res))
        sys.exit(0)

    if argv[0] == "--patch":
        if len(argv) < 3:
            print(
                "Usage: python -m kasanki_helium._auto --patch <file> <function> "
                "[--apply --expect-sha256 <hex>]",
                file=sys.stderr,
            )
            sys.exit(2)
        file_path = argv[1]
        func_name = argv[2]
        if not os.path.exists(file_path):
            print(f"No such file: {file_path}", file=sys.stderr)
            sys.exit(2)
        apply = False
        expect_sha = None
        idx = 3
        while idx < len(argv):
            if argv[idx] == "--apply":
                apply = True
                idx += 1
            elif argv[idx] == "--expect-sha256":
                if idx + 1 >= len(argv):
                    sys.exit(2)
                expect_sha = argv[idx + 1]
                idx += 2
            else:
                idx += 1
        if apply and expect_sha is None:
            sys.exit(2)
        res = patch_function(
            file_path, func_name, apply=apply, expect_sha256=expect_sha
        )
        print(json.dumps(res))
        sys.exit(0)

    name = None
    args_idx = 0
    if argv[0] == "--name":
        if len(argv) < 2:
            sys.exit(2)
        name = argv[1]
        args_idx = 2

    if args_idx >= len(argv):
        sys.exit(2)

    script_path = argv[args_idx]
    script_args = argv[args_idx + 1 :]

    run_script(script_path, script_args, name=name)


if __name__ == "__main__":
    main()
