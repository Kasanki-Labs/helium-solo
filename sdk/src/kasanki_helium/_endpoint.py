# Copyright © 2026 Kasanki Labs.
# SPDX-License-Identifier: AGPL-3.0-only
# Licensed under the GNU AGPL v3 with an attribution term: see LICENSE and CREDIT.md.

"""Finding the engine for a data folder (trace-format spec §7.5)."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

#: Overrides the system-wide data folder (for development and tests).
DATA_DIR_VARIABLE = "KASANKI_HELIUM_DATA_DIR"


#: The system-wide data folder on Linux, used when it already exists.
LINUX_SYSTEM_DATA_DIR = "/var/lib/kasanki-helium"


def default_data_dir() -> Path:
    """The default data folder, the same as the engine's default. On Linux
    only root can make the system-wide folder, so when it doesn't exist the
    folder is the user's own."""
    if sys.platform == "win32":
        return Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "Kasanki Helium"
    if sys.platform == "darwin":
        return Path("/Library/Application Support/Kasanki Helium")
    return linux_data_dir(
        Path(LINUX_SYSTEM_DATA_DIR).is_dir(),
        os.environ.get("XDG_DATA_HOME"),
        os.environ.get("HOME"),
    )


def linux_data_dir(
    system_exists: bool, xdg_data_home: str | None, home: str | None
) -> Path:
    """The Linux rule (the engine's `linux_data_dir`): the system-wide folder if
    it exists, else `$XDG_DATA_HOME/kasanki-helium`, else
    `~/.local/share/kasanki-helium`. An empty or relative folder is ignored."""
    if system_exists:
        return Path(LINUX_SYSTEM_DATA_DIR)
    if xdg_data_home and Path(xdg_data_home).is_absolute():
        return Path(xdg_data_home) / "kasanki-helium"
    if home and Path(home).is_absolute():
        return Path(home) / ".local" / "share" / "kasanki-helium"
    return Path(LINUX_SYSTEM_DATA_DIR)


def data_dir() -> Path:
    configured = os.environ.get(DATA_DIR_VARIABLE)
    return Path(configured) if configured else default_data_dir()


def _ascii_lower(text: str) -> str:
    return "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in text)


def pipe_name(full_path: str) -> str:
    """The Windows pipe name for a data folder's full path: the first 16 hex
    digits of SHA-256 over the path with ASCII letters lowercased."""
    digest = hashlib.sha256(_ascii_lower(full_path).encode("utf-8")).hexdigest()
    return f"kasanki-helium-{digest[:16]}"


def full_path(folder: Path) -> str:
    """The folder's full path as Windows reports it, without a \\\\?\\ prefix."""
    resolved = os.path.realpath(folder)
    if resolved.startswith("\\\\?\\UNC\\"):
        return "\\\\" + resolved[len("\\\\?\\UNC\\") :]
    if resolved.startswith("\\\\?\\"):
        return resolved[len("\\\\?\\") :]
    return resolved


def endpoint(folder: Path) -> str:
    """Unix: the socket file in the data folder. Windows: the named pipe."""
    if sys.platform == "win32":
        return "\\\\.\\pipe\\" + pipe_name(full_path(folder))
    return str(folder / "engine.sock")
