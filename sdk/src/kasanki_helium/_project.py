# Copyright © 2026 Kasanki Labs.
# SPDX-License-Identifier: AGPL-3.0-only
# Licensed under the GNU AGPL v3 with an attribution term: see LICENSE and CREDIT.md.

"""A project's identity: ``helium/project.toml`` holds a generated
ID and a name, and is committed, so moving or cloning the folder keeps them. A
project's board is named after the ID, and each run records it.

Recording a run only ever reads this file. Only `helium new` and the first use of a
project's board write it.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("kasanki_helium")


def project_root(start: Path | None = None) -> Path:
    """The nearest folder with a pyproject.toml, else the working directory."""
    here = (start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        if (folder / "pyproject.toml").is_file():
            return folder
    return here


RELATIVE = Path("helium") / "project.toml"
_ID = re.compile(r'^id\s*=\s*"([a-z0-9][a-z0-9_.-]{0,99})"\s*$', flags=re.MULTILINE)
_NAME = re.compile(r'^name\s*=\s*"([^"\n]+)"\s*$', flags=re.MULTILINE)
#: How long to wait for another process that's writing the file right now.
_WAITS = (0.01, 0.05, 0.2)


@dataclass(frozen=True)
class Project:
    id: str
    name: str
    root: Path


def new_id(name: str) -> str:
    """A name a person can recognise, and enough randomness that two projects with
    the same name never share a board. It never changes, even if the name does."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40].strip("-")
    return f"{slug or 'project'}-{uuid.uuid4().hex[:12]}"


def file_text(name: str, id: str | None = None) -> str:
    clean = name.replace('"', "").replace("\n", " ").strip() or "project"
    return (
        "# This project's identity, made by Helium. Commit this file: it keeps the\n"
        "# project's board and runs together when the folder moves or is cloned.\n"
        f'id = "{id or new_id(clean)}"\n'
        f'name = "{clean}"\n'
    )


def _read(root: Path) -> Project | None:
    path = root / RELATIVE
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as e:
        log.warning("Kasanki Helium: could not read %s: %s", path, e)
        return None
    found = _ID.search(text)
    if found is None:
        return None
    name = _NAME.search(text)
    return Project(found.group(1), name.group(1) if name else root.name, root)


def find(start: str | Path | None = None) -> Project | None:
    """The project this folder belongs to, if it has an identity yet."""
    return _read(project_root(Path(start) if start is not None else None))


def _pyproject_name(root: Path) -> str:
    try:
        text = (root / "pyproject.toml").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return root.name
    found = _NAME.search(text)
    return found.group(1) if found else root.name


def ensure(start: str | Path | None = None) -> Project | None:
    """The project this folder belongs to, giving it an identity on first use. A
    folder with no pyproject.toml isn't a project. Never overwrites the file, and
    never raises: a read-only folder must not stop an agent (C7)."""
    root = project_root(Path(start) if start is not None else None)
    found = _read(root)
    if found is not None:
        return found
    path = root / RELATIVE
    if not (root / "pyproject.toml").is_file():
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as f:
            f.write(file_text(_pyproject_name(root)))
    except FileExistsError:
        # Another process got there first, or a person's file can't be read.
        for wait in _WAITS:
            time.sleep(wait)
            found = _read(root)
            if found is not None:
                return found
        log.warning("Kasanki Helium: %s has no usable id; fix or delete it", path)
        return None
    except OSError as e:
        log.warning("Kasanki Helium: could not write %s: %s", path, e)
        return None
    return _read(root)
