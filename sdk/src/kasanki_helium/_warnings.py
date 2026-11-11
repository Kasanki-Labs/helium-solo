# Copyright © 2026 Kasanki Labs.
# SPDX-License-Identifier: AGPL-3.0-only
# Licensed under the GNU AGPL v3 with an attribution term: see LICENSE and CREDIT.md.

"""Warnings the SDK gives at most once per process."""

from __future__ import annotations

import threading
import warnings


class HeliumWarning(UserWarning):
    """Something was recorded in a way replay may not reproduce exactly."""


_warned: set[str] = set()
_warned_lock = threading.Lock()


def first_time(keys: set[str]) -> list[str]:
    """Those of `keys` not seen before, sorted; all are seen from now on."""
    with _warned_lock:
        new = sorted(keys - _warned)
        _warned.update(new)
    return new


def warn_once(key: str, message: str, stacklevel: int = 3) -> None:
    if first_time({key}):
        warnings.warn(message, HeliumWarning, stacklevel=stacklevel)
