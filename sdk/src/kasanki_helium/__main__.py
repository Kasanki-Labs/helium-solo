# Copyright © 2026 Kasanki Labs.
# SPDX-License-Identifier: AGPL-3.0-only
# Licensed under the GNU AGPL v3 with an attribution term: see LICENSE and CREDIT.md.

"""``python -m kasanki_helium ...``: the commands the ``helium`` program hands to
Python (replay, test, fix and the rest) are Helium's tools.
This passes them on, or says what to install."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    try:
        from kasanki_helium_tools.__main__ import main as tools_main
    except ModuleNotFoundError:
        from kasanki_helium import _tools_missing

        if not _tools_missing():
            raise
        print(
            "kasanki-helium: this command needs Helium's tools: pip install helium-solo",
            file=sys.stderr,
        )
        return 2
    return tools_main(argv)


if __name__ == "__main__":
    sys.exit(main())
