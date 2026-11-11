# Copyright © 2026 Kasanki Labs.
# SPDX-License-Identifier: AGPL-3.0-only
# Licensed under the GNU AGPL v3 with an attribution term: see LICENSE and CREDIT.md.

"""Open verifier: checks a Helium data folder without trusting Helium's own binary."""

import argparse
import io
import sys
from pathlib import Path

from kasanki_helium._ved25519 import public_key_from_pem
from kasanki_helium._vjson import JsonError, canonical
from kasanki_helium._vstore import (
    StoreReport,
    default_data_dir,
    for_terminal,
    is_imported,
    json_for_terminal,
    verify_store,
)


def verify(
    data_dir: str | Path | None = None,
    public_keys: list[str | Path] | None = None,
) -> StoreReport:
    """Verify a Helium data folder and return a StoreReport."""
    folder = default_data_dir() if data_dir is None else Path(data_dir)

    trusted_raw: list[bytes] = []
    if public_keys:
        for pk_entry in public_keys:
            p = Path(pk_entry)
            try:
                pem_text = p.read_text(encoding="utf-8")
            except OSError as e:
                raise OSError(f"{p}: {e}") from None
            try:
                raw_key = public_key_from_pem(pem_text)
            except ValueError as e:
                raise ValueError(f"{p}: {e}") from None
            trusted_raw.append(raw_key)

    return verify_store(folder, trusted_raw if trusted_raw else None)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for `python -m kasanki_helium.verify`."""
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", newline="\n")

    parser = argparse.ArgumentParser(
        prog="python -m kasanki_helium.verify",
        description="Verify a Helium data folder.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data folder (default: the system-wide service location).",
    )
    parser.add_argument(
        "--public-key",
        dest="public_keys",
        action="append",
        type=Path,
        default=[],
        help=(
            "Trust only these public key files (PEM). Default: the folder's own "
            "key. Auditors should pass the key they were given separately."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the report as JSON.",
    )

    args = parser.parse_args(argv)

    folder = args.data_dir if args.data_dir is not None else default_data_dir()

    try:
        report = verify(data_dir=folder, public_keys=args.public_keys)
    except (ValueError, TypeError, KeyError, OSError, JsonError) as e:
        sys.stderr.write(f"helium: {for_terminal(str(e))}\n")
        return 2

    if args.json:
        doc = report.to_json()
        bytes_out = canonical(doc, max_depth=8, max_bytes=1024 * 1024 * 1024)
        sys.stdout.write(json_for_terminal(bytes_out.decode("utf-8")) + "\n")
    else:
        text = str(report)
        sys.stdout.write(for_terminal(text))

        # Check for imported runs
        runs_dir = folder / "runs"
        imported_count = 0
        if runs_dir.is_dir():
            try:
                for entry in runs_dir.iterdir():
                    if entry.is_dir() and is_imported(entry.name):
                        imported_count += 1
            except OSError:
                pass
        if imported_count > 0:
            sys.stdout.write(
                f"{imported_count} imported run(s) are tamper-evident only from when "
                "Helium received them, not from when they happened\n"
            )

    return 0 if report.is_intact() else 1


if __name__ == "__main__":
    sys.exit(main())
