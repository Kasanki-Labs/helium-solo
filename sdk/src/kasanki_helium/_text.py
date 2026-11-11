# Copyright © 2026 Kasanki Labs.
# SPDX-License-Identifier: AGPL-3.0-only
# Licensed under the GNU AGPL v3 with an attribution term: see LICENSE and CREDIT.md.

"""Text from a recording is hostile: a model's
answer or a tool's result may hold terminal escape sequences, or characters
that hide or reorder what's shown. On a terminal, the ``helium`` commands
print them escaped, the way the engine's own commands do: control characters
as ``\\x1b``, hidden ones as ``\\u{202e}``. Newlines and tabs stay; so do
right-to-left letters and the joiners inside emoji.

Output that isn't a terminal (a file, a pipe, a dataset) is left exact.
"""

from __future__ import annotations

from typing import Any, TextIO

_BIDI = {0x061C, 0x200E, 0x200F, *range(0x202A, 0x202F), *range(0x2066, 0x206A)}
_ZERO_WIDTH = {
    0x00AD,
    0x034F,
    0x115F,
    0x1160,
    0x17B4,
    0x17B5,
    0x180E,
    *range(0x200B, 0x200E),
    *range(0x2060, 0x2065),
    0x3164,
    0xFEFF,
    0xFFA0,
    *range(0xFFF9, 0xFFFC),
}
_EMOJI_RANGES = (
    (0x1F000, 0x1FAFF),
    (0x2600, 0x27BF),
    (0x2300, 0x23FF),
    (0x2B00, 0x2BFF),
    (0x2190, 0x21FF),
    (0x25A0, 0x25FF),
)
_EMOJI_SINGLE = {
    0x3030,
    0x303D,
    0x3297,
    0x3299,
    0xA9,
    0xAE,
    0x2122,
    0x2139,
    0x2934,
    0x2935,
}


def _emoji(code: int | None) -> bool:
    if code is None:
        return False
    return code in _EMOJI_SINGLE or any(a <= code <= b for a, b in _EMOJI_RANGES)


def hidden(code: int, prev: int | None = None, nxt: int | None = None) -> str | None:
    """Why a character is hidden where it stands, as the engine names it, or None."""
    if code in (0x09, 0x0A) or (code == 0x0D and nxt == 0x0A):
        return None
    if code <= 0x1F or 0x7F <= code <= 0x9F:
        return "control"
    if code in _BIDI:
        return "bidi"
    if code == 0x200D and (_emoji(prev) or prev == 0xFE0F) and _emoji(nxt):
        return None
    if code in _ZERO_WIDTH:
        return "zero_width"
    if 0xE0000 <= code <= 0xE007F:
        return "tag"
    if (
        code in (0xFE0E, 0xFE0F)
        and prev is not None
        and hidden(prev) is None
        and not 0xFE00 <= prev <= 0xFE0F
    ):
        return None
    if 0xFE00 <= code <= 0xFE0F or 0xE0100 <= code <= 0xE01EF:
        return "variation"
    if code in (0x2028, 0x2029):
        return "separator"
    return None


def for_terminal(text: str) -> str:
    """``text`` with every control and hidden character escaped."""
    if text.isascii() and text.isprintable():
        return text
    codes = [ord(c) for c in text]
    out: list[str] = []
    for i, code in enumerate(codes):
        kind = hidden(
            code,
            codes[i - 1] if i else None,
            codes[i + 1] if i + 1 < len(codes) else None,
        )
        if kind is None:
            out.append(chr(code))
        elif kind == "control":
            out.append(f"\\x{code:02x}")
        else:
            out.append(f"\\u{{{code:x}}}")
    return "".join(out)


class TerminalWriter:
    """Wraps a terminal's stream so everything printed to it is escaped."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def write(self, text: str) -> int:
        self._stream.write(for_terminal(text))
        return len(text)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def guard_terminal(stream: TextIO) -> TextIO:
    """``stream`` wrapped when it's a terminal; as it is otherwise."""
    try:
        is_terminal = stream.isatty()
    except (AttributeError, ValueError):
        return stream
    return TerminalWriter(stream) if is_terminal else stream  # type: ignore[return-value]
