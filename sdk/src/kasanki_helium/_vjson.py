# Copyright © 2026 Kasanki Labs.
# SPDX-License-Identifier: AGPL-3.0-only
# Licensed under the GNU AGPL v3 with an attribution term: see LICENSE and CREDIT.md.

"""Strict JSON reader and RFC 8785 canonical JSON serializer."""

import math
from typing import Any

MAX_SAFE_INTEGER = 9007199254740991


EXPONENT_FORM_FROM = 1e21


class JsonError(Exception):
    """Raised when JSON parsing or canonicalization fails."""


def _format_ecma_number(val: float) -> str:
    """Format a float according to ECMAScript ToString (RFC 8785 §3.2.2.3)."""
    if val == 0.0:
        return "0"
    sign = "-" if math.copysign(1.0, val) < 0 else ""
    if val < 0:
        val = -val

    s = repr(val)
    if "e" in s or "E" in s:
        parts = s.lower().split("e")
        digits_part = parts[0]
        exp = int(parts[1])
    else:
        digits_part = s
        exp = 0

    if "." in digits_part:
        int_p, frac_p = digits_part.split(".")
        frac_p = frac_p.rstrip("0")
        if not frac_p:
            digits = int_p
        else:
            digits = (int_p + frac_p).lstrip("0")
            exp -= len(frac_p)
    else:
        digits = digits_part.lstrip("0")

    k = len(digits)
    n = exp + k

    if k <= n <= 21:
        res = digits + "0" * (n - k)
    elif 0 < n <= 21:
        res = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        res = "0." + "0" * (-n) + digits
    elif k == 1:
        res = digits + "e" + ("+" if n - 1 >= 0 else "") + str(n - 1)
    else:
        res = (
            digits[0]
            + "."
            + digits[1:]
            + "e"
            + ("+" if n - 1 >= 0 else "")
            + str(n - 1)
        )

    return sign + res


def _write_string(s: str, out: list[str]) -> None:
    out.append('"')
    for c in s:
        code = ord(c)
        if c == '"':
            out.append('\\"')
        elif c == "\\":
            out.append("\\\\")
        elif c == "\b":
            out.append("\\b")
        elif c == "\f":
            out.append("\\f")
        elif c == "\n":
            out.append("\\n")
        elif c == "\r":
            out.append("\\r")
        elif c == "\t":
            out.append("\\t")
        elif code < 0x20:
            out.append(f"\\u{code:04x}")
        else:
            out.append(c)
    out.append('"')


def _write_value(val: Any, depth: int, max_depth: int, out: list[str]) -> None:
    if val is None:
        out.append("null")
    elif isinstance(val, bool):
        out.append("true" if val else "false")
    elif isinstance(val, int):
        if abs(val) > MAX_SAFE_INTEGER:
            raise JsonError(
                f"whole number beyond ±{MAX_SAFE_INTEGER}; write it as a string"
            )
        out.append(str(val))
    elif isinstance(val, float):
        if not math.isfinite(val):
            raise JsonError("number out of range")
        if val.is_integer() and (MAX_SAFE_INTEGER < abs(val) < EXPONENT_FORM_FROM):
            raise JsonError(
                f"whole number beyond ±{MAX_SAFE_INTEGER}; write it as a string"
            )
        out.append(_format_ecma_number(val))
    elif isinstance(val, str):
        _write_string(val, out)
    elif isinstance(val, (list, tuple)):
        depth += 1
        if depth > max_depth:
            raise JsonError(f"JSON is nested deeper than {max_depth} levels")
        out.append("[")
        for i, item in enumerate(val):
            if i > 0:
                out.append(",")
            _write_value(item, depth, max_depth, out)
        out.append("]")
    elif isinstance(val, dict):
        depth += 1
        if depth > max_depth:
            raise JsonError(f"JSON is nested deeper than {max_depth} levels")
        out.append("{")
        sorted_keys = sorted(val.keys(), key=lambda k: k.encode("utf-16-be"))
        for i, k in enumerate(sorted_keys):
            if not isinstance(k, str):
                raise JsonError("object key must be a string")
            if i > 0:
                out.append(",")
            _write_string(k, out)
            out.append(":")
            _write_value(val[k], depth, max_depth, out)
        out.append("}")
    else:
        raise JsonError(f"unsupported type for JSON: {type(val)}")


def canonical(value: Any, *, max_depth: int = 64, max_bytes: int = 1048576) -> bytes:
    """Serialize `value` into RFC 8785 canonical JSON bytes."""
    out: list[str] = []
    _write_value(value, 0, max_depth, out)
    res = "".join(out).encode("utf-8")
    if len(res) > max_bytes:
        raise JsonError(f"JSON is larger than {max_bytes} bytes")
    return res


class _Parser:
    def __init__(self, data: bytes, text: str, max_depth: int) -> None:
        self.data = data
        self.text = text
        self.pos = 0
        self.length = len(data)
        self.max_depth = max_depth

    def _syntax(self, reason: str) -> JsonError:
        return JsonError(f"byte {self.pos}: {reason}")

    def _peek(self) -> int | None:
        if self.pos < self.length:
            return self.data[self.pos]
        return None

    def _skip_whitespace(self) -> None:
        while self.pos < self.length and self.data[self.pos] in (32, 9, 10, 13):
            self.pos += 1

    def _expect(self, b: int, reason: str) -> None:
        if self._peek() == b:
            self.pos += 1
        else:
            raise self._syntax(reason)

    def _literal(self, word: bytes, value: Any) -> Any:
        if self.data[self.pos :].startswith(word):
            self.pos += len(word)
            return value
        raise self._syntax("expected a JSON value")

    def _check_depth(self, depth: int) -> None:
        if depth > self.max_depth:
            raise JsonError(f"JSON is nested deeper than {self.max_depth} levels")

    def _string(self) -> str:
        self.pos += 1  # opening quote
        chars: list[str] = []
        while True:
            start = self.pos
            while self.pos < self.length:
                b = self.data[self.pos]
                if b == 34 or b == 92 or b < 32:  # '"', '\\', or control
                    break
                self.pos += 1

            if self.pos > start:
                chars.append(self.data[start : self.pos].decode("utf-8"))

            if self.pos >= self.length:
                raise self._syntax("unterminated string")

            b = self.data[self.pos]
            if b == 34:  # '"'
                self.pos += 1
                return "".join(chars)
            elif b == 92:  # '\\'
                self.pos += 1
                chars.append(self._escape())
            else:
                raise self._syntax("control character in string")

    def _escape(self) -> str:
        if self.pos >= self.length:
            raise self._syntax("unterminated string")
        b = self.data[self.pos]
        self.pos += 1
        if b == 34:
            return '"'
        elif b == 92:
            return "\\"
        elif b == 47:
            return "/"
        elif b == 98:
            return "\b"
        elif b == 102:
            return "\f"
        elif b == 110:
            return "\n"
        elif b == 114:
            return "\r"
        elif b == 116:
            return "\t"
        elif b == 117:  # 'u'
            start = self.pos - 2  # start at '\'
            unit = self._hex4()
            if 0xD800 <= unit <= 0xDBFF:
                # High surrogate; must be followed by \uDC00..\uDFFF
                if not self.data[self.pos :].startswith(b"\\u"):
                    raise JsonError(f"byte {start}: lone surrogate")
                self.pos += 2
                low = self._hex4()
                if not (0xDC00 <= low <= 0xDFFF):
                    raise JsonError(f"byte {start}: lone surrogate")
                code = 0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00)
            elif 0xDC00 <= unit <= 0xDFFF:
                raise JsonError(f"byte {start}: lone surrogate")
            else:
                code = unit
            try:
                return chr(code)
            except ValueError:
                raise JsonError(f"byte {start}: lone surrogate") from None
        else:
            raise self._syntax("invalid escape")

    def _hex4(self) -> int:
        digits = self.data[self.pos : self.pos + 4]
        if len(digits) < 4:
            raise self._syntax("expected four hex digits")
        unit = 0
        for d in digits:
            if 48 <= d <= 57:  # 0-9
                v = d - 48
            elif 97 <= d <= 102:  # a-f
                v = d - 97 + 10
            elif 65 <= d <= 70:  # A-F
                v = d - 65 + 10
            else:
                raise self._syntax("expected four hex digits")
            unit = (unit << 4) | v
        self.pos += 4
        return unit

    def _number(self) -> int | float:
        start = self.pos
        if self._peek() == 45:  # '-'
            self.pos += 1

        b = self._peek()
        if b == 48:  # '0'
            self.pos += 1
        elif b is not None and 49 <= b <= 57:  # '1'-'9'
            while self.pos < self.length and 48 <= self.data[self.pos] <= 57:
                self.pos += 1
        else:
            raise self._syntax("invalid number")

        whole_literal = True
        if self._peek() == 46:  # '.'
            whole_literal = False
            self.pos += 1
            start_digits = self.pos
            while self.pos < self.length and 48 <= self.data[self.pos] <= 57:
                self.pos += 1
            if self.pos == start_digits:
                raise self._syntax("expected digits after '.'")

        if self._peek() in (101, 69):  # 'e', 'E'
            whole_literal = False
            self.pos += 1
            if self._peek() in (43, 45):  # '+', '-'
                self.pos += 1
            start_digits = self.pos
            while self.pos < self.length and 48 <= self.data[self.pos] <= 57:
                self.pos += 1
            if self.pos == start_digits:
                raise self._syntax("expected digits in exponent")

        literal = self.data[start : self.pos].decode("ascii")
        if whole_literal:
            magnitude = literal.lstrip("-")
            safe = len(magnitude) <= 16 and int(magnitude) <= MAX_SAFE_INTEGER
            if not safe:
                raise JsonError(
                    f"byte {start}: whole number beyond ±{MAX_SAFE_INTEGER}; write it as a string"
                )
            return int(literal)

        try:
            val = float(literal)
        except ValueError:
            raise JsonError(f"byte {start}: number out of range") from None

        if not math.isfinite(val):
            raise JsonError(f"byte {start}: number out of range")

        if val.is_integer() and (MAX_SAFE_INTEGER < abs(val) < EXPONENT_FORM_FROM):
            raise JsonError(
                f"byte {start}: whole number beyond ±{MAX_SAFE_INTEGER}; write it as a string"
            )

        return val

    def _object(self, depth: int) -> dict[str, Any]:
        self._check_depth(depth)
        self.pos += 1  # '{'
        obj: dict[str, Any] = {}
        seen: set[str] = set()
        self._skip_whitespace()
        if self._peek() == 125:  # '}'
            self.pos += 1
            return obj

        while True:
            if self._peek() != 34:  # '"'
                raise self._syntax("expected a string key")
            key = self._string()
            if key in seen:
                raise JsonError(f'duplicate key "{key}"')
            seen.add(key)
            self._skip_whitespace()
            self._expect(58, "expected ':' after key")
            self._skip_whitespace()
            val = self._value(depth)
            obj[key] = val
            self._skip_whitespace()
            b = self._peek()
            if b == 44:  # ','
                self.pos += 1
                self._skip_whitespace()
            elif b == 125:  # '}'
                self.pos += 1
                return obj
            else:
                raise self._syntax("expected ',' or '}'")

    def _array(self, depth: int) -> list[Any]:
        self._check_depth(depth)
        self.pos += 1  # '['
        items: list[Any] = []
        self._skip_whitespace()
        if self._peek() == 93:  # ']'
            self.pos += 1
            return items

        while True:
            items.append(self._value(depth))
            self._skip_whitespace()
            b = self._peek()
            if b == 44:  # ','
                self.pos += 1
                self._skip_whitespace()
            elif b == 93:  # ']'
                self.pos += 1
                return items
            else:
                raise self._syntax("expected ',' or ']'")

    def _value(self, depth: int) -> Any:
        b = self._peek()
        if b == 123:  # '{'
            return self._object(depth + 1)
        elif b == 91:  # '['
            return self._array(depth + 1)
        elif b == 34:  # '"'
            return self._string()
        elif b == 116:  # 't'
            return self._literal(b"true", True)
        elif b == 102:  # 'f'
            return self._literal(b"false", False)
        elif b == 110:  # 'n'
            return self._literal(b"null", None)
        elif b == 45 or (b is not None and 48 <= b <= 57):  # '-', '0'-'9'
            return self._number()
        else:
            raise self._syntax("expected a JSON value")

    def parse(self) -> Any:
        self._skip_whitespace()
        val = self._value(0)
        self._skip_whitespace()
        if self.pos != self.length:
            raise self._syntax("unexpected characters after the JSON value")
        return val


def parse(data: bytes, *, max_depth: int = 64, max_bytes: int = 1048576) -> Any:
    """Parse JSON bytes strictly according to Helium rules."""
    if len(data) > max_bytes:
        raise JsonError(f"JSON is larger than {max_bytes} bytes")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise JsonError("JSON is not valid UTF-8") from None

    parser = _Parser(data, text, max_depth)
    return parser.parse()
