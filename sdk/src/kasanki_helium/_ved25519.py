# Copyright © 2026 Kasanki Labs.
# SPDX-License-Identifier: AGPL-3.0-only
# Licensed under the GNU AGPL v3 with an attribution term: see LICENSE and CREDIT.md.

"""Pure-Python strict Ed25519 signature verifier (RFC 8032)."""

import base64
import binascii
import hashlib

# Field prime p = 2^255 - 19
_P = 2**255 - 19

# Curve constant d = -121665 / 121666 mod p
_D = (-121665 * pow(121666, _P - 2, _P)) % _P
_TWO_D = (2 * _D) % _P

# Square root of -1 mod p
_I = pow(2, (_P - 1) // 4, _P)

# Subgroup prime order L
_L = 2**252 + 27742317777372353535851937790883648493

# Base point B = (bx, by)
_BY = (4 * pow(5, _P - 2, _P)) % _P
_U = (_BY * _BY - 1) % _P
_V = (_D * _BY * _BY + 1) % _P
_X2 = (_U * pow(_V, _P - 2, _P)) % _P
_BX = pow(_X2, (_P + 3) // 8, _P)
if (_BX * _BX - _X2) % _P != 0:
    _BX = (_BX * _I) % _P
if _BX % 2 != 0:
    _BX = _P - _BX

# Base point in extended coordinates: (X, Y, Z, T) where x = X/Z, y = Y/Z, T = X*Y/Z
_B_EXT = (_BX, _BY, 1, (_BX * _BY) % _P)

# SPKI DER prefix for Ed25519 (12 bytes)
_SPKI_ED25519_PREFIX = bytes.fromhex("302a300506032b6570032100")

# Point tuple in extended coordinates: (X, Y, Z, T)
_Point = tuple[int, int, int, int]


def _ext_add(p1: _Point, p2: _Point) -> _Point:
    x1, y1, z1, t1 = p1
    x2, y2, z2, t2 = p2
    a = ((y1 - x1) * (y2 - x2)) % _P
    b = ((y1 + x1) * (y2 + x2)) % _P
    c = (t1 * _TWO_D * t2) % _P
    d = (2 * z1 * z2) % _P
    e = (b - a) % _P
    f = (d - c) % _P
    g = (d + c) % _P
    h = (b + a) % _P
    return ((e * f) % _P, (g * h) % _P, (f * g) % _P, (e * h) % _P)


def _ext_double(p1: _Point) -> _Point:
    x1, y1, z1, _ = p1
    a = (x1 * x1) % _P
    b = (y1 * y1) % _P
    c = (2 * z1 * z1) % _P
    h = (a + b) % _P
    e = (h - pow(x1 + y1, 2, _P)) % _P
    g = (a - b) % _P
    f = (c + g) % _P
    return ((e * f) % _P, (g * h) % _P, (f * g) % _P, (e * h) % _P)


def _ext_scalar_mult(p1: _Point, s: int) -> _Point:
    r: _Point = (0, 1, 1, 0)
    base = p1
    while s > 0:
        if s & 1:
            r = _ext_add(r, base)
        base = _ext_double(base)
        s >>= 1
    return r


def _is_small_order(p1: _Point) -> bool:
    """True if [8]P == O (neutral element), i.e. P is in the 8-torsion subgroup."""
    p8 = p1
    for _ in range(3):
        p8 = _ext_double(p8)
    x, y, z, _ = p8
    return (x % _P == 0) and ((y - z) % _P == 0)


def _points_equal(p1: _Point, p2: _Point) -> bool:
    x1, y1, z1, _ = p1
    x2, y2, z2, _ = p2
    return ((x1 * z2 - x2 * z1) % _P == 0) and ((y1 * z2 - y2 * z1) % _P == 0)


def _decompress_point(s: bytes) -> _Point | None:
    if len(s) != 32:
        return None
    val = int.from_bytes(s, "little")
    sign = (val >> 255) & 1
    y = val & ((1 << 255) - 1)
    if y >= _P:
        return None  # non-canonical y encoding
    u = (y * y - 1) % _P
    v = (_D * y * y + 1) % _P
    if v == 0:
        if u == 0:
            x = 0
        else:
            return None
    else:
        uv7 = (u * pow(v, 7, _P)) % _P
        cand = (u * pow(v, 3, _P) * pow(uv7, (_P - 5) // 8, _P)) % _P
        vc2 = (v * cand * cand) % _P
        if vc2 == u:
            x = cand
        elif vc2 == (-u) % _P:
            x = (cand * _I) % _P
        else:
            return None
    if x == 0 and sign != 0:
        return None  # canonical x=0 must have sign 0
    if (x % 2) != sign:
        x = _P - x
    return (x, y, 1, (x * y) % _P)


def verify_strict(public_key: bytes, message: bytes, signature: bytes) -> bool:
    """Strict RFC 8032 Ed25519 verification matching ed25519-dalek verify_strict."""
    if len(public_key) != 32 or len(signature) != 64:
        return False

    r_bytes = signature[:32]
    s_bytes = signature[32:]
    s = int.from_bytes(s_bytes, "little")
    if s >= _L:
        return False  # non-canonical S (must be strictly less than L)

    a_pt = _decompress_point(public_key)
    if a_pt is None:
        return False

    r_pt = _decompress_point(r_bytes)
    if r_pt is None:
        return False

    # Reject small-order public keys or small-order R
    if _is_small_order(a_pt) or _is_small_order(r_pt):
        return False

    h = hashlib.sha512(r_bytes + public_key + message).digest()
    k = int.from_bytes(h, "little") % _L

    sb = _ext_scalar_mult(_B_EXT, s)
    ka = _ext_scalar_mult(a_pt, k)
    r_plus_ka = _ext_add(r_pt, ka)

    return _points_equal(sb, r_plus_ka)


def public_key_from_pem(text: str) -> bytes:
    """Parse an Ed25519 SubjectPublicKeyInfo PEM string and return 32 raw bytes."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not (
        lines
        and lines[0] == "-----BEGIN PUBLIC KEY-----"
        and lines[-1] == "-----END PUBLIC KEY-----"
    ):
        raise ValueError("not a valid Ed25519 PEM key")

    try:
        der = base64.b64decode("".join(lines[1:-1]))
    except (ValueError, binascii.Error):
        raise ValueError("not a valid Ed25519 PEM key") from None

    if len(der) != 44 or not der.startswith(_SPKI_ED25519_PREFIX):
        raise ValueError("not a valid Ed25519 PEM key")

    return der[12:]
