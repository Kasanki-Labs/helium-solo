# Copyright © 2026 Kasanki Labs.
# SPDX-License-Identifier: AGPL-3.0-only
# Licensed under the GNU AGPL v3 with an attribution term: see LICENSE and CREDIT.md.

"""Store verification logic mirroring the Helium Rust engine."""

import hashlib
import os
from enum import Enum
from pathlib import Path
from typing import Any

from kasanki_helium._ved25519 import verify_strict
from kasanki_helium._vjson import JsonError, canonical, parse

# =============================================================================
# helium-core/src/hash.rs
# =============================================================================

HASH_PREFIX = "sha256:"


class Domain:
    GENESIS = "helium/genesis/v0"
    EVENT = "helium/event/v0"
    EVENT_BODY = "helium/event-body/v0"
    MERKLE_LEAF = "helium/merkle-leaf/v0"
    MERKLE_NODE = "helium/merkle-node/v0"
    CHECKPOINT_GENESIS = "helium/checkpoint-genesis/v0"
    CHECKPOINT = "helium/checkpoint/v0"
    PUBLIC_KEY = "helium/public-key/v0"
    BLOB = "helium/blob/v0"
    ANCHOR = "helium/anchor/v0"
    WITNESS_GENESIS = "helium/witness-genesis/v0"
    WITNESS_ENTRY = "helium/witness-entry/v0"


def labelled_hash(domain_label: str, data: bytes) -> str:
    """Compute SHA-256(label || 0x00 || data) as 'sha256:<64 hex>'."""
    h = hashlib.sha256(domain_label.encode("utf-8") + b"\x00" + data).hexdigest()
    return f"{HASH_PREFIX}{h}"


def raw_labelled_hash(domain_label: str, data: bytes) -> bytes:
    """Compute SHA-256(label || 0x00 || data) as 32 raw bytes."""
    return hashlib.sha256(domain_label.encode("utf-8") + b"\x00" + data).digest()


def is_valid_hash(s: str) -> bool:
    """Check if s is 'sha256:' followed by 64 lowercase hex digits."""
    if not s.startswith(HASH_PREFIX) or len(s) != len(HASH_PREFIX) + 64:
        return False
    hex_part = s[len(HASH_PREFIX) :]
    return all(c in "0123456789abcdef" for c in hex_part)


# =============================================================================
# helium-core/src/keys.rs
# =============================================================================

SIGNATURE_PREFIX = "ed25519:"
CHECKPOINT_SIGNATURE_LABEL = "helium/checkpoint-signature/v0"


def compute_key_id(public_key: bytes) -> str:
    """Fingerprint of an Ed25519 public key."""
    doc = {"algorithm": "ed25519", "public_key": public_key.hex()}
    bytes_data = canonical(doc)
    return labelled_hash(Domain.PUBLIC_KEY, bytes_data)


def verify_signature(
    public_key: bytes, label: str, message: bytes, signature_str: str
) -> bool:
    """Check an Ed25519 signature formatted as 'ed25519:<128 hex>'."""
    if (
        not signature_str.startswith(SIGNATURE_PREFIX)
        or len(signature_str) != len(SIGNATURE_PREFIX) + 128
    ):
        return False
    sig_hex = signature_str[len(SIGNATURE_PREFIX) :]
    if not all(c in "0123456789abcdef" for c in sig_hex):
        return False
    sig_bytes = bytes.fromhex(sig_hex)
    m = label.encode("utf-8") + b"\x00" + message
    return verify_strict(public_key, m, sig_bytes)


# =============================================================================
# helium-core/src/chain.rs
# =============================================================================

CHAIN_FIELDS = ["run_id", "seq", "prev_hash", "hash"]


def parse_u64(s: str) -> int | None:
    """Parse a 64-bit unsigned integer written in canonical decimal string form."""
    if not s or not s.isdigit():
        return None
    if s != "0" and s.startswith("0"):
        return None
    val = int(s)
    if 0 <= val <= 18446744073709551615:
        return val
    return None


def check_uuid(field: str, value: str) -> None:
    """Check that value is a lowercase hyphenated UUID."""
    b = value.encode("ascii", "replace")
    well_formed = len(b) == 36 and all(
        b[i] == 45
        if i in (8, 13, 18, 23)
        else (48 <= b[i] <= 57 or (97 <= b[i] <= 102))
        for i in range(36)
    )
    if not well_formed:
        raise ValueError(f'{field} "{value}" is not a lowercase hyphenated UUID')


def genesis(deployment_id: str, run_id: str) -> str:
    """Starting hash of a run's hash chain."""
    check_uuid("deployment_id", deployment_id)
    check_uuid("run_id", time_ordered(run_id))
    input_doc = {"deployment_id": deployment_id, "run_id": run_id}
    data = canonical(input_doc)
    return labelled_hash(Domain.GENESIS, data)


def body_hash(event: dict[str, Any]) -> str:
    """Hash of an event's content: everything except the 4 chain fields."""
    body = {k: v for k, v in event.items() if k not in CHAIN_FIELDS}
    data = canonical(body)
    return labelled_hash(Domain.EVENT_BODY, data)


def link_record_of_event(event: dict[str, Any]) -> dict[str, str]:
    """Content-free record an event's hash is computed from."""
    seq_str = event.get("seq")
    if not isinstance(seq_str, str) or parse_u64(seq_str) is None:
        raise ValueError('"seq" is not a canonical whole number')
    run_id = event.get("run_id")
    if not isinstance(run_id, str):
        raise TypeError('"run_id" is missing or not a string')
    check_uuid("run_id", time_ordered(run_id))
    prev_hash = event.get("prev_hash")
    if not isinstance(prev_hash, str) or not is_valid_hash(prev_hash):
        raise ValueError('"prev_hash" is not a valid hash')
    b_hash = body_hash(event)
    return {
        "body_hash": b_hash,
        "prev_hash": prev_hash,
        "run_id": run_id,
        "seq": seq_str,
    }


def event_hash(event: dict[str, Any]) -> str:
    """Hash of a stored event, computed over its link record."""
    link = link_record_of_event(event)
    return labelled_hash(Domain.EVENT, canonical(link))


def following_seq(event: dict[str, Any], seq: int) -> int:
    """Seq of the event after `event` stored at `seq`."""
    gr = gap_range(event)
    if gr is not None and gr[0] == seq:
        return min(gr[1] + 1, 18446744073709551615)
    return min(seq + 1, 18446744073709551615)


def verify_run_chain(
    deployment_id: str, run_id: str, events: list[dict[str, Any]]
) -> tuple[list[str], str | None]:
    """Verify chain integrity for a run's events in stored order."""
    problems: list[str] = []
    expected_seq = 0
    previous: str | None = genesis(deployment_id, run_id)
    last_head: str | None = None

    for position, event in enumerate(events):
        seq_raw = event.get("seq")
        run_id_raw = event.get("run_id")
        prev_hash_raw = event.get("prev_hash")
        hash_raw = event.get("hash")

        # Validate chain fields presence and form
        malformed_reason: str | None = None
        seq_val: int | None = None
        if not isinstance(run_id_raw, str):
            malformed_reason = '"run_id" is missing or not a string'
        elif not isinstance(seq_raw, str):
            malformed_reason = '"seq" is missing or not a string'
        else:
            seq_val = parse_u64(seq_raw)
            if seq_val is None:
                malformed_reason = (
                    f'"seq" {seq_raw!r} is not a whole number in canonical form'
                )
        if malformed_reason is None:
            if not isinstance(prev_hash_raw, str) or not is_valid_hash(prev_hash_raw):
                malformed_reason = '"prev_hash": not a hash: expected "sha256:" followed by 64 lowercase hex digits'
            elif not isinstance(hash_raw, str) or not is_valid_hash(hash_raw):
                malformed_reason = '"hash": not a hash: expected "sha256:" followed by 64 lowercase hex digits'

        if malformed_reason is not None or seq_val is None:
            problems.append(f"event #{position}: malformed: {malformed_reason}")
            previous = None
            expected_seq += 1
            continue

        assert isinstance(run_id_raw, str)
        assert isinstance(prev_hash_raw, str)
        assert isinstance(hash_raw, str)

        if run_id_raw != run_id:
            problems.append(f"seq {seq_val}: belongs to run {run_id_raw}")

        try:
            actual_hash = event_hash(event)
            if actual_hash != hash_raw:
                problems.append(f"seq {seq_val}: content changed")
        except (ValueError, TypeError, KeyError, OSError, JsonError) as e:
            problems.append(f"event #{position}: malformed: {e}")

        if seq_val == expected_seq:
            if previous is not None and previous != prev_hash_raw:
                problems.append(
                    f"seq {seq_val}: does not follow on from the previous event"
                )
            expected_seq = following_seq(event, seq_val)
        elif seq_val > expected_seq:
            if expected_seq == seq_val - 1:
                problems.append(f"seq {expected_seq} missing")
            else:
                problems.append(f"seq {expected_seq}\u2013{seq_val - 1} missing")
            expected_seq = following_seq(event, seq_val)
        else:
            problems.append(
                f"seq {seq_val}: out of order (expected seq {expected_seq})"
            )

        previous = hash_raw
        last_head = hash_raw

    return problems, last_head


# =============================================================================
# helium-core/src/event.rs
# =============================================================================


class Stage(str, Enum):
    """Which fields an event is expected to have."""

    BODY = "body"
    CHAINED = "chained"


class EventError(ValueError):
    """Why an event breaks the rules. field is a path such as links[0].rel."""

    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"{field}: {reason}")


FORMAT_VERSION = 0

KINDS = {
    "run.start",
    "run.end",
    "llm.call",
    "tool.call",
    "clock.read",
    "random.draw",
    "id.generate",
    "message",
    "error",
    "profile",
    "policy",
    "gap",
    "run.verified",
    "reward",
    "effects",
}

EFFECT_CHANGES = {"added", "changed", "removed"}
LINK_RELS = {"parent", "caused_by", "message_to", "message_from", "branched_from"}
BLOB_ROLES = {"request", "response", "state", "chunk"}
BODY_FIELDS = [
    "format_version",
    "kind",
    "wall_time",
    "mono_time",
    "links",
    "payload",
    "blobs",
]
CHAINED_EXTRA = ["run_id", "seq", "prev_hash", "hash"]
GAP_SOURCES = {"sdk", "monitor"}
IMPORTED_PREFIX = "trace-"


def is_imported(run_id: str) -> bool:
    return run_id.startswith(IMPORTED_PREFIX)


def time_ordered(run_id: str) -> str:
    if run_id.startswith(IMPORTED_PREFIX):
        return run_id[len(IMPORTED_PREFIX) :]
    return run_id


def check_run_id(s: str) -> None:
    bare = time_ordered(s)
    check_uuid("run_id", bare)
    if not (bare[14] == "7" and bare[19] in "89ab"):
        raise ValueError(f"{s!r} is not a UUID version 7")


def check_wall_time(s: str) -> None:
    bad_msg = f"{s!r} is not UTC time as YYYY-MM-DDTHH:MM:SS.nnnnnnnnnZ"
    if (
        len(s) != 30
        or s[4] != "-"
        or s[7] != "-"
        or s[10] != "T"
        or s[13] != ":"
        or s[16] != ":"
        or s[19] != "."
        or s[29] != "Z"
    ):
        raise ValueError(bad_msg)

    parts = [s[:4], s[5:7], s[8:10], s[11:13], s[14:16], s[17:19], s[20:29]]
    if not all(p.isdigit() for p in parts):
        raise ValueError(bad_msg)

    year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
    hour, minute, second = int(parts[3]), int(parts[4]), int(parts[5])

    leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    days_in_month = {
        1: 31,
        2: 29 if leap else 28,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 30,
        10: 31,
        11: 30,
        12: 31,
    }
    dim = days_in_month.get(month)
    if dim is None or year == 0 or day == 0 or day > dim:
        raise ValueError(bad_msg)
    if hour > 23 or minute > 59 or second > 59:
        raise ValueError(bad_msg)


def wall_time_nanos(s: str) -> int:
    check_wall_time(s)
    year = int(s[0:4])
    month = int(s[5:7])
    day = int(s[8:10])
    hour = int(s[11:13])
    minute = int(s[14:16])
    second = int(s[17:19])
    nanos = int(s[20:29])

    y = year - 1 if month <= 2 else year
    era = y // 400
    yoe = y - era * 400
    doy = (153 * ((month + 9) % 12) + 2) // 5 + day - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    days = era * 146097 + doe - 719468
    seconds = days * 86400 + hour * 3600 + minute * 60 + second
    return seconds * 1000000000 + nanos


def gap_range(event: dict[str, Any]) -> tuple[int, int] | None:
    if event.get("kind") != "gap":
        return None
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None
    from_raw = payload.get("from_seq")
    to_raw = payload.get("to_seq")
    if not isinstance(from_raw, str) or not isinstance(to_raw, str):
        return None
    from_seq = parse_u64(from_raw)
    to_seq = parse_u64(to_raw)
    if from_seq is None or to_seq is None or to_seq < from_seq:
        return None
    return (from_seq, to_seq)


def _is_dotted_name(s: str) -> bool:
    if not s:
        return False
    parts = s.split(".")
    for part in parts:
        if not part or not part[0].islower():
            return False
        if not all(c.islower() or c.isdigit() or c == "_" for c in part):
            return False
    return True


def _check_name(name: str, known: set[str]) -> None:
    if name in known:
        return
    if name.startswith("custom."):
        rest = name[len("custom.") :]
        if _is_dotted_name(rest):
            return
        raise ValueError(
            f"{name!r}: custom names are lowercase letters, digits and '_', in dot-separated parts"
        )
    raise ValueError(f'{name!r} is not a known name; use "custom.<name>" for your own')


def _check_media_type(s: str) -> None:
    parts = s.split("/")
    if len(parts) != 2:
        raise ValueError(
            f'{s!r} is not a lowercase "type/subtype" media type without parameters'
        )
    t, sub = parts

    def valid_part(p: str) -> bool:
        if not (1 <= len(p) <= 127):
            return False
        if not (p[0].islower() or p[0].isdigit()):
            return False
        return all(c.islower() or c.isdigit() or c in "!#$&^_.+-" for c in p)

    if not valid_part(t) or not valid_part(sub):
        raise ValueError(
            f'{s!r} is not a lowercase "type/subtype" media type without parameters'
        )


def _exact_fields(
    obj: dict[str, Any], prefix: str, required: list[str], optional: list[str]
) -> None:
    for name in required:
        if name not in obj:
            raise EventError(f"{prefix}{name}", "missing")
    req_set = set(required)
    opt_set = set(optional)
    for name in obj:
        if name not in req_set and name not in opt_set:
            raise EventError(f"{prefix}{name}", "unknown field")


def check_event(event: dict[str, Any], stage: Stage | str) -> None:
    """Validate event fields according to spec."""
    stage_str = stage.value if isinstance(stage, Stage) else str(stage)
    expected = (
        BODY_FIELDS
        if stage_str in ("body", Stage.BODY)
        else BODY_FIELDS + CHAINED_EXTRA
    )
    _exact_fields(event, "", expected, [])

    fv = event.get("format_version")
    if not isinstance(fv, (int, float)) or isinstance(fv, bool) or fv != FORMAT_VERSION:
        raise EventError("format_version", "must be the number 0")

    kind = event.get("kind")
    if not isinstance(kind, str):
        raise EventError("kind", "must be a string")
    try:
        _check_name(kind, KINDS)
    except ValueError as e:
        raise EventError("kind", str(e)) from None

    wt = event.get("wall_time")
    if not isinstance(wt, str):
        raise EventError("wall_time", "must be a string")
    try:
        check_wall_time(wt)
    except ValueError as e:
        raise EventError("wall_time", str(e)) from None

    mono = event.get("mono_time")
    if not isinstance(mono, str) or parse_u64(mono) is None:
        raise EventError(
            "mono_time",
            "must be a whole number written as a string, with no sign or leading zeros",
        )

    payload = event.get("payload")
    if not isinstance(payload, dict):
        raise EventError("payload", "must be an object")

    links = event.get("links")
    if not isinstance(links, list):
        raise EventError("links", "must be an array")
    for i, link in enumerate(links):
        path = f"links[{i}]."
        if not isinstance(link, dict):
            raise EventError(f"links[{i}]", "must be an object")
        _exact_fields(link, path, ["rel", "run_id"], ["seq"])
        rel = link.get("rel")
        if not isinstance(rel, str):
            raise EventError(f"{path}rel", "must be a string")
        try:
            _check_name(rel, LINK_RELS)
        except ValueError as e:
            raise EventError(f"{path}rel", str(e)) from None
        link_run = link.get("run_id")
        if not isinstance(link_run, str):
            raise EventError(f"{path}run_id", "must be a string")
        try:
            check_uuid("run_id", time_ordered(link_run))
        except ValueError as e:
            raise EventError(f"{path}run_id", str(e)) from None
        if "seq" in link:
            lseq = link.get("seq")
            if not isinstance(lseq, str) or parse_u64(lseq) is None:
                raise EventError(
                    f"{path}seq",
                    "must be a whole number written as a string, with no sign or leading zeros",
                )

    blobs = event.get("blobs")
    if not isinstance(blobs, list):
        raise EventError("blobs", "must be an array")
    for i, blob in enumerate(blobs):
        path = f"blobs[{i}]."
        if not isinstance(blob, dict):
            raise EventError(f"blobs[{i}]", "must be an object")
        _exact_fields(blob, path, ["hash", "size", "media_type", "role"], [])
        bhash = blob.get("hash")
        if not isinstance(bhash, str) or not is_valid_hash(bhash):
            raise EventError(
                f"{path}hash",
                'not a hash: expected "sha256:" followed by 64 lowercase hex digits',
            )
        bsize = blob.get("size")
        if not isinstance(bsize, str) or parse_u64(bsize) is None:
            raise EventError(
                f"{path}size",
                "must be a whole number written as a string, with no sign or leading zeros",
            )
        mt = blob.get("media_type")
        if not isinstance(mt, str):
            raise EventError(f"{path}media_type", "must be a string")
        try:
            _check_media_type(mt)
        except ValueError as e:
            raise EventError(f"{path}media_type", str(e)) from None
        role = blob.get("role")
        if not isinstance(role, str):
            raise EventError(f"{path}role", "must be a string")
        try:
            _check_name(role, BLOB_ROLES)
        except ValueError as e:
            raise EventError(f"{path}role", str(e)) from None

    if stage_str in ("chained", Stage.CHAINED):
        er = event.get("run_id")
        if not isinstance(er, str):
            raise EventError("run_id", "must be a string")
        try:
            check_run_id(er)
        except ValueError as e:
            raise EventError("run_id", str(e)) from None

        es = event.get("seq")
        if not isinstance(es, str) or parse_u64(es) is None:
            raise EventError(
                "seq",
                "must be a whole number written as a string, with no sign or leading zeros",
            )

        ep = event.get("prev_hash")
        if not isinstance(ep, str) or not is_valid_hash(ep):
            raise EventError(
                "prev_hash",
                'not a hash: expected "sha256:" followed by 64 lowercase hex digits',
            )

        eh = event.get("hash")
        if not isinstance(eh, str) or not is_valid_hash(eh):
            raise EventError(
                "hash",
                'not a hash: expected "sha256:" followed by 64 lowercase hex digits',
            )


def _check_gap(event: dict[str, Any], seq: int) -> None:
    payload = event.get("payload")
    if not isinstance(payload, dict):
        raise TypeError("payload: must be an object")
    _exact_fields(payload, "payload.", ["from_seq", "to_seq", "reason", "source"], [])
    from_raw = payload.get("from_seq")
    to_raw = payload.get("to_seq")
    if not isinstance(from_raw, str) or parse_u64(from_raw) is None:
        raise ValueError(
            "payload.from_seq: must be a whole number written as a string, with no sign or leading zeros"
        )
    if not isinstance(to_raw, str) or parse_u64(to_raw) is None:
        raise ValueError(
            "payload.to_seq: must be a whole number written as a string, with no sign or leading zeros"
        )
    from_seq = int(from_raw)
    to_seq = int(to_raw)
    if from_seq != seq:
        raise ValueError(
            f"payload.from_seq: seq {seq}: a gap starts at its own seq, not {from_seq}"
        )
    if to_seq < from_seq:
        raise ValueError(f"payload.to_seq: seq {seq}: must not be before from_seq")
    if to_seq == 18446744073709551615:
        raise ValueError("payload.to_seq: leaves no seq for the next event")
    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("payload.reason: must not be empty")
    source = payload.get("source")
    if not isinstance(source, str) or source not in GAP_SOURCES:
        raise ValueError(f"payload.source: must be one of {sorted(GAP_SOURCES)!r}")


def _check_effects_payload(event: dict[str, Any], kind: str) -> None:
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return
    if kind == "effects":
        _exact_fields(payload, "payload.", ["effects"], [])
    elif kind == "tool.call" and "effects" in payload:
        pass
    else:
        return

    items = payload.get("effects")
    if not isinstance(items, list):
        raise TypeError("payload.effects: must be an array")
    for i, item in enumerate(items):
        path = f"payload.effects[{i}]"
        if not isinstance(item, dict):
            raise TypeError(f"{path}: must be an object")
        _exact_fields(item, f"{path}.", ["after", "before", "change", "target"], [])
        target = item.get("target")
        if not isinstance(target, str):
            raise TypeError(f"{path}.target: must be a string")
        if not (1 <= len(target.encode("utf-8")) <= 4096):
            raise ValueError(f"{path}.target: must be 1 to 4096 bytes")
        change = item.get("change")
        if not isinstance(change, str) or change not in EFFECT_CHANGES:
            raise ValueError(
                f"{path}.change: must be one of {sorted(EFFECT_CHANGES)!r}"
            )

        before = item.get("before")
        after = item.get("after")

        def check_hash_side(val: Any, name: str, pfx: str) -> str | None:
            if val is None:
                return None
            if not isinstance(val, str) or not is_valid_hash(val):
                raise ValueError(
                    f'{pfx}.{name}: not a hash: expected "sha256:" followed by 64 lowercase hex digits'
                )
            return val

        b_val = check_hash_side(before, "before", path)
        a_val = check_hash_side(after, "after", path)

        fits = False
        if change == "added":
            fits = b_val is None and a_val is not None
        elif change == "removed":
            fits = b_val is not None and a_val is None
        elif change == "changed":
            fits = b_val is not None and a_val is not None and b_val != a_val
        if not fits:
            raise ValueError(f"{path}: {change}: before and after don't fit it")


class RunRules:
    """Check order and transitions of events within one run."""

    def __init__(self) -> None:
        self.next_seq = 0
        self.rollout = False
        self.training = False
        self.rewarded = False
        self.effected = False
        self.last_mono = 0
        self.ended = False
        self.verified = False

    def check(self, event: dict[str, Any]) -> None:
        seq = self.next_seq
        kind = event.get("kind")
        if not isinstance(kind, str):
            kind = ""
        mono_raw = event.get("mono_time")
        mono = parse_u64(mono_raw) if isinstance(mono_raw, str) else 0
        if mono is None:
            mono = 0

        def at(reason: str) -> ValueError:
            return ValueError(f"kind: seq {seq}: {reason}")

        if self.verified:
            raise at("nothing may follow run.verified")
        if self.ended and kind != "run.verified":
            raise at(f"{kind} after run.end; only run.verified may follow")
        if self.rewarded and kind != "run.end" and not self.ended:
            raise at(f"{kind} after reward; only run.end may follow")
        if self.effected and kind not in ("reward", "run.end") and not self.ended:
            raise at(f"{kind} after effects; only reward or run.end may follow")

        required = None
        if seq == 0:
            required = "run.start"
        elif seq == 1:
            required = "profile"
        elif seq == 2 and self.rollout:
            required = "policy"

        if required is not None:
            if kind != required:
                raise at(f"must be {required}, not {kind}")
        else:
            if kind == "run.start":
                raise at("run.start only at seq 0")
            elif kind == "profile":
                raise at("profile only at seq 1")
            elif kind == "policy" and self.rollout:
                raise at("policy only at seq 2")
            elif kind == "policy":
                raise at("policy only in rollouts")
            elif kind == "run.verified" and not self.ended:
                raise at("run.verified only after run.end")
            elif kind == "reward" and not self.training:
                raise at("reward only in training episodes")
            elif kind == "reward" and self.rewarded:
                raise at("only one reward")
            elif kind == "effects" and self.effected:
                raise at("only one effects event")

        if kind == "gap":
            _check_gap(event, seq)
        _check_effects_payload(event, kind)

        if seq == 0 and mono != 0:
            raise ValueError('mono_time: seq 0: must be "0" (time since run start)')
        if mono < self.last_mono:
            raise ValueError(
                f"mono_time: seq {seq}: goes back from {self.last_mono} to {mono}"
            )

    def accept(self, event: dict[str, Any]) -> None:
        kind = event.get("kind", "")
        if self.next_seq == 0:
            links = event.get("links")
            if isinstance(links, list):
                self.rollout = any(
                    isinstance(link, dict) and link.get("rel") == "branched_from"
                    for link in links
                )
        mono_raw = event.get("mono_time")
        if isinstance(mono_raw, str):
            m = parse_u64(mono_raw)
            if m is not None:
                self.last_mono = max(self.last_mono, m)
        if kind == "policy":
            payload = event.get("payload")
            if isinstance(payload, dict):
                self.training = payload.get("training") is True
        self.rewarded = self.rewarded or (kind == "reward")
        self.effected = self.effected or (kind == "effects")
        self.ended = self.ended or (kind == "run.end")
        self.verified = self.verified or (kind == "run.verified")

        gr = gap_range(event)
        if gr is not None:
            self.next_seq = gr[1]
        self.next_seq += 1


# =============================================================================
# helium-core/src/run.rs
# =============================================================================


def verify_run(
    deployment_id: str, run_id: str, events: list[dict[str, Any]]
) -> list[str]:
    """Verify one run: chain integrity, field rules, and run structure rules."""
    problems: list[str] = []
    chain_problems, _ = verify_run_chain(deployment_id, run_id, events)
    problems.extend(chain_problems)

    try:
        check_run_id(run_id)
    except ValueError as e:
        problems.append(f"run: invalid: run_id: {e}")

    # Positions of events that had malformed chain fields
    unreadable_positions = {
        int(p.split("#")[1].split(":")[0])
        for p in chain_problems
        if p.startswith("event #") and "malformed:" in p
    }

    rules = RunRules()
    for position, event in enumerate(events):
        if position in unreadable_positions:
            continue
        try:
            check_event(event, "chained")
        except ValueError as e:
            problems.append(f"event #{position}: invalid: {e}")
            continue

        try:
            rules.check(event)
        except ValueError as e:
            problems.append(f"event #{position}: invalid: {e}")
        rules.accept(event)

    return problems


# =============================================================================
# helium-core/src/merkle.rs
# =============================================================================


class Head:
    """One run's position at a checkpoint: latest seq and that event's hash."""

    def __init__(self, run_id: str, seq: int, hash_val: str) -> None:
        self.run_id = run_id
        self.seq = seq
        self.hash = hash_val

    def to_dict(self) -> dict[str, str]:
        return {
            "hash": self.hash,
            "run_id": self.run_id,
            "seq": str(self.seq),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Head":
        if len(d) != 3:
            raise ValueError("a head has exactly hash, run_id and seq")
        h = d.get("hash")
        r = d.get("run_id")
        s = d.get("seq")
        if not (isinstance(h, str) and isinstance(r, str) and isinstance(s, str)):
            raise TypeError("head fields must be strings")
        seq_val = parse_u64(s)
        if seq_val is None:
            raise ValueError('"seq" is not a canonical whole number')
        if not is_valid_hash(h):
            raise ValueError('"hash": not a hash')
        return cls(run_id=r, seq=seq_val, hash_val=h)

    def leaf_hash(self) -> str:
        return labelled_hash(Domain.MERKLE_LEAF, canonical(self.to_dict()))


def merkle_leaf_hash(run_id: str, seq: int, event_hash_val: str) -> str:
    """Compute leaf hash of one run head."""
    head = {"hash": event_hash_val, "run_id": run_id, "seq": str(seq)}
    return labelled_hash(Domain.MERKLE_LEAF, canonical(head))


def _merkle_node(left: str, right: str) -> str:
    node = {"left": left, "right": right}
    return labelled_hash(Domain.MERKLE_NODE, canonical(node))


def _split_power_of_two(n: int) -> int:
    """Largest power of two strictly less than n (for n >= 2)."""
    return 1 << ((n - 1).bit_length() - 1)


def merkle_root(leaves: list[str]) -> str:
    """Compute the Merkle root over already-hashed leaves."""
    if not leaves:
        return labelled_hash(Domain.MERKLE_NODE, b"")
    if len(leaves) == 1:
        return leaves[0]
    k = _split_power_of_two(len(leaves))
    left = merkle_root(leaves[:k])
    right = merkle_root(leaves[k:])
    return _merkle_node(left, right)


def inclusion_proof(leaves: list[str], index: int) -> list[str] | None:
    """The audit path proving leaves[index] is in the tree, or None if index out of range."""
    if index < 0 or index >= len(leaves):
        return None
    proof: list[str] = []
    _merkle_path(leaves, index, proof)
    return proof


def _merkle_path(leaves: list[str], index: int, proof: list[str]) -> None:
    if len(leaves) <= 1:
        return
    k = _split_power_of_two(len(leaves))
    if index < k:
        _merkle_path(leaves[:k], index, proof)
        proof.append(merkle_root(leaves[k:]))
    else:
        _merkle_path(leaves[k:], index - k, proof)
        proof.append(merkle_root(leaves[:k]))


def verify_inclusion(
    leaf: str, index: int, size: int, proof: list[str], root_hash: str
) -> bool:
    """Check an audit path (RFC 9162 Section 2.1.3.2)."""
    if index < 0 or index >= size:
        return False
    fnode = index
    snode = size - 1
    r = leaf
    for p in proof:
        if snode == 0:
            return False
        if fnode & 1 == 1 or fnode == snode:
            r = _merkle_node(p, r)
            while fnode & 1 == 0 and fnode != 0:
                fnode >>= 1
                snode >>= 1
        else:
            r = _merkle_node(r, p)
        fnode >>= 1
        snode >>= 1
    return snode == 0 and r == root_hash


# =============================================================================
# helium-core/src/checkpoint.rs
# =============================================================================

CHECKPOINT_HEADER_FIELDS = [
    "deployment_id",
    "format_version",
    "key_id",
    "licence",
    "merkle_root",
    "prev_hash",
    "run_count",
    "seq",
    "time",
    "timer",
]

CLOCK_TOLERANCE_NANOS = 5 * 60 * 1000000000  # 5 minutes


def checkpoint_genesis(deployment_id: str) -> str:
    check_uuid("deployment_id", deployment_id)
    doc = {"deployment_id": deployment_id}
    return labelled_hash(Domain.CHECKPOINT_GENESIS, canonical(doc))


def checkpoint_header_hash(header: dict[str, Any]) -> str:
    return labelled_hash(Domain.CHECKPOINT, canonical(header))


def _check_licence(licence_val: Any) -> tuple[str, str | None]:
    if not isinstance(licence_val, dict):
        raise TypeError("licence: must be an object")
    if len(licence_val) != 3:
        raise ValueError("licence: exactly id, renewal_reference and status")
    _exact_fields(licence_val, "licence.", ["id", "renewal_reference", "status"], [])
    lid = licence_val.get("id")
    if lid is not None and not isinstance(lid, str):
        raise TypeError("licence.id: must be a string or null")
    if lid == "":
        raise ValueError("licence.id: must not be empty")

    status = licence_val.get("status")
    if status not in ("valid", "grace", "none"):
        raise ValueError("licence.status: must be valid, grace or none")

    rr = licence_val.get("renewal_reference")
    if rr is not None and not isinstance(rr, str):
        raise TypeError("licence.renewal_reference: must be a string or null")

    if status == "grace":
        if not rr:
            raise ValueError("licence: grace needs a renewal reference")
    else:
        if rr is not None:
            raise ValueError("licence: a renewal reference only goes with grace")

    return (status, lid)


class _ParsedCheckpoint:
    def __init__(self, obj: dict[str, Any]) -> None:
        expected = CHECKPOINT_HEADER_FIELDS + ["hash", "signature", "heads"]
        _exact_fields(obj, "", expected, [])

        fv = obj.get("format_version")
        if not isinstance(fv, (int, float)) or isinstance(fv, bool) or fv != 0:
            raise ValueError("format_version: must be the number 0")

        dep = obj.get("deployment_id")
        if not isinstance(dep, str):
            raise TypeError("deployment_id: must be a string")
        check_uuid("deployment_id", dep)
        self.deployment_id = dep

        _check_licence(obj.get("licence"))

        if obj.get("timer") is not None:
            raise ValueError("timer: must be null until the licensing step defines it")

        seq_raw = obj.get("seq")
        if not isinstance(seq_raw, str) or parse_u64(seq_raw) is None:
            raise ValueError("seq: not a canonical whole number")
        self.seq = int(seq_raw)

        time_raw = obj.get("time")
        if not isinstance(time_raw, str):
            raise TypeError("time: must be a string")
        self.time_nanos = wall_time_nanos(time_raw)

        prev_h = obj.get("prev_hash")
        if not isinstance(prev_h, str) or not is_valid_hash(prev_h):
            raise ValueError(
                'prev_hash: not a hash: expected "sha256:" followed by 64 lowercase hex digits'
            )
        self.prev_hash = prev_h

        m_root = obj.get("merkle_root")
        if not isinstance(m_root, str) or not is_valid_hash(m_root):
            raise ValueError(
                'merkle_root: not a hash: expected "sha256:" followed by 64 lowercase hex digits'
            )
        self.merkle_root = m_root

        rc_raw = obj.get("run_count")
        if not isinstance(rc_raw, str) or parse_u64(rc_raw) is None:
            raise ValueError("run_count: not a canonical whole number")
        self.run_count = int(rc_raw)

        kid = obj.get("key_id")
        if not isinstance(kid, str) or not is_valid_hash(kid):
            raise ValueError(
                'key_id: not a hash: expected "sha256:" followed by 64 lowercase hex digits'
            )
        self.key_id = kid

        h_val = obj.get("hash")
        if not isinstance(h_val, str) or not is_valid_hash(h_val):
            raise ValueError(
                'hash: not a hash: expected "sha256:" followed by 64 lowercase hex digits'
            )
        self.hash = h_val

        sig = obj.get("signature")
        if not isinstance(sig, str) or not sig.startswith("ed25519:"):
            raise ValueError("signature: must be a valid signature")
        self.signature = sig

        heads_raw = obj.get("heads")
        if not isinstance(heads_raw, list):
            raise TypeError("heads: must be an array")
        self.heads: list[tuple[str, int, str]] = []
        for i, item in enumerate(heads_raw):
            if not isinstance(item, dict):
                raise TypeError(f"heads[{i}]: must be an object")
            _exact_fields(item, f"heads[{i}].", ["hash", "run_id", "seq"], [])
            hrun = item.get("run_id")
            if not isinstance(hrun, str):
                raise TypeError(f"heads[{i}].run_id: must be a string")
            hseq_raw = item.get("seq")
            if not isinstance(hseq_raw, str) or parse_u64(hseq_raw) is None:
                raise ValueError(f"heads[{i}].seq: not a canonical whole number")
            hhash = item.get("hash")
            if not isinstance(hhash, str) or not is_valid_hash(hhash):
                raise ValueError(
                    f'heads[{i}].hash: not a hash: expected "sha256:" followed by 64 lowercase hex digits'
                )
            self.heads.append((hrun, int(hseq_raw), hhash))

        self.header = {k: obj[k] for k in CHECKPOINT_HEADER_FIELDS}


def verify_checkpoints(
    deployment_id: str,
    checkpoints: list[dict[str, Any]],
    trusted_keys: list[bytes],
) -> list[str]:
    """Verify checkpoint chain integrity, Merkle roots, signatures and timestamps."""
    problems: list[str] = []
    expected_seq = 0
    previous: str | None = checkpoint_genesis(deployment_id)
    latest_time: int | None = None

    # Map key_id -> public_key bytes
    trusted_by_id = {compute_key_id(pk): pk for pk in trusted_keys}

    for position, cp in enumerate(checkpoints):
        try:
            parsed = _ParsedCheckpoint(cp)
        except ValueError as e:
            problems.append(f"checkpoint #{position}: malformed: {e}")
            previous = None
            expected_seq += 1
            continue

        seq = parsed.seq
        if parsed.deployment_id != deployment_id:
            problems.append(
                f"checkpoint {seq}: belongs to deployment {parsed.deployment_id}"
            )

        # Check signature & hash
        actual_hash = checkpoint_header_hash(parsed.header)
        if actual_hash != parsed.hash:
            problems.append(f"checkpoint {seq}: content changed")
        else:
            matching_pk = trusted_by_id.get(parsed.key_id)
            if matching_pk is None:
                problems.append(
                    f"checkpoint {seq}: signed by an untrusted key {parsed.key_id}"
                )
            else:
                raw_hash = bytes.fromhex(parsed.hash[len(HASH_PREFIX) :])
                if not verify_signature(
                    matching_pk,
                    CHECKPOINT_SIGNATURE_LABEL,
                    raw_hash,
                    parsed.signature,
                ):
                    problems.append(f"checkpoint {seq}: signature does not match")

        # Check Merkle root & sorting
        heads = parsed.heads
        is_sorted = all(heads[i][0] < heads[i + 1][0] for i in range(len(heads) - 1))
        leaves = [merkle_leaf_hash(hrun, hseq, hhash) for hrun, hseq, hhash in heads]
        computed_root = merkle_root(leaves)
        if (
            not is_sorted
            or parsed.run_count != len(heads)
            or computed_root != parsed.merkle_root
        ):
            problems.append(
                f"checkpoint {seq}: run heads do not match the signed Merkle root"
            )

        if seq == expected_seq:
            if previous is not None and previous != parsed.prev_hash:
                problems.append(
                    f"checkpoint {seq}: does not follow on from the previous checkpoint"
                )
            expected_seq = seq + 1
        elif seq > expected_seq:
            if expected_seq == seq - 1:
                problems.append(f"checkpoint {expected_seq} missing")
            else:
                problems.append(f"checkpoints {expected_seq}\u2013{seq - 1} missing")
            expected_seq = seq + 1
        else:
            problems.append(f"checkpoint {seq}: out of order (expected {expected_seq})")

        if (
            latest_time is not None
            and latest_time - parsed.time_nanos > CLOCK_TOLERANCE_NANOS
        ):
            diff_sec = (latest_time - parsed.time_nanos) // 1000000000
            problems.append(
                f"checkpoint {seq}: time went back {diff_sec} s (clock moved backwards)"
            )
        latest_time = (
            parsed.time_nanos
            if latest_time is None
            else max(latest_time, parsed.time_nanos)
        )
        previous = parsed.hash

    return problems


def cross_check(
    checkpoints: list[dict[str, Any]],
    runs: dict[str, list[dict[str, Any]]],
) -> tuple[list[str], list[tuple[str, int, int]]]:
    """Cross-check stored runs against checkpoint commitments."""
    # Index stored runs: run_id -> (seq -> hash, run_end_seq)
    index: dict[str, tuple[dict[int, str | None], int | None]] = {}
    for run_id, events in runs.items():
        by_seq: dict[int, str | None] = {}
        end_seq: int | None = None
        for e in events:
            s_str = e.get("seq")
            if isinstance(s_str, str):
                s_val = parse_u64(s_str)
                if s_val is not None:
                    h_val = e.get("hash")
                    by_seq[s_val] = (
                        h_val
                        if isinstance(h_val, str) and is_valid_hash(h_val)
                        else None
                    )
                    if e.get("kind") == "run.end":
                        end_seq = s_val
        index[run_id] = (by_seq, end_seq)

    problems: list[str] = []
    deleted_reported: set[str] = set()
    cut_short: dict[
        str, tuple[int, int, int | None]
    ] = {}  # run_id -> (cp_seq, committed, stored_last)
    covered: dict[str, int] = {}

    for cp in checkpoints:
        try:
            parsed = _ParsedCheckpoint(cp)
        except ValueError:
            continue

        for hrun, hseq, hhash in parsed.heads:
            if hrun not in index:
                if hrun not in deleted_reported:
                    deleted_reported.add(hrun)
                    problems.append(
                        f"run {hrun}: deleted (checkpoint {parsed.seq} committed to it)"
                    )
                continue

            by_seq, _ = index[hrun]
            if hseq not in by_seq:
                stored_last = max(by_seq.keys()) if by_seq else None
                prev_cut = cut_short.get(hrun)
                if prev_cut is None or hseq > prev_cut[1]:
                    cut_short[hrun] = (parsed.seq, hseq, stored_last)
            else:
                stored_h = by_seq[hseq]
                if stored_h != hhash:
                    problems.append(
                        f"run {hrun}: seq {hseq} differs from what checkpoint {parsed.seq} committed to"
                    )
                covered[hrun] = max(covered.get(hrun, hseq), hseq)

    for run_id, (cp_seq, committed, stored_last) in sorted(cut_short.items()):
        stored_desc = (
            "no events" if stored_last is None else f"store ends at seq {stored_last}"
        )
        problems.append(
            f"run {run_id}: cut short (checkpoint {cp_seq} saw seq {committed}; {stored_desc})"
        )

    not_yet_checkpointed: list[tuple[str, int, int]] = []
    for run_id in sorted(index.keys()):
        by_seq, end_seq = index[run_id]
        if not by_seq:
            continue
        last_seq = max(by_seq.keys())
        if run_id not in covered:
            problems.append(f"run {run_id}: not covered by any checkpoint")
        else:
            c = covered[run_id]
            if c < last_seq:
                if end_seq is not None and end_seq > c:
                    problems.append(
                        f"run {run_id}: run.end (seq {end_seq}) not covered by any checkpoint"
                    )
                else:
                    not_yet_checkpointed.append((run_id, c + 1, last_seq))

    return problems, not_yet_checkpointed


# =============================================================================
# helium-core/src/text.rs
# =============================================================================


def _is_emoji(code: int) -> bool:
    return (
        0x1F000 <= code <= 0x1FAFF
        or 0x2600 <= code <= 0x27BF
        or 0x2300 <= code <= 0x23FF
        or 0x2B00 <= code <= 0x2BFF
        or 0x2190 <= code <= 0x21FF
        or code in (0x3030, 0x303D, 0x3297, 0x3299, 0x00A9, 0x00AE, 0x2122, 0x2139)
        or 0x25A0 <= code <= 0x25FF
        or code in (0x2934, 0x2935)
    )


def _hidden_kind(c: str, prev: str | None, next_char: str | None) -> str | None:
    code = ord(c)
    if code in (0x09, 0x0A):
        return None
    if code == 0x0D and next_char == "\n":
        return None
    if (0x00 <= code <= 0x1F) or (0x7F <= code <= 0x9F):
        return "control"
    if code in (
        0x061C,
        0x200E,
        0x200F,
        0x202A,
        0x202B,
        0x202C,
        0x202D,
        0x202E,
        0x2066,
        0x2067,
        0x2068,
        0x2069,
    ):
        return "bidi"
    if (
        code == 0x200D
        and prev is not None
        and (_is_emoji(ord(prev)) or ord(prev) == 0xFE0F)
        and next_char is not None
        and _is_emoji(ord(next_char))
    ):
        return None
    if code in (
        0x00AD,
        0x034F,
        0x115F,
        0x1160,
        0x17B4,
        0x17B5,
        0x180E,
        0x200B,
        0x200C,
        0x200D,
        0x2060,
        0x2061,
        0x2062,
        0x2063,
        0x2064,
        0x3164,
        0xFEFF,
        0xFFA0,
        0xFFF9,
        0xFFFA,
        0xFFFB,
    ):
        return "zero_width"
    if 0xE0000 <= code <= 0xE007F:
        return "tag"
    if code in (0xFE0E, 0xFE0F):
        if (
            prev is not None
            and _hidden_kind(prev, None, None) is None
            and not (0xFE00 <= ord(prev) <= 0xFE0F)
        ):
            return None
        return "variation"
    if (0xFE00 <= code <= 0xFE0F) or (0xE0100 <= code <= 0xE01EF):
        return "variation"
    if code in (0x2028, 0x2029):
        return "separator"
    return None


def for_terminal(s: str) -> str:
    """Escapes hidden/control characters so output is safe for terminals."""
    chars = list(s)
    out: list[str] = []
    has_hidden = False
    for i, c in enumerate(chars):
        prev = chars[i - 1] if i > 0 else None
        next_c = chars[i + 1] if i + 1 < len(chars) else None
        h = _hidden_kind(c, prev, next_c)
        if h == "control":
            has_hidden = True
            out.append(f"\\x{ord(c):02x}")
        elif h is not None:
            has_hidden = True
            out.append(f"\\u{{{ord(c):x}}}")
        else:
            out.append(c)
    return "".join(out) if has_hidden else s


def json_for_terminal(s: str) -> str:
    """Canonical JSON safe to print to a terminal with \\uXXXX escapes."""
    chars = list(s)
    out: list[str] = []
    has_hidden = False
    for i, c in enumerate(chars):
        prev = chars[i - 1] if i > 0 else None
        next_c = chars[i + 1] if i + 1 < len(chars) else None
        h = _hidden_kind(c, prev, next_c)
        if h is not None:
            has_hidden = True
            code = ord(c)
            if code <= 0xFFFF:
                out.append(f"\\u{code:04x}")
            else:
                high = 0xD800 + ((code - 0x10000) >> 10)
                low = 0xDC00 + ((code - 0x10000) & 0x3FF)
                out.append(f"\\u{high:04x}\\u{low:04x}")
        else:
            out.append(c)
    return "".join(out) if has_hidden else s


# =============================================================================
# helium-store/src/layout.rs
# =============================================================================


def default_data_dir() -> Path:
    """Default Helium data directory for this platform: the folder
    KASANKI_HELIUM_DATA_DIR names, else the system-wide one; on Linux the
    system-wide one only if it exists, else the user's own."""
    configured = os.environ.get("KASANKI_HELIUM_DATA_DIR")
    if configured:
        return Path(configured)
    if os.name == "nt":
        base = os.environ.get("ProgramData", r"C:\ProgramData")
        return Path(base) / "Kasanki Helium"
    elif sys_platform_is_darwin():
        return Path("/Library/Application Support/Kasanki Helium")
    else:
        system = Path("/var/lib/kasanki-helium")
        if system.is_dir():
            return system
        xdg = os.environ.get("XDG_DATA_HOME")
        if xdg and Path(xdg).is_absolute():
            return Path(xdg) / "kasanki-helium"
        home = os.environ.get("HOME")
        if home and Path(home).is_absolute():
            return Path(home) / ".local" / "share" / "kasanki-helium"
        return system


def sys_platform_is_darwin() -> bool:
    import sys

    return sys.platform == "darwin"


def blob_file_path(root: Path, hash_str: str) -> Path:
    hex_val = hash_str[len(HASH_PREFIX) :]
    return root / "blobs" / hex_val[:2] / hex_val


# =============================================================================
# helium-store/src/blobs.rs
# =============================================================================


def check_blob(path: Path, expected_hash: str, expected_size: int) -> str | None:
    """Returns None if blob is intact, or a problem reason string."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return "missing"
    except OSError as e:
        return str(e)

    if len(data) != expected_size:
        return f"has {len(data)} bytes, expected {expected_size}"

    actual_hash = labelled_hash(Domain.BLOB, data)
    if actual_hash != expected_hash:
        return "content changed"
    return None


# =============================================================================
# helium-store/src/segments.rs
# =============================================================================


def read_segment_lines(dir_path: Path) -> list[tuple[str, int, bytes, bool]]:
    """Read all segment-*.jsonl files in dir_path in sorted order.

    Returns tuples: (segment_filename, line_number_1_based, line_bytes, is_complete).
    """
    if not dir_path.is_dir():
        return []

    segments: list[tuple[int, Path]] = []
    try:
        entries = list(dir_path.iterdir())
    except OSError:
        return []

    for entry in entries:
        name = entry.name
        if name.startswith("segment-") and name.endswith(".jsonl"):
            digits = name[len("segment-") : -len(".jsonl")]
            if len(digits) >= 6 and digits.isdigit():
                try:
                    segments.append((int(digits), entry))
                except ValueError:
                    pass

    segments.sort(key=lambda t: t[0])
    lines: list[tuple[str, int, bytes, bool]] = []

    for _, seg_path in segments:
        try:
            data = seg_path.read_bytes()
        except OSError:
            continue

        file_name = seg_path.name
        rest = data
        number = 0
        while rest:
            number += 1
            idx = rest.find(b"\n")
            if idx != -1:
                line_bytes = rest[:idx]
                complete = True
                rest = rest[idx + 1 :]
            else:
                line_bytes = rest
                complete = False
                rest = b""
            lines.append((file_name, number, line_bytes, complete))

    return lines


# =============================================================================
# helium-store/src/store.rs
# =============================================================================


def read_records(
    dir_path: Path, max_depth: int, max_bytes: int
) -> tuple[list[dict[str, Any]], list[str]]:
    """Read records from all segments in dir_path, validating line canonical form."""
    items: list[dict[str, Any]] = []
    unreadable: list[str] = []

    for file_name, line_num, line_bytes, complete in read_segment_lines(dir_path):
        at = f"{file_name} line {line_num}"
        if not complete:
            unreadable.append(
                f"{at}: incomplete last line (cut short by a crash; quarantined when the engine next starts)"
            )
            continue

        try:
            val = parse(line_bytes, max_depth=max_depth, max_bytes=max_bytes)
        except JsonError as e:
            unreadable.append(f"{at}: {e}")
            continue

        if not isinstance(val, dict):
            unreadable.append(f"{at}: not a JSON object")
            continue

        try:
            canon_bytes = canonical(val, max_depth=max_depth, max_bytes=max_bytes)
            if canon_bytes == line_bytes:
                items.append(val)
            else:
                unreadable.append(f"{at}: not in canonical form")
        except JsonError:
            unreadable.append(f"{at}: not in canonical form")

    return items, unreadable


# =============================================================================
# helium-store/src/verify.rs
# =============================================================================


class StoreReport:
    """Verification report for a Helium data folder."""

    def __init__(self, deployment_id: str) -> None:
        self.deployment_id = deployment_id
        self.runs = 0
        self.events = 0
        self.checkpoints = 0
        self.blobs_checked = 0
        self.problems: list[str] = []
        self.not_yet_checkpointed: list[tuple[str, int, int]] = []
        self.licences: dict[str, int] = {}

    def is_intact(self) -> bool:
        return len(self.problems) == 0

    def to_json(self) -> dict[str, Any]:
        """Convert report to JSON object for CLI output."""
        return {
            "blobs_checked": self.blobs_checked,
            "checkpoints": self.checkpoints,
            "deployment_id": self.deployment_id,
            "events": self.events,
            "intact": self.is_intact(),
            "licences": {k: self.licences[k] for k in sorted(self.licences.keys())},
            "not_yet_checkpointed": [
                {"from_seq": str(f), "run_id": r, "to_seq": str(t)}
                for (r, f, t) in self.not_yet_checkpointed
            ],
            "problems": self.problems,
            "runs": self.runs,
        }

    def __str__(self) -> str:
        lines: list[str] = [
            f"Kasanki Helium verify: deployment {self.deployment_id}",
            f"Checked {self.runs} runs ({self.events} events), {self.checkpoints} checkpoints, {self.blobs_checked} blobs.",
        ]
        if self.licences:
            parts = [
                f"{count} {status}" for status, count in sorted(self.licences.items())
            ]
            lines.append(f"Licence in checkpoints: {', '.join(parts)}.")

        for r, f, t in self.not_yet_checkpointed:
            range_str = f"seq {f}" if f == t else f"seq {f}-{t}"
            lines.append(f"Not yet checkpointed: run {r} {range_str}.")

        if self.is_intact():
            lines.append("Result: INTACT")
        else:
            lines.append(f"Problems ({len(self.problems)}):")
            for p in self.problems:
                lines.append(f"  - {p}")
            lines.append("Result: PROBLEMS FOUND")

        return "\n".join(lines) + "\n"


def verify_store(
    root: Path,
    trusted_keys: list[bytes] | None = None,
) -> StoreReport:
    """Verify entire data folder at root."""
    marker = root / "deployment.json"
    if not marker.is_file():
        raise FileNotFoundError(
            f"{root} is not a Helium data folder (run 'helium init')"
        )

    try:
        marker_data = marker.read_bytes()
    except OSError as e:
        raise OSError(f"{marker}: {e}") from None

    if not marker_data.endswith(b"\n"):
        raise ValueError(f"{marker} is malformed")
    try:
        deployment_doc = parse(marker_data[:-1], max_depth=64, max_bytes=1048576)
        if not isinstance(deployment_doc, dict):
            raise TypeError(f"{marker} is malformed")
        deployment_id = deployment_doc.get("deployment_id")
        if not isinstance(deployment_id, str):
            raise TypeError(f"{marker} is malformed")
    except (ValueError, TypeError, KeyError, OSError, JsonError):
        raise ValueError(f"{marker} is malformed") from None

    # Load deployment public key if none given
    if trusted_keys is None or len(trusted_keys) == 0:
        pub_key_path = root / "keys" / "deployment.pub.pem"
        try:
            pem_text = pub_key_path.read_text(encoding="utf-8")
        except OSError as e:
            raise OSError(f"{pub_key_path}: {e}") from None
        from kasanki_helium._ved25519 import public_key_from_pem

        trusted_keys = [public_key_from_pem(pem_text)]

    report = StoreReport(deployment_id)

    runs_dir = root / "runs"
    run_ids: list[str] = []
    if runs_dir.is_dir():
        try:
            for entry in runs_dir.iterdir():
                if entry.is_dir():
                    run_ids.append(entry.name)
        except OSError as e:
            raise OSError(f"{runs_dir}: {e}") from None
    run_ids.sort()

    runs_data: dict[str, list[dict[str, Any]]] = {}

    for run_id in run_ids:
        report.runs += 1
        rdir = runs_dir / run_id
        items, unreadable = read_records(rdir, max_depth=64, max_bytes=1048576)
        report.events += len(items)

        for line_err in unreadable:
            report.problems.append(f"run {run_id}: {line_err}")

        run_probs = verify_run(deployment_id, run_id, items)
        for p in run_probs:
            report.problems.append(f"run {run_id}: {p}")

        # Check blobs
        for event in items:
            seq_val = parse_u64(event.get("seq", ""))
            at_str = f" seq {seq_val}" if seq_val is not None else ""
            blobs_list = event.get("blobs")
            if not isinstance(blobs_list, list):
                continue
            for blob_ref in blobs_list:
                if not isinstance(blob_ref, dict):
                    continue
                bhash = blob_ref.get("hash")
                bsize_str = blob_ref.get("size")
                if not isinstance(bhash, str) or not is_valid_hash(bhash):
                    continue
                if not isinstance(bsize_str, str):
                    continue
                bsize = parse_u64(bsize_str)
                if bsize is None:
                    continue

                report.blobs_checked += 1
                bpath = blob_file_path(root, bhash)
                bprob = check_blob(bpath, bhash, bsize)
                if bprob == "missing":
                    report.problems.append(
                        f"run {run_id}{at_str}: blob {bhash} missing"
                    )
                elif bprob == "content changed":
                    report.problems.append(
                        f"run {run_id}{at_str}: blob {bhash} content changed"
                    )
                elif bprob is not None:
                    report.problems.append(
                        f"run {run_id}{at_str}: blob {bhash} {bprob}"
                    )

        runs_data[run_id] = items

    checkpoints_dir = root / "checkpoints"
    cp_items, cp_unreadable = read_records(
        checkpoints_dir, max_depth=64, max_bytes=64 * 1024 * 1024
    )
    report.checkpoints = len(cp_items)

    for line_err in cp_unreadable:
        report.problems.append(f"checkpoints: {line_err}")

    cp_probs = verify_checkpoints(deployment_id, cp_items, trusted_keys)
    report.problems.extend(cp_probs)

    cross_probs, not_yet = cross_check(cp_items, runs_data)
    report.problems.extend(cross_probs)
    report.not_yet_checkpointed = not_yet

    # Licenses count
    for cp in cp_items:
        lic = cp.get("licence")
        if isinstance(lic, dict):
            status = lic.get("status")
            if not isinstance(status, str):
                status = "unknown"
            lid = lic.get("id")
            label = f"{status} ({lid})" if isinstance(lid, str) else status
            report.licences[label] = report.licences.get(label, 0) + 1

    return report
