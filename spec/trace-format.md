# Helium trace format, version 0

Copyright © 2026 Kasanki Labs. Licensed under the GNU AGPL v3 with an attribution term: see LICENSE and CREDIT.md.

This document describes the records in a Helium data folder and how to check that none of them was changed, removed or added. It matches `helium verify` and the open verifier `python -m kasanki_helium.verify` in Helium Solo 0.1.0.

## 1. Scope

This specification defines the file layout, cryptographic chaining, canonical JSON encoding, and verification rules for Helium trace records stored in a data folder.

The format version defined by this specification is 0. [S1]

The reference verifier implementations are `helium verify` in Helium Solo and the standalone verifier `python -m kasanki_helium.verify`. [S2]

This specification covers only the stored trace records and the rules applied by a verifier to check their integrity. [S3]

This specification does not cover internal engine scheduling, memory caching, IPC protocols, or client SDK recording wrappers. [S4]

## 2. The data folder

A Helium data folder contains recorded traces, signing keys, and content-addressed payloads. A verifier reads only the following files and directories:

- `deployment.json`: stores the deployment identifier and creation timestamp in canonical JSON. [S5]
- `keys/deployment.pub.pem`: contains the deployment public key in SPKI PEM format, used to verify checkpoint signatures. [S6]
- `runs/<run_id>/segment-NNNNNN.jsonl`: contains event records for each run, with one canonical JSON object per line. [S7]
- `checkpoints/segment-NNNNNN.jsonl`: contains signed checkpoint records, with one canonical JSON object per line. [S8]
- `blobs/<2 hex>/<64 hex>`: contains payload content, stored under a two-character prefix directory and named by the 64-character lowercase hex SHA-256 hash. [S9]
- A verifier ignores every other file. Each folder directly under `runs/` must be named with a run ID (section 6): any other name stops the check with exit code 2 and the message `helium: run_id "<name>" is not a lowercase hyphenated UUID`. [S10] [S137]

## 3. Segments

Segment files contain newline-delimited JSON records.

Segment files must be named `segment-NNNNNN.jsonl`, where `NNNNNN` is a zero-padded integer of at least 6 decimal digits. [S11]

Within a directory, segments are ordered strictly ascending by their numeric sequence number. [S12]

Every record within a segment file must end with a single ASCII newline character `\n` (byte 0x0A). [S14]

If the final line of a segment does not end with `\n`, the verifier treats it as an incomplete line cut short. [S15]

## 4. JSON and its canonical form

All stored JSON records must be valid UTF-8. [S16]

Object keys must be unique within an object; any duplicate key is rejected. [S17]

Lone surrogate escape sequences (`U+D800` through `U+DFFF`) are rejected. [S18]

Numbers must represent finite IEEE 754 doubles; NaN and Infinity are prohibited. [S19]

A whole number whose absolute value is above 9,007,199,254,740,991 (2^53 − 1) and below 10^21 is rejected: a value that large must be written as a string. Whole numbers from 10^21 up are written with an exponent, as RFC 8785 does, and are accepted. [S20] [S138]

The maximum nesting depth for arrays and objects is 64 levels. [S21]

The maximum size of an event record is 1 MiB (1,048,576 bytes); the maximum size of a checkpoint record is 64 MiB (67,108,864 bytes). [S22]

In canonical form (RFC 8785), object keys must be sorted lexicographically by their UTF-16 code units. [S23]

No insignificant whitespace is permitted between tokens (no spaces after colons or commas). [S24]

Numbers are formatted according to ECMAScript number-to-string rules, with zero formatted as `0`. [S25]

In strings, `"` is written `\"` and `\` is written `\\`; the control characters U+0008, U+000C, U+000A, U+000D and U+0009 are written `\b`, `\f`, `\n`, `\r` and `\t`; every other character below U+0020 is written `\u` with four lowercase hex digits; every other character is written as itself, in UTF-8. [S26]

## 5. Hashes

Every cryptographic hash is computed using SHA-256 over a purpose label: `SHA-256(label || 0x00 || data)`. [S27]

Hashes are formatted as the prefix `sha256:` followed by exactly 64 lowercase hexadecimal ASCII characters. [S28]

Helium Solo uses the following domain separation labels:

- `helium/genesis/v0`: the genesis hash of a run chain. [S29]
- `helium/event/v0`: an event hash computed over its link record. [S30]
- `helium/event-body/v0`: the hash of an event body content. [S31]
- `helium/merkle-leaf/v0`: a leaf node in a checkpoint Merkle tree. [S32]
- `helium/merkle-node/v0`: an interior node or empty root in a Merkle tree. [S33]
- `helium/checkpoint-genesis/v0`: the genesis hash of a deployment checkpoint chain. [S34]
- `helium/checkpoint/v0`: the hash of a checkpoint header. [S35]
- `helium/public-key/v0`: a public key fingerprint `key_id`. [S36]
- `helium/blob/v0`: the content address of a stored blob. [S37]

## 6. Events

Stored event records must contain exactly 11 top-level fields: [S38]

- `format_version`: must be the number 0. [S39]
- `kind`: string naming the event kind. [S40]
- `wall_time`: UTC timestamp in RFC 3339 format with 9 nanosecond digits: `YYYY-MM-DDTHH:MM:SS.nnnnnnnnnZ`. [S41]
- `mono_time`: string holding a non-negative whole number of nanoseconds elapsed since run start. [S42]
- `links`: array of causal link objects. [S43]
- `payload`: object holding kind-specific payload fields. [S44]
- `blobs`: array of blob reference objects. [S45]
- `run_id`: string holding a UUID version 7, optionally with prefix `trace-` for imported runs. [S46]
- `seq`: string holding a whole number with no sign or leading zeros. [S47]
- `prev_hash`: string holding the SHA-256 hash of the previous event, or the genesis hash for sequence 0. [S48]
- `hash`: string holding the SHA-256 hash of the link record. [S49]

The 15 standard event kinds are `run.start`, `run.end`, `llm.call`, `tool.call`, `clock.read`, `random.draw`, `id.generate`, `message`, `error`, `profile`, `policy`, `gap`, `run.verified`, `reward`, and `effects`. [S50]

Custom event kinds must match `custom.<name>`, where `<name>` consists of lowercase letters, digits, and underscores separated by dots. [S51]

Link objects must contain `rel` (one of `"parent"`, `"caused_by"`, `"message_to"`, `"message_from"`, `"branched_from"`, or `custom.<name>`), `run_id` (UUID v7), and optional `seq` (whole number as string). [S52]

Blob reference objects must contain `hash` (`sha256:...`), `size` (whole number as string), `media_type` (lowercase `type/subtype`), and `role` (one of `"request"`, `"response"`, `"state"`, `"chunk"`, or `custom.<name>`). [S53]

## 7. Runs

The genesis hash of a run is computed as `H("helium/genesis/v0", canonical({"deployment_id": deployment_id, "run_id": run_id}))`. [S54]

The body hash of an event is computed as `H("helium/event-body/v0", canonical(event without run_id, seq, prev_hash, hash))`. [S55]

The link record of an event is a canonical object containing exactly `body_hash`, `prev_hash`, `run_id`, and `seq`. [S56]

The event hash is computed as `H("helium/event/v0", canonical(link_record))`. [S57]

Event sequence numbers within a run start at 0 and increase without gaps, except where represented by a `gap` event. [S58]

A `gap` event payload must contain `from_seq`, `to_seq`, non-empty `reason`, and `source` (`"sdk"` or `"monitor"`). [S59]

In a `gap` event, `from_seq` must equal the event sequence number, `to_seq` must be greater than or equal to `from_seq`, and the next sequence is `to_seq + 1`. [S60]

Sequence 0 of a run must have kind `run.start` and `mono_time` equal to `"0"`. [S61]

Sequence 1 of a run must have kind `profile`. [S62]

A run is branched when its seq 0 event has a link with `rel` `"branched_from"`. In a branched run, seq 2 must have kind `policy`. `policy` may appear only there: never at another seq, and never in a run that isn't branched. [S63] [S139]

The event kind `run.start` is permitted only at sequence 0; `profile` is permitted only at sequence 1; `run.verified` is permitted only after `run.end`. [S64]

Monotonic time `mono_time` must not decrease between consecutive events. [S65]

After `run.end`, only `run.verified` may appear in the run. [S66]

Nothing may follow `run.verified`. [S67]

A run may contain at most one `effects` event and at most one `reward` event. [S68]

After an `effects` event, only `reward` or `run.end` may follow. [S69]

After a `reward` event, only `run.end` may follow. [S70]

Side effects payloads in `effects` or `tool.call` events contain an array of objects, each with `target` (1 to 4096 bytes), `change` (`"added"`, `"changed"`, or `"removed"`), `before`, and `after`. [S71]

For `"added"`, `before` must be null and `after` must be a hash; for `"removed"`, `before` must be a hash and `after` must be null; for `"changed"`, both must be hashes and must differ. [S72]

The event kind `reward` is permitted only after a `policy` event whose payload has `"training": true`. [S73] [S140]

## 8. Blobs

Blobs are stored at `<data_dir>/blobs/<first 2 hex digits>/<full 64 hex digits>`. [S74]

The blob filename and lookup address is the labelled hash `H("helium/blob/v0", content)`. [S75]

A verifier checks that each referenced blob exists, its stored byte length equals `size`, and its content hash equals `hash`. [S76]

## 9. Checkpoints

The genesis hash of the checkpoint chain is `H("helium/checkpoint-genesis/v0", canonical({"deployment_id": deployment_id}))`. [S77]

A checkpoint header contains exactly 10 fields: `deployment_id`, `format_version`, `key_id`, `licence`, `merkle_root`, `prev_hash`, `run_count`, `seq`, `time`, and `timer`. [S78]

A stored checkpoint record is the header plus exactly three more fields: `heads` (the run heads), `hash` and `signature`. [S143]

In a checkpoint header, `format_version` must be 0, `seq` is a whole number sequence string, and `time` is a UTC timestamp with 9 nanosecond digits. [S79]

The `licence` object contains exactly `id` (a non-empty string, or null), `status` (`"valid"`, `"grace"`, or `"none"`), and `renewal_reference` (non-empty string if status is `"grace"`, otherwise null). [S80]

The `timer` field is a string or null. [S81]

The checkpoint hash is `H("helium/checkpoint/v0", canonical(header))`. [S82]

The signature is an Ed25519 signature over `"helium/checkpoint-signature/v0" || 0x00 || header_hash`, formatted as `ed25519:` followed by 128 lowercase hex characters. [S83]

The signing key fingerprint `key_id` is computed as `H("helium/public-key/v0", canonical({"algorithm": "ed25519", "public_key": <64 hex digits>}))`. [S84]

The Merkle tree root commits to run heads; each leaf is `H("helium/merkle-leaf/v0", canonical({"hash": hash, "run_id": run_id, "seq": seq}))`. [S85]

Interior nodes in the Merkle tree are computed as `H("helium/merkle-node/v0", canonical({"left": left, "right": right}))`; an empty tree root is `H("helium/merkle-node/v0", "")`. [S86]

For `n` leaves, the tree splits at the largest power of two strictly less than `n`. [S87]

Run heads in a checkpoint must be sorted by `run_id` ascending with no duplicate run IDs. [S88]

A checkpoint's `time` must not be more than 300 seconds earlier than the latest `time` of the checkpoints before it. [S89] [S141]

The cross-check holds every checkpoint's run heads against the stored runs: [S90]

- A head whose run folder is missing: the run was deleted. [S91]
- A head whose seq isn't stored: the run was cut short. The highest seq any checkpoint committed to is reported. [S91]
- A head whose seq is stored with a different hash: the stored event differs from what the checkpoint committed to. [S90]
- A stored run that no checkpoint covers is a problem. A stored run whose `run.end` comes after the last seq a checkpoint covers is a problem too. [S92]
- A stored run with later events that no checkpoint covers yet, none of them `run.end`, is not a problem: the report lists it as not yet checkpointed. [S92]

## 10. Verifying

Inputs to a verifier are a data folder path and an optional list of trusted Ed25519 public keys; if none are supplied, the verifier trusts `keys/deployment.pub.pem`. [S93]

The verifier performs checks in sequence: reads `deployment.json`, loads trusted keys, checks every run and its blobs, checks the checkpoint chain, and cross-checks checkpoints against runs. [S94]

In text mode, the report is these lines, in this order: [S95]

```
Kasanki Helium verify: deployment <deployment_id>
Checked <runs> runs (<events> events), <checkpoints> checkpoints, <blobs> blobs.
Licence in checkpoints: <count> <status>, <count> <status>.
Not yet checkpointed: run <run_id> seq <from>-<to>.
Result: INTACT
```

The `Licence in checkpoints` line appears only when a checkpoint was read; a status is followed by ` (<id>)` when the checkpoint names a licence ID. There is one `Not yet checkpointed` line per such run, with `seq <n>` when the range is one event. When there are problems, `Result: INTACT` is replaced by `Problems (<n>):`, one `  - <problem>` line per problem, and `Result: PROBLEMS FOUND`. [S95] [S142]

When the folder holds runs whose ID starts with `trace-`, a last line follows: `<n> imported run(s) are tamper-evident only from when Helium received them, not from when they happened`. [S144]

In JSON mode (`--json`), the verifier prints one line, a canonical JSON object containing `deployment_id`, `intact`, `runs`, `events`, `checkpoints`, `blobs_checked`, `problems`, `not_yet_checkpointed`, and `licences`. [S96]

Exit codes: 0 when the data folder is intact (`Result: INTACT`); 1 when problems are found (`Result: PROBLEMS FOUND`); 2 when the check can't run, with one line `helium: <message>` on standard error and nothing on standard output. [S97] [S145]

The table below lists every problem message produced during verification:

| Problem message form | Verification failure description | Source |
|---|---|---|
| `run <run_id>: <file> line <n>: incomplete last line (cut short by a crash; quarantined when the engine next starts)` | Segment write cut short without ending newline | [S98] |
| `run <run_id>: <file> line <n>: not in canonical form` | Line does not match RFC 8785 canonical bytes | [S99] |
| `run <run_id>: <file> line <n>: not a JSON object` | Parsed JSON value is not an object | [S100] |
| `run <run_id>: <file> line <n>: <json_error>` | JSON syntax error, duplicate key, or value violation | [S101] |
| `run <run_id>: event #<pos>: malformed: <reason>` | Event has missing, unparseable, or invalid chain fields | [S102] |
| `run <run_id>: seq <seq>: belongs to run <found>` | Event `run_id` does not match the enclosing run folder | [S103] |
| `run <run_id>: seq <from> missing` | Event sequence number is missing from the chain | [S104] |
| `run <run_id>: seq <from>–<to> missing` | A range of event sequence numbers is missing | [S105] |
| `run <run_id>: seq <seq>: out of order (expected seq <expected>)` | Event appears out of sequence order | [S106] |
| `run <run_id>: seq <seq>: content changed` | Event hash does not match computed hash | [S107] |
| `run <run_id>: seq <seq>: does not follow on from the previous event` | `prev_hash` does not match hash of preceding event | [S108] |
| `run <run_id>: event #<pos>: invalid: <field>: <reason>` | Event field or payload violates format rules | [S109] |
| `run <run_id>: run: invalid: <reason>` | Run identifier violates UUID version 7 rules | [S110] |
| `run <run_id> seq <seq>: blob <hash> missing` | Referenced blob file is missing from store | [S111] |
| `run <run_id> seq <seq>: blob <hash> has <stored> bytes, expected <expected>` | Stored blob size differs from reference | [S112] |
| `run <run_id> seq <seq>: blob <hash> content changed` | Stored blob hash differs from reference | [S113] |
| `run <run_id> seq <seq>: blob <hash>: <io_error>` | File system error while reading blob | [S114] |
| `checkpoints: <file> line <n>: incomplete last line (cut short by a crash; quarantined when the engine next starts)` | Checkpoint line cut short without newline | [S115] |
| `checkpoints: <file> line <n>: not in canonical form` | Checkpoint line does not match canonical JSON bytes | [S116] |
| `checkpoints: <file> line <n>: not a JSON object` | Checkpoint line is not a JSON object | [S117] |
| `checkpoints: <file> line <n>: <json_error>` | Checkpoint line syntax or structure error | [S118] |
| `checkpoint #<pos>: malformed: <reason>` | Checkpoint header missing required fields | [S119] |
| `checkpoint <seq>: belongs to deployment <found>` | Checkpoint deployment identifier mismatch | [S120] |
| `checkpoint <from> missing` | Checkpoint sequence number is missing | [S121] |
| `checkpoints <from>–<to> missing` | A range of checkpoint sequences is missing | [S122] |
| `checkpoint <seq>: out of order (expected <expected>)` | Checkpoint appears out of sequence order | [S123] |
| `checkpoint <seq>: content changed` | Checkpoint hash does not match computed header hash | [S124] |
| `checkpoint <seq>: does not follow on from the previous checkpoint` | Checkpoint `prev_hash` does not match previous checkpoint hash | [S125] |
| `checkpoint <seq>: signed by an untrusted key <key_id>` | Checkpoint key fingerprint is not in trusted set | [S126] |
| `checkpoint <seq>: signature does not match` | Ed25519 signature verification failed | [S127] |
| `checkpoint <seq>: run heads do not match the signed Merkle root` | Checkpoint heads do not match `merkle_root` | [S128] |
| `checkpoint <seq>: time went back <seconds> s (clock moved backwards)` | Timestamp regressed by more than 300 seconds | [S129] |
| `run <run_id>: deleted (checkpoint <checkpoint> committed to it)` | Run committed in a checkpoint is absent from store | [S130] |
| `run <run_id>: cut short (checkpoint <checkpoint> saw seq <committed>; <stored>)` | Stored run ends before sequence committed in checkpoint | [S131] |
| `run <run_id>: seq <seq> differs from what checkpoint <checkpoint> committed to` | Event hash differs from head committed in checkpoint | [S132] |
| `run <run_id>: not covered by any checkpoint` | Stored run is not committed by any checkpoint | [S133] |
| `run <run_id>: run.end (seq <seq>) not covered by any checkpoint` | Finished run has no checkpoint committing to its end | [S134] |

## 11. Versioning

This section says what the version number means today.

Version 0 is the only version. Every event and checkpoint must have `format_version` 0, and a verifier reports any other value as invalid. [S135]
