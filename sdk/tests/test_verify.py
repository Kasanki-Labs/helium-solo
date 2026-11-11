# Copyright © 2026 Kasanki Labs.
# SPDX-License-Identifier: AGPL-3.0-only
# Licensed under the GNU AGPL v3 with an attribution term: see LICENSE and CREDIT.md.

"""Tests for the pure-Python open verifier."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

import kasanki_helium as helium
from kasanki_helium import _client, _endpoint
from kasanki_helium._ved25519 import (
    _L,
    _P,
    public_key_from_pem,
    verify_strict,
)
from kasanki_helium._vjson import JsonError, canonical, parse
from kasanki_helium._vstore import (
    EventError,
    Head,
    Stage,
    body_hash,
    check_event,
    compute_key_id,
    cross_check,
    event_hash,
    genesis,
    inclusion_proof,
    link_record_of_event,
    merkle_root,
    verify_checkpoints,
    verify_inclusion,
    verify_run,
)
from kasanki_helium._wire import Blob
from kasanki_helium.verify import verify

FIXTURES_DIR = Path(__file__).resolve().parent / "data" / "verify"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_vectors_path() -> Path:
    p_export = REPO_ROOT / "spec" / "test-vectors" / "v0.json"
    if p_export.is_file():
        return p_export
    p_mono = REPO_ROOT / "docs" / "spec" / "test-vectors" / "v0.json"
    if p_mono.is_file():
        return p_mono
    return p_export


def _find_rfc8785_dir() -> Path:
    p_export = Path(__file__).resolve().parent / "data" / "rfc8785"
    if p_export.is_dir():
        return p_export
    p_mono = (
        REPO_ROOT / "engine" / "crates" / "helium-core" / "tests" / "data" / "rfc8785"
    )
    if p_mono.is_dir():
        return p_mono
    return p_export


VECTORS_PATH = _find_vectors_path()
RFC8785_DIR = _find_rfc8785_dir()


def test_canonical_forms_match_test_vectors() -> None:
    v0 = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))
    for case in v0["canonical"]:
        name = case["name"]
        inp = case["input"].encode("utf-8")
        val = parse(inp)
        out = canonical(val).decode("utf-8")
        assert out == case["canonical"], f"canonical fail on {name}"


def test_invalid_inputs_are_rejected_for_the_right_reason() -> None:
    v0 = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))
    for case in v0["invalid"]:
        name = case["name"]
        data = (
            case["input"].encode("utf-8")
            if "input" in case
            else bytes.fromhex(case["input_hex"])
        )
        with pytest.raises(JsonError) as exc_info:
            parse(data)
        reason = case["reason"]
        msg = str(exc_info.value)
        if reason == "duplicate key":
            assert "duplicate key" in msg, f"{name}: {msg}"
        elif reason == "lone surrogate":
            assert "lone surrogate" in msg, f"{name}: {msg}"
        elif reason == "unsafe integer":
            assert "whole number beyond" in msg, f"{name}: {msg}"
        elif reason == "number out of range":
            assert "number out of range" in msg, f"{name}: {msg}"
        elif reason == "syntax":
            assert not any(
                w in msg
                for w in (
                    "duplicate key",
                    "lone surrogate",
                    "whole number beyond",
                    "number out of range",
                    "is nested deeper than",
                    "invalid UTF-8",
                )
            ), f"{name}: {msg}"
        elif reason == "too deep":
            assert "is nested deeper than" in msg, f"{name}: {msg}"
        elif reason == "invalid utf-8":
            assert "invalid UTF-8" in msg or "utf-8" in msg.lower(), f"{name}: {msg}"


def test_genesis_hash_matches_test_vector() -> None:
    v0 = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))
    g = v0["genesis"]
    assert genesis(g["deployment_id"], g["run_id"]) == g["hash"]


def test_chain_hashes_match_and_verify() -> None:
    v0 = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))
    g = v0["genesis"]
    dep_id = g["deployment_id"]
    run_id = g["run_id"]
    events = [entry["event"] for entry in v0["chain"]]
    for entry, event in zip(v0["chain"], events, strict=True):
        assert body_hash(event) == body_hash(json.loads(entry["canonical_body"]))
        link = link_record_of_event(event)
        assert canonical(link) == entry["canonical_link"].encode("utf-8")
        assert event_hash(event) == event["hash"]
    problems = verify_run(dep_id, run_id, events)
    assert problems == [], problems


def test_a_run_with_a_gap_matches_and_verifies() -> None:
    v0 = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))
    g = v0["genesis"]
    dep_id = g["deployment_id"]
    gap_data = v0["gap_chain"]
    gap_events = [entry["event"] for entry in gap_data["events"]]
    problems = verify_run(dep_id, gap_data["run_id"], gap_events)
    assert problems == [], problems


def test_invalid_events_are_rejected_for_the_right_field() -> None:
    v0 = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))
    check_event(v0["valid_body"], Stage.BODY)
    for case in v0["invalid_events"]:
        name = case["name"]
        with pytest.raises(EventError) as exc_info:
            check_event(case["event"], Stage.BODY)
        assert exc_info.value.field == case["field"], (
            f"{name}: got {exc_info.value.field}, expected {case['field']}"
        )


def test_merkle_root_and_proofs() -> None:
    v0 = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))
    m = v0["merkle"]
    heads = [Head.from_dict(h) for h in m["heads"]]
    leaf_hashes = [h.leaf_hash() for h in heads]
    assert leaf_hashes == m["leaf_hashes"]
    root_hash = merkle_root(leaf_hashes)
    assert root_hash == m["root"]
    assert merkle_root([]) == m["empty_root"]
    for p in m["proofs"]:
        idx = int(p["index"])
        path = p["path"]
        assert inclusion_proof(leaf_hashes, idx) == path
        assert verify_inclusion(
            leaf_hashes[idx], idx, len(leaf_hashes), path, root_hash
        )


def test_key_and_signed_checkpoints() -> None:
    v0 = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))
    g = v0["genesis"]
    dep_id = g["deployment_id"]
    run_id = g["run_id"]
    events = [entry["event"] for entry in v0["chain"]]
    k = v0["key"]
    raw_pk = public_key_from_pem(k["public_key_pem"])
    assert compute_key_id(raw_pk) == k["key_id"]
    cps = [entry["checkpoint"] for entry in v0["checkpoints"]]
    cp_problems = verify_checkpoints(dep_id, cps, [raw_pk])
    assert cp_problems == [], cp_problems
    cross_problems, not_yet = cross_check(cps, {run_id: events})
    assert cross_problems == [], cross_problems
    assert not_yet == []


def test_official_rfc8785_test_files() -> None:
    for name in ["arrays", "french", "structures", "unicode", "values", "weird"]:
        inp = (RFC8785_DIR / "input" / f"{name}.json").read_bytes()
        expected = (RFC8785_DIR / "output" / f"{name}.json").read_bytes()
        actual = canonical(parse(inp))
        assert actual == expected, f"RFC 8785 mismatch on {name}"


def test_rfc8032_vectors_and_strict_rejections() -> None:
    # TEST 1
    pk1 = bytes.fromhex(
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
    )
    msg1 = b""
    sig1 = bytes.fromhex(
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
        "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
    )
    assert verify_strict(pk1, msg1, sig1)

    # TEST 2
    pk2 = bytes.fromhex(
        "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c"
    )
    msg2 = bytes.fromhex("72")
    sig2 = bytes.fromhex(
        "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da"
        "085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"
    )
    assert verify_strict(pk2, msg2, sig2)

    # TEST 3
    pk3 = bytes.fromhex(
        "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025"
    )
    msg3 = bytes.fromhex("af82")
    sig3 = bytes.fromhex(
        "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac"
        "18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a"
    )
    assert verify_strict(pk3, msg3, sig3)

    # TEST 1024
    pk1024 = bytes.fromhex(
        "278117fc144c72340f67d0f2316e8386ceffbf2b2428c9c51fef7c597f1d426e"
    )
    msg1024 = bytes.fromhex(
        "08b8b2b733424243760fe426a4b54908632110a66c2f6591eabd3345e3e4eb98fa6e264bf09efe12ee50f8f54e9f77b1e355f6c50544e23fb1433ddf73be84d879de7c0046dc4996d9e773f4bc9efe5738829adb26c81b37c93a1b270b20329d658675fc6ea534e0810a4432826bf58c941efb65d57a338bbd2e26640f89ffbc1a858efcb8550ee3a5e1998bd177e93a7363c344fe6b199ee5d02e82d522c4feba15452f80288a821a579116ec6dad2b3b310da903401aa62100ab5d1a36553e06203b33890cc9b832f79ef80560ccb9a39ce767967ed628c6ad573cb116dbefefd75499da96bd68a8a97b928a8bbc103b6621fcde2beca1231d206be6cd9ec7aff6f6c94fcd7204ed3455c68c83f4a41da4af2b74ef5c53f1d8ac70bdcb7ed185ce81bd84359d44254d95629e9855a94a7c1958d1f8ada5d0532ed8a5aa3fb2d17ba70eb6248e594e1a2297acbbb39d502f1a8c6eb6f1ce22b3de1a1f40cc24554119a831a9aad6079cad88425de6bde1a9187ebb6092cf67bf2b13fd65f27088d78b7e883c8759d2c4f5c65adb7553878ad575f9fad878e80a0c9ba63bcbcc2732e69485bbc9c90bfbd62481d9089beccf80cfe2df16a2cf65bd92dd597b0707e0917af48bbb75fed413d238f5555a7a569d80c3414a8d0859dc65a46128bab27af87a71314f318c782b23ebfe808b82b0ce26401d2e22f04d83d1255dc51addd3b75a2b1ae0784504df543af8969be3ea7082ff7fc9888c144da2af58429ec96031dbcad3dad9af0dcbaaaf268cb8fcffead94f3c7ca495e056a9b47acdb751fb73e666c6c655ade8297297d07ad1ba5e43f1bca32301651339e22904cc8c42f58c30c04aafdb038dda0847dd988dcda6f3bfd15c4b4c4525004aa06eeff8ca61783aacec57fb3d1f92b0fe2fd1a85f6724517b65e614ad6808d6f6ee34dff7310fdc82aebfd904b01e1dc54b2927094b2db68d6f903b68401adebf5a7e08d78ff4ef5d63653a65040cf9bfd4aca7984a74d37145986780fc0b16ac451649de6188a7dbdf191f64b5fc5e2ab47b57f7f7276cd419c17a3ca8e1b939ae49e488acba6b965610b5480109c8b17b80e1b7b750dfc7598d5d5011fd2dcc5600a32ef5b52a1ecc820e308aa342721aac0943bf6686b64b2579376504ccc493d97e6aed3fb0f9cd71a43dd497f01f17c0e2cb3797aa2a2f256656168e6c496afc5fb93246f6b1116398a346f1a641f3b041e989f7914f90cc2c7fff357876e506b50d334ba77c225bc307ba537152f3f1610e4eafe595f6d9d90d11faa933a15ef1369546868a7f3a45a96768d40fd9d03412c091c6315cf4fde7cb68606937380db2eaaa707b4c4185c32eddcdd306705e4dc1ffc872eeee475a64dfac86aba41c0618983f8741c5ef68d3a101e8a3b8cac60c905c15fc910840b94c00a0b9d0"
    )
    sig1024 = bytes.fromhex(
        "0aab4c900501b3e24d7cdf4663326a3a87df5e4843b2cbdb67cbf6e460fec350"
        "aa5371b1508f9f4528ecea23c436d94b5e8fcd4f681e30a6ac00a9704a188a03"
    )
    assert verify_strict(pk1024, msg1024, sig1024)

    # TEST SHA(abc)
    pk_sha = bytes.fromhex(
        "ec172b93ad5e563bf4932c70e1245034c35467ef2efd4d64ebf819683467e2bf"
    )
    msg_sha = bytes.fromhex(
        "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a"
        "2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"
    )
    sig_sha = bytes.fromhex(
        "dc2a4459e7369633a52b1bf277839a00201009a3efbf3ecb69bea2186c26b589"
        "09351fc9ac90b3ecfdfbc7c66431e0303dca179c138ac17ad9bef1177331a704"
    )
    assert verify_strict(pk_sha, msg_sha, sig_sha)

    # Strict rejection 1: S >= L
    r_part = sig1[:32]
    s_int = int.from_bytes(sig1[32:], "little")
    bad_s = (s_int + _L).to_bytes(32, "little")
    assert not verify_strict(pk1, msg1, r_part + bad_s)

    # Strict rejection 2: small-order A
    small_order_a = (1).to_bytes(32, "little")
    assert not verify_strict(small_order_a, msg1, sig1)

    # Strict rejection 3: small-order R
    bad_r_sig = small_order_a + sig1[32:]
    assert not verify_strict(pk1, msg1, bad_r_sig)

    # Strict rejection 4: non-canonical A (y >= p)
    y_int = int.from_bytes(pk1, "little") & ((1 << 255) - 1)
    bad_pk = (y_int + _P).to_bytes(32, "little")
    assert not verify_strict(bad_pk, msg1, sig1)


def _solo_bin() -> Path:
    configured = os.environ.get("HELIUM_SOLO_BIN")
    if not configured:
        pytest.skip(
            "HELIUM_SOLO_BIN is not set: the cross-check needs the Solo helium program "
            "(cargo build -p helium-cli --no-default-features --target-dir target/solo)"
        )
    p = Path(configured)
    if not p.is_file():
        pytest.fail(f"HELIUM_SOLO_BIN={configured} does not exist")
    return p


def _lines(path: Path) -> list[bytes]:
    data = path.read_bytes()
    assert data.endswith(b"\n"), path
    return data[:-1].split(b"\n")


def _write_lines(path: Path, lines: list[bytes]) -> None:
    path.write_bytes(b"\n".join(lines) + b"\n")


@pytest.mark.slow
def test_a_folder_with_a_thousand_events_and_ten_checkpoints_verifies_in_under_ten_seconds(
    helium_bin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "speed_data"
    env = {
        **os.environ,
        "HELIUM_NO_AI_RULES": "1",
        "USERPROFILE": str(tmp_path),
        "HOME": str(tmp_path),
    }

    subprocess.run(
        [str(helium_bin), "init", "--data-dir", str(data)],
        check=True,
        capture_output=True,
        env=env,
    )
    monkeypatch.setenv(_endpoint.DATA_DIR_VARIABLE, str(data))

    log_file = (tmp_path / "speed_engine.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [str(helium_bin), "engine", "run", "--data-dir", str(data)],
        stdout=subprocess.PIPE,
        stderr=log_file,
        text=True,
        env=env,
    )
    assert proc.stdout is not None
    proc.stdout.readline()

    try:
        # Record 10 runs of 100 events each = 1,000 events total.
        for r in range(10):
            with helium.run(f"speed-agent-{r}") as run:
                for i in range(97):
                    run.event("custom.step", {"step": i})
            assert helium.flush(15)
    finally:
        proc.kill()
        proc.wait()
        log_file.close()

    t0 = time.monotonic()
    report = verify(data_dir=data)
    duration = time.monotonic() - t0

    assert report.is_intact()
    assert report.runs == 10
    assert report.events == 1000
    assert report.checkpoints >= 10
    assert duration < 10.0, f"Verification took {duration:.2f}s, expected < 10.0s"


def _create_base_cross_check_folder(
    solo_bin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, str, str, str]:
    data = tmp_path / "base_data"
    env = {
        **os.environ,
        "HELIUM_NO_AI_RULES": "1",
        "USERPROFILE": str(tmp_path),
        "HOME": str(tmp_path),
    }

    subprocess.run(
        [str(solo_bin), "init", "--data-dir", str(data)],
        check=True,
        capture_output=True,
        env=env,
    )
    monkeypatch.setenv(_endpoint.DATA_DIR_VARIABLE, str(data))

    # Record gap run via backlog while engine is not running
    _client.UNREACHABLE_SECONDS = 600
    _client.MEMORY_EVENTS = 3
    _client.BACKLOG_RETRY_SECONDS = 0.05

    def disk_full(lines: Any) -> tuple[int, OSError]:
        return 0, OSError(28, "No space left on device")

    with helium.run("gap-agent") as run_gap:
        client = _client._clients[_endpoint.endpoint(data)]
        real_write = client.backlog.write
        monkeypatch.setattr(client.backlog, "write", disk_full)
        run_gap.uuid4()
        run_gap.uuid4()
        run_gap.uuid4()
        deadline = time.monotonic() + 10
        while client._backlog_error is None:
            assert time.monotonic() < deadline
            time.sleep(0.05)
        run_gap.uuid4()
        run_gap.uuid4()
        monkeypatch.setattr(client.backlog, "write", real_write)
        assert helium.flush(15)
    assert helium.flush(15)
    gap_run_id = run_gap.run_id

    # Start engine and monitor to drain backlog
    eng_log = (tmp_path / "engine.log").open("w", encoding="utf-8")
    eng_proc = subprocess.Popen(
        [str(solo_bin), "engine", "run", "--data-dir", str(data)],
        stdout=subprocess.PIPE,
        stderr=eng_log,
        text=True,
        env=env,
    )
    assert eng_proc.stdout is not None
    eng_proc.stdout.readline()

    mon_log = (tmp_path / "monitor.log").open("w", encoding="utf-8")
    mon_proc = subprocess.Popen(
        [str(solo_bin), "monitor", "run", "--data-dir", str(data)],
        stdout=subprocess.DEVNULL,
        stderr=mon_log,
        env=env,
    )

    try:
        deadline = time.monotonic() + 15
        gap_verified = False
        while time.monotonic() < deadline:
            run_dir = data / "runs" / gap_run_id
            if run_dir.exists():
                for pf in run_dir.glob("segment-*.jsonl"):
                    if b"run.verified" in pf.read_bytes():
                        gap_verified = True
                        break
            if gap_verified:
                break
            time.sleep(0.1)
        assert gap_verified, "gap run was not verified by engine"
        mon_proc.kill()
        mon_proc.wait()
        mon_log.close()

        _client.MEMORY_EVENTS = 10000

        # Normal run
        with helium.run("normal-agent", input={"ticket": 42}) as run_norm:
            run_norm.random()
            run_norm.uuid4()
            run_norm.call_tool("add", lambda a, b: a + b, 10, b=20)
            run_norm.set_output({"sum": 30})
        assert helium.flush(10)
        norm_run_id = run_norm.run_id

        # Blob run
        blob = Blob.of(b"Sample blob content for verify cross-check tests")
        with helium.run("blob-agent") as run_blob:
            run_blob._record(
                "custom.payload",
                {"detail": "blob-test"},
                blobs=[(blob, "text/plain", "request")],
            )
        assert helium.flush(10)
        blob_run_id = run_blob.run_id
    finally:
        for cl in list(_client._clients.values()):
            cl.close()
        _client._clients.clear()
        if mon_proc.poll() is None:
            mon_proc.kill()
            mon_proc.wait()
            mon_log.close()
        eng_proc.kill()
        eng_proc.wait()
        eng_log.close()

    shutil.rmtree(data / "backlog", ignore_errors=True)
    return data, gap_run_id, norm_run_id, blob_run_id


def test_cross_check_against_rust(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    solo_bin = _solo_bin()
    base_data, _gap_id, norm_id, _blob_id = _create_base_cross_check_folder(
        solo_bin, tmp_path, monkeypatch
    )

    # Assert normal run has exactly 6 lines
    seg_norm = next((base_data / "runs" / norm_id).glob("segment-*.jsonl"))
    norm_lines = _lines(seg_norm)
    assert len(norm_lines) == 6, (
        f"expected 6 lines in normal run, got {len(norm_lines)}"
    )

    env = {
        **os.environ,
        "HELIUM_NO_AI_RULES": "1",
        "USERPROFILE": str(tmp_path),
        "HOME": str(tmp_path),
    }

    write_fixtures = os.environ.get("HELIUM_VERIFY_WRITE_FIXTURES") == "1"

    cases: list[tuple[str, Path, list[str], bytes, bool]] = []

    def make_case(name: str) -> Path:
        p = tmp_path / name
        shutil.copytree(
            base_data,
            p,
            ignore=shutil.ignore_patterns(
                "backlog", "monitor", "*.lock", "deployment.key.pem"
            ),
        )
        assert (p / "keys" / "deployment.pub.pem").is_file(), (
            f"{name}: keys/deployment.pub.pem missing"
        )
        assert not (p / "keys" / "deployment.key.pem").exists(), (
            f"{name}: keys/deployment.key.pem present"
        )
        return p

    # 01_intact
    c1 = make_case("01_intact")
    cases.append(("01_intact", c1, [], b"Result: INTACT", False))

    # 02_event_content_changed
    c2 = make_case("02_event_content_changed")
    seg2 = next((c2 / "runs" / norm_id).glob("segment-*.jsonl"))
    b_before2 = seg2.read_bytes()
    assert b_before2.count(b'"sum":30') == 1, "sum:30 must appear once in file"
    lines2 = _lines(seg2)
    assert b'"sum":30' in lines2[5], "line 5 must contain sum:30"
    lines2[5] = lines2[5].replace(b'"sum":30', b'"sum":31')
    _write_lines(seg2, lines2)
    assert seg2.read_bytes() != b_before2
    cases.append(("02_event_content_changed", c2, [], b"seq 5: content changed", False))

    # 03_line_removed
    c3 = make_case("03_line_removed")
    seg3 = next((c3 / "runs" / norm_id).glob("segment-*.jsonl"))
    b_before3 = seg3.read_bytes()
    lines3 = _lines(seg3)
    del lines3[2]
    _write_lines(seg3, lines3)
    assert seg3.read_bytes() != b_before3
    cases.append(("03_line_removed", c3, [], b"seq 2 missing", False))

    # 04_two_lines_swapped
    c4 = make_case("04_two_lines_swapped")
    seg4 = next((c4 / "runs" / norm_id).glob("segment-*.jsonl"))
    b_before4 = seg4.read_bytes()
    lines4 = _lines(seg4)
    lines4[2], lines4[3] = lines4[3], lines4[2]
    _write_lines(seg4, lines4)
    assert seg4.read_bytes() != b_before4
    cases.append(("04_two_lines_swapped", c4, [], b"out of order", False))

    # 05_incomplete_last_line
    c5 = make_case("05_incomplete_last_line")
    seg5 = next((c5 / "runs" / norm_id).glob("segment-*.jsonl"))
    b_before5 = seg5.read_bytes()
    with seg5.open("ab") as f:
        f.write(b'{"format_version":0,"kind":"run.ver')
    assert seg5.read_bytes() != b_before5
    cases.append(("05_incomplete_last_line", c5, [], b"incomplete last line", False))

    # 06_non_canonical_space
    c6 = make_case("06_non_canonical_space")
    seg6 = next((c6 / "runs" / norm_id).glob("segment-*.jsonl"))
    b_before6 = seg6.read_bytes()
    lines6 = _lines(seg6)
    assert lines6[1].count(b'"format_version":0,') == 1, (
        "format_version:0, must occur once in line 1"
    )
    lines6[1] = lines6[1].replace(b'"format_version":0,', b'"format_version": 0,')
    _write_lines(seg6, lines6)
    assert seg6.read_bytes() != b_before6
    cases.append(
        ("06_non_canonical_space", c6, [], b"line 2: not in canonical form", True)
    )

    # 07_duplicate_key
    c7 = make_case("07_duplicate_key")
    seg7 = next((c7 / "runs" / norm_id).glob("segment-*.jsonl"))
    b_before7 = seg7.read_bytes()
    lines7 = _lines(seg7)
    assert lines7[1].count(b'"format_version":0,') == 1, (
        "format_version:0, must occur once in line 1"
    )
    lines7[1] = lines7[1].replace(
        b'"format_version":0,', b'"format_version":0,"format_version":0,'
    )
    _write_lines(seg7, lines7)
    assert seg7.read_bytes() != b_before7
    cases.append(("07_duplicate_key", c7, [], b'duplicate key "format_version"', False))

    # 08_whole_run_deleted
    c8 = make_case("08_whole_run_deleted")
    shutil.rmtree(c8 / "runs" / norm_id)
    cases.append(("08_whole_run_deleted", c8, [], b"deleted (checkpoint", False))

    # 09_run_cut_short
    c9 = make_case("09_run_cut_short")
    seg9 = next((c9 / "runs" / norm_id).glob("segment-*.jsonl"))
    b_before9 = seg9.read_bytes()
    lines9 = _lines(seg9)
    lines9 = lines9[:3]
    _write_lines(seg9, lines9)
    assert seg9.read_bytes() != b_before9
    cases.append(("09_run_cut_short", c9, [], b"store ends at seq 2", False))

    # 10_blob_changed
    c10 = make_case("10_blob_changed")
    bpath10 = next(
        p
        for p in (c10 / "blobs").rglob("*")
        if p.is_file() and not p.name.endswith(".lock")
    )
    b_before10 = bpath10.read_bytes()
    bdata10 = bytearray(b_before10)
    bdata10[0] ^= 1
    bpath10.write_bytes(bytes(bdata10))
    assert bpath10.read_bytes() != b_before10
    cases.append(("10_blob_changed", c10, [], b"content changed", False))

    # 11_blob_deleted
    c11 = make_case("11_blob_deleted")
    bpath11 = next(
        p
        for p in (c11 / "blobs").rglob("*")
        if p.is_file() and not p.name.endswith(".lock")
    )
    bpath11.unlink()
    cases.append(("11_blob_deleted", c11, [], b"missing", False))

    # 12_blob_extra_bytes
    c12 = make_case("12_blob_extra_bytes")
    bpath12 = next(
        p
        for p in (c12 / "blobs").rglob("*")
        if p.is_file() and not p.name.endswith(".lock")
    )
    b_before12 = bpath12.read_bytes()
    with bpath12.open("ab") as f:
        f.write(b"extra")
    assert bpath12.read_bytes() != b_before12
    cases.append(("12_blob_extra_bytes", c12, [], b"has 53 bytes, expected 48", False))

    # 13_checkpoint_content_changed
    c13 = make_case("13_checkpoint_content_changed")
    cp_file13 = next((c13 / "checkpoints").glob("segment-*.jsonl"))
    b_before13 = cp_file13.read_bytes()
    lines13 = _lines(cp_file13)
    assert lines13[0].count(b'"time":"2026') == 1, (
        "time 2026 must appear once in line 0"
    )
    assert b'"time":"2026' in lines13[0]
    lines13[0] = lines13[0].replace(b'"time":"2026', b'"time":"2025')
    _write_lines(cp_file13, lines13)
    assert cp_file13.read_bytes() != b_before13
    cases.append(("13_checkpoint_content_changed", c13, [], b"checkpoint 0: ", False))

    # 14_checkpoint_signature_changed
    c14 = make_case("14_checkpoint_signature_changed")
    cp_file14 = next((c14 / "checkpoints").glob("segment-*.jsonl"))
    b_before14 = cp_file14.read_bytes()
    lines14 = _lines(cp_file14)
    parsed_cp = json.loads(lines14[0].decode("utf-8"))
    orig_sig = parsed_cp["signature"].encode("utf-8")
    assert orig_sig.startswith(b"ed25519:"), "signature must start with ed25519:"
    prefix_len = len(b"ed25519:")
    hex_part = orig_sig[prefix_len:]
    flipped = b"0" if hex_part[:1] != b"0" else b"1"
    mod_sig = orig_sig[:prefix_len] + flipped + hex_part[1:]
    assert len(mod_sig) == len(orig_sig)
    assert lines14[0].count(orig_sig) == 1
    lines14[0] = lines14[0].replace(orig_sig, mod_sig)
    _write_lines(cp_file14, lines14)
    assert cp_file14.read_bytes() != b_before14
    cases.append(
        (
            "14_checkpoint_signature_changed",
            c14,
            [],
            b"checkpoint 0: signature does not match",
            False,
        )
    )

    # 15_untrusted_public_key
    c15 = make_case("15_untrusted_public_key")
    other_pem15 = c15 / "other.pub.pem"
    other_pem15.write_bytes(
        b"-----BEGIN PUBLIC KEY-----\n"
        b"MCowBQYDK2VwAyEA1111111111111111111111111111111111111111110=\n"
        b"-----END PUBLIC KEY-----\n"
    )
    cases.append(
        (
            "15_untrusted_public_key",
            c15,
            ["--public-key", str(other_pem15)],
            b"signed by an untrusted key",
            False,
        )
    )

    # 16_non_uuid_run_folder
    c16 = make_case("16_non_uuid_run_folder")
    (c16 / "runs" / "not-a-uuid").mkdir()
    cases.append(
        (
            "16_non_uuid_run_folder",
            c16,
            [],
            b'helium: run_id "not-a-uuid" is not a lowercase hyphenated UUID',
            False,
        )
    )

    # 17_missing_deployment
    c17 = make_case("17_missing_deployment")
    (c17 / "deployment.json").unlink()
    cases.append(
        ("17_missing_deployment", c17, [], b"is not a Helium data folder", False)
    )

    # 18_imported_run
    c18 = make_case("18_imported_run")
    trace_id = "trace-0192f3a4-5b6c-7d7e-8f90-a1b2c3d4e5f7"
    (c18 / "runs" / trace_id).mkdir()
    cases.append(("18_imported_run", c18, [], b"imported run(s)", False))

    # 19_bad_public_key_file
    c19 = make_case("19_bad_public_key_file")
    other_pem19 = c19 / "other.pub.pem"
    other_pem19.write_bytes(b"hello\n")
    cases.append(
        (
            "19_bad_public_key_file",
            c19,
            ["--public-key", str(other_pem19)],
            b": not a valid Ed25519 PEM key\n",
            False,
        )
    )

    if write_fixtures:
        shutil.rmtree(FIXTURES_DIR, ignore_errors=True)
        FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    for case_name, case_folder, extra, expected_text, is_case_06 in cases:
        # Run Rust helium verify (text mode)
        r_text = subprocess.run(
            [str(solo_bin), "verify", "--data-dir", str(case_folder)] + extra,
            capture_output=True,
            check=False,
            env=env,
        )
        # Run Python verify (text mode)
        p_text = subprocess.run(
            [
                sys.executable,
                "-m",
                "kasanki_helium.verify",
                "--data-dir",
                str(case_folder),
            ]
            + extra,
            capture_output=True,
            check=False,
            env=env,
        )

        # Run Rust helium verify (JSON mode)
        r_json = subprocess.run(
            [str(solo_bin), "verify", "--data-dir", str(case_folder), "--json"] + extra,
            capture_output=True,
            check=False,
            env=env,
        )
        # Run Python verify (JSON mode)
        p_json = subprocess.run(
            [
                sys.executable,
                "-m",
                "kasanki_helium.verify",
                "--data-dir",
                str(case_folder),
                "--json",
            ]
            + extra,
            capture_output=True,
            check=False,
            env=env,
        )

        # Check expected text on Rust output
        target_rust = (
            r_text.stderr
            if case_name
            in (
                "16_non_uuid_run_folder",
                "17_missing_deployment",
                "19_bad_public_key_file",
            )
            else r_text.stdout
        )
        if case_name == "19_bad_public_key_file":
            assert target_rust.endswith(expected_text), (
                f"[{case_name}] Rust stderr did not end with {expected_text!r}:\n"
                f"{target_rust.decode('utf-8', errors='replace')}"
            )
        else:
            assert expected_text in target_rust, (
                f"[{case_name}] Rust output did not contain expected text {expected_text!r}:\n"
                f"{target_rust.decode('utf-8', errors='replace')}"
            )

        if not is_case_06:
            assert b"not in canonical form" not in r_text.stdout, (
                f"[{case_name}] Rust stdout unexpectedly contained 'not in canonical form':\n"
                f"{r_text.stdout.decode('utf-8', errors='replace')}"
            )
        else:
            canonical_lines = [
                line
                for line in r_text.stdout.split(b"\n")
                if b"not in canonical form" in line
            ]
            assert len(canonical_lines) == 1, (
                f"[{case_name}] expected exactly 1 line with 'not in canonical form', got {len(canonical_lines)}"
            )

        assert r_text.returncode == p_text.returncode, (
            f"[{case_name}] text returncode: rust={r_text.returncode}, py={p_text.returncode}"
        )
        assert r_text.stdout == p_text.stdout, (
            f"[{case_name}] text stdout mismatch:\n"
            f"--- Rust ({len(r_text.stdout)} bytes) ---\n{r_text.stdout.decode('utf-8', errors='replace')}\n"
            f"--- Py ({len(p_text.stdout)} bytes) ---\n{p_text.stdout.decode('utf-8', errors='replace')}"
        )
        assert r_text.stderr == p_text.stderr, (
            f"[{case_name}] text stderr mismatch:\n"
            f"--- Rust ---\n{r_text.stderr.decode('utf-8', errors='replace')}\n"
            f"--- Py ---\n{p_text.stderr.decode('utf-8', errors='replace')}"
        )

        assert r_json.returncode == p_json.returncode, (
            f"[{case_name}] json returncode: rust={r_json.returncode}, py={p_json.returncode}"
        )
        assert r_json.stdout == p_json.stdout, (
            f"[{case_name}] json stdout mismatch:\n"
            f"--- Rust ---\n{r_json.stdout.decode('utf-8', errors='replace')}\n"
            f"--- Py ---\n{p_json.stdout.decode('utf-8', errors='replace')}"
        )
        assert r_json.stderr == p_json.stderr, (
            f"[{case_name}] json stderr mismatch:\n"
            f"--- Rust ---\n{r_json.stderr.decode('utf-8', errors='replace')}\n"
            f"--- Py ---\n{p_json.stderr.decode('utf-8', errors='replace')}"
        )

        if write_fixtures:
            target_fixture = FIXTURES_DIR / case_name
            shutil.copytree(
                case_folder,
                target_fixture,
                ignore=shutil.ignore_patterns(
                    "backlog", "monitor", "*.lock", "deployment.key.pem"
                ),
            )

            exp_txt = r_text.stdout if r_text.returncode != 2 else r_text.stderr
            exp_json = r_json.stdout if r_json.returncode != 2 else r_json.stderr

            # The key file's path first, so its separator isn't baked in for other OSes.
            for path, mark in (
                (case_folder / "other.pub.pem", b"{key_file}"),
                (case_folder, b"{data_dir}"),
            ):
                exp_txt = exp_txt.replace(str(path).encode("utf-8"), mark)
                exp_json = exp_json.replace(str(path).encode("utf-8"), mark)

            (target_fixture / "expected.txt").write_bytes(exp_txt)
            (target_fixture / "expected.json").write_bytes(exp_json)
            (target_fixture / "expected_exit").write_bytes(
                f"{r_text.returncode}\n".encode()
            )


def test_the_verifier_gives_the_recorded_answer_for_each_fixture() -> None:
    the_verifier_gives_the_recorded_answer_for_each_fixture()


def the_verifier_gives_the_recorded_answer_for_each_fixture() -> None:
    if not FIXTURES_DIR.is_dir():
        pytest.skip(f"fixtures directory {FIXTURES_DIR} does not exist yet")

    fixtures = sorted([d for d in FIXTURES_DIR.iterdir() if d.is_dir()])
    assert len(fixtures) > 0, f"no fixtures found in {FIXTURES_DIR}"

    for case_dir in fixtures:
        exp_exit = int((case_dir / "expected_exit").read_bytes().strip())
        exp_txt = (case_dir / "expected.txt").read_bytes()
        exp_json = (case_dir / "expected.json").read_bytes()

        # Substitute case_dir path back
        for mark, path in (
            (b"{key_file}", case_dir / "other.pub.pem"),
            (b"{data_dir}", case_dir),
        ):
            exp_txt = exp_txt.replace(mark, str(path).encode("utf-8"))
            exp_json = exp_json.replace(mark, str(path).encode("utf-8"))

        extra_args: list[str] = []
        other_key = case_dir / "other.pub.pem"
        if other_key.exists():
            extra_args = ["--public-key", str(other_key)]

        p_text = subprocess.run(
            [sys.executable, "-m", "kasanki_helium.verify", "--data-dir", str(case_dir)]
            + extra_args,
            capture_output=True,
            check=False,
        )
        p_json = subprocess.run(
            [
                sys.executable,
                "-m",
                "kasanki_helium.verify",
                "--data-dir",
                str(case_dir),
                "--json",
            ]
            + extra_args,
            capture_output=True,
            check=False,
        )

        assert p_text.returncode == exp_exit, (
            f"{case_dir.name}: exit {p_text.returncode} != {exp_exit}"
        )
        assert p_json.returncode == exp_exit, (
            f"{case_dir.name}: exit {p_json.returncode} != {exp_exit}"
        )

        if exp_exit == 2:
            assert p_text.stderr == exp_txt, (
                f"{case_dir.name}: text stderr mismatch:\n"
                f"--- expected ---\n{exp_txt.decode('utf-8', errors='replace')}\n"
                f"--- got ---\n{p_text.stderr.decode('utf-8', errors='replace')}"
            )
            assert p_json.stderr == exp_json, (
                f"{case_dir.name}: json stderr mismatch:\n"
                f"--- expected ---\n{exp_json.decode('utf-8', errors='replace')}\n"
                f"--- got ---\n{p_json.stderr.decode('utf-8', errors='replace')}"
            )
        else:
            assert p_text.stdout == exp_txt, (
                f"{case_dir.name}: text stdout mismatch:\n"
                f"--- expected ---\n{exp_txt.decode('utf-8', errors='replace')}\n"
                f"--- got ---\n{p_text.stdout.decode('utf-8', errors='replace')}"
            )
            assert p_json.stdout == exp_json, (
                f"{case_dir.name}: json stdout mismatch:\n"
                f"--- expected ---\n{exp_json.decode('utf-8', errors='replace')}\n"
                f"--- got ---\n{p_json.stdout.decode('utf-8', errors='replace')}"
            )


@pytest.fixture
def helium_bin() -> Path:
    configured = os.environ.get("HELIUM_BIN") or os.environ.get("HELIUM_SOLO_BIN")
    if configured:
        p = Path(configured)
        if p.is_file():
            return p
        pytest.skip(f"configured binary {configured} does not exist")
    default_bin = (
        REPO_ROOT
        / "engine"
        / "target"
        / "debug"
        / ("helium.exe" if sys.platform == "win32" else "helium")
    )
    if default_bin.is_file():
        return default_bin
    pytest.skip("no helium engine binary found")
