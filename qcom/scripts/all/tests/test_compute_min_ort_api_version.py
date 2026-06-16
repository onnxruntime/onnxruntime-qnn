# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Unit tests for qcom/scripts/all/compute_min_ort_api_version.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from compute_min_ort_api_version import (
    build_since_map,
    compute_floor,
    scan_ep_source,
)

SCRIPT = Path(__file__).resolve().parents[1] / "compute_min_ort_api_version.py"


def _h(headers: Path, name: str, body: str) -> None:
    (headers / name).write_text(body)


# ---------------------------------------------------------------------------
# build_since_map: every declaration form recognized by _DECL_RE
# ---------------------------------------------------------------------------


def test_each_decl_form_is_recognized(tmp_path: Path) -> None:
    _h(
        tmp_path,
        "all_forms.h",
        """
/** \\since Version 1.5 */
void(ORT_API_CALL* CreateEnv)(int level);

/** \\since Version 1.7 */
ORT_API2_STATUS(GetTensorMutableData, _In_ OrtValue* value);

/** \\since Version 1.9 */
ORT_API_STATUS(RunOptionsSetTerminate, _In_ OrtRunOptions* options);

/** \\since Version 1.11 */
ORT_API_T(const char*, GetBuildInfoString,);

/** \\since Version 1.13 */
ORT_CLASS_RELEASE(Allocator);
""",
    )
    m = build_since_map(tmp_path)
    assert m["CreateEnv"] == 5
    assert m["GetTensorMutableData"] == 7
    assert m["RunOptionsSetTerminate"] == 9
    assert m["GetBuildInfoString"] == 11
    assert m["ReleaseAllocator"] == 13


# ---------------------------------------------------------------------------
# build_since_map: M-1 regression — duplicates must keep the MAX version
# ---------------------------------------------------------------------------


def test_duplicate_member_keeps_max_since(tmp_path: Path) -> None:
    _h(
        tmp_path,
        "a.h",
        """
/** \\since Version 1.4 */
ORT_API2_STATUS(Foo, int x);
""",
    )
    _h(
        tmp_path,
        "b.h",
        """
/** \\since Version 1.20 */
ORT_API2_STATUS(Foo, int x);
""",
    )
    assert build_since_map(tmp_path)["Foo"] == 20


def test_undocumented_duplicate_does_not_clobber(tmp_path: Path) -> None:
    """No-\\since decl is treated as version 0; must not lower a real annotation."""
    _h(
        tmp_path,
        "annotated.h",
        """
/** \\since Version 1.15 */
ORT_API2_STATUS(Bar, int x);
""",
    )
    _h(
        tmp_path,
        "raw.h",
        """
/** Plain doc, no since tag. */
ORT_API2_STATUS(Bar, int x);
""",
    )
    assert build_since_map(tmp_path)["Bar"] == 15


def test_missing_since_is_zero(tmp_path: Path) -> None:
    _h(
        tmp_path,
        "raw.h",
        """
/** Plain doc, no since tag. */
ORT_API2_STATUS(NoSince, int x);
""",
    )
    assert build_since_map(tmp_path)["NoSince"] == 0


# ---------------------------------------------------------------------------
# scan_ep_source: each call-site form _CALL_RE accepts
# ---------------------------------------------------------------------------


def test_scan_ep_source_recognizes_call_forms(tmp_path: Path) -> None:
    src = tmp_path / "ep"
    src.mkdir()
    (src / "use.cc").write_text(
        """
void f() {
  ort_api->CreateEnv(1);
  ep_api.GetEpName();
  model_editor_api->AddNode();
  compile_api.CompileModel();
  ort_api ->  SpacedCall ( );
}
"""
    )
    names = scan_ep_source(src)
    assert names == {"CreateEnv", "GetEpName", "AddNode", "CompileModel", "SpacedCall"}


# ---------------------------------------------------------------------------
# compute_floor: end-to-end max over actually-used members
# ---------------------------------------------------------------------------


def test_compute_floor_uses_max_of_called_members(tmp_path: Path) -> None:
    headers = tmp_path / "include"
    headers.mkdir()
    _h(
        headers,
        "api.h",
        """
/** \\since Version 1.5 */
ORT_API2_STATUS(Called, int);
/** \\since Version 1.30 */
ORT_API2_STATUS(NotCalled, int);
/** \\since Version 1.18 */
ORT_API2_STATUS(AlsoCalled, int);
""",
    )
    ep = tmp_path / "ep"
    ep.mkdir()
    (ep / "u.cc").write_text("void f() { ort_api->Called(1); ep_api.AlsoCalled(2); }")
    assert compute_floor(headers, ep) == 18


# ---------------------------------------------------------------------------
# Driver: --check (drift / pass / no-baseline) and --update-baseline
# ---------------------------------------------------------------------------


def _make_tree(tmp_path: Path, floor_since: int) -> tuple[Path, Path]:
    headers = tmp_path / "include"
    headers.mkdir()
    _h(
        headers,
        "api.h",
        f"""
/** \\since Version 1.{floor_since} */
ORT_API2_STATUS(OnlyCall, int);
""",
    )
    ep = tmp_path / "ep"
    ep.mkdir()
    (ep / "u.cc").write_text("void f() { ort_api->OnlyCall(1); }")
    return headers, ep


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False
    )


def test_print_emits_floor(tmp_path: Path) -> None:
    headers, ep = _make_tree(tmp_path, floor_since=9)
    r = _run(["--ort-header-root", str(headers), "--ep-source-root", str(ep), "--print"])
    assert r.returncode == 0
    assert r.stdout.strip() == "9"


def test_update_baseline_then_check_passes(tmp_path: Path) -> None:
    headers, ep = _make_tree(tmp_path, floor_since=12)
    baseline = tmp_path / "MIN_ORT_API_VERSION.txt"
    r = _run(
        [
            "--ort-header-root", str(headers),
            "--ep-source-root", str(ep),
            "--update-baseline", "--baseline", str(baseline),
        ]
    )
    assert r.returncode == 0
    assert baseline.read_text().strip() == "12"

    r = _run(
        [
            "--ort-header-root", str(headers),
            "--ep-source-root", str(ep),
            "--check", "--baseline", str(baseline),
        ]
    )
    assert r.returncode == 0


def test_check_detects_drift(tmp_path: Path) -> None:
    headers, ep = _make_tree(tmp_path, floor_since=17)
    baseline = tmp_path / "MIN_ORT_API_VERSION.txt"
    baseline.write_text("3\n")
    r = _run(
        [
            "--ort-header-root", str(headers),
            "--ep-source-root", str(ep),
            "--check", "--baseline", str(baseline),
        ]
    )
    assert r.returncode == 1
    assert "drift" in r.stderr.lower() or "computed 17" in r.stderr


def test_check_missing_baseline_returns_2(tmp_path: Path) -> None:
    headers, ep = _make_tree(tmp_path, floor_since=4)
    r = _run(
        [
            "--ort-header-root", str(headers),
            "--ep-source-root", str(ep),
            "--check", "--baseline", str(tmp_path / "does-not-exist.txt"),
        ]
    )
    assert r.returncode == 2


def test_missing_ort_header_root_returns_2(tmp_path: Path) -> None:
    ep = tmp_path / "ep"
    ep.mkdir()
    (ep / "u.cc").write_text("void f() { ort_api->X(1); }")
    r = _run(
        [
            "--ort-header-root", str(tmp_path / "missing"),
            "--ep-source-root", str(ep),
            "--print",
        ]
    )
    assert r.returncode == 2


def test_write_header_writes_define(tmp_path: Path) -> None:
    headers, ep = _make_tree(tmp_path, floor_since=22)
    out = tmp_path / "gen" / "min_api.h"
    r = _run(
        [
            "--ort-header-root", str(headers),
            "--ep-source-root", str(ep),
            "--write-header", str(out),
        ]
    )
    assert r.returncode == 0
    text = out.read_text()
    assert "#define QNN_EP_MIN_ORT_API_VERSION 22" in text


def test_write_header_is_idempotent(tmp_path: Path) -> None:
    headers, ep = _make_tree(tmp_path, floor_since=8)
    out = tmp_path / "gen" / "min_api.h"
    args = [
        "--ort-header-root", str(headers),
        "--ep-source-root", str(ep),
        "--write-header", str(out),
    ]
    _run(args)
    first_mtime = out.stat().st_mtime_ns
    _run(args)
    # Content unchanged → file not rewritten → mtime preserved.
    assert out.stat().st_mtime_ns == first_mtime


def test_unknown_call_raises(tmp_path: Path) -> None:
    headers = tmp_path / "include"
    headers.mkdir()
    _h(headers, "api.h", "/** \\since Version 1.5 */\nORT_API2_STATUS(Known, int);")
    ep = tmp_path / "ep"
    ep.mkdir()
    (ep / "u.cc").write_text("void f() { ort_api->NotInHeaders(1); }")
    with pytest.raises(RuntimeError):
        compute_floor(headers, ep)
