# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Unit tests for qcom/scripts/all/compute_min_ort_api_version.py."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from compute_min_ort_api_version import (
    build_since_map,
    compute_floor,
    fetch_ort_headers,
    parse_deps_txt,
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
# scan_ep_source: Ort:: wrapper method resolution via _WRAPPER_TO_C_API
# ---------------------------------------------------------------------------


def test_scan_ep_source_resolves_wrapper_methods(tmp_path: Path) -> None:
    src = tmp_path / "ep"
    src.mkdir()
    (src / "use.cc").write_text(
        """
#include <onnxruntime_cxx_api.h>
void f(const OrtNode* n) {
  Ort::ConstNode(n).GetDomain();
  Ort::ConstNode(n).GetOperatorType();
}
"""
    )
    names = scan_ep_source(src)
    assert "Node_GetDomain" in names
    assert "Node_GetOperatorType" in names


def test_scan_ep_source_resolves_ambiguous_method(tmp_path: Path) -> None:
    """GetName maps to Node_GetName, Graph_GetName, GetValueInfoName — all added."""
    src = tmp_path / "ep"
    src.mkdir()
    (src / "use.cc").write_text(
        """
void f(const OrtNode* n) {
  Ort::ConstNode(n).GetName();
}
"""
    )
    names = scan_ep_source(src)
    assert "Node_GetName" in names
    assert "Graph_GetName" in names
    assert "GetValueInfoName" in names


def test_scan_ep_source_tripwire_on_unknown_wrapper_method(tmp_path: Path) -> None:
    """Unknown Ort:: wrapper method call triggers RuntimeError."""
    src = tmp_path / "ep"
    src.mkdir()
    (src / "use.cc").write_text(
        """
void f(const OrtNode* n) {
  Ort::ConstNode(n).BrandNewMethod();
}
"""
    )
    with pytest.raises(RuntimeError, match="Unknown Ort:: wrapper method"):
        scan_ep_source(src)


def test_scan_ep_source_ignores_non_ort_getname(tmp_path: Path) -> None:
    """Files without Ort::Const should not resolve wrapper method names."""
    src = tmp_path / "ep"
    src.mkdir()
    (src / "use.cc").write_text(
        """
void f() {
  some_other_object.GetName();
}
"""
    )
    names = scan_ep_source(src)
    assert "Node_GetName" not in names
    assert "GetValueInfoName" not in names


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
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False)


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
            "--ort-header-root",
            str(headers),
            "--ep-source-root",
            str(ep),
            "--update-baseline",
            "--baseline",
            str(baseline),
        ]
    )
    assert r.returncode == 0
    assert baseline.read_text().strip() == "12"

    r = _run(
        [
            "--ort-header-root",
            str(headers),
            "--ep-source-root",
            str(ep),
            "--check",
            "--baseline",
            str(baseline),
        ]
    )
    assert r.returncode == 0


def test_check_detects_drift(tmp_path: Path) -> None:
    headers, ep = _make_tree(tmp_path, floor_since=17)
    baseline = tmp_path / "MIN_ORT_API_VERSION.txt"
    baseline.write_text("3\n")
    r = _run(
        [
            "--ort-header-root",
            str(headers),
            "--ep-source-root",
            str(ep),
            "--check",
            "--baseline",
            str(baseline),
        ]
    )
    assert r.returncode == 1
    assert "drift" in r.stderr.lower() or "computed 17" in r.stderr


def test_check_missing_baseline_returns_2(tmp_path: Path) -> None:
    headers, ep = _make_tree(tmp_path, floor_since=4)
    r = _run(
        [
            "--ort-header-root",
            str(headers),
            "--ep-source-root",
            str(ep),
            "--check",
            "--baseline",
            str(tmp_path / "does-not-exist.txt"),
        ]
    )
    assert r.returncode == 2


def test_missing_ort_header_root_returns_2(tmp_path: Path) -> None:
    ep = tmp_path / "ep"
    ep.mkdir()
    (ep / "u.cc").write_text("void f() { ort_api->X(1); }")
    r = _run(
        [
            "--ort-header-root",
            str(tmp_path / "missing"),
            "--ep-source-root",
            str(ep),
            "--print",
        ]
    )
    assert r.returncode == 2


def test_write_header_writes_define(tmp_path: Path) -> None:
    headers, ep = _make_tree(tmp_path, floor_since=22)
    out = tmp_path / "gen" / "min_api.h"
    r = _run(
        [
            "--ort-header-root",
            str(headers),
            "--ep-source-root",
            str(ep),
            "--write-header",
            str(out),
        ]
    )
    assert r.returncode == 0
    text = out.read_text()
    assert "#define QNN_EP_MIN_ORT_API_VERSION 22" in text


def test_write_header_is_idempotent(tmp_path: Path) -> None:
    headers, ep = _make_tree(tmp_path, floor_since=8)
    out = tmp_path / "gen" / "min_api.h"
    args = [
        "--ort-header-root",
        str(headers),
        "--ep-source-root",
        str(ep),
        "--write-header",
        str(out),
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


def test_partial_unknown_raises(tmp_path: Path) -> None:
    headers = tmp_path / "include"
    headers.mkdir()
    _h(headers, "api.h", "/** \\since Version 1.5 */\nORT_API2_STATUS(Known, int);")
    ep = tmp_path / "ep"
    ep.mkdir()
    (ep / "u.cc").write_text("void f() { ort_api->Known(1); ort_api->NotInHeaders(2); }")
    with pytest.raises(RuntimeError, match="not found in"):
        compute_floor(headers, ep)


# ---------------------------------------------------------------------------
# parse_deps_txt + fetch_ort_headers: header fetch from cmake/deps.txt
# ---------------------------------------------------------------------------


def test_parse_deps_txt_finds_named_dep(tmp_path: Path) -> None:
    deps = tmp_path / "deps.txt"
    deps.write_text(
        "# comment\n"
        "\n"
        "ort_core;https://example.invalid/ort_core.zip;abcdef1234567890abcdef1234567890abcdef12\n"
        "extensions;https://example.invalid/ext.zip;0011223344556677889900112233445566778899\n"
    )
    url, sha = parse_deps_txt(deps, "ort_core")
    assert url == "https://example.invalid/ort_core.zip"
    assert sha == "abcdef1234567890abcdef1234567890abcdef12"


def test_parse_deps_txt_raises_on_missing(tmp_path: Path) -> None:
    deps = tmp_path / "deps.txt"
    deps.write_text("other;https://x/y.zip;deadbeef\n")
    with pytest.raises(RuntimeError):
        parse_deps_txt(deps, "ort_core")


def _build_fake_ort_core_zip(tmp_path: Path) -> tuple[Path, str]:
    """Build a zip that mimics a GitHub source-archive: top-level dir contains
    include/onnxruntime/core/session/onnxruntime_c_api.h. Return (zip path, sha1)."""
    archive_root = tmp_path / "onnxruntime-fakecommit"
    inc = archive_root / "include" / "onnxruntime" / "core" / "session"
    inc.mkdir(parents=True)
    (inc / "onnxruntime_c_api.h").write_text("/** \\since Version 1.7 */\nORT_API2_STATUS(FetchTest, int);\n")
    zip_path = tmp_path / "ort_core.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in archive_root.rglob("*"):
            zf.write(p, p.relative_to(tmp_path))
    sha1 = hashlib.sha1(zip_path.read_bytes()).hexdigest()
    return zip_path, sha1


def test_fetch_ort_headers_downloads_verifies_extracts(tmp_path: Path) -> None:
    zip_path, sha1 = _build_fake_ort_core_zip(tmp_path)
    deps = tmp_path / "deps.txt"
    deps.write_text(f"ort_core;{zip_path.as_uri()};{sha1}\n")
    cache = tmp_path / "cache"

    include_dir = fetch_ort_headers(deps, cache_root=cache)
    assert include_dir.is_dir()
    assert (include_dir / "onnxruntime" / "core" / "session" / "onnxruntime_c_api.h").is_file()
    # Cached under the SHA1 slot.
    assert include_dir.parent.name == f"ort_core-{sha1}"


def test_fetch_ort_headers_is_cached(tmp_path: Path) -> None:
    zip_path, sha1 = _build_fake_ort_core_zip(tmp_path)
    deps = tmp_path / "deps.txt"
    deps.write_text(f"ort_core;{zip_path.as_uri()};{sha1}\n")
    cache = tmp_path / "cache"

    first = fetch_ort_headers(deps, cache_root=cache)
    # Drop a sentinel into the include dir; a second fetch must keep it
    # (i.e., must NOT re-download or wipe the cached tree).
    sentinel = first / "_sentinel.txt"
    sentinel.write_text("ok")
    second = fetch_ort_headers(deps, cache_root=cache)
    assert second == first
    assert sentinel.is_file()


def test_fetch_ort_headers_rejects_sha1_mismatch(tmp_path: Path) -> None:
    zip_path, _ = _build_fake_ort_core_zip(tmp_path)
    deps = tmp_path / "deps.txt"
    deps.write_text(f"ort_core;{zip_path.as_uri()};{'0' * 40}\n")
    with pytest.raises(RuntimeError, match="sha1 mismatch"):
        fetch_ort_headers(deps, cache_root=tmp_path / "cache")


def test_main_fetch_from_deps_txt_end_to_end(tmp_path: Path) -> None:
    """Driver: --fetch-from-deps-txt fetches the archive and computes the floor
    against EP source the caller supplies."""
    zip_path, sha1 = _build_fake_ort_core_zip(tmp_path)
    # Build a synthetic repo layout the script can walk.
    repo = tmp_path / "repo"
    (repo / "cmake").mkdir(parents=True)
    (repo / "cmake" / "deps.txt").write_text(f"ort_core;{zip_path.as_uri()};{sha1}\n")
    ep = repo / "onnxruntime" / "core" / "providers" / "qnn"
    ep.mkdir(parents=True)
    (ep / "use.cc").write_text("void f() { ort_api->FetchTest(1); }")

    # Stage the script into qcom/scripts/all/ under repo so __file__ parents[3]
    # resolves to `repo`.
    script_dst_dir = repo / "qcom" / "scripts" / "all"
    script_dst_dir.mkdir(parents=True)
    script_dst = script_dst_dir / "compute_min_ort_api_version.py"
    script_dst.write_text(SCRIPT.read_text())

    env = {
        **os.environ,
        "QNN_EP_LINT_CACHE": str(tmp_path / "cache"),
    }
    r = subprocess.run(
        [sys.executable, str(script_dst), "--fetch-from-deps-txt", "--print"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "7"


# ---------------------------------------------------------------------------
# Exit-code contract: this is what the lint adapter routes on.
# 0 = success, 1 = drift, 2 = cannot compute. Locked down by these tests.
# ---------------------------------------------------------------------------


def test_exit_code_success_is_zero(tmp_path: Path) -> None:
    headers, ep = _make_tree(tmp_path, floor_since=5)
    r = _run(["--ort-header-root", str(headers), "--ep-source-root", str(ep), "--print"])
    assert r.returncode == 0


def test_exit_code_drift_is_one(tmp_path: Path) -> None:
    headers, ep = _make_tree(tmp_path, floor_since=10)
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("9\n")
    r = _run(
        [
            "--ort-header-root",
            str(headers),
            "--ep-source-root",
            str(ep),
            "--check",
            "--baseline",
            str(baseline),
        ]
    )
    assert r.returncode == 1


@pytest.mark.parametrize(
    "extra_args",
    [
        # baseline file missing
        ["--check"],
        # --update-baseline without --baseline
        ["--update-baseline"],
        # --check without --baseline
        # (these all hit the configuration-error path, exit 2)
    ],
)
def test_exit_code_config_error_is_two(tmp_path: Path, extra_args: list[str]) -> None:
    headers, ep = _make_tree(tmp_path, floor_since=3)
    r = _run(["--ort-header-root", str(headers), "--ep-source-root", str(ep), *extra_args])
    assert r.returncode == 2


def test_exit_code_unparseable_baseline_is_two(tmp_path: Path) -> None:
    headers, ep = _make_tree(tmp_path, floor_since=3)
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("not-a-number\n")
    r = _run(
        [
            "--ort-header-root",
            str(headers),
            "--ep-source-root",
            str(ep),
            "--check",
            "--baseline",
            str(baseline),
        ]
    )
    assert r.returncode == 2


def test_exit_code_missing_headers_is_two(tmp_path: Path) -> None:
    ep = tmp_path / "ep"
    ep.mkdir()
    (ep / "u.cc").write_text("void f() { ort_api->X(1); }")
    r = _run(
        [
            "--ort-header-root",
            str(tmp_path / "missing"),
            "--ep-source-root",
            str(ep),
            "--print",
        ]
    )
    assert r.returncode == 2


def test_exit_code_sha1_mismatch_is_two(tmp_path: Path) -> None:
    """--fetch-from-deps-txt must surface SHA1 mismatch as exit 2, not 1."""
    zip_path, _real_sha = _build_fake_ort_core_zip(tmp_path)
    repo = tmp_path / "repo"
    (repo / "cmake").mkdir(parents=True)
    # Intentionally wrong sha so the fetch verifier rejects.
    (repo / "cmake" / "deps.txt").write_text(f"ort_core;{zip_path.as_uri()};{'0' * 40}\n")
    ep = repo / "onnxruntime" / "core" / "providers" / "qnn"
    ep.mkdir(parents=True)
    (ep / "use.cc").write_text("void f() { ort_api->FetchTest(1); }")
    script_dst_dir = repo / "qcom" / "scripts" / "all"
    script_dst_dir.mkdir(parents=True)
    script_dst = script_dst_dir / "compute_min_ort_api_version.py"
    script_dst.write_text(SCRIPT.read_text())

    env = {**os.environ, "QNN_EP_LINT_CACHE": str(tmp_path / "cache")}
    r = subprocess.run(
        [sys.executable, str(script_dst), "--fetch-from-deps-txt", "--print"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 2, r.stderr
    assert "sha1 mismatch" in r.stderr.lower()


# ---------------------------------------------------------------------------
# --lintrunner: drift -> ort-api-floor-drift, env error -> ort-api-floor-check-error.
# Always exits 0 and emits one JSON finding on stdout.
# ---------------------------------------------------------------------------


def test_lintrunner_routes_drift_to_floor_drift(tmp_path: Path) -> None:
    """Real drift (computed != baseline) -> exit 0 with an 'ort-api-floor-drift' finding."""
    headers, ep = _make_tree(tmp_path, floor_since=17)
    baseline = tmp_path / "MIN_ORT_API_VERSION.txt"
    baseline.write_text("3\n")
    r = _run(
        [
            "--ort-header-root",
            str(headers),
            "--ep-source-root",
            str(ep),
            "--lintrunner",
            "--baseline",
            str(baseline),
        ]
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout.strip())
    assert payload["name"] == "ort-api-floor-drift"
    assert payload["code"] == "MIN-ORT-API-VERSION"
    assert "computed 17" in payload["description"] or "drift" in payload["description"].lower()


def test_lintrunner_routes_env_error_to_check_error(tmp_path: Path) -> None:
    """Cannot compute (missing headers) -> exit 0 with an 'ort-api-floor-check-error'
    finding, so reviewers can tell it apart from a real baseline-refresh request."""
    ep = tmp_path / "ep"
    ep.mkdir()
    (ep / "u.cc").write_text("void f() { ort_api->X(1); }")
    baseline = tmp_path / "MIN_ORT_API_VERSION.txt"
    baseline.write_text("3\n")
    r = _run(
        [
            "--ort-header-root",
            str(tmp_path / "missing"),
            "--ep-source-root",
            str(ep),
            "--lintrunner",
            "--baseline",
            str(baseline),
        ]
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout.strip())
    assert payload["name"] == "ort-api-floor-check-error"


def test_lintrunner_pass_emits_nothing(tmp_path: Path) -> None:
    """Baseline matches -> exit 0 and no finding on stdout."""
    headers, ep = _make_tree(tmp_path, floor_since=11)
    baseline = tmp_path / "MIN_ORT_API_VERSION.txt"
    baseline.write_text("11\n")
    r = _run(
        [
            "--ort-header-root",
            str(headers),
            "--ep-source-root",
            str(ep),
            "--lintrunner",
            "--baseline",
            str(baseline),
        ]
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == ""


def test_lintrunner_ignores_trailing_paths(tmp_path: Path) -> None:
    """Trailing file paths (lintrunner @{{PATHSFILE}} convention) are accepted
    and ignored; the check stays project-wide."""
    headers, ep = _make_tree(tmp_path, floor_since=6)
    baseline = tmp_path / "MIN_ORT_API_VERSION.txt"
    baseline.write_text("6\n")
    r = _run(
        [
            "--ort-header-root",
            str(headers),
            "--ep-source-root",
            str(ep),
            "--lintrunner",
            "--baseline",
            str(baseline),
            "some/file.cc",
            "another/file.h",
        ]
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == ""
