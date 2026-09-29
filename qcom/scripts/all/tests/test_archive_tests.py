# qcom/scripts/all/test_archive_tests.py
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import logging
import tarfile
import zipfile
from pathlib import Path

import pytest
from archive_tests import (
    PerArchAcceptRules,
    _iter_msvc_redist,
    archive_linux,
    archive_windows,
)


def _touch(p: Path, content: str = "") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


@pytest.fixture
def per_arch_rules() -> PerArchAcceptRules:
    return PerArchAcceptRules()


# --- accept-list rule unit tests ---


@pytest.mark.parametrize(
    "name",
    [
        "onnxruntime.dll",
        "onnxruntime_providers_qnn.dll",
        "libonnxruntime.so",
        "libonnxruntime.so.1.24.4",
        "libQnnHtp.so",
        "QnnHtpV73Skel.cat",
        "MockGenie.dll",
    ],
)
def test_accepts_top_level_binaries_by_extension(per_arch_rules, name):
    assert per_arch_rules.accept_top_level(name)


@pytest.mark.parametrize(
    "name",
    [
        "onnxruntime_provider_test",
        "onnxruntime_perf_test",
        "onnxruntime_plugin_ep_onnx_test",
    ],
)
def test_accepts_linux_top_level_executables_no_extension(per_arch_rules, tmp_path, name):
    p = tmp_path / name
    _touch(p)
    p.chmod(p.stat().st_mode | 0o100)  # +x
    assert per_arch_rules.accept_top_level_with_path(p)


@pytest.mark.parametrize(
    "name",
    [
        "CTestTestfile.cmake",
        "run_tests.ps1",
        "run_tests.sh",
        "ctest",
        "ctest.exe",
        "python_test_files.txt",
        "onnxruntime_test_python.py",
        "onnxruntime_test_python_compile_api.py",
    ],
)
def test_accepts_top_level_test_drivers(per_arch_rules, name):
    assert per_arch_rules.accept_top_level(name)


@pytest.mark.parametrize(
    "rel",
    [
        "quantization/run_tests.py",
        "quantization/__init__.py",
    ],
)
def test_accepts_quantization_tree(per_arch_rules, rel):
    assert per_arch_rules.accept_repo_relative(Path(rel))


@pytest.mark.parametrize(
    "rel",
    [
        "_deps/onnx-src/onnx/backend/test/data/pytorch-converted/test_x/model.onnx",
        "testdata/float32/model.onnx",
        "_deps/abseil_cpp-src/foo/bar.cc",
        "bin/something",
        "lib/libfoo.a",
        "dist/onnxruntime_qnn-1.24.4.whl",
        "docs/index.html",
        "LICENSE",
        "Privacy.md",
        "Qualcomm_LICENSE.pdf",
        "ThirdPartyNotices.txt",
        "compile_commands.json",
        "build.ninja",
        "CMakeFiles/foo.txt",
        "CMakeCache.txt",
    ],
)
def test_rejects_dropped_paths(per_arch_rules, rel):
    assert not per_arch_rules.accept_repo_relative(Path(rel))


# --- providers_qnn anywhere-rule ---


@pytest.mark.parametrize(
    "rel",
    [
        "subdir/libonnxruntime_providers_qnn.so",
        "deeper/path/onnxruntime_providers_qnn.dll",
    ],
)
def test_accepts_providers_qnn_in_any_subdir(per_arch_rules, rel):
    assert per_arch_rules.accept_repo_relative(Path(rel))


# Non-shared-library variants must NOT be re-bundled — PDBs ship via upload_pdb_archive,
# .lib/.exp/.a are link-only artifacts that bloat the per-arch archive.
@pytest.mark.parametrize(
    "rel",
    [
        "subdir/onnxruntime_providers_qnn.lib",
        "subdir/onnxruntime_providers_qnn.exp",
        "subdir/onnxruntime_providers_qnn.pdb",
        "_deps/abseil_cpp-build/libonnxruntime_providers_qnn.a",
    ],
)
def test_rejects_providers_qnn_non_shared_library(per_arch_rules, rel):
    assert not per_arch_rules.accept_repo_relative(Path(rel))


# AAR-build test APKs must be included so test_aar.py runs the instrumentation suite
# instead of silently skipping with "AAR APKs not in test archive".
@pytest.mark.parametrize(
    "rel",
    [
        "java/androidtest/android/app/build/outputs/apk/debug/app-debug.apk",
        "java/androidtest/android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk",
    ],
)
def test_accepts_aar_test_apks(per_arch_rules, rel):
    assert per_arch_rules.accept_repo_relative(Path(rel))


# --- end-to-end archive smoke ---


def test_archive_linux_excludes_testdata(tmp_path):
    """Build a fake Release tree, run archive_linux, confirm testdata is NOT in the archive."""
    repo_root = tmp_path / "repo"
    build_root = repo_root / "build"
    plat = "linux-x86_64"
    rel = build_root / plat / "Release"
    _touch(rel / "libonnxruntime.so", "B")
    _touch(rel / "libQnnHtp.so", "B")
    _touch(rel / "onnxruntime_provider_test", "B")
    (rel / "onnxruntime_provider_test").chmod(0o755)
    _touch(rel / "CTestTestfile.cmake", "ctest")
    _touch(rel / "run_tests.sh", "#!/bin/sh")
    (rel / "run_tests.sh").chmod(0o755)
    _touch(rel / "python_test_files.txt", "")
    _touch(rel / "onnxruntime_test_python.py", "")
    _touch(rel / "quantization/__init__.py", "")
    _touch(rel / "testdata/float32/model.onnx", "DROP-ME")
    _touch(rel / "_deps/onnx-src/onnx/backend/test/data/pytorch-converted/foo/model.onnx", "DROP-ME")
    _touch(rel / "_deps/onnx-src/onnx/backend/test/data/node/foo/model.onnx", "DROP-ME")
    _touch(rel / "lib/libfoo.a", "DROP-ME")
    _touch(rel / "subdir/libonnxruntime_providers_qnn.so", "B")
    _touch(repo_root / "qcom/scripts/all/foo.py", "B")

    archive_linux(target_platform=plat, config="Release", repo_root=repo_root)

    archive = build_root / f"onnxruntime-tests-{plat}.tar.bz2"
    assert archive.exists()

    with tarfile.open(archive, "r:bz2") as tf:
        names = sorted(m.name for m in tf.getmembers() if m.isfile())

    # Sanity: binaries + scripts present
    assert any(n.endswith("Release/libonnxruntime.so") for n in names)
    assert any(n.endswith("Release/CTestTestfile.cmake") for n in names)
    assert any(n.endswith("Release/run_tests.sh") for n in names)
    assert any(n.endswith("Release/onnxruntime_provider_test") for n in names)
    assert any(n.endswith("subdir/libonnxruntime_providers_qnn.so") for n in names)
    assert any(n.endswith("qcom/scripts/all/foo.py") for n in names)

    # CRUCIAL: testdata is NOT inside the archive
    assert not any("testdata/" in n for n in names), [n for n in names if "testdata" in n]
    assert not any("pytorch-converted" in n for n in names)
    assert not any("backend/test/data/node" in n for n in names)
    assert not any(n.endswith("libfoo.a") for n in names)


# --- Windows MSVC redist bundling ---


def _make_fake_windows_tree(repo_root: Path, plat: str) -> None:
    """Minimal single-config build/ tree the archive_windows path can consume."""
    rel = repo_root / "build" / plat / "Release"
    _touch(rel / "onnxruntime.dll", "B")
    _touch(rel / "onnxruntime_providers_qnn.dll", "B")
    _touch(rel / "onnxruntime_provider_test.exe", "B")
    _touch(rel / "CTestTestfile.cmake", "ctest")
    _touch(rel / "run_tests.ps1", "ps")
    _touch(repo_root / "qcom/scripts/all/foo.py", "B")


def _make_fake_windows_tree_nested(repo_root: Path, plat: str) -> None:
    """Multi-config (VS) build/ tree: binaries nest at build/<plat>/Release/Release/."""
    outer = repo_root / "build" / plat / "Release"
    nested = outer / "Release"
    _touch(nested / "onnxruntime.dll", "B")
    _touch(nested / "onnxruntime_providers_qnn.dll", "B")
    _touch(nested / "onnxruntime_provider_test.exe", "B")
    _touch(outer / "CTestTestfile.cmake", "ctest")
    _touch(outer / "run_tests.ps1", "ps")
    _touch(repo_root / "qcom/scripts/all/foo.py", "B")


def _make_fake_vc_redist(redist_root: Path, arch_subdir: str) -> None:
    crt = redist_root / arch_subdir / "Microsoft.VC143.CRT"
    for name in ("msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
        _touch(crt / name, "REDIST")
    # Files we should NOT bundle (concrt140, vccorlib140, .props metadata, etc.)
    _touch(crt / "concrt140.dll", "X")
    _touch(crt / "Microsoft.VC143.CRT.manifest", "X")


@pytest.mark.parametrize(
    "plat,arch_subdir",
    [
        ("windows-x86_64", "x64"),
        ("windows-arm64", "arm64"),
        ("windows-arm64ec", "arm64"),  # arm64ec loads the native ARM64 CRT
        ("windows-arm64x", "arm64"),  # arm64x is a hybrid arm64+arm64ec set; native ARM64 CRT
    ],
)
def test_archive_windows_bundles_msvc_redist(tmp_path, plat, arch_subdir):
    repo_root = tmp_path / "repo"
    _make_fake_windows_tree(repo_root, plat)
    redist_root = tmp_path / "redist"
    _make_fake_vc_redist(redist_root, arch_subdir)

    archive_windows(target_platform=plat, config="Release", repo_root=repo_root, vc_redist_dir=redist_root)

    archive = repo_root / "build" / f"onnxruntime-tests-{plat}.zip"
    assert archive.exists()
    with zipfile.ZipFile(archive, "r") as zf:
        names = sorted(zf.namelist())

    redist_prefix = f"build/{plat}/Release/"
    for name in ("msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
        assert f"{redist_prefix}{name}" in names, f"missing {name} in {names}"
    # Out-of-scope redist files must not leak in.
    assert f"{redist_prefix}concrt140.dll" not in names
    assert not any(n.endswith(".manifest") for n in names)


def test_archive_windows_redist_colocates_with_binaries_nested_layout(tmp_path):
    """Regression: in the VS multi-config layout binaries nest at Release/Release/, so the
    redist must land there too — otherwise the loader can't find it on a clean machine."""
    repo_root = tmp_path / "repo"
    plat = "windows-arm64"
    _make_fake_windows_tree_nested(repo_root, plat)
    redist_root = tmp_path / "redist"
    _make_fake_vc_redist(redist_root, "arm64")

    archive_windows(target_platform=plat, config="Release", repo_root=repo_root, vc_redist_dir=redist_root)

    archive = repo_root / "build" / f"onnxruntime-tests-{plat}.zip"
    with zipfile.ZipFile(archive, "r") as zf:
        names = set(zf.namelist())

    # onnxruntime.dll lands in the doubled Release/Release/ dir; the redist must be in the SAME dir.
    bin_dir = f"build/{plat}/Release/Release/"
    assert f"{bin_dir}onnxruntime.dll" in names
    for name in ("msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll"):
        assert f"{bin_dir}{name}" in names, f"redist not co-located with onnxruntime.dll: {name}"
    # And NOT stranded in the outer Release/ directory off the loader's search path.
    assert f"build/{plat}/Release/msvcp140.dll" not in names


def test_archive_windows_omits_redist_when_dir_not_passed(tmp_path):
    repo_root = tmp_path / "repo"
    plat = "windows-x86_64"
    _make_fake_windows_tree(repo_root, plat)

    archive_windows(target_platform=plat, config="Release", repo_root=repo_root, vc_redist_dir=None)

    archive = repo_root / "build" / f"onnxruntime-tests-{plat}.zip"
    with zipfile.ZipFile(archive, "r") as zf:
        names = zf.namelist()
    assert not any("msvcp140" in n or "vcruntime140" in n for n in names)


def test_archive_windows_raises_when_redist_supplied_but_incomplete(tmp_path):
    """When --vc-redist-dir is given but the runtime can't be bundled, fail loudly rather
    than ship a silently-broken archive. Here the arch dir doesn't exist under the redist root."""
    repo_root = tmp_path / "repo"
    plat = "windows-arm64"  # maps to arm64 subdir
    _make_fake_windows_tree(repo_root, plat)
    redist_root = tmp_path / "redist"
    _make_fake_vc_redist(redist_root, "x64")  # populate a DIFFERENT arch -> arm64 not found

    with pytest.raises(RuntimeError, match="incomplete"):
        archive_windows(target_platform=plat, config="Release", repo_root=repo_root, vc_redist_dir=redist_root)


def test_iter_msvc_redist_unmapped_platform_yields_nothing(tmp_path, caplog):
    """An unmapped target_platform warns and yields no DLLs (the warn-then-empty branch)."""
    redist_root = tmp_path / "redist"
    _make_fake_vc_redist(redist_root, "x64")

    with caplog.at_level(logging.WARNING):
        result = list(_iter_msvc_redist(redist_root, "windows-mips"))

    assert result == []
    assert any("No MSVC redist arch mapping" in r.message for r in caplog.records)


def test_archive_windows_x64_floor_requires_vcruntime140_1(tmp_path):
    """x64 onnxruntime.dll imports vcruntime140_1.dll, so the completeness guard must reject a
    redist that lacks it — even though arm64 tolerates its absence."""
    repo_root = tmp_path / "repo"
    plat = "windows-x86_64"
    _make_fake_windows_tree(repo_root, plat)
    redist_root = tmp_path / "redist"
    crt = redist_root / "x64" / "Microsoft.VC143.CRT"
    # Everything the x64 floor needs EXCEPT vcruntime140_1.dll.
    for name in ("msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll"):
        _touch(crt / name, "REDIST")

    with pytest.raises(RuntimeError, match="vcruntime140_1.dll"):
        archive_windows(target_platform=plat, config="Release", repo_root=repo_root, vc_redist_dir=redist_root)


def test_archive_windows_arm64_floor_tolerates_missing_vcruntime140_1(tmp_path):
    """arm64 onnxruntime.dll does NOT import vcruntime140_1.dll; a redist without it must still
    succeed (the arm64 floor must not over-require the x64-only DLL)."""
    repo_root = tmp_path / "repo"
    plat = "windows-arm64"
    _make_fake_windows_tree(repo_root, plat)
    redist_root = tmp_path / "redist"
    crt = redist_root / "arm64" / "Microsoft.VC143.CRT"
    for name in ("msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll"):
        _touch(crt / name, "REDIST")

    archive_windows(target_platform=plat, config="Release", repo_root=repo_root, vc_redist_dir=redist_root)

    archive = repo_root / "build" / f"onnxruntime-tests-{plat}.zip"
    with zipfile.ZipFile(archive, "r") as zf:
        names = zf.namelist()
    for name in ("msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll"):
        assert f"build/{plat}/Release/{name}" in names


def test_archive_windows_dedupes_multiple_crt_versions(tmp_path):
    """With side-by-side CRT version folders, the numerically-highest is bundled exactly once
    (VC9 < VC143 — must not sort lexicographically)."""
    repo_root = tmp_path / "repo"
    plat = "windows-x86_64"
    _make_fake_windows_tree(repo_root, plat)
    redist_root = tmp_path / "redist"
    # VC9 would sort AFTER VC143 lexicographically; the numeric sort must still pick VC143.
    for ver in ("Microsoft.VC9.CRT", "Microsoft.VC142.CRT", "Microsoft.VC143.CRT"):
        crt = redist_root / "x64" / ver
        for name in ("msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
            _touch(crt / name, "REDIST")

    archive_windows(target_platform=plat, config="Release", repo_root=repo_root, vc_redist_dir=redist_root)

    archive = repo_root / "build" / f"onnxruntime-tests-{plat}.zip"
    with zipfile.ZipFile(archive, "r") as zf:
        names = zf.namelist()
    # Each redist DLL must appear exactly once despite three CRT folders existing.
    for name in ("msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
        assert names.count(f"build/{plat}/Release/{name}") == 1, names
