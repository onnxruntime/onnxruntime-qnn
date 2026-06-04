# qcom/scripts/all/test_archive_tests.py
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import tarfile
from pathlib import Path

import pytest
from archive_tests import (
    PerArchAcceptRules,
    archive_linux,
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
