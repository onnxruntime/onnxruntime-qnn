# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
from archive_testdata import _HANDLES
from extract_testdata import MAPPING, extract


def _make_archive(path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for arcname, content in files.items():
            zf.writestr(arcname, content)


def test_mapping_contains_all_four_handles():
    assert set(MAPPING.keys()) == {"testdata", "pytorch-converted", "pytorch-operator", "node"}


def test_handles_match_extract_mapping():
    """Contract: archive_testdata._HANDLES and extract_testdata.MAPPING must declare the same handle set.

    qdc_runner.py also imports MAPPING; a key drift between producer (archive_testdata) and
    consumers (extract_testdata, qdc_runner) silently breaks testdata layout on at least one platform.
    """
    assert set(_HANDLES.keys()) == set(MAPPING.keys())


def test_extract_places_files_at_expected_paths(tmp_path):
    archive = tmp_path / "td.zip"
    _make_archive(
        archive,
        {
            "testdata/a.onnx": "A",
            "pytorch-converted/b.pb": "B",
            "pytorch-operator/c.pb": "C",
            "node/d.pb": "D",
        },
    )
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    extract(archive=archive, target_platform="linux-x86_64", repo_root=repo_root)
    assert (repo_root / "build/linux-x86_64/Release/testdata/a.onnx").read_text() == "A"
    assert (
        repo_root / "build/linux-x86_64/Release/_deps/onnx-src/onnx/backend/test/data/pytorch-converted/b.pb"
    ).read_text() == "B"
    assert (
        repo_root / "build/linux-x86_64/Release/_deps/onnx-src/onnx/backend/test/data/pytorch-operator/c.pb"
    ).read_text() == "C"
    assert (repo_root / "cmake/external/onnx/onnx/backend/test/data/node/d.pb").read_text() == "D"


def test_extract_rejects_unknown_handle(tmp_path):
    archive = tmp_path / "td.zip"
    _make_archive(archive, {"unexpected/a.onnx": "A"})
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    with pytest.raises(ValueError, match="unexpected"):
        extract(archive=archive, target_platform="linux-x86_64", repo_root=repo_root)


def test_extract_supports_tar_bz2(tmp_path):
    archive = tmp_path / "td.tar.bz2"
    src = tmp_path / "src"
    (src / "testdata").mkdir(parents=True)
    (src / "testdata" / "a.onnx").write_text("A")
    (src / "pytorch-converted").mkdir()
    (src / "pytorch-converted" / "b.pb").write_text("B")
    (src / "pytorch-operator").mkdir()
    (src / "pytorch-operator" / "c.pb").write_text("C")
    (src / "node").mkdir()
    (src / "node" / "d.pb").write_text("D")
    with tarfile.open(archive, "w:bz2") as tf:
        for p in src.glob("**/*"):
            if p.is_file():
                tf.add(p, str(p.relative_to(src)))
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    extract(archive=archive, target_platform="linux-aarch64_oe_gcc11_2", repo_root=repo_root)
    assert (repo_root / "build/linux-aarch64_oe_gcc11_2/Release/testdata/a.onnx").read_text() == "A"


def test_extract_cli_repo_root(tmp_path):
    """Verify --repo-root directs output to a custom location."""
    archive = tmp_path / "td.zip"
    _make_archive(archive, {"testdata/a.onnx": "A"})
    custom_root = tmp_path / "custom_root"
    custom_root.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent.parent / "extract_testdata.py"),
            "--target-platform",
            "linux-x86_64",
            "--archive",
            str(archive),
            "--repo-root",
            str(custom_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (custom_root / "build/linux-x86_64/Release/testdata/a.onnx").exists()


def test_extract_uses_release_release_for_multiconfig_windows(tmp_path):
    """When Release/Release/ already exists (multi-config VS build), testdata goes there."""
    archive = tmp_path / "td.zip"
    _make_archive(archive, {"testdata/a.onnx": "A", "node/d.pb": "D"})
    repo_root = tmp_path / "repo"
    # Simulate a previously-extracted per-arch archive that has Release/Release/ binaries.
    (repo_root / "build/windows-arm64/Release/Release").mkdir(parents=True)
    extract(archive=archive, target_platform="windows-arm64", repo_root=repo_root)
    # testdata must land inside Release/Release/, not Release/
    assert (repo_root / "build/windows-arm64/Release/Release/testdata/a.onnx").read_text() == "A"
    assert not (repo_root / "build/windows-arm64/Release/testdata").exists()
    # node is repo-root-relative; unaffected by multi-config detection
    assert (repo_root / "cmake/external/onnx/onnx/backend/test/data/node/d.pb").read_text() == "D"
