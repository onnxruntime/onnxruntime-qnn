# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import hashlib
import shutil
import tarfile
import zipfile
from pathlib import Path

import pytest
from archive_testdata import OrtCoreDep, download_and_verify, parse_deps_txt, stage_sources, write_archives

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_parse_deps_txt_returns_url_and_sha1(tmp_path):
    deps_file = tmp_path / "deps.txt"
    deps_file.write_text(
        "# Comment\n"
        "abseil_cpp;https://example.com/abseil.zip;deadbeef\n"
        "ort_core;https://github.com/microsoft/onnxruntime/archive/refs/tags/v1.24.4.zip;c05dbfeb4841d9d1d0760ddb26d30d6d24352a67\n"
        "onnx;https://example.com/onnx.zip;cafebabe\n"
    )
    result = parse_deps_txt(deps_file)
    assert isinstance(result, OrtCoreDep)
    assert result.url == "https://github.com/microsoft/onnxruntime/archive/refs/tags/v1.24.4.zip"
    assert result.sha1 == "c05dbfeb4841d9d1d0760ddb26d30d6d24352a67"


def test_parse_deps_txt_raises_when_ort_core_missing(tmp_path):
    deps_file = tmp_path / "deps.txt"
    deps_file.write_text("abseil_cpp;https://example.com/abseil.zip;deadbeef\n")
    with pytest.raises(ValueError, match="ort_core"):
        parse_deps_txt(deps_file)


def _make_fake_zip(path: Path) -> str:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("hello.txt", "world")
    return hashlib.sha1(path.read_bytes()).hexdigest()


def test_download_and_verify_uses_cache_when_sha_matches(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    fake_zip = cache_dir / "ort_core.zip"
    sha1 = _make_fake_zip(fake_zip)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("urlretrieve should not be called when cache is valid")

    monkeypatch.setattr("archive_testdata.urlretrieve", fail_if_called)

    result = download_and_verify(
        url="https://example.invalid/should-not-fetch.zip",
        sha1=sha1,
        cache_path=fake_zip,
    )
    assert result == fake_zip


def test_download_and_verify_removes_stale_cache_and_redownloads(tmp_path, monkeypatch):
    """Stale cached zip (wrong SHA) is deleted and re-fetched automatically."""
    stale_zip = tmp_path / "ort_core.zip"
    # Write a zip with different content from what we'll claim the "fresh" download has.
    with zipfile.ZipFile(stale_zip, "w") as zf:
        zf.writestr("stale.txt", "old content")

    # A second zip that urlretrieve will "download" — has a known SHA1.
    fresh_zip = tmp_path / "fresh.zip"
    fresh_sha1 = _make_fake_zip(fresh_zip)  # {"hello.txt": "world"}

    calls: list[str] = []

    def fake_urlretrieve(url: str, path: str) -> None:
        calls.append(path)
        shutil.copy(str(fresh_zip), path)

    monkeypatch.setattr("archive_testdata.urlretrieve", fake_urlretrieve)

    result = download_and_verify(
        url="https://example.invalid/",
        sha1=fresh_sha1,
        cache_path=stale_zip,
    )
    assert len(calls) == 1, "urlretrieve must be called exactly once for the stale-cache case"
    assert result == stale_zip
    assert stale_zip.exists()


def test_download_and_verify_raises_on_fresh_download_sha_mismatch(tmp_path, monkeypatch):
    """When the freshly downloaded file doesn't match the expected SHA, raise ValueError."""
    cache = tmp_path / "ort_core.zip"
    downloaded = tmp_path / "downloaded.zip"
    _make_fake_zip(downloaded)  # has a real SHA1 ≠ "0"*40

    def fake_urlretrieve(url: str, path: str) -> None:
        shutil.copy(str(downloaded), path)

    monkeypatch.setattr("archive_testdata.urlretrieve", fake_urlretrieve)

    with pytest.raises(ValueError, match="SHA1 mismatch"):
        download_and_verify(
            url="https://example.invalid/",
            sha1="0" * 40,
            cache_path=cache,
        )


def _make_tree(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def test_stage_sources_lays_out_four_handles(tmp_path):
    ort_core_src = tmp_path / "ort_core"
    onnx_src = tmp_path / "onnx"
    _make_tree(ort_core_src / "onnxruntime/test/testdata", {"a.onnx": "TD-A"})
    _make_tree(onnx_src / "backend/test/data/pytorch-converted", {"x.pb": "PC-X"})
    _make_tree(onnx_src / "backend/test/data/pytorch-operator", {"y.pb": "PO-Y"})
    _make_tree(onnx_src / "backend/test/data/node", {"z.pb": "ND-Z"})

    stage = tmp_path / "stage"
    stage_sources(
        stage_root=stage,
        ort_core_src=ort_core_src,
        onnx_src=onnx_src,
    )
    assert (stage / "testdata" / "a.onnx").read_text() == "TD-A"
    assert (stage / "pytorch-converted" / "x.pb").read_text() == "PC-X"
    assert (stage / "pytorch-operator" / "y.pb").read_text() == "PO-Y"
    assert (stage / "node" / "z.pb").read_text() == "ND-Z"


def test_write_archives_emits_zip_and_tar_bz2(tmp_path):
    stage = tmp_path / "stage"
    _make_tree(stage / "testdata", {"a.onnx": "A"})
    _make_tree(stage / "pytorch-converted", {"b.pb": "B"})
    _make_tree(stage / "pytorch-operator", {"c.pb": "C"})
    _make_tree(stage / "node", {"d.pb": "D"})

    out = tmp_path / "out"
    out.mkdir()
    written = write_archives(stage_root=stage, output_dir=out)

    assert sorted(p.name for p in written) == ["onnxruntime-testdata.tar.bz2", "onnxruntime-testdata.zip"]

    with zipfile.ZipFile(out / "onnxruntime-testdata.zip") as z:
        names = sorted(z.namelist())
    assert names == ["node/d.pb", "pytorch-converted/b.pb", "pytorch-operator/c.pb", "testdata/a.onnx"]

    with tarfile.open(out / "onnxruntime-testdata.tar.bz2", "r:bz2") as t:
        names = sorted(m.name for m in t.getmembers() if m.isfile())
    assert names == ["node/d.pb", "pytorch-converted/b.pb", "pytorch-operator/c.pb", "testdata/a.onnx"]
