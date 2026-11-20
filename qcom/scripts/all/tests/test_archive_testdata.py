# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import hashlib
import http.client
import json
import shutil
import tarfile
import zipfile
from pathlib import Path

import archive_testdata
import pytest
from archive_testdata import (
    OrtCoreDep,
    download_and_verify,
    ort_core_cache_path,
    parse_deps_txt,
    stage_sources,
    write_archives,
)
from extract_testdata import extract
from ort_build_identity import RUNTIME_VERSION_FILE, TESTDATA_IDENTITY_FILE, build_identity
from ort_build_identity import testdata_identity as make_testdata_identity

REPO_ROOT = Path(__file__).resolve().parents[3]


def _sha1_of_file(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


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


def test_ort_core_cache_path_matches_filecache_layout(tmp_path):
    """The cache path must match package_manager.py's FileCache (<root>/ort_core/<url-basename>) so a
    zip already fetched by fetch_cmake_deps.py is reused rather than re-downloaded."""
    url = "https://github.com/microsoft/onnxruntime/archive/refs/tags/v1.27.0.zip"
    assert ort_core_cache_path(tmp_path, url) == tmp_path / "ort_core" / "v1.27.0.zip"


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
    """A persistent SHA mismatch is retried, then raises RuntimeError once attempts are exhausted."""
    cache = tmp_path / "ort_core.zip"
    downloaded = tmp_path / "downloaded.zip"
    _make_fake_zip(downloaded)  # has a real SHA1 ≠ "0"*40

    calls: list[str] = []

    def fake_urlretrieve(url: str, path: str) -> None:
        calls.append(path)
        shutil.copy(str(downloaded), path)

    monkeypatch.setattr("archive_testdata.urlretrieve", fake_urlretrieve)
    monkeypatch.setattr("archive_testdata.time.sleep", lambda _s: None)  # keep the retry loop fast

    with pytest.raises(RuntimeError, match="after 3 attempt"):
        download_and_verify(
            url="https://example.invalid/",
            sha1="0" * 40,
            cache_path=cache,
        )
    assert len(calls) == 3, "a persistent SHA mismatch should exhaust all download attempts"
    assert not cache.exists(), "the bad partial file must not be left in the cache"


def test_download_and_verify_retries_incomplete_read_then_succeeds(tmp_path, monkeypatch):
    """The observed CI failure (transient http.client.IncompleteRead) is retried and then succeeds."""
    cache = tmp_path / "ort_core.zip"
    good_zip = tmp_path / "good.zip"
    good_sha1 = _make_fake_zip(good_zip)

    attempts: list[str] = []

    def flaky_urlretrieve(url: str, path: str) -> None:
        attempts.append(path)
        if len(attempts) == 1:
            # Simulate a truncated transfer: write a partial file, then raise like urllib would.
            Path(path).write_bytes(b"partial")
            raise http.client.IncompleteRead(b"partial", 3451)
        shutil.copy(str(good_zip), path)

    monkeypatch.setattr("archive_testdata.urlretrieve", flaky_urlretrieve)
    monkeypatch.setattr("archive_testdata.time.sleep", lambda _s: None)

    result = download_and_verify(url="https://example.invalid/", sha1=good_sha1, cache_path=cache)
    assert len(attempts) == 2, "must retry after the first IncompleteRead"
    assert result == cache
    assert _sha1_of_file(cache) == good_sha1


def test_download_and_verify_content_addressed_cache_hit_skips_download(tmp_path, monkeypatch):
    """A cache file already matching the expected SHA1 is returned without any network call."""
    persistent = tmp_path / "ort_core-cachekey.zip"
    sha1 = _make_fake_zip(persistent)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("urlretrieve must not run on a persistent-cache hit")

    monkeypatch.setattr("archive_testdata.urlretrieve", fail_if_called)

    result = download_and_verify(url="https://example.invalid/", sha1=sha1, cache_path=persistent)
    assert result == persistent


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


@pytest.mark.parametrize("extension", ["zip", "tar.bz2"])
def test_prebuilt_archive_uses_selected_assets_and_checks_identity(tmp_path, monkeypatch, extension):
    sdk = tmp_path / "sdk"
    assets = tmp_path / "assets"
    _make_tree(sdk, {"VERSION_NUMBER": "1.25.0"})
    _make_tree(assets, {"VERSION_NUMBER": "1.25.0", "onnxruntime/test/testdata/selected.onnx": "selected"})
    for handle in ("pytorch-converted", "pytorch-operator", "node"):
        _make_tree(tmp_path / f"cmake/external/onnx/onnx/backend/test/data/{handle}", {"model.onnx": handle})

    def unexpected_fetch(*args, **kwargs):
        raise AssertionError("Prebuilt testdata must not request ORT source")

    monkeypatch.setattr(archive_testdata, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(archive_testdata, "parse_deps_txt", unexpected_fetch)
    monkeypatch.setattr(archive_testdata, "download_and_verify", unexpected_fetch)
    monkeypatch.setenv("ORT_PREBUILT_ROOT", str(sdk))
    monkeypatch.setenv("ORT_TEST_ASSETS_ROOT", str(assets))
    monkeypatch.setattr(
        "sys.argv", ["archive_testdata.py", f"--build-dir={tmp_path / 'stage'}", f"--output-dir={tmp_path / 'out'}"]
    )
    assert archive_testdata.main() == 0
    archive = tmp_path / f"out/onnxruntime-testdata.{extension}"
    if extension == "zip":
        with zipfile.ZipFile(archive) as zf:
            assert json.loads(zf.read(TESTDATA_IDENTITY_FILE)) == json.loads(
                make_testdata_identity(str(sdk), str(assets))
            )
    destination = tmp_path / "device"
    extract(archive, "android-aarch64", destination)
    models = destination / "build/android-aarch64/Release/testdata"
    assert (models / "selected.onnx").read_text() == "selected"

    # Local build identities change on relocation; portable model compatibility does not.
    relocated_sdk = tmp_path / "other/sdk"
    relocated_assets = tmp_path / "other/assets"
    shutil.copytree(sdk, relocated_sdk)
    shutil.copytree(assets, relocated_assets)
    assert build_identity(str(sdk), str(assets)) != build_identity(str(relocated_sdk), str(relocated_assets))
    for selected_sdk, selected_assets in (
        (relocated_sdk, assets),
        (sdk, relocated_assets),
        (relocated_sdk, relocated_assets),
    ):
        monkeypatch.setenv("ORT_PREBUILT_ROOT", str(selected_sdk))
        monkeypatch.setenv("ORT_TEST_ASSETS_ROOT", str(selected_assets))
        extract(archive, "android-aarch64", destination)
        assert (models / "selected.onnx").read_text() == "selected"

    # Archive-only consumers need the version carried by their per-arch binaries.
    monkeypatch.delenv("ORT_PREBUILT_ROOT")
    monkeypatch.delenv("ORT_TEST_ASSETS_ROOT")
    with pytest.raises(ValueError, match="runtime version"):
        extract(archive, "android-aarch64", destination)
    version_file = models.parent / RUNTIME_VERSION_FILE
    version_file.write_text("1.25.0", encoding="utf-8")
    extract(archive, "android-aarch64", destination)
    version_file.write_text("1.24.0", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the selected ORT"):
        extract(archive, "android-aarch64", destination)
    assert (models / "selected.onnx").read_text() == "selected"
    version_file.unlink()

    # Equal SDK/assets versions still cannot consume an archive from another version.
    (relocated_sdk / "VERSION_NUMBER").write_text("1.26.0", encoding="utf-8")
    (relocated_assets / "VERSION_NUMBER").write_text("1.26.0", encoding="utf-8")
    monkeypatch.setenv("ORT_PREBUILT_ROOT", str(relocated_sdk))
    monkeypatch.setenv("ORT_TEST_ASSETS_ROOT", str(relocated_assets))
    with pytest.raises(ValueError, match="does not match the selected ORT"):
        extract(archive, "android-aarch64", destination)
    assert (models / "selected.onnx").read_text() == "selected"

    # An archive from the old selection is rejected before altering device testdata.
    replacement = tmp_path / "replacement"
    _make_tree(replacement, {"VERSION_NUMBER": "1.25.0", "onnxruntime/test/testdata/B.onnx": "B"})
    monkeypatch.setenv("ORT_PREBUILT_ROOT", str(sdk))
    monkeypatch.setenv("ORT_TEST_ASSETS_ROOT", str(replacement))
    with pytest.raises(ValueError, match="does not match the selected ORT"):
        extract(archive, "android-aarch64", destination)
    assert (models / "selected.onnx").read_text() == "selected"
