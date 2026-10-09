# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Hermetic contract tests for snapshot-golden preflight."""

import json
import os
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "prepare_snapshot_goldens.sh"
QAIRT_VERSION = "2.50.40"
ORT_VERSION = "1.29.0"


def write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.fixture
def build_dir(tmp_path: Path) -> Path:
    qairt_root = tmp_path / "qairt"
    qairt_root.mkdir()
    (qairt_root / "sdk.yaml").write_text(f"version: {QAIRT_VERSION}\n")

    ort_root = tmp_path / "ort"
    ort_root.mkdir()
    (ort_root / "VERSION_NUMBER").write_text(f"{ORT_VERSION}\n")

    build = tmp_path / "build"
    build.mkdir()
    (build / "CMakeCache.txt").write_text(
        f"onnxruntime_QNN_HOME:PATH={qairt_root}\nonnxruntime_ORT_HOME:PATH={ort_root}\n"
    )
    return build


def create_zip(path: Path, manifest: object | None) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        if manifest is not None:
            if isinstance(manifest, str):
                archive.writestr("manifest.json", manifest)
            else:
                archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("snapshot/builder/opbuilder/clip/Case.json", "{}")


def run_preflight(
    tmp_path: Path,
    build_dir: Path,
    source_zip: Path | None,
    *,
    download_fails: bool = False,
    fake_unzip: str | None = None,
    initial_github_env: str = "",
    golden_dir: Path | None = None,
    working_dir: Path | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    write_executable(
        fake_bin / "jf",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [ "${FAKE_JF_FAIL:-}" = "1" ]; then exit 1; fi\n'
        'cp "${FAKE_GOLDENS_ZIP}" "${@: -1}/goldens.zip"\n',
    )
    if fake_unzip is not None:
        write_executable(fake_bin / "unzip", fake_unzip)

    if golden_dir is None:
        golden_dir = tmp_path / "goldens"
    github_env = tmp_path / "github_env"
    github_env.write_text(initial_github_env)
    env = os.environ.copy()
    env.update(
        {
            "BUILD_ARTIFACTORY_REPO": "test-repo",
            "GITHUB_ENV": str(github_env),
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
        }
    )
    env.pop("QNN_UT_SNAPSHOT_GOLDEN_DIR", None)
    if source_zip is not None:
        env["FAKE_GOLDENS_ZIP"] = str(source_zip)
    if download_fails:
        env["FAKE_JF_FAIL"] = "1"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            f"--build-dir={build_dir}",
            f"--golden-dir={golden_dir}",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=working_dir,
    )
    return result, github_env, golden_dir


@pytest.mark.parametrize(
    "mode",
    ["download", "corrupt", "missing_manifest", "malformed_manifest", "qairt_mismatch", "ort_mismatch"],
)
def test_invalid_preflight_never_enables_golden_store(tmp_path: Path, build_dir: Path, mode: str) -> None:
    source_zip = tmp_path / "goldens.zip"
    if mode == "corrupt":
        source_zip.write_bytes(b"not a zip")
    elif mode == "missing_manifest":
        create_zip(source_zip, None)
    elif mode == "malformed_manifest":
        create_zip(source_zip, "not json")
    else:
        manifest = {"qairt_version": QAIRT_VERSION, "ort_version": ORT_VERSION}
        if mode == "qairt_mismatch":
            manifest["qairt_version"] = "different"
        if mode == "ort_mismatch":
            manifest["ort_version"] = "different"
        create_zip(source_zip, manifest)

    result, github_env, golden_dir = run_preflight(
        tmp_path,
        build_dir,
        None if mode == "download" else source_zip,
        download_fails=mode == "download",
    )

    assert result.returncode == 0, result.stderr
    assert github_env.read_text().endswith("QNN_UT_SNAPSHOT_GOLDEN_DIR=\n")
    assert not golden_dir.exists()


def test_extract_failure_never_enables_golden_store(tmp_path: Path, build_dir: Path) -> None:
    source_zip = tmp_path / "goldens.zip"
    create_zip(source_zip, {"qairt_version": QAIRT_VERSION, "ort_version": ORT_VERSION})
    real_unzip = shutil.which("unzip")
    assert real_unzip is not None
    fake_unzip = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'if [ "$1" = "-tqq" ]; then exec "{real_unzip}" "$@"; fi\n'
        'if [ "$1" = "-q" ]; then mkdir -p "$4"; touch "$4/partial"; exit 1; fi\n'
        "exit 2\n"
    )

    result, github_env, golden_dir = run_preflight(tmp_path, build_dir, source_zip, fake_unzip=fake_unzip)

    assert result.returncode == 0, result.stderr
    assert github_env.read_text().endswith("QNN_UT_SNAPSHOT_GOLDEN_DIR=\n")
    assert not golden_dir.exists()


def test_failed_preflight_clears_previously_exported_golden_store(tmp_path: Path, build_dir: Path) -> None:
    source_zip = tmp_path / "goldens.zip"
    create_zip(source_zip, {"qairt_version": "different", "ort_version": ORT_VERSION})
    previous = "QNN_UT_SNAPSHOT_GOLDEN_DIR=/previous/goldens\n"

    result, github_env, golden_dir = run_preflight(
        tmp_path,
        build_dir,
        source_zip,
        initial_github_env=previous,
    )

    assert result.returncode == 0, result.stderr
    assert not golden_dir.exists()
    assert github_env.read_text() == previous + "QNN_UT_SNAPSHOT_GOLDEN_DIR=\n"


def test_aligned_archive_enables_golden_store(tmp_path: Path, build_dir: Path) -> None:
    source_zip = tmp_path / "goldens.zip"
    create_zip(source_zip, {"qairt_version": QAIRT_VERSION, "ort_version": ORT_VERSION})

    result, github_env, golden_dir = run_preflight(tmp_path, build_dir, source_zip)

    assert result.returncode == 0, result.stderr
    assert github_env.read_text() == f"QNN_UT_SNAPSHOT_GOLDEN_DIR=\nQNN_UT_SNAPSHOT_GOLDEN_DIR={golden_dir}\n"
    assert (golden_dir / "manifest.json").is_file()
    assert (golden_dir / "snapshot/builder/opbuilder/clip/Case.json").is_file()


def test_relative_golden_dir_exports_absolute_store_for_provider_working_dir(tmp_path: Path, build_dir: Path) -> None:
    source_zip = tmp_path / "goldens.zip"
    create_zip(source_zip, {"qairt_version": QAIRT_VERSION, "ort_version": ORT_VERSION})
    relative_golden_dir = Path("build/linux-x86_64/snapshot-goldens")
    provider_working_dir = tmp_path / "build/linux-x86_64/RelWithDebInfo"
    provider_working_dir.mkdir(parents=True)

    result, github_env, _ = run_preflight(
        tmp_path,
        build_dir,
        source_zip,
        golden_dir=relative_golden_dir,
        working_dir=tmp_path,
    )

    expected_golden_dir = (tmp_path / relative_golden_dir).resolve()
    assert result.returncode == 0, result.stderr
    assert github_env.read_text() == (
        f"QNN_UT_SNAPSHOT_GOLDEN_DIR=\nQNN_UT_SNAPSHOT_GOLDEN_DIR={expected_golden_dir}\n"
    )
    exported_golden_dir = Path(github_env.read_text().splitlines()[-1].split("=", 1)[1])
    assert exported_golden_dir.is_absolute()
    assert (provider_working_dir / exported_golden_dir / "manifest.json").is_file()
