# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Hermetic publisher contract tests for snapshot-golden artifacts."""

import json
import os
import stat
import subprocess
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "publish_goldens.sh"
QAIRT_VERSION = "2.50.40"
ORT_VERSION = "1.29.0"
REPO = "test-repo"
SUBPATH = "ci/snapshot-goldens-pr923-test"


def write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def create_publisher_inputs(tmp_path: Path) -> tuple[Path, Path]:
    qairt_root = tmp_path / "qairt"
    qairt_root.mkdir()
    (qairt_root / "sdk.yaml").write_text(f"version: {QAIRT_VERSION}\n")

    ort_root = tmp_path / "ort"
    ort_root.mkdir()
    (ort_root / "VERSION_NUMBER").write_text(f"{ORT_VERSION}\n")

    build_dir = tmp_path / "build"
    bin_dir = build_dir / "RelWithDebInfo"
    bin_dir.mkdir(parents=True)
    provider_test = bin_dir / "onnxruntime_provider_test"
    provider_test.touch()
    provider_test.chmod(provider_test.stat().st_mode | stat.S_IXUSR)
    (bin_dir / "CMakeCache.txt").write_text(
        f"onnxruntime_QNN_HOME:PATH={qairt_root}\nonnxruntime_ORT_HOME:PATH={ort_root}\n"
    )

    results_dir = bin_dir / "snapshot_accuracy_results"
    results_dir.mkdir()
    (results_dir / "accuracy_results.json").write_text(
        json.dumps(
            {
                "testsuites": [
                    {
                        "name": "QnnAcc_Clip_AccuracyTest",
                        "testsuite": [{"result": "COMPLETED", "status": "RUN", "failures": []}],
                    }
                ]
            }
        )
    )

    golden_dir = tmp_path / "goldens"
    golden_file = golden_dir / "snapshot/builder/opbuilder/clip/Case.json"
    golden_file.parent.mkdir(parents=True)
    golden_file.write_text("{}")
    return build_dir, golden_dir


def test_publish_sidecars_match_zip_manifest_and_upload_in_order(tmp_path: Path) -> None:
    build_dir, golden_dir = create_publisher_inputs(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    upload_log = tmp_path / "uploads.txt"
    fake_store = tmp_path / "store"
    write_executable(
        fake_bin / "jf",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        '[ "$1" = "rt" ] && [ "$2" = "upload" ] && [ "$3" = "--flat" ]\n'
        'printf "%s\\n" "$5" >> "${FAKE_JF_UPLOAD_LOG}"\n'
        'destination="${FAKE_JF_STORE}/$5"\n'
        'mkdir -p "$(dirname "${destination}")"\n'
        'cp "$4" "${destination}"\n',
    )

    env = os.environ.copy()
    env.update(
        {
            "BUILD_ARTIFACTORY_REPO": REPO,
            "JF_URL": "https://example.invalid/artifactory",
            "JF_ACCESS_TOKEN": "test-token",
            "FAKE_JF_UPLOAD_LOG": str(upload_log),
            "FAKE_JF_STORE": str(fake_store),
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "GITHUB_RUN_ID": "12345",
            "GITHUB_RUN_ATTEMPT": "1",
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            f"--build-dir={build_dir}",
            f"--golden-dir={golden_dir}",
            f"--repo-subpath={SUBPATH}",
            "--skip-regen",
            "--publish",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    uploads = upload_log.read_text().splitlines()
    assert len(uploads) == 4
    archive_dir = uploads[0].removesuffix("/manifest.json")
    assert uploads == [
        f"{archive_dir}/manifest.json",
        f"{archive_dir}/goldens.zip",
        f"{REPO}/{SUBPATH}/latest/manifest.json",
        f"{REPO}/{SUBPATH}/latest/goldens.zip",
    ]

    archive_manifest = fake_store / f"{archive_dir}/manifest.json"
    archive_zip = fake_store / f"{archive_dir}/goldens.zip"
    latest_manifest = fake_store / f"{REPO}/{SUBPATH}/latest/manifest.json"
    latest_zip = fake_store / f"{REPO}/{SUBPATH}/latest/goldens.zip"
    assert archive_manifest.read_bytes() == latest_manifest.read_bytes()
    assert archive_zip.read_bytes() == latest_zip.read_bytes()

    with zipfile.ZipFile(latest_zip) as archive:
        assert archive.read("manifest.json") == latest_manifest.read_bytes()
