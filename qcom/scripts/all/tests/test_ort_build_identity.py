# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import importlib
import os
import zipfile
from pathlib import Path

import pytest
from extract_testdata import extract
from ort_build_identity import (
    TESTDATA_IDENTITY_FILE,
    build_identity,
    validate_test_inputs,
    validate_testdata_archive,
)
from ort_build_identity import testdata_identity as make_testdata_identity
from reconcile_prebuilt_tests import reconcile


@pytest.fixture
def selected_inputs(tmp_path):
    sdk = tmp_path / "sdk"
    assets = tmp_path / "assets"
    sdk.mkdir()
    (sdk / "VERSION_NUMBER").write_text("1.25.0\n", encoding="utf-8")
    testdata = assets / "onnxruntime/test/testdata"
    testdata.mkdir(parents=True)
    (assets / "VERSION_NUMBER").write_text("1.25.0\n", encoding="utf-8")
    (testdata / "model.onnx").write_bytes(b"selected model")
    return sdk, assets


@pytest.mark.parametrize("change", ("contents", "name", "added", "deleted"))
def test_archive_checks_actual_payload_before_replacing_models(tmp_path, monkeypatch, selected_inputs, change):
    sdk, assets = selected_inputs
    monkeypatch.setenv("ORT_PREBUILT_ROOT", str(sdk))
    monkeypatch.setenv("ORT_TEST_ASSETS_ROOT", str(assets))
    files = {"testdata/model.onnx": b"selected model"}
    if change == "contents":
        files["testdata/model.onnx"] = b"other model"
    elif change == "name":
        files["testdata/renamed.onnx"] = files.pop("testdata/model.onnx")
    elif change == "added":
        files["testdata/other.onnx"] = b"other"
    else:
        files.clear()
    archive = tmp_path / "testdata.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(TESTDATA_IDENTITY_FILE, make_testdata_identity(str(sdk), str(assets)))
        output.writestr("testdata/", "")
        for name, content in files.items():
            output.writestr(name, content)
    models = tmp_path / "device/build/linux-x86_64/Release/testdata"
    models.mkdir(parents=True)
    (models / "keep.onnx").write_bytes(b"keep")
    with pytest.raises(ValueError, match="rerun archive_testdata"):
        extract(archive, "linux-x86_64", tmp_path / "device")
    assert list(models.iterdir()) == [models / "keep.onnx"]
    assert (models / "keep.onnx").read_bytes() == b"keep"


def test_digest_ignores_timestamp_and_detects_in_place_updates(selected_inputs):
    sdk, assets = selected_inputs
    before = make_testdata_identity(str(sdk), str(assets))
    model = assets / "onnxruntime/test/testdata/model.onnx"
    os.utime(model, (1, 1))
    assert make_testdata_identity(str(sdk), str(assets)) == before
    model.write_bytes(b"changed model")
    assert make_testdata_identity(str(sdk), str(assets)) != before


@pytest.mark.parametrize("invalid", ("sdk_version_missing", "assets_version_missing", "empty", "mismatch", "testdata"))
def test_shared_validation_for_preflight_archive_and_reconciliation(tmp_path, monkeypatch, selected_inputs, invalid):
    sdk, assets = selected_inputs
    if invalid == "sdk_version_missing":
        (sdk / "VERSION_NUMBER").unlink()
    elif invalid == "assets_version_missing":
        (assets / "VERSION_NUMBER").unlink()
    elif invalid == "empty":
        (assets / "VERSION_NUMBER").write_text(" \n", encoding="utf-8")
    elif invalid == "mismatch":
        (assets / "VERSION_NUMBER").write_text("1.24.0", encoding="utf-8")
    else:
        (assets / "onnxruntime/test/testdata").rename(assets / "other-testdata")

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    entrypoint = importlib.import_module("build_and_test")
    failures = []
    for validate in (
        lambda: validate_test_inputs(str(sdk), str(assets)),
        lambda: entrypoint.get_ort_test_assets_root(assets, sdk, [], False),
        lambda: make_testdata_identity(str(sdk), str(assets)),
        lambda: reconcile(tmp_path / "build", "Release", str(sdk), str(assets)),
    ):
        with pytest.raises((ValueError, FileNotFoundError)) as error:
            validate()
        failures.append((type(error.value), str(error.value)))
    assert len(set(failures)) == 1


def test_reconciliation_preserves_local_sdk_identity_and_instrumentation(tmp_path, monkeypatch, selected_inputs):
    sdk, assets = selected_inputs
    build = tmp_path / "build"
    config = build / "RelWithDebInfo"
    config.mkdir(parents=True)
    marker = build / "ort-build-identity-RelWithDebInfo.txt"
    marker.write_text(build_identity(str(sdk), str(assets)), encoding="utf-8")
    (config / "CMakeCache.txt").write_text(
        f"CMAKE_CACHEFILE_DIR:INTERNAL={config.as_posix()}\n"
        "CMAKE_COMMAND:INTERNAL=cmake\n"
        "CMAKE_HOME_DIRECTORY:INTERNAL=original-source\n"
        "ENABLE_COVERAGE:BOOL=ON\n",
        encoding="utf-8",
    )
    commands = []
    monkeypatch.setattr("reconcile_prebuilt_tests.subprocess.run", lambda command, **kwargs: commands.append(command))
    reconcile(build, "RelWithDebInfo", str(sdk), str(assets))
    assert len(commands) == 2
    assert commands[1][-2:] == ["--target", "ort_test_assets_target"]
    assert not any("ENABLE_COVERAGE" in arg for arg in commands[0])
    assert (config / "CMakeCache.txt").read_text(encoding="utf-8").endswith("ENABLE_COVERAGE:BOOL=ON\n")

    relocated_sdk = tmp_path / "relocated-sdk"
    relocated_sdk.mkdir()
    (relocated_sdk / "VERSION_NUMBER").write_text("1.25.0", encoding="utf-8")
    with pytest.raises(ValueError, match="required coverage/ASan instrumentation"):
        reconcile(build, "RelWithDebInfo", str(relocated_sdk), str(assets))
    assert len(commands) == 2


def test_archive_only_rejects_obsolete_path_identity(tmp_path, monkeypatch, selected_inputs):
    sdk, assets = selected_inputs
    monkeypatch.delenv("ORT_PREBUILT_ROOT", raising=False)
    monkeypatch.delenv("ORT_TEST_ASSETS_ROOT", raising=False)
    (tmp_path / TESTDATA_IDENTITY_FILE).write_text(build_identity(str(sdk), str(assets)), encoding="utf-8")
    with pytest.raises(ValueError, match="rerun archive_testdata"):
        validate_testdata_archive(tmp_path, "1.25.0")
