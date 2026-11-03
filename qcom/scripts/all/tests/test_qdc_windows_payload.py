# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import hashlib
import io
import json
import tarfile
import zipfile

import pytest
import qdc_windows_payload
from qdc_windows_payload import prepare_windows_payload


@pytest.fixture
def payload_inputs(tmp_path):
    installer = tmp_path / "python.exe"
    installer.write_bytes(b"fake installer")
    info = {"python_version": "3.12.10", "python_sha256": hashlib.sha256(installer.read_bytes()).hexdigest()}
    data = tmp_path / "testdata.zip"
    with zipfile.ZipFile(data, "w") as archive:
        for handle in ("testdata", "node", "pytorch-converted", "pytorch-operator"):
            archive.writestr(f"{handle}/sentinel.txt", handle)
    models = tmp_path / "models"
    for suite in ("float32", "qdq", "qdq-with-context-cache"):
        folder = models / "testdata" / suite
        folder.mkdir(parents=True)
        (folder / "model.onnx").write_bytes(b"model")
    return {"testdata_archive": data, "models_dir": models, "installer": installer, "python_info": info}


def write_archive(path, config="Release", multi_config=False, omit=""):
    root = f"build/windows-arm64/{config}"
    binaries = f"{root}/{config}" if multi_config else root
    contents = {
        f"{root}/run_tests.ps1": b"runner",
        f"{root}/ctest.exe": b"ctest",
        f"{root}/CTestTestfile.cmake": b"ctest plan",
        "qcom/scripts/windows/utils.ps1": b"original utils",
    }
    for name in (
        "onnxruntime_provider_test.exe",
        "onnxruntime_plugin_ep_onnx_test.exe",
        "onnxruntime.dll",
        "onnxruntime_providers_qnn.dll",
        "QnnHtp.dll",
        "QnnSystem.dll",
        "QnnHtpV81Stub.dll",
        "libQnnHtpV81Skel.so",
        "libqnnhtpv81.cat",
        "msvcp140.dll",
        "msvcp140_1.dll",
        "vcruntime140.dll",
    ):
        if name != omit:
            contents[f"{binaries}/{name}"] = name.encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in contents.items():
            archive.writestr(name, content)
    return contents


@pytest.mark.parametrize("config", ("Release", "Debug", "RelWithDebInfo"))
@pytest.mark.parametrize("multi_config", (False, True))
@pytest.mark.parametrize("testdata_format", ("zip", "tar.bz2"))
def test_preserves_archive_and_places_data_next_to_binaries(
    tmp_path, payload_inputs, config, multi_config, testdata_format
):
    if testdata_format == "tar.bz2":
        data = tmp_path / "testdata.tar.bz2"
        with zipfile.ZipFile(payload_inputs["testdata_archive"]) as source, tarfile.open(data, "w:bz2") as target:
            for name in source.namelist():
                content = source.read(name)
                info = tarfile.TarInfo(name)
                info.size = len(content)
                # Leave a full day after the ZIP epoch so local-time conversion cannot
                # move this timestamp back into 1979 on runners west of UTC.
                info.mtime = 315619200  # 1980-01-02 00:00:00 UTC
                target.addfile(info, io.BytesIO(content))
        payload_inputs["testdata_archive"] = data
    source = tmp_path / "tests.zip"
    originals = write_archive(source, config, multi_config)
    original_zip = source.read_bytes()
    result = prepare_windows_payload(source, tmp_path / "output", config=config, **payload_inputs)
    assert source.read_bytes() == original_zip
    root = f"build/windows-arm64/{config}"
    binaries = f"{root}/{config}" if multi_config else root
    with zipfile.ZipFile(result) as payload:
        assert len(payload.namelist()) == len(set(payload.namelist()))
        for name, content in originals.items():
            assert payload.read(name) == content
        assert payload.read(f"{binaries}/testdata/sentinel.txt") == b"testdata"
        assert payload.read("cmake/external/onnx/onnx/backend/test/data/node/sentinel.txt") == b"node"
        assert payload.read(f"{root}/model_tests/onnx_models/testdata/qdq/model.onnx") == b"model"
        assert "model_tests/onnx_models/testdata/qdq/model.onnx" not in payload.namelist()
        assert payload.read(f"{root}/utils.ps1")
        assert payload.read("installers/python-arm64.exe") == payload_inputs["installer"].read_bytes()
        settings = json.loads(payload.read("qdc_windows_config.json"))
        assert settings["build_config"] == config
        assert settings["expected_htp_arch"] == 81
        assert settings["test_mode"] == "full"


def test_missing_v81_runtime_fails_before_downloading(tmp_path, payload_inputs, monkeypatch):
    monkeypatch.setattr(qdc_windows_payload, "fetch_python_installer", lambda: pytest.fail("Unexpected download"))
    source = tmp_path / "tests.zip"
    write_archive(source, omit="libqnnhtpv81.cat")
    with pytest.raises(ValueError, match="libqnnhtpv81.cat"):
        prepare_windows_payload(source, tmp_path / "output", payload_inputs["testdata_archive"])


def test_smoke_does_not_download_or_bundle_model_zoo(tmp_path, payload_inputs, monkeypatch):
    monkeypatch.setattr(qdc_windows_payload, "PackageManager", lambda _: pytest.fail("Unexpected model download"))
    source = tmp_path / "tests.zip"
    write_archive(source)
    payload_inputs["models_dir"] = None
    result = prepare_windows_payload(source, tmp_path / "output", test_mode="smoke", **payload_inputs)
    with zipfile.ZipFile(result) as payload:
        assert not any("model_tests/onnx_models" in n for n in payload.namelist())
        assert json.loads(payload.read("qdc_windows_config.json"))["test_mode"] == "smoke"


def test_corrupt_installer_is_rejected(tmp_path, payload_inputs):
    source = tmp_path / "tests.zip"
    write_archive(source)
    payload_inputs["installer"].write_bytes(b"corrupted download")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        prepare_windows_payload(source, tmp_path / "output", **payload_inputs)


def test_missing_model_suite_is_rejected(tmp_path, payload_inputs):
    source = tmp_path / "tests.zip"
    write_archive(source)
    suite = payload_inputs["models_dir"] / "testdata/qdq-with-context-cache"
    (suite / "model.onnx").unlink()
    suite.rmdir()
    with pytest.raises(ValueError, match="qdq-with-context-cache"):
        prepare_windows_payload(source, tmp_path / "output", **payload_inputs)


def test_missing_testdata_handle_is_rejected(tmp_path, payload_inputs):
    source = tmp_path / "tests.zip"
    write_archive(source)
    with zipfile.ZipFile(payload_inputs["testdata_archive"], "w") as archive:
        for handle in ("testdata", "pytorch-converted", "pytorch-operator"):
            archive.writestr(f"{handle}/sentinel.txt", handle)
    with pytest.raises(ValueError, match="missing node"):
        prepare_windows_payload(source, tmp_path / "output", **payload_inputs)
