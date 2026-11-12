# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import hashlib
import io
import json
import tarfile
import zipfile

import pytest
import qdc_windows_payload
from qdc_windows_payload import DEFAULT_WINDOWS_HTP_ARCH, prepare_windows_payload, validate_windows_results


@pytest.fixture
def payload_inputs(tmp_path, htp_arch):
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
    return {
        "testdata_archive": data,
        "models_dir": models,
        "installer": installer,
        "python_info": info,
        "htp_arch": htp_arch,
    }


@pytest.fixture(params=(73, 81))
def htp_arch(request):
    return request.param


def write_archive(path, config="Release", multi_config=False, omit="", *, htp_arch):
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
        f"QnnHtpV{htp_arch}Stub.dll",
        f"libQnnHtpV{htp_arch}Skel.so",
        f"libqnnhtpv{htp_arch}.cat",
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
    originals = write_archive(source, config, multi_config, htp_arch=payload_inputs["htp_arch"])
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
        assert settings["expected_htp_arch"] == payload_inputs["htp_arch"]
        assert settings["test_mode"] == "full"


@pytest.mark.parametrize("runtime", ("QnnHtpV{arch}Stub.dll", "libQnnHtpV{arch}Skel.so", "libqnnhtpv{arch}.cat"))
def test_missing_htp_runtime_fails_before_downloading(tmp_path, payload_inputs, monkeypatch, runtime):
    monkeypatch.setattr(qdc_windows_payload, "fetch_python_installer", lambda: pytest.fail("Unexpected download"))
    source = tmp_path / "tests.zip"
    htp_arch = payload_inputs["htp_arch"]
    missing_runtime = runtime.format(arch=htp_arch)
    write_archive(source, omit=missing_runtime, htp_arch=htp_arch)
    with pytest.raises(ValueError, match=missing_runtime):
        prepare_windows_payload(source, tmp_path / "output", payload_inputs["testdata_archive"], htp_arch=htp_arch)


def test_wrong_htp_runtime_is_rejected(tmp_path, payload_inputs):
    source = tmp_path / "tests.zip"
    other_arch = 73 if payload_inputs["htp_arch"] == 81 else 81
    write_archive(source, htp_arch=other_arch)
    with pytest.raises(ValueError, match="Incomplete Windows QDC test archive"):
        prepare_windows_payload(source, tmp_path / "output", **payload_inputs)


def test_default_htp_arch_is_preserved(tmp_path, payload_inputs):
    source = tmp_path / "tests.zip"
    write_archive(source, htp_arch=DEFAULT_WINDOWS_HTP_ARCH)
    inputs = {key: value for key, value in payload_inputs.items() if key != "htp_arch"}
    result = prepare_windows_payload(source, tmp_path / "output", **inputs)
    with zipfile.ZipFile(result) as payload:
        assert json.loads(payload.read("qdc_windows_config.json"))["expected_htp_arch"] == DEFAULT_WINDOWS_HTP_ARCH


def test_smoke_does_not_download_or_bundle_model_zoo(tmp_path, payload_inputs, monkeypatch):
    monkeypatch.setattr(qdc_windows_payload, "PackageManager", lambda _: pytest.fail("Unexpected model download"))
    source = tmp_path / "tests.zip"
    write_archive(source, htp_arch=payload_inputs["htp_arch"])
    payload_inputs["models_dir"] = None
    result = prepare_windows_payload(source, tmp_path / "output", test_mode="smoke", **payload_inputs)
    with zipfile.ZipFile(result) as payload:
        assert not any("model_tests/onnx_models" in n for n in payload.namelist())
        assert json.loads(payload.read("qdc_windows_config.json"))["test_mode"] == "smoke"


def test_corrupt_installer_is_rejected(tmp_path, payload_inputs):
    source = tmp_path / "tests.zip"
    write_archive(source, htp_arch=payload_inputs["htp_arch"])
    payload_inputs["installer"].write_bytes(b"corrupted download")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        prepare_windows_payload(source, tmp_path / "output", **payload_inputs)


def test_missing_model_suite_is_rejected(tmp_path, payload_inputs):
    source = tmp_path / "tests.zip"
    write_archive(source, htp_arch=payload_inputs["htp_arch"])
    suite = payload_inputs["models_dir"] / "testdata/qdq-with-context-cache"
    (suite / "model.onnx").unlink()
    suite.rmdir()
    with pytest.raises(ValueError, match="qdq-with-context-cache"):
        prepare_windows_payload(source, tmp_path / "output", **payload_inputs)


def test_missing_testdata_handle_is_rejected(tmp_path, payload_inputs):
    source = tmp_path / "tests.zip"
    write_archive(source, htp_arch=payload_inputs["htp_arch"])
    with zipfile.ZipFile(payload_inputs["testdata_archive"], "w") as archive:
        for handle in ("testdata", "pytorch-converted", "pytorch-operator"):
            archive.writestr(f"{handle}/sentinel.txt", handle)
    with pytest.raises(ValueError, match="missing node"):
        prepare_windows_payload(source, tmp_path / "output", **payload_inputs)


@pytest.fixture
def windows_results(tmp_path):
    logs = tmp_path / "nested" / "QDC_logs"
    logs.mkdir(parents=True)
    (logs / "bootstrap.results.xml").write_text(
        '<testsuites><testsuite tests="1" failures="0"><testcase classname="QDCWindows" '
        'name="BootstrapAndTests" /></testsuite></testsuites>'
    )
    (logs / "required-reports.results.txt").write_text(
        json.dumps({"test_mode": "smoke", "build_config": "Release", "ctest_reports": []})
    )
    (logs / "htp-smoke.results.xml").write_text(
        '<testsuite><testcase classname="QnnHTPBackendTests" name="BinaryOp_Add4D" '
        'status="run" result="completed" /></testsuite>'
    )
    return tmp_path


def test_valid_windows_results(windows_results):
    validate_windows_results(windows_results)


@pytest.mark.parametrize(
    "report",
    (
        '<testsuites failures="1" />',
        '<testsuite errors="1" />',
        '<testsuite><testcase><failure message="inference failed" /></testcase></testsuite>',
        '<testsuite><testcase><error message="setup failed" /></testcase></testsuite>',
        '<testsuite failures="unknown" />',
        "<testsuite",
        "<unexpected />",
    ),
)
def test_failed_or_invalid_windows_results(windows_results, report):
    (windows_results / "provider.results.xml").write_text(report)
    with pytest.raises(ValueError, match="Invalid or failed"):
        validate_windows_results(windows_results)


@pytest.mark.parametrize("name", ("bootstrap.results.xml", "htp-smoke.results.xml"))
def test_missing_required_smoke_results(windows_results, name):
    (windows_results / "nested/QDC_logs" / name).unlink()
    with pytest.raises(ValueError, match="reports are missing"):
        validate_windows_results(windows_results)


def test_skipped_bootstrap_results(windows_results):
    report = windows_results / "nested/QDC_logs/bootstrap.results.xml"
    report.write_text(
        '<testsuite><testcase classname="QDCWindows" name="BootstrapAndTests"><skipped /></testcase></testsuite>'
    )
    with pytest.raises(ValueError, match="Bootstrap did not report a completed test"):
        validate_windows_results(windows_results)


@pytest.fixture
def full_windows_results(windows_results):
    logs = windows_results / "nested/QDC_logs"
    ctest_reports = ["onnxruntime_provider_test.exe.Release.results.xml", "other_unit_test.exe.Release.results.xml"]
    (logs / "required-reports.results.txt").write_text(
        json.dumps({"test_mode": "full", "build_config": "Release", "ctest_reports": ctest_reports})
    )
    names = ctest_reports + [
        f"{suite}_model_tests.results.xml" for suite in ("node", "float32", "qdq", "qdq-with-context-cache")
    ]
    for name in names:
        (logs / name).write_text(
            '<testsuite><testcase classname="QnnHTPBackendTests" name="Test" '
            'status="run" result="completed" /></testsuite>'
        )
    return windows_results


def test_valid_full_windows_results(full_windows_results):
    validate_windows_results(full_windows_results)


@pytest.mark.parametrize(
    "name",
    (
        "onnxruntime_provider_test.exe.Release.results.xml",
        "other_unit_test.exe.Release.results.xml",
        "node_model_tests.results.xml",
        "float32_model_tests.results.xml",
        "qdq_model_tests.results.xml",
        "qdq-with-context-cache_model_tests.results.xml",
    ),
)
def test_missing_required_full_results(full_windows_results, name):
    (full_windows_results / "nested/QDC_logs" / name).unlink()
    with pytest.raises(ValueError, match="reports are missing"):
        validate_windows_results(full_windows_results)


@pytest.mark.parametrize(
    "cases",
    (
        "",
        '<testcase status="notrun" result="suppressed" />',
        '<testcase classname="QnnHTPBackendTests"><skipped /></testcase>',
        '<testcase classname="QnnCPUBackendTests" status="run" result="completed" />',
    ),
)
def test_no_completed_htp_tests(full_windows_results, cases):
    report = full_windows_results / "nested/QDC_logs/onnxruntime_provider_test.exe.Release.results.xml"
    report.write_text(f"<testsuite>{cases}</testsuite>")
    with pytest.raises(ValueError, match="No completed, non-skipped"):
        validate_windows_results(full_windows_results)


def test_empty_ctest_manifest_is_rejected(full_windows_results):
    (full_windows_results / "nested/QDC_logs/required-reports.results.txt").write_text(
        json.dumps({"test_mode": "full", "build_config": "Release", "ctest_reports": []})
    )
    with pytest.raises(ValueError, match="CTest report list is missing or empty"):
        validate_windows_results(full_windows_results)


def test_missing_report_manifest_is_rejected(windows_results):
    (windows_results / "nested/QDC_logs/required-reports.results.txt").unlink()
    with pytest.raises(ValueError, match="exactly one"):
        validate_windows_results(windows_results)


def test_duplicate_reports_are_rejected(windows_results):
    (windows_results / "bootstrap.results.xml").write_text("<testsuite />")
    with pytest.raises(ValueError, match="Duplicate"):
        validate_windows_results(windows_results)
