# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import subprocess
import tempfile
from pathlib import Path

import pytest
from device.adb_device import AdbDevice
from qdc_helpers import TestBase

CONFIG = TestBase.config()

_APK_BASE = Path(CONFIG.host_build_root) / "java" / "androidtest" / "android" / "app" / "build" / "outputs" / "apk"
_APP_APK = _APK_BASE / "debug" / "app-debug.apk"
_ANDROIDTEST_APK = _APK_BASE / "androidTest" / "debug" / "app-debug-androidTest.apk"

_INSTRUMENTATION_TARGET = "ai.onnxruntime.example.javavalidator.test/androidx.test.runner.AndroidJUnitRunner"
_RESULTS_XML_NAME = "test_aar.results.xml"


class TestAar(TestBase):
    def test_aar_instrumentation(self) -> None:
        # Non-AAR builds legitimately produce no APKs, so missing-both is a valid
        # skip. But "only one present" is always a build bug (e.g. gradle ran
        # halfway); hard-fail so CI surfaces it instead of silently skipping.
        app_exists = _APP_APK.exists()
        androidtest_exists = _ANDROIDTEST_APK.exists()
        assert app_exists == androidtest_exists, (
            f"Partial AAR test build: {_APP_APK} exists={app_exists}, "
            f"{_ANDROIDTEST_APK} exists={androidtest_exists}. Both should be present "
            "together (when AAR is built) or both absent (when only the non-AAR archive is built)."
        )
        if not app_exists:
            pytest.skip("AAR APKs not in test archive (expected for non-AAR Android builds).")

        # APK installation requires an ADB connection; SshDevice does not support it.
        # AdbDevice is the only DeviceBase subclass with install(), and AAR tests by
        # definition run on Android hardware accessed via adb.
        device = self.device
        assert isinstance(device, AdbDevice), (
            f"AAR instrumentation test requires an adb-connected device, got {type(device).__name__}."
        )
        device.install(_APP_APK)
        device.install(_ANDROIDTEST_APK)

        # `am instrument -w` exits 0 even on test failure, so parse output. Pass -r
        # to get machine-readable key/value blocks terminated by INSTRUMENTATION_CODE,
        # which is more robust than grepping for the human-readable "OK (N tests)" or
        # "FAILURES!!!" strings that can change across AndroidX test-runner versions.
        output = device.shell(
            [f"am instrument -w -r {_INSTRUMENTATION_TARGET}"],
            capture_output=True,
        )
        text = "\n".join(output or [])
        print(text)

        # Convert the per-method status blocks into JUnit XML and push it to the
        # device's QDC log directory so it is uploaded as a job artifact.
        log_to_xml = Path(self.config().host_qcom_scripts_path) / "all" / "instrumentation_to_junit_xml.py"
        with tempfile.TemporaryDirectory(prefix="InstrumentationXml-") as tmpdir:
            tmppath = Path(tmpdir)
            raw_log = tmppath / "test_aar.instrumentation.txt"
            raw_log.write_text(text, encoding="utf-8")
            results_xml = tmppath / _RESULTS_XML_NAME
            with open(results_xml, "w") as xml_out:
                subprocess.run([str(log_to_xml), str(raw_log)], stdout=xml_out, check=True)
            self.device.push(results_xml, Path(self.config().qdc_log_path) / _RESULTS_XML_NAME)

        # INSTRUMENTATION_CODE values: -1 = success, 0 = failure, other = runner error.
        assert "INSTRUMENTATION_CODE: -1" in text, (
            "AndroidJUnit instrumentation did not report success (expected 'INSTRUMENTATION_CODE: -1' in output)."
        )
