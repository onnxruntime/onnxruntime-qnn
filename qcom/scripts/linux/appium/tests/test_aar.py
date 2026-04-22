# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
from qdc_helpers import TestBase

CONFIG = TestBase.config()

_APK_BASE = Path(CONFIG.host_build_root) / "java" / "androidtest" / "android" / "app" / "build" / "outputs" / "apk"
_APP_APK = _APK_BASE / "debug" / "app-debug.apk"
_ANDROIDTEST_APK = _APK_BASE / "androidTest" / "debug" / "app-debug-androidTest.apk"

_INSTRUMENTATION_TARGET = "ai.onnxruntime.example.javavalidator.test/androidx.test.runner.AndroidJUnitRunner"


class TestAar(TestBase):
    def test_aar_instrumentation(self) -> None:
        if not _APP_APK.exists() or not _ANDROIDTEST_APK.exists():
            pytest.skip("AAR APKs not found in test archive; skipping instrumentation test.")

        self.device.install(_APP_APK)
        self.device.install(_ANDROIDTEST_APK)

        # `am instrument -w` exits 0 even on test failure, so parse output.
        output = self.device.shell(
            [f"am instrument -w {_INSTRUMENTATION_TARGET}"],
            capture_output=True,
        )
        text = "\n".join(output or [])
        print(text)
        assert "FAILURES!!!" not in text, "AndroidJUnit instrumentation reported failures."
        assert "OK (" in text, "AndroidJUnit instrumentation did not report OK."
