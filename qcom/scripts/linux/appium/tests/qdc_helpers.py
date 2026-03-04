# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import subprocess
import tempfile
from pathlib import Path

from device import DeviceBase, device_from_url
from ort_test_config import OrtTestConfig, default_test_config


class TestBase:
    @staticmethod
    def config() -> OrtTestConfig:
        return default_test_config()

    @property
    def device(self) -> DeviceBase:
        return device_from_url(self.config().device_url)

    def clean_device(self):
        self.device.shell(["rm", "-rf", self.config().qdc_log_path])
        self.device.shell(["mkdir", "-p", self.config().qdc_log_path])

        if self.config().clean_build:
            self.device.shell(["rm", "-rf", self.config().device_runtime_path])
            self.device.shell(["mkdir", "-p", self.config().device_runtime_path])

        if self.config().clean_onnx_model_tests:
            self.device.shell(["rm", "-fr", self.config().device_onnx_model_test_path])

    def prepare_ort_tests(self):
        self.clean_device()

        # Push binaries from qdc_host_path to /data/local/tmp
        for item in Path(self.config().qdc_host_path).iterdir():
            self.device.push(item, Path(self.config().device_runtime_path))

        # Push ONNX test models
        self.device.shell(["mkdir", "-p", f"{self.config().device_onnx_model_test_path}"])
        for item in Path(self.config().host_onnx_model_test_path).iterdir():
            self.device.push(item.resolve(), Path(self.config().device_onnx_model_test_path))

        # Builds sometimes come from Windows, where executable bits are not set.
        if (Path(self.config().host_build_root) / "lib").exists():
            # fmt: off
            self.device.shell(
                [
                    "find", f"{self.config().device_runtime_path}/lib",
                    "-type", "f",
                    "-exec", "chmod", "+x", "{}",
                    "\\;",
                ],
            )
            # fmt: on
        self.device.shell([f"sh -c 'chmod +x {self.config().device_build_root}/*'"])

    def copy_logs(self):
        self.device.shell(
            [f"sh -c 'cp {self.config().test_results_device_glob} {self.config().qdc_log_path}'"],
        )

        # This is a bit convoluted, but we need to convert our model test logs to JUnit XML files and
        # then have those included in the QDC job's artifacts. Therefore, we pull relevant logs, process
        # them on the host, and then put them back on the device where QDC will find them.
        device_results_log_glob = (
            f"{self.config().device_results_root}/{self.config().model_test_results_filename_glob}"
        )
        log_files = self.device.shell(["ls", device_results_log_glob], capture_output=True)
        assert log_files is not None
        log_files = [x for x in log_files if x != ""]
        log_to_xml = Path(self.config().host_qcom_scripts_path) / "all" / "model_test_log_to_junit_xml.py"
        with tempfile.TemporaryDirectory(prefix="ModelTestLogToXml-") as tmpdir:
            tmppath = Path(tmpdir)
            for log_filename in log_files:
                host_log_path = tmppath / Path(log_filename).name
                host_xml_path = host_log_path.with_suffix(".xml")
                self.device.pull(Path(log_filename), host_log_path)
                with open(host_xml_path, "w") as results_xml:
                    subprocess.run([log_to_xml, host_log_path], stdout=results_xml, check=True)
                self.device.push(host_xml_path, Path(self.config().qdc_log_path))
