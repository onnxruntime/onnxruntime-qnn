#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Generate and validate Custom Op contexts with the build's native provider tests."""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def run_test(build: Path, name: str, directory: Path, target: dict[str, int], report: Path) -> None:
    env = os.environ.copy()
    env.pop("ORT_QNN_CUSTOM_OP_DOMAINS", None)
    env["ORT_QNN_UDO_OP_PACKAGE_DIR"] = str(build / "custom_op/op_packages/htp")
    env["ORT_QNN_UDO_CONTEXT_DIR"] = str(directory)
    env["ORT_QNN_UDO_CONTEXT_TARGET"] = ":".join(str(target[key]) for key in ("htp_arch", "soc_model", "vtcm_mb"))
    env["LD_LIBRARY_PATH"] = str(build) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    subprocess.run(
        [
            str(build / "onnxruntime_provider_test"),
            f"--gtest_filter=QnnHTPUdoTests.{name}",
            f"--gtest_output=xml:{report}",
        ],
        cwd=build,
        env=env,
        check=True,
    )
    result = ET.parse(report).getroot()
    if any(int(result.get(key, "0")) != value for key, value in (("tests", 1), ("failures", 0), ("disabled", 0))):
        raise RuntimeError(f"Custom Op {name} did not pass: {report}")
    if result.findall(".//skipped"):
        raise RuntimeError(f"Custom Op {name} was skipped: {report}")


def prepare(host_build: Path, destination: Path, targets_file: Path, qairt_sdk: Path) -> None:
    host_build = host_build.resolve()
    destination = destination.resolve()
    targets = json.loads(targets_file.read_text())
    if not (host_build / "onnxruntime_provider_test").is_file():
        raise RuntimeError(f"Build the Linux x86_64 provider tests first: {host_build}")
    sdk_stamp = host_build.parent / f"qairt-sdk-path-{host_build.name}.txt"
    if not sdk_stamp.is_file() or Path(sdk_stamp.read_text().strip()).resolve() != qairt_sdk.resolve():
        raise RuntimeError("The context producer and destination must use the same QAIRT SDK")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="custom-op-contexts-", dir=destination) as temporary:
        stage = Path(temporary)
        for arch, target in targets.items():
            if arch != str(target["htp_arch"]) or any(
                not isinstance(value, int) or value <= 0 for value in target.values()
            ):
                raise ValueError(f"Invalid target: {arch}: {target}")
            directory = stage / arch
            directory.mkdir()
            run_test(host_build, "UDO_AOT_GenerateContext", directory, target, stage / f"generate-{arch}.xml")
            run_test(host_build, "UDO_AOT_LoadContext", directory, target, stage / f"load-{arch}.xml")
            (directory / "target.json").write_text(json.dumps(target, indent=2) + "\n")
            (directory / "target.txt").write_text(
                ":".join(str(target[key]) for key in ("htp_arch", "soc_model", "vtcm_mb")) + "\n"
            )
        contexts = destination / "contexts"
        if contexts.exists():
            shutil.rmtree(contexts)
        contexts.mkdir()
        for arch in targets:
            shutil.move(str(stage / arch), contexts / arch)
    for name in ("run_custom_op_tests.sh", "README.md"):
        shutil.copy2(Path(__file__).parent / "custom_op" / name, destination / name)
    shutil.copy2(targets_file, destination / "targets.json")
    (destination / "sdk-version.txt").write_text(qairt_sdk.name + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-build", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--qairt-sdk", type=Path, required=True)
    parser.add_argument("--targets", type=Path, default=Path(__file__).with_name("custom_op_targets.json"))
    args = parser.parse_args()
    prepare(args.host_build, args.destination, args.targets, args.qairt_sdk)


if __name__ == "__main__":
    main()
