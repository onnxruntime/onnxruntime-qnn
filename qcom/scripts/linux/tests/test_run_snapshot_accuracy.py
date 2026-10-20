# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Hermetic tests for the snapshot/accuracy runner."""

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "run_snapshot_accuracy.sh"


def write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_converts_gtest_report_paths_for_windows_provider_binary(tmp_path: Path) -> None:
    build_dir = tmp_path / "build"
    bin_dir = build_dir / "Release"
    bin_dir.mkdir(parents=True)

    provider = bin_dir / "onnxruntime_provider_test.exe"
    write_executable(
        provider,
        f"""#!{sys.executable}
import json
import sys
from pathlib import Path

args = sys.argv[1:]
if "--gtest_list_tests" in args:
    test_filter = next(arg for arg in args if arg.startswith("--gtest_filter="))
    if "QnnSnapshot_" in test_filter:
        print("QnnSnapshot_Clip_OpBuilderTest.\\n  Case")
    else:
        print("QnnAcc_Clip_AccuracyTest.\\n  Case")
    raise SystemExit(0)

output_arg = next(arg for arg in args if arg.startswith("--gtest_output=json:"))
output_path = output_arg.removeprefix("--gtest_output=json:")
if not output_path.startswith("WIN::"):
    raise SystemExit(91)
output_path = Path(output_path.removeprefix("WIN::"))
output_path.parent.mkdir(parents=True, exist_ok=True)
suite = "QnnAcc_Clip_AccuracyTest" if any("QnnAcc_" in arg for arg in args) else "QnnSnapshot_Clip_OpBuilderTest"
output_path.write_text(json.dumps({{"testsuites": [{{"name": suite, "failures": 0, "errors": 0}}]}}))
""",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    write_executable(
        fake_bin / "cygpath",
        '#!/usr/bin/env bash\ntest "$1" = "-w"\nprintf "WIN::%s\\n" "$2"\n',
    )

    env = os.environ | {
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "PYTHON_BIN": sys.executable,
    }
    result = subprocess.run(
        ["bash", str(SCRIPT), f"--build-dir={build_dir}", "--generate-goldens"],
        cwd=SCRIPT.parents[3],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    results_dir = bin_dir / "snapshot_accuracy_results"
    assert json.loads((results_dir / "snapshot_results.json").read_text())["testsuites"]
    assert json.loads((results_dir / "accuracy_results.json").read_text())["testsuites"]
