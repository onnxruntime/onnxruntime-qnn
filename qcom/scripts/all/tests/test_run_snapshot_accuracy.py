# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Hermetic tests for the platform-agnostic snapshot/accuracy runner."""

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "run_snapshot_accuracy.sh"
BASH = (
    Path(os.environ.get("PROGRAMW6432", r"C:\Program Files")) / "Git/bin/bash.exe"
    if sys.platform == "win32"
    else "bash"
)


def to_bash_path(path: Path) -> str:
    if sys.platform != "win32":
        return str(path)
    resolved = path.resolve()
    return f"/{resolved.drive[0].lower()}/{resolved.as_posix()[3:]}"


def write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def create_fake_windows_build(tmp_path: Path) -> tuple[Path, Path, Path]:
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
test_filter = next(arg for arg in args if arg.startswith("--gtest_filter="))
if "--gtest_list_tests" in args:
    if "QnnSnapshot_" in test_filter:
        print("QnnSnapshot_Clip_OpBuilderTest.\\n  Case")
    else:
        print("QnnAcc_Clip_AccuracyTest.\\n  Case")
    raise SystemExit(0)

output_arg = next(arg for arg in args if arg.startswith("--gtest_output=json:"))
output_path = output_arg.removeprefix("--gtest_output=json:")
if output_path.startswith("WIN::"):
    output_path = output_path.removeprefix("WIN::")
elif sys.platform != "win32" or not Path(output_path).is_absolute():
    raise SystemExit(91)
output_path = Path(output_path)
output_path.parent.mkdir(parents=True, exist_ok=True)

if "QnnSnapshot_" in test_filter:
    testsuites = [{{
        "name": "QnnSnapshot_Clip_OpBuilderTest",
        "failures": 0,
        "errors": 0,
        "testsuite": [{{"name": "Case", "status": "RUN", "result": "COMPLETED"}}],
    }}]
elif "ZZZ_NoCasesSelected" in test_filter:
    testsuites = []
else:
    testsuites = [{{
        "name": "QnnAcc_Clip_AccuracyTest",
        "failures": 0,
        "errors": 0,
        "testsuite": [{{"name": "Case", "status": "RUN", "result": "COMPLETED"}}],
    }}]
output_path.write_text(json.dumps({{"testsuites": testsuites}}))
""",
    )
    write_executable(bin_dir / "onnxruntime_provider_test", "#!/usr/bin/env bash\nexit 92\n")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    write_executable(
        fake_bin / "cygpath",
        '#!/usr/bin/env bash\ntest "$1" = "-w"\nprintf "WIN::%s\\n" "$2"\n',
    )
    return build_dir, bin_dir, fake_bin


def run_runner(build_dir: Path, fake_bin: Path, *args: str, extra_env: dict[str, str] | None = None):
    env = os.environ | {
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "PYTHON_BIN": sys.executable,
    }
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [str(BASH), to_bash_path(SCRIPT), f"--build-dir={to_bash_path(build_dir)}", *args],
        cwd=SCRIPT.parents[3],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_prefers_windows_provider_binary_and_converts_gtest_report_paths(tmp_path: Path) -> None:
    build_dir, bin_dir, fake_bin = create_fake_windows_build(tmp_path)

    result = run_runner(build_dir, fake_bin, "--generate-goldens")

    assert result.returncode == 0, result.stderr
    results_dir = bin_dir / "snapshot_accuracy_results"
    assert json.loads((results_dir / "snapshot_results.json").read_text())["testsuites"]
    assert json.loads((results_dir / "accuracy_results.json").read_text())["testsuites"]


def test_normal_validation_uses_shared_gate_full_accuracy_fallback(tmp_path: Path) -> None:
    build_dir, bin_dir, fake_bin = create_fake_windows_build(tmp_path)

    result = run_runner(
        build_dir,
        fake_bin,
        extra_env={"QNN_UT_QAIRT_VERSION": "2.51.0", "QNN_UT_ORT_VERSION": "1.30.0"},
    )

    assert result.returncode == 0, result.stderr
    results_dir = bin_dir / "snapshot_accuracy_results"
    summary = (results_dir / "accuracy_gate_summary.txt").read_text()
    assert "MISMATCH/absent => full accuracy run" in summary
    assert "run 1, skip 0" in summary
    accuracy = json.loads((results_dir / "accuracy_results.json").read_text())
    assert len(accuracy["testsuites"][0]["testsuite"]) == 1


def test_normal_validation_skips_accuracy_for_matched_snapshot(tmp_path: Path) -> None:
    build_dir, bin_dir, fake_bin = create_fake_windows_build(tmp_path)
    golden_dir = tmp_path / "goldens"
    golden_dir.mkdir()
    (golden_dir / "manifest.json").write_text(json.dumps({"qairt_version": "2.51.0", "ort_version": "1.30.0"}))

    result = run_runner(
        build_dir,
        fake_bin,
        extra_env={
            "QNN_UT_QAIRT_VERSION": "2.51.0",
            "QNN_UT_ORT_VERSION": "1.30.0",
            "QNN_UT_SNAPSHOT_GOLDEN_DIR": str(golden_dir),
        },
    )

    assert result.returncode == 0, result.stderr
    results_dir = bin_dir / "snapshot_accuracy_results"
    summary = (results_dir / "accuracy_gate_summary.txt").read_text()
    assert "-> MATCH" in summary
    assert "run 0, skip 1" in summary
    accuracy = json.loads((results_dir / "accuracy_results.json").read_text())
    assert accuracy["testsuites"] == []
