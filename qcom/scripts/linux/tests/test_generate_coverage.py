# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Hermetic regression tests for generate_coverage.sh routing inputs."""

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "generate_coverage.sh"
REPO_ROOT = SCRIPT.parents[3]


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.mark.parametrize(
    ("skip_accuracy", "expected_exit"),
    [(False, 0), (True, 1)],
    ids=["accuracy-fallback", "accuracy-skipped"],
)
def test_failed_snapshot_uses_accuracy_fallback_or_fails_when_accuracy_skipped(
    tmp_path: Path, skip_accuracy: bool, expected_exit: int
) -> None:
    """A failed snapshot must not silently bypass all correctness validation."""
    build_dir = tmp_path / "build"
    config_dir = build_dir / "RelWithDebInfo"
    config_dir.mkdir(parents=True)
    (build_dir / "coverage.gcno").touch()
    stale_json = config_dir / "snapshot_results.json"
    stale_json.write_text('{"testsuites": []}\n')
    (config_dir / "accuracy_filter.txt").write_text("stale-filter\n")
    (config_dir / "accuracy_gate_summary.txt").write_text("stale-summary\n")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    lcov_bin = tmp_path / "lcov-bin"
    lcov_bin.mkdir()
    filter_log = tmp_path / "accuracy_filters.txt"

    _write_executable(
        config_dir / "onnxruntime_provider_test",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'case " $* " in\n'
        '  *" --gtest_list_tests "*) printf "QnnAcc_Clip_AccuracyTest.\\n  Case/Clip_f32\\n" ;;\n'
        '  *" --gtest_filter=QnnSnapshot_* "*) exit 1 ;;\n'
        '  *" --gtest_filter=QnnAcc_* "*) echo "$*" >> "${FAKE_FILTER_LOG}" ;;\n'
        "esac\n",
    )
    _write_executable(
        fake_bin / "python3",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'if [ "${{1:-}}" = "{REPO_ROOT}/qcom/scripts/all/package_manager.py" ]; then\n'
        '  case " $* " in *" --print-bin-dir "*) echo "${FAKE_LCOV_BIN}" ;; esac\n'
        "  exit 0\n"
        "fi\n"
        f'exec "{sys.executable}" "$@"\n',
    )
    _write_executable(
        lcov_bin / "lcov",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [ "${1:-}" = "--version" ]; then echo "lcov fake"; exit 0; fi\n'
        'out=""; while [ "$#" -gt 0 ]; do [ "$1" = "--output-file" ] && { out="$2"; shift; }; shift; done\n'
        '[ -z "${out}" ] || { mkdir -p "$(dirname "${out}")"; : > "${out}"; }\n',
    )
    _write_executable(
        lcov_bin / "genhtml",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'while [ "$#" -gt 0 ]; do [ "$1" = "--output-directory" ] && { mkdir -p "$2"; exit 0; }; shift; done\n',
    )
    _write_executable(
        lcov_bin / "lcov_cobertura",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'while [ "$#" -gt 0 ]; do [ "$1" = "--output" ] && { : > "$2"; exit 0; }; shift; done\n',
    )

    env = os.environ | {
        "FAKE_FILTER_LOG": str(filter_log),
        "FAKE_LCOV_BIN": str(lcov_bin),
        "ORT_BUILD_TOOLS_PATH": str(tmp_path / "tools"),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
    }
    command = ["bash", str(SCRIPT), f"--build-dir={build_dir}"]
    if skip_accuracy:
        command.append("--skip-accuracy")
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == expected_exit, result.stderr
    assert not stale_json.exists()
    assert not (config_dir / "accuracy_filter.txt").exists()
    assert not (config_dir / "accuracy_gate_summary.txt").exists()
    if skip_accuracy:
        assert "Snapshot test phase failed while accuracy was skipped" in result.stderr
        assert (config_dir / "coverage/coverage.xml").is_file()
        assert not filter_log.exists()
    else:
        assert "--gtest_filter=QnnAcc_*" in filter_log.read_text()
