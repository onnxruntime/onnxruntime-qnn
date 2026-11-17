# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import io
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
from report_qnn_ep_sizes import PLATFORMS, collect_sizes, format_summary


@pytest.fixture
def packages(tmp_path):
    for platform, suffix in PLATFORMS.items():
        # ARM64X uses the Windows multi-config Release/Release/dist layout.
        dist = tmp_path / platform / "Release" / ("Release/dist" if platform == "windows-arm64x" else "dist")
        dist.mkdir(parents=True)
        archive = dist / f"onnxruntime-qnn-{suffix.replace('*', '2.9.0')}"
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
                package.writestr("onnxruntime_providers_qnn.dll", b"a" * 1000)
                package.writestr("QnnHtpPrepare.dll", b"a" * 2000)
        else:
            with tarfile.open(archive, "w:gz") as package:
                entry = tarfile.TarInfo("libonnxruntime_providers_qnn.so")
                entry.size = 1000
                package.addfile(entry, io.BytesIO(b"a" * 1000))
    return tmp_path


def test_uncompressed_sizes_exclude_pdb_debug_and_dependencies(packages):
    arm64 = packages / "windows-arm64"
    (arm64 / "Release/dist/onnxruntime-qnn-2.9.0-win-arm64-pdb.zip").write_bytes(b"ignored")
    (arm64 / "Debug/dist").mkdir(parents=True)
    (arm64 / "Debug/dist/onnxruntime-qnn-2.9.0-Debug-win-arm64.zip").write_bytes(b"ignored")
    (arm64 / "RelWithDebInfo/dist").mkdir(parents=True)
    (arm64 / "RelWithDebInfo/dist/onnxruntime-qnn-2.9.0-RelWithDebInfo-win-arm64.zip").write_bytes(b"ignored")
    rows = collect_sizes(packages)
    assert [row["platform"] for row in rows] == list(PLATFORMS)
    assert all(row["size_bytes"] == 1000 for row in rows)


@pytest.mark.parametrize(
    "problem", ["missing_archive", "duplicate_archive", "missing_library", "duplicate_library", "empty_library"]
)
def test_invalid_package_fails(packages, problem):
    archive = next((packages / "windows-arm64/Release").rglob("*.zip"))
    if problem == "missing_archive":
        archive.unlink()
    elif problem == "duplicate_archive":
        archive.with_name(archive.name.replace("2.9.0", "2.8.0")).write_bytes(archive.read_bytes())
    else:
        with zipfile.ZipFile(archive, "w") as package:
            if problem != "missing_library":
                package.writestr("onnxruntime_providers_qnn.dll", b"" if problem == "empty_library" else b"a")
            if problem == "duplicate_library":
                package.writestr("nested/onnxruntime_providers_qnn.dll", b"b")
    with pytest.raises(ValueError, match="expected one"):
        collect_sizes(packages)


def test_summary_growth_shrink_unchanged_and_missing_baseline():
    report = {
        "qa_tag": "nightly-current",
        "commit_sha": "abc",
        "sizes": [
            {"platform": platform, "size_bytes": size}
            for platform, size in [("growth", 1100), ("shrink", 900), ("unchanged", 1000), ("new", 1000)]
        ],
    }
    baseline = {
        "qa_tag": "nightly-previous",
        "sizes": [{"platform": platform, "size_bytes": 1000} for platform in ["growth", "shrink", "unchanged"]],
    }
    summary, alerts = format_summary(report, baseline)
    assert not alerts
    assert "+100 | +10.00%" in summary
    assert "-100 | -10.00%" in summary
    assert "+0 | +0.00%" in summary
    assert "| new | 1,000 | 0.001 | N/A | N/A | N/A |" in summary
    assert "unavailable" in format_summary(report, None)[0]
    baseline["sizes"][0]["size_bytes"] = 0
    with pytest.raises(ValueError, match="Invalid baseline size"):
        format_summary(report, baseline)


@pytest.mark.parametrize("same_ref", [True, False, None])
def test_cli_writes_snapshots_and_job_summary(packages, monkeypatch, same_ref):
    monkeypatch.setenv("GITHUB_SHA", "abc123")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    summary_path = packages / "job-summary.md"
    summary_path.write_text("Existing summary\n")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_path))
    output_path = packages / "job-output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))
    baseline_path = packages / "previous.json"
    baseline_path.write_text(
        json.dumps(
            {
                "qa_tag": "nightly-previous",
                "ref": "refs/heads/main" if same_ref else "refs/heads/other",
                "sizes": [{"platform": platform, "size_bytes": 800} for platform in PLATFORMS],
            }
        )
    )
    script = Path(__file__).resolve().parents[1] / "report_qnn_ep_sizes.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--build-dir",
            str(packages),
            "--qa-tag",
            "nightly-current",
            *(["--baseline", str(baseline_path)] if same_ref is not None else []),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((packages / "qnn-ep-sizes.json").read_text())
    assert report["commit_sha"] == "abc123"
    assert report["baseline_qa_tag"] == ("nightly-previous" if same_ref else None)
    assert len(report["sizes"]) == 6
    summary = (packages / "qnn-ep-sizes.md").read_text()
    assert ("+200 | +25.00%" in summary) == bool(same_ref)
    assert summary_path.read_text() == "Existing summary\n" + summary
    assert output_path.read_text() == "size_threshold_exceeded=false\n"


@pytest.mark.parametrize(
    "limit,delta,exceeded",
    [(0.5, 500000, False), (0.5, 500001, True), (0.5, -600000, False), (0, 0, False), (0, 1, True)],
)
def test_growth_limit_boundary(limit, delta, exceeded):
    report = {"qa_tag": "current", "commit_sha": "abc", "sizes": [{"platform": "arm64", "size_bytes": 1000000 + delta}]}
    baseline = {"qa_tag": "previous", "sizes": [{"platform": "arm64", "size_bytes": 1000000}]}
    summary, alerts = format_summary(report, baseline, limit)
    assert bool(alerts) == exceeded
    assert ("### Size growth alerts" in summary) == exceeded
    if exceeded:
        assert "arm64" in alerts[0]
        assert f"{delta:,} bytes" in alerts[0]


def test_cli_alerts_save_reports_and_output_without_failing(packages, monkeypatch):
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    output_path = packages / "job-output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))
    baseline = packages / "previous.json"
    baseline.write_text(
        json.dumps(
            {
                "qa_tag": "nightly-previous",
                "ref": "refs/heads/main",
                "sizes": [{"platform": platform, "size_bytes": 800} for platform in PLATFORMS],
            }
        )
    )
    script = Path(__file__).resolve().parents[1] / "report_qnn_ep_sizes.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--build-dir",
            str(packages),
            "--qa-tag",
            "nightly-current",
            "--baseline",
            str(baseline),
            "--growth-limit-mb",
            "0.000199",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((packages / "qnn-ep-sizes.json").read_text())
    assert report["growth_limit_mb"] == 0.000199
    assert len(report["alerts"]) == 6
    assert result.stdout.count("::warning::") == 6
    assert "### Size growth alerts" in (packages / "qnn-ep-sizes.md").read_text()
    assert output_path.read_text() == "size_threshold_exceeded=true\n"


@pytest.mark.parametrize("limit", ["-1", "nan", "inf"])
def test_invalid_limit_rejected_before_writing_reports(packages, limit):
    script = Path(__file__).resolve().parents[1] / "report_qnn_ep_sizes.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--build-dir",
            str(packages),
            "--qa-tag",
            "nightly-current",
            f"--growth-limit-mb={limit}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "finite, nonnegative" in result.stderr
    assert not (packages / "qnn-ep-sizes.json").exists()
