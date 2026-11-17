#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Report uncompressed Release QNN EP binary sizes from nightly ZIP/TGZ packages."""

import argparse
import json
import math
import os
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

PLATFORMS = {
    "windows-arm64": "*-win-arm64.zip",
    "windows-arm64ec": "*-win-arm64ec.zip",
    "windows-arm64x": "*-win-arm64x.zip",
    "windows-x86_64": "*-win-x64.zip",
    "linux-aarch64_manylinux_2_34": "*-linux-aarch64.tgz",
    "linux-x86_64_ubuntu_22_04": "*-linux-x64.tgz",
}


def collect_sizes(build_dir: Path) -> list[dict]:
    rows = []
    for platform, suffix in PLATFORMS.items():
        archives = sorted((build_dir / platform / "Release").rglob(f"onnxruntime-qnn-{suffix}"))
        if len(archives) != 1:
            raise ValueError(f"{platform}: expected one Release archive, found {len(archives)}: {archives}")
        archive = archives[0]
        library = "onnxruntime_providers_qnn.dll" if archive.suffix == ".zip" else "libonnxruntime_providers_qnn.so"
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as package:
                sizes = [
                    entry.file_size
                    for entry in package.infolist()
                    if not entry.is_dir() and PurePosixPath(entry.filename).name == library
                ]
        else:
            with tarfile.open(archive, "r:gz") as package:
                sizes = [
                    entry.size for entry in package if entry.isfile() and PurePosixPath(entry.name).name == library
                ]
        if len(sizes) != 1 or sizes[0] <= 0:
            raise ValueError(f"{archive}: expected one nonempty {library}, found sizes {sizes}")
        rows.append({"platform": platform, "archive": archive.name, "library": library, "size_bytes": sizes[0]})
    return rows


def format_summary(report: dict, baseline: dict | None, growth_limit_mb: float = 0.5) -> tuple[str, list[str]]:
    previous_sizes = {}
    if baseline is not None:
        for row in baseline["sizes"]:
            size = row["size_bytes"]
            if type(size) is not int or size <= 0:
                raise ValueError(f"Invalid baseline size for {row['platform']}: {size}")
            if row["platform"] in previous_sizes:
                raise ValueError(f"Duplicate baseline platform: {row['platform']}")
            previous_sizes[row["platform"]] = size

    lines = [
        "## Nightly QNN EP binary sizes",
        "",
        f"Current: `{report['qa_tag']}`; commit: `{report['commit_sha']}`.",
        f"Baseline: `{baseline['qa_tag']}`." if baseline else "Baseline: unavailable (no prior snapshot loaded).",
        "",
        "Uncompressed Release binaries only. 1 MB = 1,000,000 bytes.",
        f"Growth limit: more than {growth_limit_mb:g} MB per platform. Artifacts publish before the size check.",
        "",
        "| Platform | Size (bytes) | Size (MB) | Previous (MB) | Change (bytes) | Change (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    alerts = []
    for row in report["sizes"]:
        size = row["size_bytes"]
        previous = previous_sizes.get(row["platform"])
        comparison = "N/A | N/A | N/A"
        if previous is not None:
            delta = size - previous
            comparison = f"{previous / 1_000_000:.3f} | {delta:+,} | {delta / previous * 100:+.2f}%"
            if delta > growth_limit_mb * 1_000_000:
                alerts.append(
                    f"{row['platform']}: increased by {delta:,} bytes ({delta / 1_000_000:.6f} MB), "
                    f"exceeding the {growth_limit_mb:g} MB limit."
                )
        lines.append(f"| {row['platform']} | {size:,} | {size / 1_000_000:.3f} | {comparison} |")
    if alerts:
        lines.extend(["", "### Size growth alerts", "", *[f"- {alert}" for alert in alerts]])
    return "\n".join(lines) + "\n", alerts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--qa-tag", required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--growth-limit-mb", type=float, default=0.5)
    args = parser.parse_args()
    if not math.isfinite(args.growth_limit_mb) or args.growth_limit_mb < 0:
        parser.error("--growth-limit-mb must be a finite, nonnegative number")

    report = {
        "qa_tag": args.qa_tag,
        "commit_sha": os.environ.get("GITHUB_SHA", "local"),
        "ref": os.environ.get("GITHUB_REF", "local"),
        "ep_version": (Path(__file__).resolve().parents[3] / "VERSION_NUMBER").read_text().strip(),
        "sizes": collect_sizes(args.build_dir),
    }
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    if baseline is not None and baseline["ref"] != report["ref"]:
        print("::warning::Previous snapshot is from a different ref; reporting absolute sizes only.")
        baseline = None
    summary, alerts = format_summary(report, baseline, args.growth_limit_mb)
    report["baseline_qa_tag"] = baseline["qa_tag"] if baseline else None
    report["growth_limit_mb"] = args.growth_limit_mb
    report["alerts"] = alerts
    (args.build_dir / "qnn-ep-sizes.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.build_dir / "qnn-ep-sizes.md").write_text(summary, encoding="utf-8")
    if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary_path).open("a", encoding="utf-8") as output:
            output.write(summary)
    if output_path := os.environ.get("GITHUB_OUTPUT"):
        with Path(output_path).open("a", encoding="utf-8") as output:
            output.write(f"size_threshold_exceeded={str(bool(alerts)).lower()}\n")
    for alert in alerts:
        print(f"::warning::{alert}")
    print(summary)


if __name__ == "__main__":
    main()
