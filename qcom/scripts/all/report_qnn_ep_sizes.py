#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Report uncompressed Release QNN EP binary sizes from nightly ZIP/TGZ packages."""

import argparse
import json
import math
import os
import re
import tarfile
import zipfile
from datetime import date, datetime, timedelta, timezone
from html import escape
from pathlib import Path, PurePosixPath
from string import Template

PLATFORMS = {
    "windows-arm64": "*-win-arm64.zip",
    "windows-arm64ec": "*-win-arm64ec.zip",
    "windows-arm64x": "*-win-arm64x.zip",
    "windows-x86_64": "*-win-x64.zip",
    "linux-aarch64_manylinux_2_34": "*-linux-aarch64.tgz",
    "linux-x86_64_ubuntu_22_04": "*-linux-x64.tgz",
}
HISTORY_HEADERS = [
    "Date",
    "Windows ARM64",
    "Windows ARM64EC",
    "Windows ARM64X",
    "Windows x64",
    "Linux aarch64 (manylinux_2_34)",
    "Linux x64 (Ubuntu 22.04)",
]
CHART_COLORS = ["#2563eb", "#d97706", "#9333ea", "#0891b2", "#15803d", "#475569"]


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


def snapshot_sizes(snapshot: dict) -> dict[str, int]:
    sizes = {}
    for row in snapshot["sizes"]:
        size = row["size_bytes"]
        if type(size) is not int or size <= 0:
            raise ValueError(f"Invalid snapshot size for {row['platform']}: {size}")
        if row["platform"] in sizes:
            raise ValueError(f"Duplicate snapshot platform: {row['platform']}")
        sizes[row["platform"]] = size
    return sizes


def snapshot_date(snapshot: dict) -> date | None:
    if "date" in snapshot:
        return date.fromisoformat(snapshot["date"])
    if match := re.match(r"nightly-(\d{4}-\d{2}-\d{2})-", snapshot["qa_tag"]):
        return date.fromisoformat(match[1])
    return None


def build_history(report: dict, baseline: dict | None) -> list[dict]:
    # shortcut: a missing baseline restarts history; reload retained reports when gap recovery is required.
    current_date = date.fromisoformat(report["date"])
    cutoff = current_date - timedelta(days=59)
    snapshots = [*baseline.get("history", []), baseline] if baseline else []
    daily = {}
    for snapshot in [*snapshots, report]:
        run_date = snapshot_date(snapshot)
        if snapshot["ref"] != report["ref"] or run_date is None or not cutoff <= run_date <= current_date:
            continue
        snapshot_sizes(snapshot)
        daily[run_date] = {
            "date": run_date.isoformat(),
            "qa_tag": snapshot["qa_tag"],
            "commit_sha": snapshot.get("commit_sha", "unknown"),
            "ref": snapshot["ref"],
            "ep_version": snapshot.get("ep_version", "unknown"),
            "sizes": snapshot["sizes"],
        }
    return [daily[run_date] for run_date in sorted(daily)]


def history_rows(history: list[dict]) -> list[list[str]]:
    rows = []
    first_sizes = {}
    previous_sizes = {}
    for snapshot in history:
        current_sizes = snapshot_sizes(snapshot)
        cells = []
        for platform in PLATFORMS:
            size = current_sizes.get(platform)
            previous = previous_sizes.get(platform)
            cell = "N/A"
            if size is not None:
                first_sizes.setdefault(platform, size)
                cell = f"{size / 1_000_000:.3f} (N/A)"
                if previous is not None:
                    delta = size - previous
                    cell = f"{size / 1_000_000:.3f} ({delta / 1_000_000:+.3f}, {delta / previous * 100:+.2f}%)"
            cells.append(cell)
        rows.append([snapshot["date"], *cells])
        previous_sizes = current_sizes
    if len(history) > 1:
        cells = []
        for platform in PLATFORMS:
            first = first_sizes.get(platform)
            last = previous_sizes.get(platform)
            cell = "N/A"
            if first is not None and last is not None:
                delta = last - first
                cell = f"{delta / 1_000_000:+.3f} MB ({delta / first * 100:+.2f}%)"
            cells.append(cell)
        rows.append(["Window growth", *cells])
    return rows


def format_history(history: list[dict]) -> str:
    lines = [
        "## QNN EP size history (last 60 calendar days)",
        "",
        "Each cell: size in MB (change in MB and % versus the previous available date). Missing dates are omitted.",
        "",
        f"| {' | '.join(HISTORY_HEADERS)} |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in history_rows(history):
        label = "**Window growth**" if row[0] == "Window growth" else row[0]
        lines.append(f"| {label} | {' | '.join(row[1:])} |")
    if len(history) > 1:
        lines.extend(
            [
                "",
                "Window growth compares the latest size with the first available size for each platform in this window.",
            ]
        )
    return "\n".join(lines) + "\n"


def trend_svg(history: list[dict], growth: bool) -> str:
    dates = [date.fromisoformat(snapshot["date"]) for snapshot in history]
    sizes = [snapshot_sizes(snapshot) for snapshot in history]
    series = []
    for platform in PLATFORMS:
        first = next((row[platform] for row in sizes if platform in row), None)
        series.append(
            [
                ((row[platform] / first - 1) * 100 if growth else row[platform] / 1_000_000)
                if platform in row
                else None
                for row in sizes
            ]
        )
    values = [value for points in series for value in points if value is not None]
    low, high = min([0, *values]), max([0, *values])
    span = high - low or 1
    magnitude = 10 ** math.floor(math.log10(span / 4))
    step = next(scale * magnitude for scale in (1, 2, 2.5, 5, 10) if scale * magnitude >= span / 4)
    low, high = math.floor(low / step) * step, math.ceil(high / step) * step
    if low == high:
        high = low + step * 4
    days = (dates[-1] - dates[0]).days or 1
    parts = [
        '<svg viewBox="0 0 960 300" role="img" aria-label="'
        + ("Window growth in percent" if growth else "Binary size in MB")
        + '">',
        "<desc>QNN EP measurements by platform and date. Exact values are available in the history table.</desc>",
    ]
    for tick in range(round((high - low) / step) + 1):
        value = low + step * tick
        y = 240 - (value - low) / (high - low) * 200
        parts.append(
            f'<path d="M70 {y}H930" stroke="#e2e8f0"/><text x="58" y="{y + 4}" text-anchor="end">{value:g}{"%" if growth else ""}</text>'
        )
    for index in sorted({0, len(dates) // 2, len(dates) - 1}):
        x = 70 + (dates[index] - dates[0]).days / days * 860 if len(dates) > 1 else 500
        parts.append(f'<text x="{x:.2f}" y="273" text-anchor="middle">{dates[index].isoformat()}</text>')
    for platform, label, color, points in zip(PLATFORMS, HISTORY_HEADERS[1:], CHART_COLORS, series, strict=True):
        path = []
        connected = False
        circles = []
        for index, value in enumerate(points):
            if value is None:
                connected = False
                continue
            x = 70 + (dates[index] - dates[0]).days / days * 860 if len(dates) > 1 else 500
            y = 240 - (value - low) / (high - low) * 200
            path.append(f"{'L' if connected else 'M'}{x:.2f},{y:.2f}")
            connected = True
            tooltip = (
                f"{label} | {dates[index]} | {sizes[index][platform]:,} bytes | {value:.3f}{'%' if growth else ' MB'}"
            )
            circles.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"><title>{escape(tooltip)}</title></circle>'
            )
        parts.append(f'<path d="{" ".join(path)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        parts.extend(circles)
    return "".join([*parts, "</svg>"])


def format_html(report: dict) -> str:
    history = report["history"]
    current = snapshot_sizes(history[-1])
    previous = snapshot_sizes(history[-2]) if len(history) > 1 else {}
    cards = []
    legend = []
    for platform, label, color in zip(PLATFORMS, HISTORY_HEADERS[1:], CHART_COLORS, strict=True):
        size = current.get(platform)
        delta = size - previous[platform] if size is not None and platform in previous else None
        flagged = delta is not None and delta > report["growth_limit_mb"] * 1_000_000
        change = f"{delta / 1_000_000:+.3f} MB vs previous date" if delta is not None else "No previous measurement"
        cards.append(
            f'<article class="card {"alert-card" if flagged else ""}"><h2>{label}</h2><strong>{size / 1_000_000:.3f} MB</strong><p>{change}{" · ABOVE LIMIT" if flagged else ""}</p></article>'
            if size is not None
            else f'<article class="card"><h2>{label}</h2><strong>N/A</strong></article>'
        )
        legend.append(f'<span><i style="background:{color}"></i>{label}</span>')
    rows = []
    for row in history_rows(history):
        cells = f'<th scope="row">{escape(row[0])}</th>' + "".join(f"<td>{escape(cell)}</td>" for cell in row[1:])
        rows.append(f'<tr class="{"total" if row[0] == "Window growth" else ""}">{cells}</tr>')
    alerts = report["alerts"]
    status = (
        f"{len(alerts)} platform{'s' if len(alerts) != 1 else ''} above the daily growth limit"
        if alerts
        else "No platforms above the daily growth limit"
    )
    template = Template(Path(__file__).with_name("qnn_ep_sizes.html").read_text(encoding="utf-8"))
    return template.substitute(
        date=escape(report["date"]),
        qa_tag=escape(report["qa_tag"]),
        commit=escape(report["commit_sha"]),
        version=escape(report["ep_version"]),
        limit=f"{report['growth_limit_mb']:g}",
        status=escape(status),
        status_class="warning" if alerts else "healthy",
        alerts="".join(f"<li>{escape(alert)}</li>" for alert in alerts),
        cards="".join(cards),
        legend="".join(legend),
        growth_chart=trend_svg(history, True),
        size_chart=trend_svg(history, False),
        headers="".join(f'<th scope="col">{escape(header)}</th>' for header in HISTORY_HEADERS),
        rows="".join(rows),
        start=escape(history[0]["date"]),
        count=len(history),
    )


def format_summary(report: dict, baseline: dict | None, growth_limit_mb: float = 0.5) -> tuple[str, list[str]]:
    previous_sizes = snapshot_sizes(baseline) if baseline is not None else {}

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
    report["date"] = (snapshot_date(report) or datetime.now(timezone.utc).date()).isoformat()
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    if baseline is not None and baseline["ref"] != report["ref"]:
        print("::warning::Previous snapshot is from a different ref; reporting absolute sizes only.")
        baseline = None
    report["history"] = build_history(report, baseline)
    if baseline is not None and snapshot_date(baseline) is not None:
        baseline = report["history"][-2] if len(report["history"]) > 1 else None
    summary, alerts = format_summary(report, baseline, args.growth_limit_mb)
    summary += "\n" + format_history(report["history"])
    report["baseline_qa_tag"] = baseline["qa_tag"] if baseline else None
    report["growth_limit_mb"] = args.growth_limit_mb
    report["alerts"] = alerts
    (args.build_dir / "qnn-ep-sizes.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.build_dir / "qnn-ep-sizes.md").write_text(summary, encoding="utf-8")
    (args.build_dir / "qnn-ep-sizes.html").write_text(format_html(report), encoding="utf-8")
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
