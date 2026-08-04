#!/usr/bin/env python3
"""Fetch per-file coverage snapshots from Artifactory and generate an interactive HTML trend report.

Requirements:
    pip install openpyxl plotly
    JFrog CLI (jf) must be configured and authenticated before running this script.
    In CI this is done by the jfrog/setup-jfrog-cli@v5 action.

Environment variables (required for Artifactory mode):
    BUILD_ARTIFACTORY_REPO  e.g. aisw-ortqnnep-generic-virtual

Usage:
    python scripts/gen_coverage_trend.py [--days 30] [--output coverage_trend.html]
    python scripts/gen_coverage_trend.py --local-dir /path/to/xlsx_files/  # offline mode
"""

import argparse
import datetime
import io
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import openpyxl
except ImportError:
    sys.exit("openpyxl not found — run: pip install openpyxl")

try:
    import plotly.graph_objects as go
except ImportError:
    sys.exit("plotly not found — run: pip install plotly")


NIGHTLY_PREFIX = "qa/nightly-"
REPORT_FILENAME = "per_file_coverage.xlsx"
# Number of most-changed files shown by default in the per-file chart and summary tables.
TOP_N = 10


# ── JFrog CLI helpers ─────────────────────────────────────────────────────────


def _get_repo() -> str:
    repo = os.environ.get("BUILD_ARTIFACTORY_REPO", "")
    if not repo:
        sys.exit("Set BUILD_ARTIFACTORY_REPO environment variable before running this script.")
    return repo


def _jf(*args: str) -> str:
    """Run a jf CLI command and return stdout. Raises on non-zero exit."""
    cmd = ["jf", *args]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"jf command failed: {' '.join(cmd)}\n{result.stderr.strip()}")
    return result.stdout


def list_nightly_reports(repo: str, days: int) -> list[tuple[datetime.date, str]]:
    """Return sorted list of (date, full_artifactory_path) for the last `days` days.

    Uses 'jf rt search' which respects the JFrog CLI configuration set up by
    jfrog/setup-jfrog-cli@v5 in CI (no explicit credentials needed here).
    """
    pattern = f"{repo}/{NIGHTLY_PREFIX}*/{REPORT_FILENAME}"
    raw = _jf("rt", "search", pattern)
    items = json.loads(raw) if raw.strip() else []

    cutoff = datetime.date.today() - datetime.timedelta(days=days)
    results = []
    for item in items:
        # item["path"] = "repo/qa/nightly-2026-08-02-63420a834b/per_file_coverage.xlsx"
        parts = item["path"].split("/")
        folder = parts[-2]  # "nightly-2026-08-02-63420a834b"
        date_part = folder[len("nightly-") : len("nightly-") + 10]
        try:
            d = datetime.date.fromisoformat(date_part)
        except ValueError:
            continue
        if d >= cutoff:
            results.append((d, item["path"]))

    return sorted(results)


def fetch_reports(repo: str, entries: list[tuple[datetime.date, str]]) -> list[tuple[datetime.date, str, dict]]:
    """Download all nightly xlsx files in one 'jf rt download' call and parse them.

    jf rt download preserves the folder hierarchy under tmpdir, mirroring the
    pattern used by qcom/scripts/artifactory/artifactory.py:Artifactory.download().
    """
    if not entries:
        return []

    pattern = f"{repo}/{NIGHTLY_PREFIX}*/{REPORT_FILENAME}"
    with tempfile.TemporaryDirectory(prefix="CoverageTrend-") as tmpdir:
        print(f"  Downloading {len(entries)} report(s) via jf rt download…")
        _jf("rt", "download", pattern, f"{tmpdir}/")

        snapshots = []
        for d, full_path in entries:
            # full_path = "repo/qa/nightly-date-sha/per_file_coverage.xlsx"
            # jf rt download strips the repo prefix, so local path is:
            # tmpdir/qa/nightly-date-sha/per_file_coverage.xlsx
            rel = "/".join(full_path.split("/")[1:])
            local = Path(tmpdir) / rel
            if not local.exists():
                print(f"  WARNING: expected file not found after download: {rel}")
                continue
            data = parse_xlsx(local.read_bytes())
            if data is None:
                print(f"  Skipping {local.name} (unrecognised schema)")
                continue
            folder = full_path.split("/")[-2]
            snapshots.append((d, folder, data))
            print(f"  {folder}: {len(data)} files")

    return snapshots


# ── xlsx parsing ──────────────────────────────────────────────────────────────


def parse_xlsx(content: bytes) -> dict[str, tuple[float, float]] | None:
    """Return {short_filename: (line_pct, branch_pct)} excluding .h files.

    Returns None if the xlsx does not have the expected per-file coverage schema
    (file / line_pct / branch_pct columns), so callers can skip unrelated files.
    """
    wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    col = {name: idx for idx, name in enumerate(header) if name is not None}
    if not {"file", "line_pct", "branch_pct"}.issubset(col):
        wb.close()
        return None
    data: dict[str, tuple[float, float]] = {}
    for row in rows:
        fname = row[col["file"]]
        if not fname or fname.endswith(".h"):
            continue
        line_pct = float(row[col["line_pct"]]) if row[col["line_pct"]] is not None else 0.0
        branch_pct = float(row[col["branch_pct"]]) if row[col["branch_pct"]] is not None else 0.0
        data[fname.split("/")[-1]] = (line_pct, branch_pct)
    wb.close()
    return data


def load_local_reports(local_dir: str) -> list[tuple[datetime.date, str, dict]]:
    """Load all per_file_coverage*.xlsx from a local directory (offline mode)."""
    results = []
    for p in sorted(Path(local_dir).glob("*.xlsx")):
        if p.name.startswith("~$"):  # skip Excel temp lock files
            continue
        # Try to extract date from filename, e.g. per_file_coverage_0706.xlsx → fallback to mtime
        stem = p.stem  # per_file_coverage_MMDD or per_file_coverage_YYYYMMDD
        date_str = stem.split("_")[-1]
        try:
            if len(date_str) == 4:  # MMDD
                d = datetime.date(datetime.date.today().year, int(date_str[:2]), int(date_str[2:]))
            elif len(date_str) == 8:  # YYYYMMDD
                d = datetime.date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:]))
            else:
                raise ValueError
        except (ValueError, IndexError):
            d = datetime.date.fromtimestamp(p.stat().st_mtime)
        data = parse_xlsx(p.read_bytes())
        if data is None:
            print(f"  Skipping {p.name} (unrecognised schema)")
            continue
        results.append((d, str(p), data))
    return sorted(results)


def make_mock_snapshots(days: int = 30) -> list[tuple[datetime.date, str, dict]]:
    """Generate synthetic time-series data for local chart validation.

    Simulates realistic coverage patterns:
    - Steady files: small noise around a stable baseline
    - Improving file: gradual increase over the period
    - Regressing file: step-down drop midway
    - New file: appears halfway through the period
    - Removed file: disappears halfway through the period
    """

    random.seed(42)

    # Representative QNN EP filenames with starting baselines
    file_baselines: dict[str, float] = {
        "qnn_execution_provider.cc": 75.0,
        "qnn_backend_manager.cc": 68.0,
        "qnn_model.cc": 85.0,
        "qnn_model_wrapper.cc": 92.0,
        "onnx_ctx_model_helper.cc": 87.0,
        "qnn_ep_utils.cc": 78.0,  # will improve
        "qnn_utils.cc": 79.0,  # will improve
        "ort_api.cc": 74.0,  # will improve
        "conv_op_builder.cc": 90.0,
        "simple_op_builder.cc": 85.0,
        "resize_op_builder.cc": 96.0,
        "lstm_op_builder.cc": 2.0,  # will improve (dramatic)
        "qnn_htp_power_config_manager.cc": 95.0,  # will regress
        "qnn_quant_params_wrapper.cc": 96.0,
        "qnn_profile_serializer.cc": 59.0,
        "gelu_fusion.cc": 83.0,
        "qnn_node_group.cc": 100.0,
        "scale_softmax_fusion.cc": 94.0,
    }
    # Files with a clear upward trend over the period (target gain over 30 days)
    improving: dict[str, float] = {
        "lstm_op_builder.cc": 77.0,
        "ort_api.cc": 14.8,
        "qnn_utils.cc": 12.9,
        "qnn_ep_utils.cc": 12.0,
    }

    today = datetime.date.today()
    snapshots = []

    for i in range(days):
        d = today - datetime.timedelta(days=days - 1 - i)
        t = i / max(days - 1, 1)  # 0.0 … 1.0
        data: dict[str, tuple[float, float]] = {}

        for fname, base in file_baselines.items():
            if fname == "qnn_htp_power_config_manager.cc":
                line = base if t < 0.5 else base - 42.0 + random.uniform(-0.3, 0.3)
            elif fname in improving:
                line = base + improving[fname] * t + random.uniform(-0.3, 0.3)
            else:
                line = base + random.uniform(-0.8, 0.8)
            line = max(0.0, min(100.0, line))
            branch = line * random.uniform(0.48, 0.58)
            data[fname] = (round(line, 1), round(branch, 1))

        # New file appears in second half
        if t >= 0.5:
            prog = (t - 0.5) / 0.5
            data["l2_norm_fusion.cc"] = (round(60.0 + 28.0 * prog, 1), round(25.0 + 10.0 * prog, 1))

        # File disappears in second half
        if t < 0.5:
            data["old_deprecated_builder.cc"] = (round(95.0 + random.uniform(-0.5, 0.5), 1), round(48.0, 1))

        folder = f"nightly-{d}-mock"
        snapshots.append((d, folder, data))

    print(f"Mock mode: generated {len(snapshots)} snapshots across {days} days")
    return snapshots


def _make_aggregate_chart(
    snapshots: list[tuple[datetime.date, str, dict]],
) -> go.Figure:
    """Line chart: avg line_pct and avg branch_pct over time."""
    dates, avg_lines, avg_branches, n_files = [], [], [], []
    for d, _, data in snapshots:
        if not data:
            continue
        dates.append(d)
        avg_lines.append(sum(v[0] for v in data.values()) / len(data))
        avg_branches.append(sum(v[1] for v in data.values()) / len(data))
        n_files.append(len(data))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=avg_lines,
            name="avg line coverage",
            mode="lines+markers",
            line=dict(color="#2196F3", width=2),
            marker=dict(size=6),
            customdata=list(zip(avg_branches, n_files, strict=False)),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "avg line: <b>%{y:.1f}%</b><br>"
                "avg branch: %{customdata[0]:.1f}%<br>"
                "files tracked: %{customdata[1]}"
                "<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=avg_branches,
            name="avg branch coverage",
            mode="lines+markers",
            line=dict(color="#FF9800", width=2, dash="dot"),
            marker=dict(size=6),
            hovertemplate=("<b>%{x}</b><br>avg branch: <b>%{y:.1f}%</b><extra></extra>"),
        )
    )
    fig.update_layout(
        title="Aggregate Coverage Trend (avg across all .cc files)",
        xaxis_title="Date",
        yaxis_title="Coverage (%)",
        yaxis=dict(range=[0, 105]),
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified",
        height=400,
        margin=dict(l=60, r=20, t=60, b=80),
    )
    return fig


def _make_perfile_chart(
    snapshots: list[tuple[datetime.date, str, dict]],
    top_n: int,
) -> go.Figure:
    """Per-file line_pct trend. Top N most volatile shown by default; all others hidden."""
    # Collect all files and their values over time
    all_files: set[str] = set()
    for _, _, data in snapshots:
        all_files.update(data.keys())

    dates = [s[0] for s in snapshots]

    # Compute volatility = max - min of line_pct across snapshots (files present in ≥2 snapshots)
    volatility: dict[str, float] = {}
    file_series: dict[str, list[float | None]] = {}
    for f in sorted(all_files):
        vals = [s[2].get(f, (None, None))[0] for s in snapshots]
        file_series[f] = vals
        present = [v for v in vals if v is not None]
        volatility[f] = (max(present) - min(present)) if len(present) >= 2 else 0.0

    top_files = sorted(volatility, key=lambda f: -volatility[f])[:top_n]
    top_set = set(top_files)

    fig = go.Figure()
    for f in sorted(all_files):
        vals = file_series[f]
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=vals,
                name=f,
                mode="lines+markers",
                visible=True if f in top_set else "legendonly",
                marker=dict(size=5),
                hovertemplate=(f"<b>{f}</b><br>%{{x}}<br>line: <b>%{{y:.1f}}%</b><extra></extra>"),
            )
        )

    fig.update_layout(
        title=(f"Per-file Line Coverage Trend (top {top_n} most volatile shown — click legend to show/hide others)"),
        xaxis_title="Date",
        yaxis_title="Line Coverage (%)",
        yaxis=dict(range=[0, 105]),
        height=500,
        legend=dict(
            orientation="v",
            x=1.01,
            y=1,
            font=dict(size=10),
        ),
        margin=dict(l=60, r=220, t=60, b=60),
        hovermode="closest",
    )
    return fig


def _make_latest_table(
    snapshots: list[tuple[datetime.date, str, dict]],
) -> go.Figure:
    """Table of all files in the most recent snapshot, sorted by line_pct ascending."""
    if not snapshots:
        return go.Figure()
    latest_date, latest_label, latest_data = snapshots[-1]

    prev_data: dict[str, tuple[float, float]] = {}
    prev_date = None
    if len(snapshots) >= 2:
        prev_date, _, prev_data_raw = snapshots[-2]
        prev_data = prev_data_raw

    rows = []
    for fname, (line_val, branch_val) in sorted(latest_data.items(), key=lambda x: x[1][0]):
        prev_l = prev_data.get(fname, (None,))[0]
        if prev_l is not None:
            delta = f"{line_val - prev_l:+.1f}%"
        else:
            delta = "new"
        rows.append((fname, f"{line_val:.1f}%", f"{branch_val:.1f}%", delta))

    files, lines, branches, deltas = zip(*rows, strict=False) if rows else ([], [], [], [])
    delta_colors = []
    for d in deltas:
        if d == "new":
            delta_colors.append("#E3F2FD")
        elif d.startswith("+"):
            delta_colors.append("#C6EFCE")
        elif d.startswith("-"):
            delta_colors.append("#FFC7CE")
        else:
            delta_colors.append("white")

    fig = go.Figure(
        go.Table(
            header=dict(
                values=["<b>file</b>", "<b>line_pct</b>", "<b>branch_pct</b>", f"<b>Δ vs {prev_date or 'prev'}</b>"],
                fill_color="#D9D9D9",
                align="left",
                font=dict(size=12),
            ),
            cells=dict(
                values=[list(files), list(lines), list(branches), list(deltas)],
                fill_color=["white", "white", "white", delta_colors],
                align=["left", "right", "right", "right"],
                font=dict(size=11),
                height=24,
            ),
        )
    )
    fig.update_layout(
        title=f"Latest Snapshot ({latest_date}) — all .cc files, sorted by line_pct ↑",
        height=max(400, min(len(rows) * 26 + 100, 900)),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def _make_top_changes_table(
    snapshots: list[tuple[datetime.date, str, dict]],
    top_n: int,
) -> go.Figure:
    """Side-by-side tables: top N improved and top N regressed files (first → last snapshot)."""
    if len(snapshots) < 2:
        return go.Figure()

    first_data = snapshots[0][2]
    last_data = snapshots[-1][2]
    first_date = snapshots[0][0]
    last_date = snapshots[-1][0]

    deltas: list[tuple[str, float, float, float]] = []  # (file, delta, first_val, last_val)
    for fname in set(first_data) & set(last_data):
        delta = last_data[fname][0] - first_data[fname][0]
        deltas.append((fname, delta, first_data[fname][0], last_data[fname][0]))

    improved = sorted([d for d in deltas if d[1] > 0], key=lambda x: -x[1])[:top_n]
    regressed = sorted([d for d in deltas if d[1] < 0], key=lambda x: x[1])[:top_n]

    def _pad(lst: list, n: int) -> list:
        return lst + [("", 0.0, 0.0, 0.0)] * (n - len(lst))

    imp = _pad(improved, top_n)
    reg = _pad(regressed, top_n)

    fig = go.Figure(
        go.Table(
            columnwidth=[3, 1, 1, 0.3, 3, 1, 1],
            header=dict(
                values=[
                    f"<b>improved file (top {top_n})</b>",
                    f"<b>{first_date}</b>",
                    f"<b>{last_date}</b>",
                    "",
                    f"<b>regressed file (top {top_n})</b>",
                    f"<b>{first_date}</b>",
                    f"<b>{last_date}</b>",
                ],
                fill_color=["#C6EFCE", "#C6EFCE", "#C6EFCE", "white", "#FFC7CE", "#FFC7CE", "#FFC7CE"],
                align="left",
                font=dict(size=12),
            ),
            cells=dict(
                values=[
                    [r[0] for r in imp],
                    [f"{r[2]:.1f}%" if r[0] else "" for r in imp],
                    [f"{r[3]:.1f}%  (+{r[1]:.1f}%)" if r[0] else "" for r in imp],
                    [""] * top_n,
                    [r[0] for r in reg],
                    [f"{r[2]:.1f}%" if r[0] else "" for r in reg],
                    [f"{r[3]:.1f}%  ({r[1]:.1f}%)" if r[0] else "" for r in reg],
                ],
                fill_color=[
                    "white",
                    "white",
                    "#C6EFCE",
                    "white",
                    "white",
                    "white",
                    "#FFC7CE",
                ],
                align=["left", "right", "right", "left", "left", "right", "right"],
                font=dict(size=11),
                height=24,
            ),
        )
    )
    fig.update_layout(
        title=f"Top {top_n} Most Improved / Regressed Files ({first_date} → {last_date})",
        height=top_n * 28 + 120,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def generate_html(
    snapshots: list[tuple[datetime.date, str, dict]],
    output_path: str,
    days: int,
) -> None:
    agg_fig = _make_aggregate_chart(snapshots)
    perfile_fig = _make_perfile_chart(snapshots, TOP_N)
    top_changes_fig = _make_top_changes_table(snapshots, TOP_N)
    table_fig = _make_latest_table(snapshots)

    agg_html = agg_fig.to_html(full_html=False, include_plotlyjs="cdn")
    perfile_html = perfile_fig.to_html(full_html=False, include_plotlyjs=False)
    top_changes_html = top_changes_fig.to_html(full_html=False, include_plotlyjs=False)
    table_html = table_fig.to_html(full_html=False, include_plotlyjs=False)

    earliest = snapshots[0][0] if snapshots else "—"
    latest = snapshots[-1][0] if snapshots else "—"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>QNN EP Coverage Trend</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #fafafa; color: #333; }}
    h1 {{ font-size: 1.4em; margin-bottom: 4px; }}
    .meta {{ color: #666; font-size: 0.9em; margin-bottom: 24px; }}
    .section {{ background: white; border: 1px solid #e0e0e0; border-radius: 6px;
               padding: 8px; margin-bottom: 20px; }}
  </style>
</head>
<body>
  <h1>QNN EP — Coverage Trend Report</h1>
  <div class="meta">
    Period: {earliest} → {latest} &nbsp;|&nbsp;
    Snapshots: {len(snapshots)} &nbsp;|&nbsp;
    Generated: {datetime.date.today()}
  </div>

  <div class="section">{agg_html}</div>
  <div class="section">{top_changes_html}</div>
  <div class="section">{perfile_html}</div>
  <div class="section">{table_html}</div>
</body>
</html>"""

    Path(output_path).write_text(html, encoding="utf-8")
    print(f"Report written to: {output_path}  ({len(snapshots)} snapshots, {earliest} → {latest})")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=30, help="How many days back to fetch (default: 30)")
    parser.add_argument("--output", default="coverage_trend.html", help="Output HTML file path")
    parser.add_argument(
        "--local-dir",
        metavar="DIR",
        help="Load xlsx files from a local directory instead of Artifactory (offline mode)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Generate synthetic data for local chart/HTML validation (no jf CLI or Artifactory needed)",
    )
    args = parser.parse_args()

    if args.mock:
        snapshots = make_mock_snapshots(args.days)
    elif args.local_dir:
        print(f"Offline mode: loading xlsx files from {args.local_dir}")
        snapshots = load_local_reports(args.local_dir)
    else:
        repo = _get_repo()
        print(f"Searching Artifactory for nightly reports (last {args.days} days, repo={repo})…")
        entries = list_nightly_reports(repo, args.days)
        print(f"  Found {len(entries)} snapshots")
        if not entries:
            sys.exit("No snapshots found — check BUILD_ARTIFACTORY_REPO and that jf CLI is authenticated.")
        snapshots = fetch_reports(repo, entries)

    if not snapshots:
        sys.exit("No data loaded.")

    generate_html(snapshots, args.output, args.days)


if __name__ == "__main__":
    main()
