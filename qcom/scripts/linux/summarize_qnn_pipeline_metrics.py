#!/usr/bin/env python3
"""Summarize QNN legacy-pipeline JSONL metrics by event and GTest case."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def parse_gtest_duration_us(value: str) -> int:
    if not value.endswith("s"):
        raise ValueError(f"Unsupported GTest duration: {value}")
    return round(float(value[:-1]) * 1_000_000)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gtest-json", type=Path)
    args = parser.parse_args()

    events: list[dict[str, object]] = []
    current_test_by_pid: dict[int, tuple[str, str]] = {}
    test_status: dict[tuple[str, str], str] = {}
    test_duration_us: dict[tuple[str, str], int] = {}

    with args.metrics.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at line {line_number}: {error}") from error

            pid = int(event["pid"])
            event_name = str(event["event"])
            if event_name == "test_start":
                current_test_by_pid[pid] = (str(event["suite"]), str(event["test"]))
            elif event_name == "test_end":
                test_key = (str(event["suite"]), str(event["test"]))
                test_status[test_key] = str(event["status"])
                test_duration_us[test_key] = int(event.get("duration_us", 0))
                current_test_by_pid.pop(pid, None)
            else:
                suite, test = current_test_by_pid.get(pid, ("<unattributed>", "<unattributed>"))
                event["suite"] = suite
                event["test"] = test
                events.append(event)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_event: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_test_event: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for event in events:
        event_name = str(event["event"])
        suite = str(event["suite"])
        test = str(event["test"])
        by_event[event_name].append(event)
        by_test_event[(suite, test, event_name)].append(event)

    with (args.output_dir / "event_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "event",
                "count",
                "tests",
                "total_duration_us",
                "mean_duration_us",
                "p50_duration_us",
                "p95_duration_us",
                "max_duration_us",
            ]
        )
        for event_name in sorted(by_event):
            event_rows = by_event[event_name]
            durations = [int(row.get("duration_us", 0)) for row in event_rows]
            tests = {(str(row["suite"]), str(row["test"])) for row in event_rows}
            writer.writerow(
                [
                    event_name,
                    len(event_rows),
                    len(tests),
                    sum(durations),
                    round(statistics.fmean(durations), 2) if durations else 0,
                    percentile(durations, 0.50),
                    percentile(durations, 0.95),
                    max(durations, default=0),
                ]
            )

    with (args.output_dir / "per_test_event.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["suite", "test", "status", "event", "count", "total_duration_us"])
        for (suite, test, event_name), event_rows in sorted(by_test_event.items()):
            writer.writerow(
                [
                    suite,
                    test,
                    test_status.get((suite, test), "unknown"),
                    event_name,
                    len(event_rows),
                    sum(int(row.get("duration_us", 0)) for row in event_rows),
                ]
            )

    event_names = sorted(by_event)
    event_count_by_test = {
        (suite, test, event_name): len(event_rows) for (suite, test, event_name), event_rows in by_test_event.items()
    }
    with (args.output_dir / "per_test_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        event_columns = [
            column for event_name in event_names for column in (f"{event_name}_count", f"{event_name}_duration_us")
        ]
        writer.writerow(["suite", "test", "status", "elapsed_us", *event_columns])
        for suite, test in sorted(test_status):
            row: list[object] = [suite, test, test_status[(suite, test)], test_duration_us.get((suite, test), 0)]
            for event_name in event_names:
                event_rows = by_test_event.get((suite, test, event_name), [])
                row.extend(
                    [
                        len(event_rows),
                        sum(int(event.get("duration_us", 0)) for event in event_rows),
                    ]
                )
            writer.writerow(row)

    with (args.output_dir / "event_repetition_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "event",
                "total_count",
                "tests_with_event",
                "tests_with_multiple",
                "max_count_per_test",
                "invocations_beyond_first",
                "beyond_first_share_pct",
            ]
        )
        for event_name in event_names:
            counts = [
                event_count_by_test.get((suite, test, event_name), 0)
                for suite, test in test_status
                if event_count_by_test.get((suite, test, event_name), 0) > 0
            ]
            total_count = sum(counts)
            tests_with_event = len(counts)
            invocations_beyond_first = total_count - tests_with_event
            writer.writerow(
                [
                    event_name,
                    total_count,
                    tests_with_event,
                    sum(count > 1 for count in counts),
                    max(counts, default=0),
                    invocations_beyond_first,
                    round(invocations_beyond_first * 100 / total_count, 2) if total_count else 0,
                ]
            )

    signature_events = [
        "cpu_session_create",
        "cpu_inference",
        "qnn_compile_total",
        "qnn_graph_execute",
        "output_comparison",
    ]
    signatures = Counter(
        tuple(event_count_by_test.get((suite, test, event_name), 0) for event_name in signature_events)
        for suite, test in test_status
    )
    with (args.output_dir / "pipeline_signature_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([*[f"{event_name}_count" for event_name in signature_events], "tests", "share_pct"])
        for signature, count in signatures.most_common():
            writer.writerow([*signature, count, round(count * 100 / len(test_status), 2)])

    event_totals = {
        event_name: sum(int(row.get("duration_us", 0)) for row in event_rows)
        for event_name, event_rows in by_event.items()
    }
    total_test_duration_us = sum(test_duration_us.values())
    measured_run_duration_us = total_test_duration_us
    gtest_data: dict[str, object] | None = None
    if args.gtest_json:
        gtest_data = json.loads(args.gtest_json.read_text(encoding="utf-8"))
        measured_run_duration_us = parse_gtest_duration_us(str(gtest_data["time"]))
    top_level_stages = [
        ("CPU session creation", "cpu_session_create"),
        ("CPU reference inference", "cpu_inference"),
        ("QNN capability / partition", "qnn_capability_partition"),
        ("QNN graph compile", "qnn_compile_total"),
        ("QNN graph execution", "qnn_graph_execute"),
        ("Output comparison", "output_comparison"),
    ]
    measured_top_level_us = sum(event_totals.get(event_name, 0) for _, event_name in top_level_stages)
    other_test_time_us = max(measured_run_duration_us - measured_top_level_us, 0)
    with (args.output_dir / "stage_breakdown.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["stage", "event", "count", "tests", "total_duration_us", "share_of_test_elapsed_pct"])
        for stage, event_name in top_level_stages:
            event_rows = by_event.get(event_name, [])
            duration_us = event_totals.get(event_name, 0)
            tests = {(str(row["suite"]), str(row["test"])) for row in event_rows}
            writer.writerow(
                [
                    stage,
                    event_name,
                    len(event_rows),
                    len(tests),
                    duration_us,
                    round(duration_us * 100 / measured_run_duration_us, 2) if measured_run_duration_us else 0,
                ]
            )
        writer.writerow(
            [
                "Other test setup / framework overhead",
                "",
                "",
                "",
                other_test_time_us,
                round(other_test_time_us * 100 / measured_run_duration_us, 2) if measured_run_duration_us else 0,
            ]
        )
        writer.writerow(
            [
                "Total measured run",
                "gtest_json" if args.gtest_json else "test_end",
                len(test_duration_us),
                len(test_duration_us),
                measured_run_duration_us,
                100.0,
            ]
        )

    if gtest_data is not None:
        suite_status_counts = Counter((suite, status) for (suite, _), status in test_status.items())
        with (args.output_dir / "suite_summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["suite", "listed", "disabled", "executed", "passed", "skipped", "elapsed_us", "share_pct"])
            for suite in gtest_data["testsuites"]:
                suite_name = str(suite["name"])
                listed = int(suite["tests"])
                disabled = int(suite["disabled"])
                elapsed_us = parse_gtest_duration_us(str(suite["time"]))
                writer.writerow(
                    [
                        suite_name,
                        listed,
                        disabled,
                        listed - disabled,
                        suite_status_counts[(suite_name, "passed")],
                        suite_status_counts[(suite_name, "skipped")],
                        elapsed_us,
                        round(elapsed_us * 100 / measured_run_duration_us, 2) if measured_run_duration_us else 0,
                    ]
                )

    compile_total_us = event_totals.get("qnn_compile_total", 0)
    compile_children = [
        ("Graph translation / compose", "qnn_graph_translation_compose"),
        ("Graph finalize", "qnn_graph_finalize"),
    ]
    measured_compile_children_us = sum(event_totals.get(event_name, 0) for _, event_name in compile_children)
    other_compile_us = max(compile_total_us - measured_compile_children_us, 0)
    with (args.output_dir / "qnn_compile_breakdown.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["stage", "event", "count", "total_duration_us", "share_of_qnn_compile_pct"])
        for stage, event_name in compile_children:
            duration_us = event_totals.get(event_name, 0)
            writer.writerow(
                [
                    stage,
                    event_name,
                    len(by_event.get(event_name, [])),
                    duration_us,
                    round(duration_us * 100 / compile_total_us, 2) if compile_total_us else 0,
                ]
            )
        writer.writerow(
            [
                "Other compile overhead",
                "",
                "",
                other_compile_us,
                round(other_compile_us * 100 / compile_total_us, 2) if compile_total_us else 0,
            ]
        )
        writer.writerow(
            [
                "Total QNN compile",
                "qnn_compile_total",
                len(by_event.get("qnn_compile_total", [])),
                compile_total_us,
                100.0,
            ]
        )

    unattributed = sum(str(event["suite"]) == "<unattributed>" for event in events)
    print(f"events={len(events)} tests={len(test_status)} unattributed={unattributed}")
    print(args.output_dir / "event_summary.csv")
    print(args.output_dir / "per_test_event.csv")
    print(args.output_dir / "per_test_summary.csv")
    print(args.output_dir / "event_repetition_summary.csv")
    print(args.output_dir / "pipeline_signature_summary.csv")
    print(args.output_dir / "stage_breakdown.csv")
    print(args.output_dir / "qnn_compile_breakdown.csv")
    if gtest_data is not None:
        print(args.output_dir / "suite_summary.csv")


if __name__ == "__main__":
    main()
