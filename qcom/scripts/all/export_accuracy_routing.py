# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Export observed two-pass runner selection for future target consumers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

SNAPSHOT_GROUP = re.compile(r"^QnnSnapshot_(.+?)_(?:OpBuilder|Session)(_\w+)?Test$")
ACCURACY_GROUP = re.compile(r"^QnnAcc_(.+?)_Accuracy(?:_\w+)?Test$")
GOLDEN_PROFILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def parse_test_list(text: str) -> list[tuple[str, str]]:
    cases = []
    suite = ""
    for line in text.splitlines():
        name = line.split("#", 1)[0].strip()
        if not name:
            continue
        if not line[0].isspace():
            suite = name.removesuffix(".")
        elif suite and not any(part.startswith("DISABLED_") for part in (suite + "/" + name).split("/")):
            cases.append((suite, name))
    if len(cases) != len(set(cases)):
        raise ValueError("Duplicate test identity in test list")
    return sorted(cases)


def snapshot_status(case: dict) -> str:
    if case.get("errors") not in (None, [], 0):
        return "untrusted"
    failures = case.get("failures", [])
    if not isinstance(failures, list) or any(
        not isinstance(failure, dict) or not isinstance(failure.get("failure"), str) for failure in failures
    ):
        return "untrusted"
    if case.get("status") != "RUN" or case.get("result") != "COMPLETED":
        return "untrusted"
    messages = "\n".join(failure["failure"] for failure in failures)
    if "QNN_SNAPSHOT_DRIFT" in messages:
        return "drift"
    if "QNN_GOLDEN_ABSENT" in messages:
        return "untrusted"
    return "untrusted" if failures else "match"


def load_snapshot_cases(path: Path) -> dict[tuple[str, str], str]:
    result = {}
    data = json.loads(path.read_text(encoding="utf-8"))
    for suite in data["testsuites"]:
        if suite.get("errors", 0):
            raise ValueError("Snapshot suite contains errors")
        for case in suite.get("testsuite", []):
            identity = (suite["name"], case["name"])
            if identity in result:
                raise ValueError("Duplicate Snapshot testcase identity")
            result[identity] = snapshot_status(case)
    return result


def read_manifest(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return {}
        manifest = {}
        for key in ("qairt_version", "ort_version", "golden_profile"):
            field = value.get(key)
            valid = isinstance(field, str) and bool(field.strip())
            if key == "golden_profile":
                valid = valid and GOLDEN_PROFILE.fullmatch(field) is not None
            manifest[key] = field if valid else None
        return manifest
    except (OSError, ValueError):
        return {}


def export_result(args: argparse.Namespace) -> dict:
    snapshot_cases = parse_test_list(args.snapshot_list.read_text(encoding="utf-8"))
    accuracy_cases = parse_test_list(args.accuracy_list.read_text(encoding="utf-8"))
    if not snapshot_cases or not accuracy_cases:
        raise ValueError("Snapshot and Accuracy test universes must be nonempty")
    try:
        statuses = load_snapshot_cases(args.snapshot_json)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        statuses = {}
    # A nonzero process exit without any recorded failed case is incomplete
    # execution evidence, even if the report contains some passing cases.
    if args.snapshot_exit and not any(status != "match" for status in statuses.values()):
        statuses = {}

    manifest = read_manifest(args.manifest)
    versions_match = bool(
        args.qairt_version
        and args.ort_version
        and args.qairt_version == manifest.get("qairt_version")
        and args.ort_version == manifest.get("ort_version")
        and manifest.get("golden_profile")
    )
    snapshots = []
    paired_snapshots: dict[tuple[str, str], list[dict]] = {}
    for suite, name in snapshot_cases:
        match = SNAPSHOT_GROUP.fullmatch(suite)
        if not match:
            raise ValueError(f"Unrecognized Snapshot suite: {suite}")
        record = {"test_id": f"{suite}.{name}", "status": statuses.get((suite, name), "untrusted")}
        snapshots.append(record)
        accuracy_suite = f"QnnAcc_{match[1]}_Accuracy{match[2] or ''}Test"
        paired_snapshots.setdefault((accuracy_suite, name), []).append(record)

    decisions = []
    for suite, name in accuracy_cases:
        match = ACCURACY_GROUP.fullmatch(suite)
        if not match:
            raise ValueError(f"Unrecognized Accuracy suite: {suite}")
        evidence = paired_snapshots.get((suite, name), [])
        if not evidence:
            status = "unmapped"
        elif any(case["status"] == "drift" for case in evidence):
            status = "drift"
        elif all(case["status"] == "match" for case in evidence) and versions_match:
            status = "match"
        else:
            status = "untrusted"
        action = "skip" if status == "match" and args.mode == "compare" else "run"
        if not args.selection_complete:
            action = "unknown"
        decisions.append(
            {
                "accuracy_test_id": f"{suite}.{name}",
                "snapshot_test_ids": [case["test_id"] for case in evidence],
                "status": status,
                "runner_action": action,
            }
        )

    return {
        "evidence_granularity": "testcase",
        "decision_granularity": "testcase",
        "producer": {
            "source_commit_sha": args.source_commit,
            "repository": os.environ.get("GITHUB_REPOSITORY"),
            "workflow_ref": os.environ.get("GITHUB_WORKFLOW_REF"),
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "job": os.environ.get("GITHUB_JOB"),
            "workflow_sha": os.environ.get("GITHUB_SHA"),
            "qairt_version": args.qairt_version or None,
            "ort_version": args.ort_version or None,
            "golden_profile": manifest.get("golden_profile"),
            "target_htp_arch": None,
        },
        "golden_manifest": {key: manifest.get(key) for key in ("qairt_version", "ort_version", "golden_profile")},
        "versions_match": versions_match,
        "mode": args.mode,
        "selection_complete": args.selection_complete,
        "runner_exit_code": args.runner_exit,
        "snapshot_exit_code": args.snapshot_exit,
        "snapshots": snapshots,
        "accuracy_decisions": decisions,
    }


def write_routing_result(result: dict, output: Path) -> None:
    """Finalize the artifact digest and atomically publish its JSON."""
    payload = json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    result["payload_sha256"] = hashlib.sha256(payload).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot-json", "snapshot-list", "accuracy-list", "manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--qairt-version", default="")
    parser.add_argument("--ort-version", default="")
    parser.add_argument("--emit-filter-file", type=Path)
    parser.add_argument("--mode", choices=("compare", "force_accuracy", "generate_goldens"), required=True)
    parser.add_argument("--selection-complete", action="store_true")
    parser.add_argument("--runner-exit", type=int, required=True)
    parser.add_argument("--snapshot-exit", type=int, required=True)
    args = parser.parse_args()
    result = export_result(args)
    if args.emit_filter_file:
        selected = [case["accuracy_test_id"] for case in result["accuracy_decisions"] if case["runner_action"] == "run"]
        args.emit_filter_file.write_text(":".join(selected) or "QnnAcc_ZZZ_NoCasesSelected.None", encoding="utf-8")
    write_routing_result(result, args.output)


if __name__ == "__main__":
    main()
