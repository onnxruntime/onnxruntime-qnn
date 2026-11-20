# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Capture build versions and observed target Accuracy evidence for shadow routing."""

import argparse
import json
import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def identity() -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    return {
        "source_commit_sha": commit,
        "repository": os.environ.get("GITHUB_REPOSITORY"),
        "workflow_ref": os.environ.get("GITHUB_WORKFLOW_REF"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
    }


def collect_build(bin_dir: Path) -> dict:
    resolver = Path(__file__).with_name("resolve_tool_versions.sh")
    process = subprocess.run(
        ["bash", str(resolver), f"--bin-dir={bin_dir}"], capture_output=True, text=True, check=False
    )
    versions = dict(line.split("=", 1) for line in process.stdout.splitlines() if "=" in line)
    return {
        "schema_version": 1,
        "identity": identity(),
        "qairt_version": versions.get("qairt"),
        "ort_version": versions.get("ort"),
    }


def collect_evidence(build_context: Path, reports_dir: Path, started: Path) -> dict:
    build = json.loads(build_context.read_text(encoding="utf-8"))
    start_time = float(started.read_text(encoding="utf-8"))
    cases = {}
    arches = set()
    errors = []
    # Reports must come from this test invocation rather than a prior runner job.
    for path in sorted(reports_dir.rglob("*.results.xml")):
        if path.stat().st_mtime < start_time:
            continue
        try:
            root = ET.parse(path).getroot()
            for suite in root.iter("testsuite"):
                if not suite.get("name", "").startswith("QnnAcc_"):
                    continue
                for case in suite.findall("testcase"):
                    name = suite.attrib["name"] + "." + case.attrib["name"]
                    if name in cases:
                        raise ValueError(f"Duplicate target testcase: {name}")
                    passed = (
                        case.get("status") == "run"
                        and case.get("result") == "completed"
                        and case.find("failure") is None
                        and case.find("error") is None
                        and case.find("skipped") is None
                    )
                    cases[name] = "passed" if passed else "not_passed"
        except (ET.ParseError, KeyError, ValueError) as error:
            errors.append(str(error))
    for path in reports_dir.rglob("LastTest.log"):
        if path.stat().st_mtime >= start_time:
            arches.update(int(value) for value in re.findall(r"HTP arch:\s*(\d+)", path.read_text(errors="replace")))
    if not cases:
        errors.append("No fresh Accuracy testcase reports")
    if len(arches) != 1:
        errors.append("Target HTP architecture is missing or ambiguous")
    return {
        "schema_version": 1,
        "identity": identity(),
        "build": build,
        "target_platform": "windows-arm64",
        "target_htp_arch": next(iter(arches)) if len(arches) == 1 else None,
        "accuracy_outcomes": cases,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--bin-dir", type=Path)
    modes.add_argument("--mark-start", action="store_true")
    modes.add_argument("--build-context", type=Path)
    parser.add_argument("--reports-dir", type=Path)
    parser.add_argument("--started", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.unlink(missing_ok=True)
    if args.mark_start:
        args.output.write_text(str(time.time()), encoding="utf-8")
        return
    if args.bin_dir:
        result = collect_build(args.bin_dir)
    else:
        if args.reports_dir is None or args.started is None:
            parser.error("Evidence capture requires --reports-dir and --started")
        result = collect_evidence(args.build_context, args.reports_dir, args.started)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
