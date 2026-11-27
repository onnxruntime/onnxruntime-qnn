# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Select local Accuracy testcases from the current invocation's verdict JSON."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from export_accuracy_routing import GOLDEN_PROFILE, SNAPSHOT_GROUP, parse_test_list, routing_payload_sha256


def routing_only_failure(path: Path, exit_code: int) -> bool:
    if exit_code == 0:
        return True
    if exit_code != 1:
        return False  # crashes, timeouts, and setup errors are not golden drift
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("errors", 0) or any(suite.get("errors", 0) for suite in data["testsuites"]):
            return False
        failures = [
            failure
            for suite in data["testsuites"]
            for case in suite.get("testsuite", [])
            for failure in case.get("failures", [])
        ]
        return bool(failures) and all(
            isinstance(failure, dict)
            and isinstance(failure.get("failure"), str)
            and ("QNN_SNAPSHOT_DRIFT" in failure["failure"] or "QNN_GOLDEN_ABSENT" in failure["failure"])
            for failure in failures
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def select_cases(result: dict, cases: list[tuple[str, str]], source_commit: str, qairt: str, ort: str) -> list[str]:
    if routing_payload_sha256(result) != result.get("payload_sha256"):
        raise ValueError("Routing verdict digest mismatch")
    identities = [f"{suite}.{name}" for suite, name in cases]
    for case in result["accuracy_decisions"]:
        if (
            not isinstance(case.get("accuracy_test_id"), str)
            or case.get("status") not in ("match", "drift", "untrusted", "unmapped")
            or not isinstance(case.get("snapshot_test_ids"), list)
            or any(not isinstance(identity, str) for identity in case["snapshot_test_ids"])
        ):
            raise ValueError("Malformed Accuracy evidence")
    decisions = {case["accuracy_test_id"]: case for case in result["accuracy_decisions"]}
    if len(decisions) != len(result["accuracy_decisions"]) or set(decisions) != set(identities):
        raise ValueError("Routing verdict does not cover the registered Accuracy universe")
    producer = result["producer"]
    manifest = result["golden_manifest"]
    profile = manifest.get("golden_profile")
    same_invocation = all(
        producer.get(field) == os.environ[environment]
        for field, environment in (
            ("repository", "GITHUB_REPOSITORY"),
            ("workflow_ref", "GITHUB_WORKFLOW_REF"),
            ("run_id", "GITHUB_RUN_ID"),
            ("run_attempt", "GITHUB_RUN_ATTEMPT"),
            ("job", "GITHUB_JOB"),
        )
        if environment in os.environ
    )
    trusted = (
        same_invocation
        and producer.get("source_commit_sha") == source_commit
        and bool(qairt)
        and bool(ort)
        and producer.get("qairt_version") == manifest.get("qairt_version") == qairt
        and producer.get("ort_version") == manifest.get("ort_version") == ort
        and isinstance(profile, str)
        and GOLDEN_PROFILE.fullmatch(profile) is not None
        and producer.get("golden_profile") == profile
        and result.get("versions_match") is True
        and result.get("selection_complete") is True
        and result.get("mode") == "compare"
        and result.get("decision_granularity") == "testcase"
    )
    if not trusted:
        return identities
    if any(
        not isinstance(case.get("test_id"), str) or case.get("status") not in ("match", "drift", "untrusted")
        for case in result["snapshots"]
    ):
        raise ValueError("Malformed Snapshot evidence")
    snapshots = {case["test_id"]: case["status"] for case in result["snapshots"]}
    if len(snapshots) != len(result["snapshots"]):
        raise ValueError("Duplicate Snapshot evidence")
    required: dict[str, set[str]] = {}
    for identity in snapshots:
        suite, name = identity.split(".", 1)
        match = SNAPSHOT_GROUP.fullmatch(suite)
        if match:
            accuracy = f"QnnAcc_{match[1]}_Accuracy{match[2] or ''}Test.{name}"
            required.setdefault(accuracy, set()).add(identity)
    selected = []
    for identity in identities:
        decision = decisions[identity]
        expected = required.get(identity, set())
        paired = decision.get("snapshot_test_ids", [])
        match = (
            decision.get("status") == "match"
            and bool(expected)
            and len(paired) == len(set(paired))
            and set(paired) == expected
            and all(snapshots[test_id] == "match" for test_id in expected)
        )
        if not match:
            selected.append(identity)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verdict", type=Path)
    parser.add_argument("--accuracy-list", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-commit", default="")
    parser.add_argument("--qairt-version", default="")
    parser.add_argument("--ort-version", default="")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--snapshot-json", type=Path)
    parser.add_argument("--snapshot-exit", type=int, default=0)
    parser.add_argument("--check-snapshot-exit", action="store_true")
    args = parser.parse_args()
    if args.check_snapshot_exit:
        raise SystemExit(0 if routing_only_failure(args.snapshot_json, args.snapshot_exit) else 1)
    cases = parse_test_list(args.accuracy_list.read_text(encoding="utf-8"))
    if not cases:
        raise ValueError("No registered Accuracy tests")
    result = json.loads(args.verdict.read_text(encoding="utf-8"))
    selected = select_cases(result, cases, args.source_commit, args.qairt_version, args.ort_version)
    args.output.write_text(":".join(selected) or "QnnAcc_ZZZ_NoCasesSelected.None", encoding="utf-8")
    if args.summary:
        args.summary.write_text(
            f"Accuracy routing: {len(cases)} cases; run {len(selected)}, skip {len(cases) - len(selected)}\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
