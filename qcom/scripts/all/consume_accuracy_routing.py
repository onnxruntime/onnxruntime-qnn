# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Observe Windows ARM64 v73 routing proposals while retaining target execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

IDENTITY_FIELDS = ("source_commit_sha", "repository", "workflow_ref", "run_id", "run_attempt")


def validate_inputs(artifact: dict, target: dict, policy: dict, expected: dict) -> list[str]:
    reasons = []
    if artifact.get("schema_version") != 1 or artifact.get("decision_granularity") != "operator_group":
        reasons.append("unsupported_artifact_schema")
    payload = {key: value for key, value in artifact.items() if key != "payload_sha256"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if artifact.get("payload_sha256") != digest:
        reasons.append("artifact_integrity_failure")
    producer = artifact.get("producer", {})
    for label, value in (
        ("producer", producer),
        ("target", target.get("identity", {})),
        ("target_build", target.get("build", {}).get("identity", {})),
    ):
        if any(not expected.get(key) or value.get(key) != expected[key] for key in IDENTITY_FIELDS):
            reasons.append(f"{label}_identity_mismatch")
    if producer.get("job") != "build":
        reasons.append("unexpected_producer_job")
    if artifact.get("mode") != "compare" or artifact.get("selection_complete") is not True:
        reasons.append("incomplete_or_noncompare_selection")
    if artifact.get("runner_exit_code") != 0:
        reasons.append("producer_run_failed")
    if target.get("schema_version") != 1 or target.get("build", {}).get("schema_version") != 1:
        reasons.append("unsupported_target_schema")
    if target.get("errors"):
        reasons.append("target_evidence_incomplete")
    if (
        policy.get("schema_version") != 1
        or policy.get("default") != "unresolved"
        or not isinstance(policy.get("cases"), dict)
    ):
        reasons.append("invalid_policy_schema")
    if policy.get("target_platform") != "windows-arm64" or policy.get("target_htp_arch") != 73:
        reasons.append("unsupported_target_policy")
    if target.get("target_platform") != "windows-arm64" or target.get("target_htp_arch") != 73:
        reasons.append("target_profile_mismatch")
    if producer.get("target_htp_arch") != 73 or producer.get("golden_profile") != policy.get("producer_golden_profile"):
        reasons.append("producer_profile_mismatch")
    manifest = artifact.get("golden_manifest", {})
    for key in ("qairt_version", "ort_version"):
        value = producer.get(key)
        if not value or value != manifest.get(key) or value != target.get("build", {}).get(key):
            reasons.append(f"{key}_mismatch")
    if artifact.get("versions_match") is not True or manifest.get("golden_profile") != producer.get("golden_profile"):
        reasons.append("untrusted_golden_manifest")
    return reasons


def select_shadow(artifact: dict, target: dict, policy: dict, expected: dict) -> dict:
    reasons = validate_inputs(artifact, target, policy, expected)
    decisions = {}
    snapshots = {}
    for record in artifact.get("snapshots", []):
        name = record["test_id"]
        if name in snapshots:
            raise ValueError("Duplicate Snapshot identity")
        snapshots[name] = record["status"]
    for decision in artifact.get("accuracy_decisions", []):
        name = decision["accuracy_test_id"]
        if name in decisions:
            raise ValueError("Duplicate Accuracy decision")
        decisions[name] = decision
    results = []
    for name, outcome in sorted(target.get("accuracy_outcomes", {}).items()):
        case_policy = policy.get("cases", {}).get(name, {})
        classification = case_policy.get("responsibility", "unresolved")
        decision = decisions.get(name, {})
        case_reasons = list(reasons)
        if classification == "always_run_platform_specific":
            case_reasons.append("always_run_platform_specific")
        elif classification != "simulator_gated_common":
            case_reasons.append("unresolved_responsibility")
        elif case_policy.get("reviewed") is not True or not case_policy.get("equivalence_evidence"):
            case_reasons.append("missing_reviewed_equivalence")
        snapshot_ids = case_policy.get("snapshot_test_ids", [])
        if not snapshot_ids or sorted(snapshot_ids) != sorted(decision.get("snapshot_test_ids", [])):
            case_reasons.append("mapping_missing_or_inconsistent")
        if (
            decision.get("status") != "match"
            or decision.get("runner_action") != "skip"
            or any(snapshots.get(identity) != "match" for identity in snapshot_ids)
        ):
            case_reasons.append("no_trusted_graph_match")
        would_skip = not case_reasons
        results.append(
            {
                "accuracy_test_id": name,
                "proposed_action": "skip" if would_skip else "run",
                "actual_outcome": outcome,
                "false_skip_candidate": would_skip and outcome != "passed",
                "reasons": sorted(set(case_reasons)),
            }
        )
    return {
        "schema_version": 1,
        "mode": "shadow",
        "target_execution_changed": False,
        "fallback_reasons": reasons,
        "cases": results,
        "would_skip": sum(case["proposed_action"] == "skip" for case in results),
        "would_run": sum(case["proposed_action"] == "run" for case in results),
        "false_skip_candidates": sum(case["false_skip_candidate"] for case in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact", "target-evidence", "policy", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    expected = {
        "source_commit_sha": args.source_commit,
        "repository": os.environ.get("GITHUB_REPOSITORY"),
        "workflow_ref": os.environ.get("GITHUB_WORKFLOW_REF"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
    }
    inputs = []
    read_errors = []
    for label, path in (("artifact", args.artifact), ("target", args.target_evidence), ("policy", args.policy)):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Expected a JSON object")
        except (OSError, ValueError):
            value = {}
            read_errors.append(f"{label}_missing_or_malformed")
        inputs.append(value)
    try:
        artifact, target, policy = inputs
        result = select_shadow(artifact, target, policy, expected)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        result = {
            "schema_version": 1,
            "mode": "shadow",
            "target_execution_changed": False,
            "fallback_reasons": ["input_or_selection_failure"],
            "detail": str(error),
            "cases": [],
            "would_skip": 0,
            "would_run": None,
            "false_skip_candidates": 0,
        }
    result["fallback_reasons"].extend(read_errors)
    result["expected_identity"] = expected
    if not inputs[1].get("accuracy_outcomes"):
        result["would_run"] = None
        result["fallback_reasons"].append("target_accuracy_universe_unavailable")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Shadow routing: would skip={result['would_skip']}, would run={result['would_run']}, "
        f"false-skip candidates={result['false_skip_candidates']}"
    )
    for case in result["cases"]:
        print(
            f"would {case['proposed_action']}: {case['accuracy_test_id']} ({', '.join(case['reasons']) or 'trusted match'})"
        )
    for reason in result["fallback_reasons"]:
        print(f"Fallback: {reason}")
    if result["false_skip_candidates"]:
        print("::warning::Shadow routing found target cases that would be skipped without a passing outcome.")


if __name__ == "__main__":
    main()
