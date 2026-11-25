# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Export Snapshot evidence from existing provider CTest XML without rerunning tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from export_accuracy_routing import export_result


def read_provider_xml(path: Path) -> dict:
    root = ET.parse(path).getroot()
    if root.tag != "testsuites":
        raise ValueError("Expected GTest testsuites XML")
    if root.get("errors", "0") != "0":
        raise ValueError("Provider report contains errors")
    suites = []
    for suite in root.findall("testsuite"):
        if not suite.attrib["name"].startswith("QnnSnapshot_"):
            continue
        if suite.get("errors", "0") != "0":
            raise ValueError("Snapshot suite contains errors")
        cases = []
        for case in suite.findall("testcase"):
            cases.append(
                {
                    "name": case.attrib["name"],
                    "status": case.get("status", "").upper(),
                    "result": "SKIPPED" if case.find("skipped") is not None else case.get("result", "").upper(),
                    "failures": [
                        {"failure": failure.get("message", "") + (failure.text or "")}
                        for failure in [*case.findall("failure"), *case.findall("error")]
                    ],
                }
            )
        suites.append({"name": suite.attrib["name"], "testsuite": cases})
    return {"testsuites": suites}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-exit", type=int, required=True)
    args = parser.parse_args()
    build_dir = args.build_dir.resolve()
    report = build_dir / "onnxruntime_provider_test.results.xml"
    snapshot_data = read_provider_xml(report)
    repo = Path(__file__).resolve().parents[3]
    env = os.environ | {"LD_LIBRARY_PATH": str(build_dir) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")}
    binary = build_dir / "onnxruntime_provider_test"
    resolver = repo / "qcom/scripts/all/resolve_tool_versions.sh"
    versions = subprocess.run(
        ["bash", str(resolver), f"--bin-dir={build_dir}", "both"], capture_output=True, text=True, check=False
    )
    current = dict(line.split("=", 1) for line in versions.stdout.splitlines() if "=" in line)
    with tempfile.TemporaryDirectory(prefix="qnn-provider-observation-") as directory:
        temporary = Path(directory)
        for filename, test_filter in (("snapshot.txt", "QnnSnapshot_*"), ("accuracy.txt", "QnnAcc_*")):
            listing = subprocess.check_output(
                [str(binary), "--gtest_list_tests", f"--gtest_filter={test_filter}"], cwd=build_dir, env=env, text=True
            )
            (temporary / filename).write_text(listing, encoding="utf-8")
        snapshot_json = temporary / "snapshot.json"
        snapshot_json.write_text(json.dumps(snapshot_data), encoding="utf-8")
        golden_dir = os.environ.get("QNN_UT_SNAPSHOT_GOLDEN_DIR")
        result = export_result(
            argparse.Namespace(
                snapshot_json=snapshot_json,
                snapshot_list=temporary / "snapshot.txt",
                accuracy_list=temporary / "accuracy.txt",
                manifest=Path(golden_dir) / "manifest.json" if golden_dir else temporary / "no-manifest.json",
                qairt_version=current.get("qairt", ""),
                ort_version=current.get("ort", ""),
                source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                target_groups="",
                mode="compare",
                selection_complete=False,
                runner_exit=args.validation_exit,
                snapshot_exit=args.validation_exit,
            )
        )
    # CTest observation is not a two-pass routing decision. Consumers must not
    # treat it as skip authorization, even if individual Snapshot cases match.
    result["mode"] = "provider_observation"
    result["decision_granularity"] = "none"
    result["snapshot_exit_code"] = None  # CTest gives an aggregate provider verdict.
    result["provider_validation_exit_code"] = args.validation_exit
    payload = json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    result["payload_sha256"] = hashlib.sha256(payload).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output.with_suffix(args.output.suffix + ".tmp")
    staging.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    staging.replace(args.output)


if __name__ == "__main__":
    main()
