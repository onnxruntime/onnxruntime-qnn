#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
#
# Extract op-builder group names from a gtest JSON report produced by the
# QnnUnit_*_Snapshot*/QnnUnit_*_SessionSnapshot* suites.
#
# Suite naming is op-first: QnnUnit_<Op>_<Tier>[_<Variant>]Test, where <Tier>
# is Snapshot or SessionSnapshot. The op is recovered as the segment(s)
# between "QnnUnit_" and the first tier token, so op names may themselves
# contain underscores (e.g. Gelu_Fusion) without ambiguity.
#
# Usage:
#   python3 extract_snapshot_groups.py <snapshot_results.json> [--failures-only]

import argparse
import json
import re
from pathlib import Path

PATTERN = re.compile(r"^QnnUnit_(.+?)_(?:SessionSnapshot|Snapshot)(?:_\w+)?Test$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract op-builder group names from a snapshot gtest JSON report.")
    p.add_argument("snapshot_json", type=Path, help="Path to the gtest --gtest_output=json: file")
    p.add_argument(
        "--failures-only",
        action="store_true",
        help="Only include groups whose suite had failures or errors (default: include all groups)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.snapshot_json) as f:
        data = json.load(f)

    ops = set()
    for suite in data.get("testsuites", []):
        if args.failures_only and not (suite.get("failures", 0) > 0 or suite.get("errors", 0) > 0):
            continue
        m = PATTERN.match(suite["name"])
        if m:
            ops.add(m.group(1))

    print(",".join(sorted(ops)))


if __name__ == "__main__":
    main()
