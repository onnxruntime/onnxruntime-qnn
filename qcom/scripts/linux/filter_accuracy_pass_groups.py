#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
#
# Filter accuracy-PASSING op groups from a gtest JSON report.
#
# Input:  path to accuracy_results.json (gtest --gtest_output=json)
# Output: one passing group name per line (sorted), to stdout.
# Exit:   0 = at least one group passed; 1 = zero groups passed; 2 = parse error.
#
# A group passes iff every QnnUnit_<Group>_Accuracy[_<Variant>]Test case has
# result=COMPLETED and no failures. SKIPPED / NOTRUN cases count as not passed.

import json
import re
import sys


def filter_pass_groups(data):
    """Return sorted list of group names where all accuracy cases passed."""
    pattern = re.compile(r"^QnnUnit_(.+?)_Accuracy(?:_\w+)?Test")
    group_all_pass = {}

    for suite in data.get("testsuites", []):
        m = pattern.match(suite.get("name", ""))
        if not m:
            continue
        group = m.group(1)
        for case in suite.get("testsuite", []):
            result = case.get("result")
            status = case.get("status")
            has_failures = bool(case.get("failures"))
            passed = (result == "COMPLETED") and (not has_failures)
            if result == "SKIPPED" or status == "NOTRUN":
                passed = False
            group_all_pass[group] = group_all_pass.get(group, True) and passed
        group_all_pass.setdefault(group, False)

    return sorted(g for g, ok in group_all_pass.items() if ok)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <accuracy_results.json>", file=sys.stderr)
        return 2

    try:
        with open(sys.argv[1]) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error reading {sys.argv[1]}: {e}", file=sys.stderr)
        return 2

    groups = filter_pass_groups(data)
    if not groups:
        return 1

    for g in groups:
        print(g)
    return 0


if __name__ == "__main__":
    sys.exit(main())
