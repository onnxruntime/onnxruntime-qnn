# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Verify that snapshot results routed accuracy tests safely."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

VERSION_RE = re.compile(r"^\[VERSION\].*-> (MATCH|MISMATCH/absent => full accuracy run)$")
ROUTING_RE = re.compile(r"^\[ROUTING\] (\d+) accuracy case\(s\): run (\d+), skip (\d+)\.$")


def verify_summary(summary: str) -> str:
    lines = summary.splitlines()
    version_line = next((line for line in lines if line.startswith("[VERSION]")), None)
    if version_line is None:
        raise ValueError("Accuracy gate summary did not contain a version verdict.")

    version_match = VERSION_RE.fullmatch(version_line)
    if version_match is None:
        raise ValueError(f"Accuracy gate summary contained an unknown version verdict: {version_line}")

    routing_line = next((line for line in lines if line.startswith("[ROUTING]")), None)
    routing_match = ROUTING_RE.fullmatch(routing_line or "")
    if routing_match is None:
        raise ValueError("Accuracy gate summary did not contain a valid routing tally.")

    total_count, run_count, skip_count = (int(value) for value in routing_match.groups())
    if run_count + skip_count != total_count:
        raise ValueError("Accuracy gate routing tally is inconsistent.")

    if version_match.group(1) == "MATCH":
        if skip_count == 0:
            raise ValueError("Matched goldens did not selectively skip any accuracy cases.")
        return f"Verified matched snapshot-golden gating with {skip_count} skipped accuracy case(s)."

    if run_count != total_count or skip_count != 0:
        raise ValueError("Version-mismatched or absent goldens did not fall back to the full accuracy suite.")
    return f"Verified full-accuracy fallback with all {total_count} accuracy case(s) executed."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()

    if not args.summary.is_file():
        parser.error(f"Accuracy gate summary was not produced: {args.summary}")

    try:
        message = verify_summary(args.summary.read_text(encoding="utf-8"))
    except ValueError as exc:
        parser.error(str(exc))

    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
