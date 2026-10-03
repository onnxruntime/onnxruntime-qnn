#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Apply QNN-owned, per-case exclusions before invoking an ONNX model runner.

The upstream onnxruntime_plugin_ep_onnx_test runner accepts a test-suite directory
but has no per-test-case exclusion argument.  Keep exclusions here, in the QNN
test harness, rather than carrying a patch to ONNX Runtime core.
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Mapping
from pathlib import Path

# Keys are QNN backend names, then suite names. Keep this list deliberately
# narrow: it records the behavior formerly hard-coded in the upstream ORT test
# runner without removing cases that a different QNN backend can execute.
QNN_MODEL_TEST_EXCLUSIONS: Mapping[str, Mapping[str, Mapping[str, str]]] = {
    "cpu": {
        "node": {
            "test_roialign_aligned_false": "QNN CPU does not support RoiAlign opset 22.",
            "test_roialign_aligned_true": "QNN CPU does not support RoiAlign opset 22.",
            "test_roialign_mode_max": "QNN CPU does not support RoiAlign opset 22.",
        },
    },
    "htp": {
        "node": {
            "test_roialign_mode_max": "QNN HTP does not support RoiAlign mode=max.",
        },
    },
}


def filter_model_test_suite(source: Path, destination: Path, suite: str, backend: str) -> dict[str, str]:
    """Copy *source* to *destination*, excluding QNN-disabled case directories.

    The source is never modified. A non-empty destination is rejected so an old
    filtered suite cannot accidentally hide a changed exclusion list.
    """
    if not source.is_dir():
        raise ValueError(f"Model test suite does not exist: {source}")
    if destination.exists():
        raise ValueError(f"Filtered model test destination already exists: {destination}")

    exclusions = dict(QNN_MODEL_TEST_EXCLUSIONS.get(backend, {}).get(suite, {}))
    destination.mkdir(parents=True)
    skipped: dict[str, str] = {}
    for item in source.iterdir():
        if item.name in exclusions:
            skipped[item.name] = exclusions[item.name]
            continue
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    return skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a QNN-filtered ONNX model test suite.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--suite", required=True, help="Model test suite name, for example node.")
    parser.add_argument("--backend", required=True, help="QNN backend name, for example cpu or htp.")
    args = parser.parse_args()

    skipped = filter_model_test_suite(args.source, args.destination, args.suite, args.backend)
    for name, reason in skipped.items():
        print(f"QNN model-test exclusion: {name}: {reason}")
    print(f"QNN model-test filter: excluded {len(skipped)} case(s) from {args.source}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
