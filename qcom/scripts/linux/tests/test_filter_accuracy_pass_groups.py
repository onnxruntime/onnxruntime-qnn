# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Tests for selecting accuracy-verified QNN EP golden groups."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "filter_accuracy_pass_groups.py"
SPEC = importlib.util.spec_from_file_location("filter_accuracy_pass_groups", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def suite(name, *cases):
    return {"name": name, "testsuite": list(cases)}


def completed_case(**overrides):
    case = {"result": "COMPLETED", "status": "RUN", "failures": []}
    case.update(overrides)
    return case


@pytest.mark.parametrize(
    "testsuites, expected",
    [
        (
            [suite("QnnAcc_Clip_AccuracyTest", completed_case())],
            ["Clip"],
        ),
        (
            [suite("QnnAcc_Conv_AccuracyTest", completed_case(failures=[{}]))],
            [],
        ),
        (
            [suite("QnnAcc_Gelu_AccuracyTest", completed_case(result="SKIPPED"))],
            [],
        ),
        (
            [suite("QnnAcc_Gelu_AccuracyTest", completed_case(status="NOTRUN"))],
            [],
        ),
        (
            [
                suite("QnnAcc_Resize_AccuracyTest", completed_case()),
                suite("QnnAcc_Resize_Accuracy_Fp16Test", completed_case(failures=[{}])),
                suite("UnrelatedSuite", completed_case()),
            ],
            [],
        ),
        (
            [
                suite("QnnAcc_Clip_AccuracyTest", completed_case()),
                suite("UnrelatedSuite", completed_case(failures=[{}])),
            ],
            ["Clip"],
        ),
    ],
    ids=["passing", "failure", "skipped", "notrun", "mixed_variants", "unrelated_suite"],
)
def test_filter_pass_groups(testsuites, expected):
    assert MODULE.filter_pass_groups({"testsuites": testsuites}) == expected
