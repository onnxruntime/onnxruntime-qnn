#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import argparse
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal

StatusT = Literal["notrun", "run"]
ResultT = Literal["complete", "skipped", "suppressed"]

# DISABLED_ --> notrun, suppressed
# GTEST_SKIP() --> run, skipped
# failed test --> run, complete, has <failure message="..." type=""><![CDATA[...]]></failure> subelement


class TestCase:
    def __init__(
        self,
        suite_name: str,
        name: str,
        file: Path,
    ) -> None:
        self.suite_name = suite_name
        self.name = name
        self.file = file
        self.status: StatusT = "run"
        self.result: ResultT = "complete"
        self.failure_messages: list[str] = []

    @property
    def is_disabled(self) -> bool:
        return self.status == "notrun"

    @property
    def is_failure(self) -> bool:
        return len(self.failure_messages) > 0

    @property
    def is_skipped(self) -> bool:
        return self.result == "skipped"

    def to_xml(self) -> ET.Element:
        xml = ET.Element(
            "testcase",
            {
                "name": self.name,
                "file": str(self.file),
                # line
                "status": self.status,
                "result": self.result,
                "time": "0.",
                # timestamp
                "classname": self.suite_name,
            },
        )

        if self.is_failure:
            ET.SubElement(xml, "failure", {"message": "\n".join(self.failure_messages), "type": ""})
        return xml


class TestSuite:
    def __init__(self, name: str) -> None:
        self.name = name
        self.cases: dict[str, TestCase] = {}

    @property
    def is_disabled(self) -> bool:
        return any(c.is_disabled for c in self.cases.values())

    @property
    def is_failure(self) -> bool:
        return any(c.is_failure for c in self.cases.values())

    def to_xml(self) -> ET.Element:
        xml = ET.Element(
            "testsuite",
            {
                "name": self.name,
                "tests": str(len(self.cases)),
                "failures": str(sum(c.is_failure for c in self.cases.values())),
                "disabled": str(sum(c.is_disabled for c in self.cases.values())),
                "skipped": str(sum(c.is_skipped for c in self.cases.values())),
                # errors
                "time": "0.",
                # timestamp
            },
        )
        for case in self.cases.values():
            xml.append(case.to_xml())
        return xml


def parse(log_path: Path) -> list[TestSuite]:
    load_exp = re.compile(r"^Load Test Case: ([-\\.\w]+) in (.*)$")
    end_of_suite_exp = re.compile(r"^result:$")
    errors_with_messages = [
        re.compile(r".* RunImpl] ([-\\.\w]+):(.*)"),
        re.compile(r".* operator\(\)] ([-\\.\w]+):Non-zero status code .* Status Message: (.*)"),
    ]
    errors_without_messages = [
        re.compile(r"test ([-\\.\w]+) failed, please fix it"),
    ]

    test_suites: list[TestSuite] = []

    test_suite: TestSuite | None = None
    for line in log_path.read_text().splitlines():
        if (x := load_exp.match(line)) is not None:
            case, test_dir_str = x.groups()
            test_dir = Path(test_dir_str)
            suite_name = str(test_dir.parent)
            model_path = test_dir / f"{case}.onnx"
            if test_suite is None:
                test_suite = TestSuite(suite_name)
            test_suite.cases[case] = TestCase(suite_name, case, model_path)
            continue
        if end_of_suite_exp.match(line) is not None:
            assert test_suite is not None, "test_suite is none"
            test_suites.append(test_suite)
            test_suite = None
            continue
        for exp in errors_with_messages:
            if (x := exp.match(line)) is not None:
                assert test_suite is not None, "test_suite is none"
                test_suite.cases[x.group(1)].failure_messages.append(x.group(2))
        for exp in errors_without_messages:
            if (x := exp.match(line)) is not None:
                assert test_suite is not None, "test_suite is none"
                test_suite.cases[x.group(1)].failure_messages.append(line)
    if test_suite is not None:
        test_suites.append(test_suite)
    return test_suites


def main(log_path: Path) -> None:
    test_suites = parse(log_path)
    xml = ET.Element(
        "testsuites",
        attrib={
            "name": "AllTests",
            "tests": str(len(test_suites)),
            "failures": str(sum(ts.is_failure for ts in test_suites)),
            "disabled": str(sum(ts.is_disabled for ts in test_suites)),
            # errors
            "time": "0.",
        },
    )
    for test_suite in test_suites:
        subtree = test_suite.to_xml()
        xml.append(subtree)
    ET.indent(xml)
    xml_bytes: bytes = ET.tostring(xml, encoding="utf-8", xml_declaration=True)
    print(xml_bytes.decode())


if __name__ == "__main__":
    log_format = "[%(asctime)s] [model_test_log_to_junit.py] [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_format, force=True)

    parser = argparse.ArgumentParser()

    parser.add_argument("log_path", type=Path, help="Path to a onnx_test_runner log.")

    args = parser.parse_args()

    main(args.log_path)
