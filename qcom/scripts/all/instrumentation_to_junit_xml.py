#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Convert Android ``am instrument -w -r`` stdout into JUnit XML.

Each test method is represented by one or more "status" blocks of the form:

    INSTRUMENTATION_STATUS: class=com.example.FooTest
    INSTRUMENTATION_STATUS: test=testBar
    INSTRUMENTATION_STATUS: stack=... (multi-line; failures/errors only)
    INSTRUMENTATION_STATUS_CODE: <code>

Status codes (AndroidX test runner):
    1  = test started (ignored; a final record follows)
    0  = passed
   -1  = errored
   -2  = failed (assertion)
   -3  = ignored (skipped)
"""

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal

__all__ = [
    "InstrumentationTestCase",
    "build_xml",
    "parse",
]

StatusT = Literal["passed", "failed", "errored", "skipped"]

_STATUS_CODE_START = 1
_STATUS_CODE_OK = 0
_STATUS_CODE_ERROR = -1
_STATUS_CODE_FAILURE = -2
_STATUS_CODE_IGNORED = -3

_STATUS_KV_RE = re.compile(r"^INSTRUMENTATION_STATUS:\s+([\w.]+)=(.*)$")
_STATUS_CODE_RE = re.compile(r"^INSTRUMENTATION_STATUS_CODE:\s*(-?\d+)\s*$")


class InstrumentationTestCase:
    """A single Java test method reported by the instrumentation runner."""

    def __init__(self, class_name: str, name: str) -> None:
        self.class_name = class_name
        self.name = name
        self.status: StatusT = "passed"
        self.stack: str | None = None

    @property
    def failure_message(self) -> str:
        if not self.stack:
            return self.status
        # First non-empty line is typically the exception class + message.
        for line in self.stack.splitlines():
            if line.strip():
                return line.strip()
        return self.status

    def to_xml(self) -> ET.Element:
        attrs = {
            "classname": self.class_name,
            "name": self.name,
            "time": "0.",
        }
        el = ET.Element("testcase", attrs)
        if self.status == "failed":
            fail = ET.SubElement(el, "failure", {"message": self.failure_message, "type": ""})
            if self.stack:
                fail.text = self.stack
        elif self.status == "errored":
            err = ET.SubElement(el, "error", {"message": self.failure_message, "type": ""})
            if self.stack:
                err.text = self.stack
        elif self.status == "skipped":
            ET.SubElement(el, "skipped")
        return el


def parse(text: str) -> list[InstrumentationTestCase]:
    """Parse ``am instrument -w -r`` output into a list of test cases."""
    cases: list[InstrumentationTestCase] = []
    fields: dict[str, str] = {}
    active_key: str | None = None

    for line in text.splitlines():
        m = _STATUS_CODE_RE.match(line)
        if m:
            code = int(m.group(1))
            _finalize(cases, fields, code)
            fields = {}
            active_key = None
            continue

        # INSTRUMENTATION_RESULT / INSTRUMENTATION_CODE are the trailing summary
        # block; they don't belong to a single testcase so drop any active key.
        if line.startswith(("INSTRUMENTATION_RESULT:", "INSTRUMENTATION_CODE:")):
            active_key = None
            continue

        m = _STATUS_KV_RE.match(line)
        if m:
            key, value = m.group(1), m.group(2)
            fields[key] = value
            active_key = key
            continue

        # Continuation line belonging to the most recent key (stack traces and
        # multi-line stream values arrive as plain text with no prefix).
        if active_key is not None:
            existing = fields.get(active_key, "")
            # Avoid a stray leading newline when the key's initial value was empty
            # (e.g. `stream=` followed by content on the next line).
            fields[active_key] = f"{existing}\n{line}" if existing else line

    return cases


def _finalize(cases: list[InstrumentationTestCase], fields: dict[str, str], code: int) -> None:
    if code == _STATUS_CODE_START:
        return  # outcome arrives in a later block with the same class/test
    class_name = fields.get("class") or "UnknownClass"
    name = fields.get("test") or "unknownTest"
    tc = InstrumentationTestCase(class_name, name)
    stack = fields.get("stack")
    if stack is not None:
        tc.stack = stack.strip() or None
    if code == _STATUS_CODE_OK:
        tc.status = "passed"
    elif code == _STATUS_CODE_FAILURE:
        tc.status = "failed"
    elif code == _STATUS_CODE_ERROR:
        tc.status = "errored"
    elif code == _STATUS_CODE_IGNORED:
        tc.status = "skipped"
    else:
        tc.status = "errored"
        if not tc.stack:
            tc.stack = f"Unknown INSTRUMENTATION_STATUS_CODE: {code}"
    cases.append(tc)


def build_xml(cases: list[InstrumentationTestCase], suite_name: str) -> ET.Element:
    tests = len(cases)
    failures = sum(c.status == "failed" for c in cases)
    errors = sum(c.status == "errored" for c in cases)
    skipped = sum(c.status == "skipped" for c in cases)

    attrs = {
        "name": suite_name,
        "tests": str(tests),
        "failures": str(failures),
        "errors": str(errors),
        "skipped": str(skipped),
        "time": "0.",
    }
    suites = ET.Element("testsuites", attrs)
    suite = ET.SubElement(suites, "testsuite", attrs)
    for c in cases:
        suite.append(c.to_xml())
    return suites


def main(log_path: Path, suite_name: str) -> None:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    cases = parse(text)
    xml = build_xml(cases, suite_name)
    ET.indent(xml)
    xml_bytes: bytes = ET.tostring(xml, encoding="UTF-8", xml_declaration=True)
    sys.stdout.write(xml_bytes.decode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert `am instrument -w -r` output to JUnit XML on stdout.")
    parser.add_argument("log_path", type=Path, help="Path to raw instrumentation output.")
    parser.add_argument("--suite-name", default="test_aar", help="Name to embed in <testsuite>/<testsuites>.")
    args = parser.parse_args()

    main(args.log_path, args.suite_name)
