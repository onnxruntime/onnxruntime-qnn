#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Lintrunner adapter: flag includes of private ORT Core headers in QNN EP source,
and flag unapproved include directories in the QNN EP cmake target.

Two checks are performed depending on file type:

C++ (.h / .cc): Any #include whose path starts with core/ but not core/providers/qnn/
is a private ORT Core header dependency. Public ORT headers (onnxruntime_c_api.h, etc.)
are always included by filename only, never via a core/ path, so this rule has no
false positives for legitimate includes.

CMake (.cmake): Every path in target_include_directories(onnxruntime_providers_qnn ...)
must be in APPROVED_CMAKE_INCLUDES. Any path not on that list triggers an error,
forcing deliberate review before a new include directory is accepted.
"""

from __future__ import annotations

import argparse
import json
import os
import re

LINTER_CODE = "PRIVATE-ORT-HEADERS"

# ---------------------------------------------------------------------------
# C++ check
# ---------------------------------------------------------------------------

# Matches any #include of a core/ path that is NOT the QNN EP's own code.
# Anchored to start-of-line so commented-out includes (// #include ...) are not flagged.
_FORBIDDEN = re.compile(r'^\s*#\s*include\s*["<]core/(?!providers/qnn/)', re.MULTILINE)


def check_cc_file(path: str) -> list[dict]:
    messages = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, start=1):
                if _FORBIDDEN.search(line):
                    messages.append(
                        {
                            "path": path,
                            "line": lineno,
                            "char": None,
                            "code": LINTER_CODE,
                            "severity": "error",
                            "name": "private-ort-header",
                            "original": None,
                            "replacement": None,
                            "description": (
                                f"Private ORT Core header: {line.strip()!r}. "
                                "Copy the required code into onnxruntime/core/providers/qnn/common/ instead."
                            ),
                        }
                    )
    except OSError as exc:
        messages.append(_io_error(path, exc))
    return messages


# ---------------------------------------------------------------------------
# CMake check
# ---------------------------------------------------------------------------

# Exact set of include directories approved for onnxruntime_providers_qnn.
# Any deviation (addition or removal) requires updating this list after careful review.
APPROVED_CMAKE_INCLUDES: frozenset[str] = frozenset(
    {
        "${CMAKE_CURRENT_BINARY_DIR}",
        "${ONNXRUNTIME_APPLICATION_INCLUDE_ROOT}/core/session",
        "${onnxruntime_QNN_HOME}/include/QNN",
        "${onnxruntime_QNN_HOME}/include",
    }
)

_CMAKE_TARGET = "onnxruntime_providers_qnn"
_CMAKE_KEYWORDS = frozenset({"PRIVATE", "PUBLIC", "INTERFACE", "BEFORE", "SYSTEM"})

# Matches the start of a target_include_directories call for our target.
_TID_RE = re.compile(
    r"target_include_directories\s*\(\s*" + re.escape(_CMAKE_TARGET) + r"\b",
    re.IGNORECASE,
)


def _strip_cmake_comments(text: str) -> str:
    return re.sub(r"#[^\n]*", "", text)


def check_cmake_file(path: str) -> list[dict]:
    messages = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read()

        for call_m in _TID_RE.finditer(content):
            call_line = content[: call_m.start()].count("\n") + 1

            # Find the matching closing paren using a depth counter.
            open_pos = content.index("(", call_m.start())
            depth, i = 1, open_pos + 1
            while i < len(content) and depth:
                if content[i] == "(":
                    depth += 1
                elif content[i] == ")":
                    depth -= 1
                i += 1
            args_text = content[open_pos + 1 : i - 1]

            for token in _strip_cmake_comments(args_text).split():
                if token == _CMAKE_TARGET or token in _CMAKE_KEYWORDS:
                    continue
                if token not in APPROVED_CMAKE_INCLUDES:
                    # Find line number of this specific token within the call.
                    tok_pos = content.find(token, open_pos)
                    tok_line = content[:tok_pos].count("\n") + 1 if tok_pos >= 0 else call_line
                    messages.append(
                        {
                            "path": path,
                            "line": tok_line,
                            "char": None,
                            "code": LINTER_CODE,
                            "severity": "error",
                            "name": "unapproved-cmake-include",
                            "original": None,
                            "replacement": None,
                            "description": (
                                f"Unapproved include directory {token!r} added to "
                                f"{_CMAKE_TARGET}. Update APPROVED_CMAKE_INCLUDES in "
                                "qcom/linters/check_private_ort_headers.py only after "
                                "careful review."
                            ),
                        }
                    )
    except OSError as exc:
        messages.append(_io_error(path, exc))
    return messages


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _io_error(path: str, exc: OSError) -> dict:
    return {
        "path": path,
        "line": None,
        "char": None,
        "code": LINTER_CODE,
        "severity": "error",
        "name": "io-error",
        "original": None,
        "replacement": None,
        "description": str(exc),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flag private ORT Core header includes and unapproved cmake include dirs.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args()
    for path in args.filenames:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".cmake" or os.path.basename(path) == "CMakeLists.txt":
            msgs = check_cmake_file(path)
        else:
            msgs = check_cc_file(path)
        for msg in msgs:
            print(json.dumps(msg), flush=True)


if __name__ == "__main__":
    main()
