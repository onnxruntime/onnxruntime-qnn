#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Lintrunner adapter: flag includes of private ORT Core headers in QNN EP source,
and flag unapproved include directories in the QNN EP cmake target.

Three checks are performed depending on file type:

C++ (.h / .cc): Any #include whose path starts with core/ but not core/providers/qnn/
is a private ORT Core header dependency. Public ORT headers (onnxruntime_c_api.h, etc.)
are always included by filename only, never via a core/ path, so this rule has no
false positives for legitimate includes.

CMake (.cmake): Every path in target_include_directories(onnxruntime_providers_qnn ...),
target_include_directories(onnxruntime_provider_test ...) must be in APPROVED_CMAKE_INCLUDES.
Any path not on that list triggers an error, forcing deliberate review before a new include
directory is accepted.

CMake (.cmake): ONNXRUNTIME_APPLICATION_INCLUDES may only be assigned via set() with
paths from APPROVED_APP_INCLUDES. Any list(APPEND ONNXRUNTIME_APPLICATION_INCLUDES ...)
call is always an error, as is any set() that introduces an unapproved path.
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

# Approved include directories per cmake target.
# Any deviation (addition or removal) requires updating this mapping after careful review.
APPROVED_CMAKE_INCLUDES: dict[str, frozenset[str]] = {
    "onnxruntime_providers_qnn": frozenset(
        {
            "${CMAKE_CURRENT_BINARY_DIR}",
            "${onnxruntime_QNN_HOME}/include/QNN",
            "${onnxruntime_QNN_HOME}/include",
            "${ONNXRUNTIME_APPLICATION_INCLUDES}",
        }
    ),
    "onnxruntime_provider_test": frozenset(
        {
            "${ONNXRUNTIME_APPLICATION_INCLUDES}",
        }
    ),
}

_CMAKE_KEYWORDS = frozenset({"PRIVATE", "PUBLIC", "INTERFACE", "BEFORE", "SYSTEM"})

# Matches the start of a target_include_directories call for any watched target.
_TID_RE = re.compile(
    r"target_include_directories\s*\(\s*(" + "|".join(re.escape(t) for t in APPROVED_CMAKE_INCLUDES) + r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# ONNXRUNTIME_APPLICATION_INCLUDES guard
# ---------------------------------------------------------------------------

_APP_INCLUDES_VAR = "ONNXRUNTIME_APPLICATION_INCLUDES"

# Approved paths that may appear inside set(ONNXRUNTIME_APPLICATION_INCLUDES ...).
# Mirrors the two branches in cmake/external/onnxruntime_prebuilt.cmake exactly.
# Update only after careful review of that file.
APPROVED_APP_INCLUDES: frozenset[str] = frozenset(
    {
        # onnxruntime_ORT_HOME (prebuilt) branch
        '"${onnxruntime_ORT_HOME}/include"',
        # Debian/Ubuntu system-package layout: headers under include/onnxruntime/
        '"${onnxruntime_ORT_HOME}/include/onnxruntime"',
        # build-from-source branch
        '"${ORT_SOURCE_DIR}/include/onnxruntime/core/session"',
        '"${ORT_SOURCE_DIR}/include/onnxruntime/core/providers/cpu"',
        # Comment tokens (stripped before token scan, kept here for documentation only)
    }
)

# Matches list(APPEND ONNXRUNTIME_APPLICATION_INCLUDES ...) — always forbidden.
_APP_INCLUDES_APPEND_RE = re.compile(
    r"\blist\s*\(\s*APPEND\s+" + re.escape(_APP_INCLUDES_VAR) + r"\b",
    re.IGNORECASE,
)

# Matches set(ONNXRUNTIME_APPLICATION_INCLUDES ...) — allowed only with approved paths.
_APP_INCLUDES_SET_RE = re.compile(
    r"\bset\s*\(\s*" + re.escape(_APP_INCLUDES_VAR) + r"\b",
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
            cmake_target = call_m.group(1)
            approved = APPROVED_CMAKE_INCLUDES[cmake_target]
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
                if token == cmake_target or token in _CMAKE_KEYWORDS:
                    continue
                if token not in approved:
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
                                f"{cmake_target}. Update APPROVED_CMAKE_INCLUDES in "
                                "qcom/linters/check_private_ort_headers.py only after "
                                "careful review."
                            ),
                        }
                    )
    except OSError as exc:
        messages.append(_io_error(path, exc))
    return messages


def check_app_includes(path: str) -> list[dict]:
    """Flag any list(APPEND ONNXRUNTIME_APPLICATION_INCLUDES ...) and any
    set(ONNXRUNTIME_APPLICATION_INCLUDES ...) that contains unapproved paths."""
    messages = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read()

        # list(APPEND ...) is always forbidden.
        for m in _APP_INCLUDES_APPEND_RE.finditer(content):
            line = content[: m.start()].count("\n") + 1
            messages.append(
                {
                    "path": path,
                    "line": line,
                    "char": None,
                    "code": LINTER_CODE,
                    "severity": "error",
                    "name": "app-includes-extended",
                    "original": None,
                    "replacement": None,
                    "description": (
                        f"list(APPEND {_APP_INCLUDES_VAR} ...) is forbidden. "
                        f"Update APPROVED_APP_INCLUDES in "
                        "qcom/linters/check_private_ort_headers.py only after "
                        "careful review of cmake/external/onnxruntime_prebuilt.cmake."
                    ),
                }
            )

        # set(...) is allowed only with approved path tokens.
        for m in _APP_INCLUDES_SET_RE.finditer(content):
            call_line = content[: m.start()].count("\n") + 1
            open_pos = content.index("(", m.start())
            depth, i = 1, open_pos + 1
            while i < len(content) and depth:
                if content[i] == "(":
                    depth += 1
                elif content[i] == ")":
                    depth -= 1
                i += 1
            args_text = content[open_pos + 1 : i - 1]

            # Re-tokenize preserving quoted strings so paths with spaces stay intact.
            for tok_m in re.finditer(r'"[^"]*"|\S+', _strip_cmake_comments(args_text)):
                token = tok_m.group()
                if token == _APP_INCLUDES_VAR:
                    continue
                if token not in APPROVED_APP_INCLUDES:
                    tok_pos = content.find(token, open_pos)
                    tok_line = content[:tok_pos].count("\n") + 1 if tok_pos >= 0 else call_line
                    messages.append(
                        {
                            "path": path,
                            "line": tok_line,
                            "char": None,
                            "code": LINTER_CODE,
                            "severity": "error",
                            "name": "unapproved-app-include",
                            "original": None,
                            "replacement": None,
                            "description": (
                                f"Unapproved path {token!r} in "
                                f"set({_APP_INCLUDES_VAR} ...). "
                                "Update APPROVED_APP_INCLUDES in "
                                "qcom/linters/check_private_ort_headers.py only after "
                                "careful review of cmake/external/onnxruntime_prebuilt.cmake."
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
            msgs = check_cmake_file(path) + check_app_includes(path)
        else:
            msgs = check_cc_file(path)
        for msg in msgs:
            print(json.dumps(msg), flush=True)


if __name__ == "__main__":
    main()
