#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Compute the minimum ORT API version QNN EP requires.

Walks the ORT C API headers for ``\\since Version 1.X`` annotations on every
function-pointer / status-returning declaration, scans QNN EP source for every
call into one of those, and emits the maximum.

Modes:
  --write-header PATH  Write a C header defining QNN_EP_MIN_ORT_API_VERSION to
                       the computed value. Used by CMake at configure time so
                       the build always picks up the correct floor.
  --check              Compare the computed value against a baseline file and
                       exit non-zero on mismatch. Used in lint to make floor
                       bumps a visible, reviewable PR diff. Pair with
                       --baseline FILE.
  --update-baseline    Rewrite the baseline file to match the computed value.
                       Dev convenience for legitimate floor bumps.

Limitations: scans direct API calls (``ort_api->X``, ``ep_api.X`` etc.). Calls
through ``Ort::`` C++ wrappers in ``onnxruntime_cxx_api.h`` are not seen here;
if those become a concern, extend the patterns or wrap them.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys

# ---------------------------------------------------------------------------
# Header parsing: build {api_member: minor_since_version}
# ---------------------------------------------------------------------------

_SINCE_RE = re.compile(r"\\since\s+Version\s+1\.(\d+)")

# Every form ORT uses to declare a versioned member of OrtApi / OrtEpApi /
# OrtModelEditorApi / OrtCompileApi. Extend if upstream adds a new macro.
_DECL_RE = re.compile(
    r"(?:"
    r"\w+\s*\(\s*ORT_API_CALL\s*\*?\s*(\w+)\s*\)"  # RetType(ORT_API_CALL* Name)
    r"|ORT_API2_STATUS\s*\(\s*(\w+)\s*[,)]"  # ORT_API2_STATUS(Name, ...)
    r"|ORT_API_STATUS\s*\(\s*(\w+)\s*[,)]"  # ORT_API_STATUS(Name, ...)
    r"|ORT_API_T\s*\(\s*[^,]+,\s*(\w+)\s*[,)]"  # ORT_API_T(rettype, Name, ...)
    r"|ORT_CLASS_RELEASE\s*\(\s*(\w+)\s*\)"  # ORT_CLASS_RELEASE(Foo) -> ReleaseFoo
    r")"
)


def build_since_map(header_root: pathlib.Path) -> dict[str, int]:
    """Walk every .h under header_root, pair each /** ... */ doc block with the
    next declaration that follows it, and record the highest version found per
    member name. Members without a \\since are treated as 1.0 (predate the
    annotation convention)."""
    result: dict[str, int] = {}
    for path in header_root.rglob("*.h"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        pos = 0
        while True:
            cstart = text.find("/**", pos)
            if cstart < 0:
                break
            cend = text.find("*/", cstart)
            if cend < 0:
                break
            block = text[cstart : cend + 2]
            m_since = _SINCE_RE.search(block)
            rest = text[cend + 2 :]
            next_block = rest.find("/**")
            scope = rest if next_block < 0 else rest[:next_block]
            m_decl = _DECL_RE.search(scope)
            if m_decl:
                name = next((g for g in m_decl.groups() if g), None)
                if name:
                    if "ORT_CLASS_RELEASE" in m_decl.group(0):
                        name = "Release" + name
                    version = int(m_since.group(1)) if m_since else 0
                    if name not in result or result[name] < version:
                        result[name] = version
            pos = cend + 2
    return result


# ---------------------------------------------------------------------------
# EP source scanning: collect every API member name actually called
# ---------------------------------------------------------------------------

_CALL_RE = re.compile(r"\b(?:ort_api|ep_api|model_editor_api|compile_api)\s*[.>\-]+\s*([A-Z]\w+)\s*\(")


def scan_ep_source(ep_root: pathlib.Path) -> set[str]:
    names: set[str] = set()
    for ext in ("*.cc", "*.h", "*.cpp", "*.hpp"):
        for path in ep_root.rglob(ext):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for m in _CALL_RE.finditer(text):
                names.add(m.group(1))
    return names


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def compute_floor(ort_header_root: pathlib.Path, ep_source_root: pathlib.Path) -> int:
    since_map = build_since_map(ort_header_root)
    if not since_map:
        raise RuntimeError(
            f"No ORT API declarations found under {ort_header_root}. "
            "Run a build first so the ORT headers are present, or pass a "
            "different --ort-header-root."
        )
    used = scan_ep_source(ep_source_root)
    if not used:
        raise RuntimeError(f"No ORT API calls found under {ep_source_root}. Path looks wrong.")
    versions = [since_map[n] for n in used if n in since_map]
    if not versions:
        raise RuntimeError(
            "EP source references API methods that are not in the \\since map. "
            "Either the regex missed a declaration form (check qcom/scripts/all/"
            "compute_min_ort_api_version.py) or those methods come from a header "
            "outside --ort-header-root."
        )
    return max(versions)


def find_ort_header_root(repo_root: pathlib.Path) -> pathlib.Path | None:
    """Walk known build output locations for the ORT core source bundle."""
    for build_dir in sorted(repo_root.glob("build/*/")):
        for config in ("Release", "Debug", "RelWithDebInfo"):
            candidate = build_dir / config / "_deps" / "ort_core-src" / "include"
            if candidate.is_dir():
                return candidate
    return None


def write_header(out_path: pathlib.Path, value: int) -> None:
    content = (
        "// AUTOGENERATED by qcom/scripts/all/compute_min_ort_api_version.py.\n"
        "// Do not edit by hand. CMake regenerates this on every configure.\n"
        "#pragma once\n"
        f"#define QNN_EP_MIN_ORT_API_VERSION {value}\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.is_file() and out_path.read_text() == content:
        return  # avoid touching mtime when unchanged
    out_path.write_text(content)


def read_baseline(baseline_path: pathlib.Path) -> int | None:
    if not baseline_path.is_file():
        return None
    raw = baseline_path.read_text().strip()
    if not raw.isdigit():
        return None
    return int(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ort-header-root",
        type=pathlib.Path,
        default=None,
        help="Root of ORT C API headers (default: search build/*/Release/_deps/ort_core-src/include)",
    )
    parser.add_argument(
        "--ep-source-root",
        type=pathlib.Path,
        default=None,
        help="Root of QNN EP source (default: <repo>/onnxruntime/core/providers/qnn)",
    )
    parser.add_argument("--write-header", type=pathlib.Path, help="Write header to this path")
    parser.add_argument("--check", action="store_true", help="Compare against --baseline; exit 1 on drift")
    parser.add_argument("--update-baseline", action="store_true", help="Rewrite --baseline to match computed value")
    parser.add_argument("--baseline", type=pathlib.Path, help="Baseline file (one integer)")
    parser.add_argument("--print", action="store_true", help="Print computed floor and exit")
    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parents[3]
    ort_header_root = args.ort_header_root or find_ort_header_root(repo_root)
    ep_source_root = args.ep_source_root or (repo_root / "onnxruntime" / "core" / "providers" / "qnn")

    if ort_header_root is None or not ort_header_root.is_dir():
        print(
            "error: ORT header root not found. Run a build first so ort_core is fetched, "
            "or pass --ort-header-root explicitly.",
            file=sys.stderr,
        )
        return 2
    if not ep_source_root.is_dir():
        print(f"error: EP source root not found at {ep_source_root}", file=sys.stderr)
        return 2

    try:
        floor = compute_floor(ort_header_root, ep_source_root)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.print:
        print(floor)
        return 0

    if args.write_header:
        write_header(args.write_header, floor)

    if args.update_baseline:
        if not args.baseline:
            print("error: --update-baseline requires --baseline", file=sys.stderr)
            return 2
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(f"{floor}\n")
        print(f"baseline updated: {args.baseline} -> {floor}")
        return 0

    if args.check:
        if not args.baseline:
            print("error: --check requires --baseline", file=sys.stderr)
            return 2
        baseline = read_baseline(args.baseline)
        if baseline is None:
            print(
                f"error: baseline {args.baseline} missing or not an integer. Run with --update-baseline to seed it.",
                file=sys.stderr,
            )
            return 2
        if baseline != floor:
            try:
                baseline_for_msg = args.baseline.relative_to(repo_root)
            except ValueError:
                baseline_for_msg = args.baseline
            print(
                f"error: ORT API floor drift. Baseline {args.baseline} says {baseline}, "
                f"computed {floor}.\n"
                "  - If the bump is intentional (the EP started using a newer API), run:\n"
                f"      python {os.path.relpath(__file__, repo_root)} "
                f"--update-baseline --baseline {baseline_for_msg}\n"
                "    and commit the result so the floor change appears in the PR diff.\n"
                "  - If unintentional, revert the call site that added the new API.",
                file=sys.stderr,
            )
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
