#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Compute the minimum ORT API version QNN EP requires.

Walks the ORT C API headers for ``\\since Version 1.X`` annotations on every
function-pointer / status-returning declaration, scans QNN EP source for every
call into one of those, and emits the maximum.

Modes:
  --write-header PATH      Write a C header defining QNN_EP_MIN_ORT_API_VERSION
                           to the computed value. Used by CMake at configure
                           time so the build always picks up the correct floor.
  --check                  Compare the computed value against a baseline file
                           and exit non-zero on mismatch. Used in lint to make
                           floor bumps a visible, reviewable PR diff. Pair with
                           --baseline FILE.
  --update-baseline        Rewrite the baseline file to match the computed
                           value. Dev convenience for legitimate floor bumps.
  --fetch-from-deps-txt    If no header root is available (e.g., lint on a
                           clean checkout with no build tree), download
                           ort_core from cmake/deps.txt, verify its SHA1, and
                           use the archive's include/ subtree. Cached.
  --lintrunner             Run --check and emit a lintrunner JSON finding on
                           stdout instead of a non-zero exit, always exiting 0.
                           Drift (exit 1) becomes an ``ort-api-floor-drift``
                           finding; cannot-compute (exit 2) becomes an
                           ``ort-api-floor-check-error`` finding so reviewers
                           can tell an environment problem apart from a real
                           baseline-refresh request. Trailing file paths
                           (lintrunner convention) are accepted and ignored;
                           the check is project-wide.

Exit codes:
  0  Success: floor printed/written, or --check found baseline matches.
  1  Drift detected by --check: computed floor disagrees with baseline. The
     baseline must be refreshed in the same PR so the bump is reviewable.
  2  Cannot compute or configuration error: missing/unreadable headers,
     missing --baseline file, SHA1 mismatch on fetch, fetch I/O failure,
     unknown API form, EP source not found. NOT drift — caller (e.g., the
     lint adapter) should surface this as an environment problem, not as a
     baseline-refresh request.

Limitations: scans direct API calls (``ort_api->X``, ``ep_api.X`` etc.) and resolves
``Ort::`` C++ wrapper method calls via a hardcoded mapping to their backing C APIs.
Unknown wrapper methods trigger a hard error (tripwire) to prevent silent blind spots.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import pathlib
import re
import shutil
import ssl
import sys
import tempfile
import urllib.request
import zipfile

# Lintrunner integration: in --lintrunner mode the script emits a lintrunner
# JSON finding instead of exiting non-zero, routing the --check exit codes to
# distinct finding names (drift vs cannot-compute) and always exiting 0 so
# lintrunner does not treat the check as a crashed adapter.
LINTER_CODE = "MIN-ORT-API-VERSION"


def _lint_json(description: str, name: str, path: str) -> dict:
    return {
        "path": path,
        "line": None,
        "char": None,
        "code": LINTER_CODE,
        "severity": "error",
        "name": name,
        "original": None,
        "replacement": None,
        "description": description,
    }


# ---------------------------------------------------------------------------
# Header parsing: build {api_member: minor_since_version}
# ---------------------------------------------------------------------------

_SINCE_RE = re.compile(r"\\since\s+Version\s+1\.(\d+)")

# Every form ORT uses to declare a versioned member of OrtApi / OrtEpApi /
# OrtModelEditorApi / OrtCompileApi. Extend if upstream adds a new macro.
_DECL_RE = re.compile(
    r"(?:"
    r"[\w\s\*]+\(\s*ORT_API_CALL\s*\*?\s*(\w+)\s*\)"  # [const] RetType[*](ORT_API_CALL* Name)
    r"|ORT_API2_STATUS\s*\(\s*(\w+)\s*[,)]"  # ORT_API2_STATUS(Name, ...)
    r"|ORT_API_STATUS\s*\(\s*(\w+)\s*[,)]"  # ORT_API_STATUS(Name, ...)
    r"|ORT_API_T\s*\(\s*[^,]+,\s*(\w+)\s*[,)]"  # ORT_API_T(rettype, Name, ...)
    r"|ORT_CLASS_RELEASE\s*\(\s*(\w+)\s*\)"  # ORT_CLASS_RELEASE(Foo) -> ReleaseFoo
    r")"
)

# Bare ORT_CLASS_RELEASE entries that appear outside /** */ doc blocks.
_CLASS_RELEASE_RE = re.compile(r"\bORT_CLASS_RELEASE\s*\(\s*(\w+)\s*\)")


def build_since_map(header_root: pathlib.Path) -> dict[str, int]:
    """Walk every .h under header_root, pair each /** ... */ doc block with the
    next declaration that follows it, and record the highest version found per
    member name. Members without a \\since are treated as version 0 (predate the
    annotation convention)."""
    result: dict[str, int] = {}
    for path in header_root.rglob("*.h"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        pos = 0
        while True:
            cstart = text.find("/*", pos)
            if cstart < 0:
                break
            cend = text.find("*/", cstart)
            if cend < 0:
                break
            block = text[cstart : cend + 2]
            m_since = _SINCE_RE.search(block)
            rest = text[cend + 2 :]
            next_block = rest.find("/*")
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
        # Second pass: collect bare ORT_CLASS_RELEASE entries not inside /** */ blocks.
        # These produce Release<X> names at version 0 (predate the \since convention).
        for m_rel in _CLASS_RELEASE_RE.finditer(text):
            name = "Release" + m_rel.group(1)
            if name not in result:
                result[name] = 0
    return result


# ---------------------------------------------------------------------------
# EP source scanning: collect every API member name actually called
# ---------------------------------------------------------------------------

_CALL_RE = re.compile(r"\b(?:ort_api|ep_api|model_editor_api|compile_api)\s*[.>\-]+\s*([A-Z]\w+)\s*\(")

# Mapping from Ort:: C++ wrapper method names to their backing C API member(s).
# Verified against onnxruntime_cxx_inline.h. Ambiguous names (e.g. GetName used
# on Node, Graph, and ValueInfo) list all possible backing APIs — floor takes
# max(\since) which is the safe (over-estimate) direction.
_WRAPPER_TO_C_API: dict[str, list[str]] = {
    "GetId": ["Node_GetId"],
    "GetName": ["Node_GetName", "Graph_GetName", "GetValueInfoName"],
    "GetDomain": ["Node_GetDomain"],
    "GetOperatorType": ["Node_GetOperatorType"],
    "GetSinceVersion": ["Node_GetSinceVersion"],
    "GetAttributeByName": ["Node_GetAttributeByName"],
    "GetInputs": ["Node_GetInputs", "Graph_GetInputs"],
    "GetOutputs": ["Node_GetOutputs", "Graph_GetOutputs"],
    "GetImplicitInputs": ["Node_GetImplicitInputs"],
    "GetAttributes": ["Node_GetAttributes"],
    "GetSubgraphs": ["Node_GetSubgraphs"],
    "GetGraph": ["Node_GetGraph"],
    "GetNodes": ["Graph_GetNodes"],
    "GetInitializers": ["Graph_GetInitializers"],
    "GetConsumers": ["ValueInfo_GetValueConsumers"],
    "GetProducerNode": ["ValueInfo_GetValueProducer"],
    "GetInitializerValue": ["ValueInfo_GetInitializerValue"],
    "IsConstantInitializer": ["ValueInfo_IsConstantInitializer"],
    "IsFromOuterScope": ["ValueInfo_IsFromOuterScope"],
    "IsRequiredGraphInput": ["ValueInfo_IsRequiredGraphInput"],
    "IsOptionalGraphInput": ["ValueInfo_IsOptionalGraphInput"],
    "IsGraphOutput": ["ValueInfo_IsGraphOutput"],
    "TypeInfo": ["GetValueInfoTypeInfo"],
    "GetTensorTypeAndShapeInfo": ["CastTypeInfoToTensorInfo", "GetTensorTypeAndShape"],
    "GetElementType": ["GetTensorElementType"],
    "GetShape": ["GetDimensions"],
    "GetElementCount": ["GetTensorShapeElementCount"],
    "GetDimensionsCount": ["GetDimensionsCount"],
    "IsOK": [],
}

_WRAPPER_METHOD_RE = re.compile(r"\.\s*(" + "|".join(re.escape(k) for k in _WRAPPER_TO_C_API) + r")\s*\(")

# Matches Ort::Const<Type>(x).Method( or variable.Method( where the variable
# was declared as an Ort:: wrapper type. We use this to detect unknown methods.
_ORT_WRAPPER_CALL_RE = re.compile(
    r"Ort::(?:Const)?\w+(?:<[^>]*>)?\s*(?:\([^)]*\)|&?\s*\w+)\s*[.)]\s*\.?\s*([A-Z]\w+)\s*\("
)


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
            if "Ort::Const" in text:
                for m in _WRAPPER_METHOD_RE.finditer(text):
                    method = m.group(1)
                    for c_api in _WRAPPER_TO_C_API[method]:
                        names.add(c_api)
                for m in _ORT_WRAPPER_CALL_RE.finditer(text):
                    method = m.group(1)
                    if method not in _WRAPPER_TO_C_API:
                        raise RuntimeError(
                            f"Unknown Ort:: wrapper method '{method}' in {path.relative_to(ep_root)}. "
                            "Update _WRAPPER_TO_C_API in compute_min_ort_api_version.py."
                        )
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
    unknown = sorted(n for n in used if n not in since_map)
    if unknown:
        raise RuntimeError(
            f"EP source calls API methods not found in \\since map: {unknown}. "
            "Either _DECL_RE missed a declaration form, or these come from "
            "headers outside --ort-header-root."
        )
    version_zero = sorted(n for n in used if since_map[n] == 0)
    if version_zero:
        print(
            f"warning: {len(version_zero)} EP-used API member(s) resolved at version 0 "
            f"(no \\since annotation): {version_zero[:5]}{'...' if len(version_zero) > 5 else ''}",
            file=sys.stderr,
        )
    return max(since_map[n] for n in used)


def find_ort_header_root(repo_root: pathlib.Path) -> pathlib.Path | None:
    """Walk known build output locations for the ORT core source bundle."""
    for build_dir in sorted(repo_root.glob("build/*/")):
        for config in ("Release", "Debug", "RelWithDebInfo"):
            candidate = build_dir / config / "_deps" / "ort_core-src" / "include"
            if candidate.is_dir():
                return candidate
    return None


# ---------------------------------------------------------------------------
# Fetch ORT headers directly from the cmake/deps.txt pin so lint works on a
# clean checkout (CI lint runner, no build tree). Cached by SHA1 under a user
# cache dir; downloaded at most once per pin.
# ---------------------------------------------------------------------------


def parse_deps_txt(deps_path: pathlib.Path, name: str) -> tuple[str, str]:
    """Return (url, sha1) for `name` from cmake/deps.txt. Raises if missing."""
    for row in csv.reader(deps_path.read_text(encoding="utf-8").splitlines(), delimiter=";"):
        if row and not row[0].startswith("#") and row[0].strip() == name:
            return row[1].strip(), row[2].strip()
    raise RuntimeError(f"{name} not found in {deps_path}")


def _cache_root() -> pathlib.Path:
    env = os.environ.get("QNN_EP_LINT_CACHE")
    if env:
        return pathlib.Path(env)
    if sys.platform == "win32":
        base = pathlib.Path(os.environ.get("LOCALAPPDATA", pathlib.Path.home() / "AppData" / "Local"))
    else:
        base = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache"))
    return base / "qnn-ep-min-ort"


def fetch_ort_headers(deps_path: pathlib.Path, cache_root: pathlib.Path | None = None) -> pathlib.Path:
    """Download and extract the ort_core archive pinned in cmake/deps.txt; return
    the include/ subtree. Cache keyed by SHA1 so a verified extract is reused
    across runs. Set QNN_EP_LINT_CACHE to override the cache location."""
    url, sha1 = parse_deps_txt(deps_path, "ort_core")
    root = (cache_root or _cache_root()) / f"ort_core-{sha1}"
    include_dir = root / "include"
    if include_dir.is_dir():
        return include_dir

    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ort_core-dl-", dir=str(root)) as tmp:
        tmp_path = pathlib.Path(tmp)
        archive = tmp_path / "ort_core.zip"
        with (
            urllib.request.urlopen(
                url, context=ssl.create_default_context(cafile=__import__("certifi").where())
            ) as resp,
            open(archive, "wb") as out,
        ):
            shutil.copyfileobj(resp, out)
        digest = hashlib.sha1(archive.read_bytes()).hexdigest()
        if digest.lower() != sha1.lower():
            raise RuntimeError(f"ort_core sha1 mismatch: expected {sha1}, got {digest}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp_path)
        # GitHub source archives extract to <repo>-<sha>/ — find that directory.
        extracted = next((p for p in tmp_path.iterdir() if p.is_dir() and (p / "include").is_dir()), None)
        if extracted is None:
            raise RuntimeError(f"ort_core archive has no include/ subtree at {url}")
        # Move just the include tree into the stable cache slot.
        shutil.move(str(extracted / "include"), str(include_dir))
    return include_dir


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
    parser = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars="@")
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
    parser.add_argument(
        "--fetch-from-deps-txt",
        action="store_true",
        help="If no header root is given/found, download ort_core from cmake/deps.txt and use its include/ tree. Caches by SHA1 under $QNN_EP_LINT_CACHE or the platform user cache dir.",
    )
    parser.add_argument(
        "--lintrunner",
        action="store_true",
        help="Emit a lintrunner JSON finding instead of a non-zero exit; implies --check and always exits 0.",
    )
    parser.add_argument("filenames", nargs="*", help="Ignored; accepted for lintrunner @{{PATHSFILE}} convention")
    args = parser.parse_args()

    if args.lintrunner:
        return run_lintrunner(args)

    return run(args)


def run_lintrunner(args: argparse.Namespace) -> int:
    """Wrap run() with --check semantics, translating its exit code into a
    lintrunner JSON finding on stdout. Always returns 0 so lintrunner treats a
    real drift / environment failure as a finding, not as a crashed adapter."""
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    args.check = True
    if args.baseline is None:
        args.baseline = repo_root / "qcom" / "MIN_ORT_API_VERSION.txt"
    if not args.fetch_from_deps_txt and args.ort_header_root is None:
        args.fetch_from_deps_txt = True
    try:
        baseline_path = os.path.relpath(args.baseline, repo_root)
    except ValueError:
        baseline_path = str(args.baseline)

    buf = io.StringIO()
    with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
        code = run(args)
    captured = buf.getvalue().strip()

    if code == 0:
        return 0
    if code == 1:
        name = "ort-api-floor-drift"
        description = captured or "ORT API floor drift"
    else:
        name = "ort-api-floor-check-error"
        description = captured or f"cannot compute ORT API floor (exit {code})"
    print(json.dumps(_lint_json(description, name=name, path=baseline_path)), flush=True)
    return 0


def run(args: argparse.Namespace) -> int:
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    ort_header_root = args.ort_header_root or find_ort_header_root(repo_root)
    ep_source_root = args.ep_source_root or (repo_root / "onnxruntime" / "core" / "providers" / "qnn")

    if (ort_header_root is None or not ort_header_root.is_dir()) and args.fetch_from_deps_txt:
        try:
            ort_header_root = fetch_ort_headers(repo_root / "cmake" / "deps.txt")
        except (OSError, RuntimeError) as exc:
            print(f"error: failed to fetch ort_core headers: {exc}", file=sys.stderr)
            return 2

    if ort_header_root is None or not ort_header_root.is_dir():
        print(
            "error: ORT header root not found. Run a build first so ort_core is fetched, "
            "or pass --ort-header-root explicitly, or re-run with --fetch-from-deps-txt.",
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
