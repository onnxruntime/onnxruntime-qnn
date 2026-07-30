#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
#
# Accuracy-routing gate for QNN EP unit tests (PR4).
#
# Decides which QnnUnit_Accuracy_* cases the coverage run must execute, given
# the snapshot-phase results and the golden store's version manifest. The core
# invariant: a FIXED QNN backend version means an unchanged graph structure
# means unchanged numerics -- so accuracy may be skipped for a case ONLY when
# both hold: (1) its paired snapshot passed (structure unchanged) AND (2) the
# golden store's recorded QAIRT version matches the current one (backend
# unchanged). Any other situation runs accuracy.
#
# Two levels:
#   Level 0 (global): golden manifest version != current QAIRT version, or no
#       manifest at all -> run ALL accuracy (version-mismatch). Snapshot still
#       ran fully upstream; the version conjunct is a must-run reason of equal
#       standing to drift.
#   Level 1 (per-case, only when Level 0 matches): for each accuracy case find
#       its paired snapshot case (mechanical suite-prefix swap, same
#       Case/<name>). Run if the snapshot drifted (failed), was skipped
#       (golden absent / backend unavailable -> no PASS proof), or has no
#       paired snapshot case at all (unmapped -> union safety). Skip only when
#       the paired snapshot passed.
#
# Reasons are collected as a UNION with no short-circuit: a case runs iff its
# reason set is non-empty. This keeps the developer-facing drift summary honest
# (drift is reported even when version-mismatch already forced a full run).
#
# Pure functions + a thin CLI so the decision logic is unit-tested with
# synthetic fixtures (see tests/test_accuracy_gate.py) -- no binary, no build.

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field

# gtest full name = "<suite>.<name>", e.g.
#   QnnUnit_Snapshot_ClipPlainTest.Case/Clip_f32
# The snapshot->accuracy mapping swaps the tier token in the suite and keeps
# Case/<name> verbatim (see clip_specs.h: names are aligned by construction).
_SNAPSHOT_TIER_RE = re.compile(r"_(?:Snapshot|SessionSnapshot)_")

# Snapshot status buckets.
PASSED = "PASSED"
DRIFT = "DRIFT"  # golden byte-mismatch -> structure changed
SKIPPED = "SKIPPED"  # golden absent OR backend unavailable -> no PASS proof

# Per-case run reasons (union; empty set == skip).
R_DRIFT = "drift"
R_ABSENT = "absent"  # paired snapshot skipped (golden absent / backend down)
R_UNMAPPED = "unmapped"  # no paired snapshot case exists
R_VERSION = "version-mismatch"  # Level-0 global override


@dataclass
class CaseDecision:
    suite: str
    name: str
    run: bool
    reasons: set[str] = field(default_factory=set)

    @property
    def full_name(self) -> str:
        return f"{self.suite}.{self.name}"


@dataclass(frozen=True)
class ToolVersions:
    """The (QAIRT backend, ORT runtime) version pair that together fix numerics.

    The skip invariant is "FIXED backend AND runtime => structure unchanged =>
    numerics unchanged". Both axes are load-bearing: a QAIRT up-level can change
    backend numerics, and an ORT up-level can change graph pre-processing / the
    live reference the accuracy tier compares against -- either can shift numbers
    even when the emitted graph structure (what snapshot checks) is byte-identical.
    So the golden store's manifest stamps BOTH, and a match requires BOTH equal.
    """

    qairt: str | None = None
    ort: str | None = None

    def complete(self) -> bool:
        return self.qairt is not None and self.ort is not None

    def matches(self, other: ToolVersions) -> bool:
        # A missing field on either side -> cannot prove that axis is unchanged
        # -> not a match -> full accuracy run (safe).
        return self.complete() and other.complete() and self == other


@dataclass
class GateResult:
    run_set: list[CaseDecision]
    skip_set: list[CaseDecision]
    drift_cases: list[tuple[str, str]]  # (suite, name) of snapshot cases that drifted
    current: ToolVersions
    manifest: ToolVersions
    version_match: bool


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def classify_testcase(tc: dict) -> str:
    """Bucket one gtest JSON testcase into PASSED / DRIFT / SKIPPED.

    Schema (locked against a real run):
      PASSED : result == "COMPLETED", no "failures" key.
      DRIFT  : has non-empty "failures" (result is still "COMPLETED" -- do NOT
               rely on result alone).
      SKIPPED: result == "SKIPPED" (carries a "skipped" message), or the test
               did not run (status "NOTRUN") -- either way, no PASS proof.
    """
    if tc.get("failures"):
        return DRIFT
    if tc.get("result") == "SKIPPED" or tc.get("status") == "NOTRUN":
        return SKIPPED
    if tc.get("result") == "COMPLETED":
        return PASSED
    # Unknown/absent result -> treat as no PASS proof (safe: run accuracy).
    return SKIPPED


def load_snapshot_results(json_path: str) -> dict[tuple[str, str], str]:
    """Parse a gtest JSON report into {(suite, name): status}.

    Keys use the snapshot suite name as-emitted; mapping to accuracy suites
    happens in compute_run_set via derive_accuracy_suite.
    """
    with open(json_path) as f:
        data = json.load(f)
    out: dict[tuple[str, str], str] = {}
    for suite in data.get("testsuites", []):
        suite_name = suite.get("name", "")
        for tc in suite.get("testsuite", []):
            name = tc.get("name", "")
            out[(suite_name, name)] = classify_testcase(tc)
    return out


def parse_accuracy_list(text: str) -> list[tuple[str, str]]:
    """Parse `--gtest_list_tests` output into [(suite, name), ...].

    gtest --gtest_list_tests format:
        QnnUnit_Accuracy_ClipPlainTest.
          Case/Clip_f32  # GetParam() = ...
          Case/Clip_int32
    A line with no leading whitespace ending in '.' is a suite; indented lines
    are cases under the most recent suite. Trailing '# ...' comments are
    stripped.
    """
    cases: list[tuple[str, str]] = []
    suite = None
    for raw in text.splitlines():
        if not raw.strip():
            continue
        if not raw[0].isspace():
            # Suite header line: "SuiteName." (may have trailing comment).
            suite = raw.strip().rstrip(".").strip()
            continue
        if suite is None:
            continue
        case = raw.strip()
        case = case.split("#", 1)[0].strip()  # drop "# GetParam() = ..."
        if case:
            cases.append((suite, case))
    return cases


def derive_accuracy_suite(snapshot_suite: str) -> str:
    """Map a snapshot suite name to its paired accuracy suite name.

    QnnUnit_Snapshot_ClipPlainTest        -> QnnUnit_Accuracy_ClipPlainTest
    QnnUnit_SessionSnapshot_ClipQDQFloat.. -> QnnUnit_Accuracy_ClipQDQFloat..
    """
    return _SNAPSHOT_TIER_RE.sub("_Accuracy_", snapshot_suite, count=1)


def load_manifest_versions(golden_root: str | None) -> ToolVersions:
    """Read qairt_version + ort_version from <golden_root>/manifest.json.

    Missing file / field / malformed json -> that axis is None (treated as
    version-mismatch by ToolVersions.matches -> safe full run).
    """
    if not golden_root:
        return ToolVersions()
    path = os.path.join(golden_root, "manifest.json")
    if not os.path.isfile(path):
        return ToolVersions()
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return ToolVersions()
    qairt = data.get("qairt_version")
    ort = data.get("ort_version")
    return ToolVersions(
        qairt=str(qairt) if qairt else None,
        ort=str(ort) if ort else None,
    )


def detect_qairt_version() -> str | None:
    """Current QAIRT version. Env override first (CI/tests inject it); else fall
    back to the QAIRT SDK's sdk.yaml. Returns None if undeterminable -- callers
    treat None like no-manifest (safe: full accuracy run)."""
    env = os.environ.get("QNN_UT_QAIRT_VERSION")
    if env:
        return env.strip()
    for var in ("QAIRT_SDK_ROOT", "QNN_SDK_ROOT", "SNPE_ROOT"):
        root = os.environ.get(var)
        if not root:
            continue
        sdk_yaml = os.path.join(root, "sdk.yaml")
        if os.path.isfile(sdk_yaml):
            try:
                with open(sdk_yaml) as f:
                    for line in f:
                        # sdk.yaml is a flat "key: value" file; the QAIRT product
                        # version is under a version-ish key. Kept intentionally
                        # loose -- pinned when a real SDK is available on CI.
                        m = re.match(r"\s*(?:version|Version)\s*:\s*(.+?)\s*$", line)
                        if m:
                            return m.group(1).strip().strip("'\"")
            except OSError:
                pass
    return None


def detect_ort_version() -> str | None:
    """Current ORT (runtime) version. Env override first (CI/tests inject it);
    else read the ORT prebuilt's VERSION_NUMBER. Returns None if undeterminable
    -- treated like a version-mismatch (safe: full accuracy run).

    NOTE: the real shared version-getter is PR2's qnn_ut_version.sh; until it
    lands, this mirrors detect_qairt_version's env-first + loose-file fallback so
    both sides can be sourced identically later."""
    env = os.environ.get("QNN_UT_ORT_VERSION")
    if env:
        return env.strip()
    root = os.environ.get("ORT_PREBUILT_ROOT")
    if root:
        for rel in ("VERSION_NUMBER", "VERSION"):
            vf = os.path.join(root, rel)
            if os.path.isfile(vf):
                try:
                    with open(vf) as f:
                        v = f.readline().strip()
                    if v:
                        return v
                except OSError:
                    pass
    return None


def detect_current_versions() -> ToolVersions:
    return ToolVersions(qairt=detect_qairt_version(), ort=detect_ort_version())


# ---------------------------------------------------------------------------
# Core decision
# ---------------------------------------------------------------------------
def compute_run_set(
    accuracy_cases: Iterable[tuple[str, str]],
    snapshot_map: dict[tuple[str, str], str],
    version_match: bool,
) -> tuple[list[CaseDecision], list[CaseDecision]]:
    """Decide run/skip per accuracy case.

    accuracy_cases : [(accuracy_suite, "Case/<name>"), ...]
    snapshot_map   : {(snapshot_suite, "Case/<name>"): status}
    version_match  : Level-0 result (False => every case runs, reason version).

    Returns (run_set, skip_set) as lists of CaseDecision.
    """
    # Re-key snapshot results by their *accuracy* suite so a case can be looked
    # up by the accuracy identity. Both Snapshot and SessionSnapshot collapse
    # onto the same accuracy suite (QDQFloat), which is correct: either source
    # passing is proof for that accuracy case.
    by_accuracy: dict[tuple[str, str], str] = {}
    for (snap_suite, name), status in snapshot_map.items():
        acc_suite = derive_accuracy_suite(snap_suite)
        key = (acc_suite, name)
        # If two sources map to the same accuracy case, a PASS wins only if the
        # other is also PASS; any non-PASS (drift/skip) forces a run. Merge by
        # keeping the "most run-worthy" status: DRIFT > SKIPPED > PASSED.
        prev = by_accuracy.get(key)
        by_accuracy[key] = _merge_status(prev, status)

    run_set: list[CaseDecision] = []
    skip_set: list[CaseDecision] = []
    for suite, name in accuracy_cases:
        reasons: set[str] = set()
        if not version_match:
            reasons.add(R_VERSION)
        status = by_accuracy.get((suite, name))
        if status is None:
            reasons.add(R_UNMAPPED)
        elif status == DRIFT:
            reasons.add(R_DRIFT)
        elif status == SKIPPED:
            reasons.add(R_ABSENT)
        # status == PASSED contributes no reason.
        decision = CaseDecision(suite=suite, name=name, run=bool(reasons), reasons=reasons)
        (run_set if decision.run else skip_set).append(decision)
    return run_set, skip_set


def _merge_status(a: str | None, b: str) -> str:
    order = {PASSED: 0, SKIPPED: 1, DRIFT: 2}
    if a is None:
        return b
    return a if order[a] >= order[b] else b


def run_gate(
    snapshot_json: str | None,
    golden_root: str | None,
    accuracy_cases: list[tuple[str, str]],
    current: ToolVersions | None = None,
) -> GateResult:
    """End-to-end gate: load inputs, resolve versions, compute decisions."""
    snapshot_map: dict[tuple[str, str], str] = {}
    if snapshot_json and os.path.isfile(snapshot_json):
        snapshot_map = load_snapshot_results(snapshot_json)

    if current is None:
        current = detect_current_versions()
    manifest = load_manifest_versions(golden_root)
    # Match iff BOTH qairt AND ort are known on both sides and equal. Any missing
    # axis / mismatch (incl. no manifest, undeterminable current) -> full run.
    version_match = current.matches(manifest)

    run_set, skip_set = compute_run_set(accuracy_cases, snapshot_map, version_match)

    drift_cases = [(suite, name) for (suite, name), status in snapshot_map.items() if status == DRIFT]
    return GateResult(
        run_set=run_set,
        skip_set=skip_set,
        drift_cases=sorted(drift_cases),
        current=current,
        manifest=manifest,
        version_match=version_match,
    )


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
# Matches no real test so an empty run-set does NOT degrade to "run everything".
_EMPTY_FILTER_SENTINEL = "QnnUnit_Accuracy_ZZZ_NoCasesSelected.None"


def build_gtest_filter(run_set: list[CaseDecision]) -> str:
    if not run_set:
        return _EMPTY_FILTER_SENTINEL
    return ":".join(d.full_name for d in run_set)


def render_summary(result: GateResult) -> str:
    lines: list[str] = []
    lines.append("=== Accuracy-routing gate summary ===")

    # (1) TOP actionable: snapshot drift, independent of version.
    if result.drift_cases:
        lines.append("")
        lines.append(f"[DRIFT] {len(result.drift_cases)} snapshot case(s) detected graph drift:")
        for suite, name in result.drift_cases:
            lines.append(f"    - {suite}.{name}")
        lines.append("  Action: run_snapshot_accuracy.sh to verify, then --update-goldens to accept.")
    else:
        lines.append("[DRIFT] none -- all snapshot goldens matched.")

    # (2) Version verdict -- both axes shown; a mismatch on EITHER forces a full run.
    lines.append("")
    verdict = "MATCH" if result.version_match else "MISMATCH/absent => full accuracy run"
    lines.append(
        f"[VERSION] qairt: current={result.current.qairt or '<unknown>'} "
        f"manifest={result.manifest.qairt or '<none>'} | "
        f"ort: current={result.current.ort or '<unknown>'} "
        f"manifest={result.manifest.ort or '<none>'} "
        f"-> {verdict}"
    )

    # (3) Routing tally.
    def _bucket(r: str) -> int:
        return sum(1 for d in result.run_set if r in d.reasons)

    total = len(result.run_set) + len(result.skip_set)
    lines.append("")
    lines.append(f"[ROUTING] {total} accuracy case(s): run {len(result.run_set)}, skip {len(result.skip_set)}.")
    if result.run_set:
        lines.append(
            "  run by reason: "
            f"drift={_bucket(R_DRIFT)} "
            f"absent={_bucket(R_ABSENT)} "
            f"unmapped={_bucket(R_UNMAPPED)} "
            f"version-mismatch={_bucket(R_VERSION)}"
        )
    lines.append("=====================================")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="QNN EP accuracy-routing gate.")
    p.add_argument("--snapshot-json", default="", help="gtest JSON report from the snapshot phase.")
    p.add_argument("--golden-root", default="", help="Golden store root (holds manifest.json).")
    p.add_argument(
        "--accuracy-list-file",
        required=True,
        help="File with `--gtest_list_tests` output for QnnUnit_Accuracy_*.",
    )
    p.add_argument("--emit-filter-file", default="", help="Write the computed gtest filter here.")
    p.add_argument("--emit-summary-file", default="", help="Write the summary text here.")
    args = p.parse_args(argv)

    with open(args.accuracy_list_file) as f:
        accuracy_cases = parse_accuracy_list(f.read())

    result = run_gate(
        snapshot_json=args.snapshot_json or None,
        golden_root=args.golden_root or None,
        accuracy_cases=accuracy_cases,
    )

    gtest_filter = build_gtest_filter(result.run_set)
    summary = render_summary(result)

    if args.emit_filter_file:
        with open(args.emit_filter_file, "w") as f:
            f.write(gtest_filter + "\n")
    if args.emit_summary_file:
        with open(args.emit_summary_file, "w") as f:
            f.write(summary)

    sys.stdout.write(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
