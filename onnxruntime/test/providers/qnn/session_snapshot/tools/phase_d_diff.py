#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
#
# Phase D — overlap audit tool for QNN EP unit-test snapshot goldens.
#
# Walks pairs of golden JSON files and produces a triage report.  Diff between
# two goldens is decomposed into 4 dimensions:
#
#   T (topology)  — graph shape: per-node op_type, package, connectivity.
#   M (metadata)  — type system: data_type, encoding, axis, dataFormat, etc.
#   A (attribute) — numeric values: dims, scales, offsets, scalar params.
#   D (data)      — initializer hash (params_data_hash).
#
# Plus an "uncategorized" bucket (U) for paths that don't match any pattern —
# these surface in the report so you can refine the classifiers per op.
#
# Verdict pattern table:
#
#   T>0                      → KEEP (different graph structure, different code path)
#   T=0, M>0                 → KEEP (different dispatch branch, e.g. static vs dynamic)
#   T=0, M=0, A>0 or D>0     → MERGE candidate (same code path, varying knob)
#   T=0, M=0, A=0, D=0       → SUSPICIOUS (byte-identical — verify path divergence)
#
# Tier 1 (size delta) intentionally NOT used in this version — see
# note_migrate_conv.md §7.3 for the rationale.  Tier 2 (md5) and Tier 3
# (structural diff) are both implemented.
#
# Usage:
#   python phase_d_diff.py [--top N] <golden_dir> [<golden_dir2> ...]
#
# Output: text report on stdout, sorted by (T, M, A, D) ascending.

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Path classification.  Patterns are evaluated in order; first match wins.
# ---------------------------------------------------------------------------

TOPOLOGY_PATTERNS = [
    r"^op_types(\[\d+\])?$",                             # top-level op_types summary array
    r"^graph\.nodes\.[^.]+$",                            # whole node added/removed (topology change)
    r"^graph\.tensors\.[^.]+$",                          # whole tensor added/removed
    r"^graph\.nodes\.[^.]+\.type$",                      # op type per node (Conv2d, etc)
    r"^graph\.nodes\.[^.]+\.package$",                   # op package
    r"^graph\.nodes\.[^.]+\.input_names(\[\d+\])?$",     # node input connectivity
    r"^graph\.nodes\.[^.]+\.output_names(\[\d+\])?$",    # node output connectivity
]

METADATA_PATTERNS = [
    # Top-level tensor metadata
    r"^graph\.tensors\.[^.]+\.data_type$",
    r"^graph\.tensors\.[^.]+\.dataFormat$",
    r"^graph\.tensors\.[^.]+\.axis_format$",
    r"^graph\.tensors\.[^.]+\.src_axis_format$",
    r"^graph\.tensors\.[^.]+\.type$",                    # tensor type (APP_WRITE/STATIC/...)
    r"^graph\.tensors\.[^.]+\.quant_params\.encoding$",
    r"^graph\.tensors\.[^.]+\.quant_params\.definition$",
    r"^graph\.tensors\.[^.]+\.quant_params\.[^.]+$",     # whole encoding subtree presence (scale_offset / axis_scale_offset / bw_*)
    r"^graph\.tensors\.[^.]+\.quant_params\.[^.]+\.axis$",
    r"^graph\.tensors\.[^.]+\.quant_params\.[^.]+\.num_scale_offsets$",
    r"^graph\.tensors\.[^.]+\.quant_params\.[^.]+\.numElements$",
    r"^graph\.tensors\.[^.]+\.quant_params\.[^.]+\.bitwidth$",
    # Per-node tensor_params metadata (dilation/pad_amount/stride etc tensors)
    r"^graph\.nodes\.[^.]+\.tensor_params\.[^.]+$",      # whole tensor_param presence (output_padding etc)
    r"^graph\.nodes\.[^.]+\.tensor_params\.[^.]+\.[^.]+\.data_type$",
    r"^graph\.nodes\.[^.]+\.tensor_params\.[^.]+\.[^.]+\.dataFormat$",
    r"^graph\.nodes\.[^.]+\.tensor_params\.[^.]+\.[^.]+\.axis_format$",
    r"^graph\.nodes\.[^.]+\.tensor_params\.[^.]+\.[^.]+\.src_axis_format$",
    r"^graph\.nodes\.[^.]+\.tensor_params\.[^.]+\.[^.]+\.type$",
    r"^graph\.nodes\.[^.]+\.tensor_params\.[^.]+\.[^.]+\.quant_params\.encoding$",
    r"^graph\.nodes\.[^.]+\.tensor_params\.[^.]+\.[^.]+\.quant_params\.definition$",
]

ATTRIBUTE_PATTERNS = [
    # Top-level tensor numeric values
    r"^graph\.tensors\.[^.]+\.dims(\[\d+\])?$",
    r"^graph\.tensors\.[^.]+\.params_count$",
    r"^graph\.tensors\.[^.]+\.data(\[.+\])?$",          # inline static tensor data (integration dumps)
    r"^graph\.tensors\.[^.]+\.quant_params\.scale_offset\.(scale|offset)$",
    r"^graph\.tensors\.[^.]+\.quant_params\.[^.]+\.scale_offsets\[\d+\]\.(scale|offset)$",
    r"^graph\.tensors\.[^.]+\.quant_params\.[^.]+\.scale_offsets\[\d+\]$", # whole entry added/removed (channel count diff)
    r"^graph\.tensors\.[^.]+\.quant_params\.[^.]+\.num_elements$",
    r"^graph\.tensors\.[^.]+\.quant_params\.[^.]+\.scales(\[\d+\])?$",
    r"^graph\.tensors\.[^.]+\.quant_params\.[^.]+\.offsets(\[\d+\])?$",
    # Node scalar/tensor param values
    r"^graph\.nodes\.[^.]+\.scalar_params\..+$",         # whole scalar_param subtree
    r"^graph\.nodes\.[^.]+\.tensor_params\.[^.]+\.[^.]+\.data(\[.+\])?$",
    r"^graph\.nodes\.[^.]+\.tensor_params\.[^.]+\.[^.]+\.dims(\[\d+\])?$",
    r"^graph\.nodes\.[^.]+\.tensor_params\.[^.]+\.[^.]+\.params_count$",
    r"^graph\.nodes\.[^.]+\.tensor_params\.[^.]+\.[^.]+\.quant_params\.scale_offset\.(scale|offset)$",
]

DATA_PATTERNS = [
    r"^graph\.tensors\.[^.]+\.params_data_hash$",
]

IGNORED_PATTERNS = [
    r"^Total MACs per inference$",
    r"^Total parameters$",
    r"^converter_command$",
    r"^copyright_str$",
    r"^graph\.nodes\.[^.]+\.macs_per_inference$",
    r"^graph\.tensors\.[^.]+\.id$",                                        # tensor allocation id (noise)
    r"^graph\.nodes\.[^.]+\.tensor_params\.[^.]+\.[^.]+\.id$",             # tensor_params sub-tensor id (noise)
]

_DIM_TABLES = [
    ("T", TOPOLOGY_PATTERNS),
    ("M", METADATA_PATTERNS),
    ("A", ATTRIBUTE_PATTERNS),
    ("D", DATA_PATTERNS),
    ("I", IGNORED_PATTERNS),
]

# Pre-compile patterns once.
_DIM_TABLES = [(d, [re.compile(p) for p in pats]) for d, pats in _DIM_TABLES]


def classify(path):
    for dim, regexes in _DIM_TABLES:
        for r in regexes:
            if r.match(path):
                return dim
    return "U"


# ---------------------------------------------------------------------------
# JSON walker.  Returns list of (path, val_a, val_b) for leaves that differ
# OR keys present on only one side.  List ordering is preserved (positional).
# ---------------------------------------------------------------------------

def diff_paths(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            sub = f"{prefix}.{k}" if prefix else k
            if k not in a:
                out.append((sub, None, b[k]))
            elif k not in b:
                out.append((sub, a[k], None))
            else:
                out.extend(diff_paths(a[k], b[k], sub))
        return out
    if isinstance(a, list) and isinstance(b, list):
        out = []
        n = max(len(a), len(b))
        for i in range(n):
            sub = f"{prefix}[{i}]"
            if i >= len(a):
                out.append((sub, None, b[i]))
            elif i >= len(b):
                out.append((sub, a[i], None))
            else:
                out.extend(diff_paths(a[i], b[i], sub))
        return out
    # Type mismatch or scalar leaf
    if a != b:
        return [(prefix, a, b)]
    return []


# ---------------------------------------------------------------------------
# Per-pair scoring + verdict
# ---------------------------------------------------------------------------

def score_pair(diffs):
    counts = defaultdict(int)
    for path, _, _ in diffs:
        counts[classify(path)] += 1
    return counts


def verdict(counts):
    t, m, a, d = counts["T"], counts["M"], counts["A"], counts["D"]
    if t > 0:
        return "KEEP-topology"
    if m > 0:
        return "KEEP-metadata"
    if a > 0 or d > 0:
        return "MERGE-candidate"
    return "SUSPICIOUS"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_files(dirs):
    files = []
    for d in dirs:
        for f in sorted(Path(d).rglob("*.json")):
            if not f.name.endswith("_tensor_log.json"):
                files.append(f)
    return files


def md5_of(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="Golden directories to scan")
    ap.add_argument("--top", type=int, default=0,
                    help="Show only the top N pairs by (T,M,A,D) ascending.  0=all")
    ap.add_argument("--show-uncategorized", action="store_true",
                    help="Print uncategorized path samples (refines patterns)")
    args = ap.parse_args()

    files = collect_files(args.dirs)
    if len(files) < 2:
        print("Need at least 2 golden files", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {len(files)} goldens, {len(files)*(len(files)-1)//2} pairs")

    # Tier 2: hash all files.  Group identical hashes for fast SUSPICIOUS lookup.
    hashes = {f: md5_of(f) for f in files}
    by_hash = defaultdict(list)
    for f, h in hashes.items():
        by_hash[h].append(f)
    identical_groups = {h: g for h, g in by_hash.items() if len(g) >= 2}

    # Pre-load JSONs once
    blobs = {f: json.loads(f.read_text()) for f in files}

    # All pairs
    rows = []
    uncategorized = defaultdict(int)  # path → count
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            a, b = files[i], files[j]
            if hashes[a] == hashes[b]:
                # Tier 2 hit — byte identical
                rows.append({
                    "a": a, "b": b,
                    "counts": defaultdict(int),
                    "verdict": "SUSPICIOUS",
                    "diffs": [],
                })
                continue
            diffs = diff_paths(blobs[a], blobs[b])
            counts = score_pair(diffs)
            for path, _, _ in diffs:
                if classify(path) == "U":
                    uncategorized[path] += 1
            rows.append({
                "a": a, "b": b,
                "counts": counts,
                "verdict": verdict(counts),
                "diffs": diffs,
            })

    def label(p):
        return f"{p.parent.name}/{p.name}" if p.parent.name != "." else p.name

    # Sort by (T, M, A, D) ascending — most similar pairs first.
    def key(r):
        c = r["counts"]
        return (c["T"], c["M"], c["A"], c["D"], label(r["a"]), label(r["b"]))
    rows.sort(key=key)

    # ----------------------------------------------------------------------
    # Output
    # ----------------------------------------------------------------------

    # Verdict count summary
    verdict_counts = defaultdict(int)
    for r in rows:
        verdict_counts[r["verdict"]] += 1
    print()
    print("=" * 80)
    print("Verdict summary")
    print("=" * 80)
    for v in ("SUSPICIOUS", "MERGE-candidate", "KEEP-metadata", "KEEP-topology"):
        print(f"  {v:<18}  {verdict_counts[v]:>4}")
    print(f"  {'(total)':<18}  {sum(verdict_counts.values()):>4}")
    print()
    print("=" * 80)
    print("Tier 2 — byte-identical groups (md5 match)")
    print("=" * 80)
    if not identical_groups:
        print("(none)")
    else:
        for h, group in identical_groups.items():
            print(f"  Group ({len(group)} files, hash={h[:12]}):")
            for f in group:
                print(f"    {label(f)}")
            print()

    print("=" * 80)
    print("Tier 3 — pairwise structural diff, sorted by (T,M,A,D) ascending")
    print("=" * 80)
    print()
    print(f"{'T':>2} {'M':>2} {'A':>2} {'D':>2} {'U':>2}  {'verdict':<18}  pair")
    print("-" * 80)

    n_show = args.top if args.top > 0 else len(rows)
    for r in rows[:n_show]:
        c = r["counts"]
        print(f"{c['T']:>2} {c['M']:>2} {c['A']:>2} {c['D']:>2} {c['U']:>2}  "
              f"{r['verdict']:<18}  {label(r['a'])}  vs  {label(r['b'])}")

    # Detail for SUSPICIOUS + MERGE-candidate (top of list always)
    print()
    print("=" * 80)
    print("Detail for SUSPICIOUS + MERGE-candidate pairs")
    print("=" * 80)
    for r in rows:
        if r["verdict"] not in ("SUSPICIOUS", "MERGE-candidate"):
            continue
        print()
        print(f"--- {label(r['a'])}  vs  {label(r['b'])}  ({r['verdict']}) ---")
        if r["verdict"] == "SUSPICIOUS":
            print("  byte-identical, no field-level diffs")
            continue
        # Show up to 10 diff paths grouped by dimension
        by_dim = defaultdict(list)
        for path, va, vb in r["diffs"]:
            by_dim[classify(path)].append((path, va, vb))
        for dim in ("A", "D", "U"):
            items = by_dim.get(dim, [])
            if not items:
                continue
            print(f"  [{dim}] ({len(items)} paths):")
            for path, va, vb in items[:5]:
                a_repr = repr(va)[:40]
                b_repr = repr(vb)[:40]
                print(f"    {path}  =  {a_repr}  vs  {b_repr}")
            if len(items) > 5:
                print(f"    ... ({len(items) - 5} more)")

    if args.show_uncategorized and uncategorized:
        print()
        print("=" * 80)
        print(f"Uncategorized path patterns ({len(uncategorized)} unique paths)")
        print("=" * 80)
        print("Add new regex to TOPOLOGY/METADATA/ATTRIBUTE/DATA/IGNORED_PATTERNS")
        print("to refine classification.  Top 30:")
        sorted_paths = sorted(uncategorized.items(), key=lambda x: -x[1])[:30]
        for path, n in sorted_paths:
            print(f"  ({n:>4}x)  {path}")


if __name__ == "__main__":
    main()
