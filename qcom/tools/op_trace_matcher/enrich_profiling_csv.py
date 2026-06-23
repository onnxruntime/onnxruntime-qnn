# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Profiling CSV enrichment tool.

Takes a QNN EP profiling CSV and a merged framework op trace JSON
(`qnn_op_trace.json` extended with `original_sources[]` by
`source_to_optimized_matcher.py`), and produces a CSV that traces every QNN
op back to the user-authored ONNX op names. Behavior adapts to the input:

  - BASIC profiling (no NODE-level events) -> verbatim byte-for-byte copy
    via shutil.copyfile; there is nothing to enrich.
  - DETAILED/OPTRACE with `ONNX Source Ops` present and populated (run-time
    `qnn.enable_framework_op_trace` was on, JIT trace landed) -> append a
    parallel `Original ONNX Source Ops` column with the pre-optimization
    ONNX op names; preserve the runtime-written cells.
  - DETAILED/OPTRACE with `ONNX Source Ops` present but every NODE row's cell
    empty (AOT Phase 2 with framework op trace requested but no sidecar
    found) -> fill the existing column from the merged trace and append
    `Original ONNX Source Ops`.
  - DETAILED/OPTRACE with `ONNX Source Ops` missing (run-time tracing was
    off entirely) -> synthesize both `ONNX Source Ops` and
    `Original ONNX Source Ops` at the end of the row.

The merged trace can be supplied either as a pre-computed JSON file
(--merged-trace) or computed inline by invoking the matcher as a library
(--source-model + --optimized-model + --qnn-trace).

Column naming, semicolon delimiter, and the `:OpId_{N}` suffix-stripping
behavior match `Serializer::LookupOnnxSources()` in qnn_profile_serializer.cc.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import shutil
import sys
from pathlib import Path

# Mode B (inline matcher) needs the `onnx` package and the sibling matcher
# module. They are not required for Mode A (pre-computed merged trace), which
# is pure-stdlib, so the imports are guarded — failing to load them only
# disables Mode B and surfaces a helpful error at use time.
#
# Predefine the availability flags so any import failure (sibling missing,
# `onnx` missing, etc.) still leaves these names defined. Without this,
# downstream references would raise a confusing NameError instead of a
# diagnosable ImportError.
_MATCHER_AVAILABLE = False
_MATCHER_IMPORT_ERROR: str | None = None
_TRACE_TYPE_OP: str | None = None
Matcher = None  # type: ignore[assignment]
join_qnn_trace = None  # type: ignore[assignment]

# Try the package-relative import first (set when invoked via `python -m
# qcom.tools.op_trace_matcher.enrich_profiling_csv`); fall back to a sibling
# absolute import for direct-script invocation. The package's __init__.py
# makes the relative path work without a sys.path mutation.
try:
    try:
        from .source_to_optimized_matcher import _TRACE_TYPE_OP  # type: ignore[no-redef]
    except ImportError:
        # Direct-script invocation: ensure the file's own directory is on
        # sys.path for the sibling lookup, then revert.
        _THIS_DIR = str(Path(__file__).resolve().parent)
        sys.path.insert(0, _THIS_DIR)
        try:
            from source_to_optimized_matcher import _TRACE_TYPE_OP  # type: ignore[no-redef]
        finally:
            with contextlib.suppress(ValueError):
                sys.path.remove(_THIS_DIR)

    # Schema constants imported successfully. Now try the heavier Mode B
    # imports (onnx + matcher classes).
    try:
        import onnx

        try:
            from .source_to_optimized_matcher import Matcher, join_qnn_trace  # type: ignore[no-redef]
        except ImportError:
            from source_to_optimized_matcher import Matcher, join_qnn_trace  # type: ignore[no-redef]
        _MATCHER_AVAILABLE = True
    except ImportError as _e:
        _MATCHER_IMPORT_ERROR = str(_e)
except ImportError as _e:
    # Even the schema-constant import failed (sibling renamed / missing).
    # _MATCHER_AVAILABLE stays False; record the error so use-time messages
    # can point at the real cause instead of a NameError.
    _MATCHER_IMPORT_ERROR = str(_e)


# Public API for import-as-a-library use. Underscore-prefixed helpers
# (_format_op_sources, _strip_op_id_suffix, _COL_*) remain private; tests reach
# for them by name but they are not part of the supported surface.
__all__ = [
    "build_lookups",
    "enrich",
]


# CSV column names from `Serializer::InitCsvFile()` in
# qnn_profile_serializer.cc. The QNN EP omits `ONNX Source Ops` when
# `op_trace_lookup` is null (BASIC profiling, framework op trace disabled,
# or AOT runs without a sidecar).
_COL_EVENT_IDENTIFIER = "Event Identifier"
_COL_ONNX_SOURCE_OPS = "ONNX Source Ops"
_COL_ORIGINAL_ONNX_SOURCE_OPS = "Original ONNX Source Ops"


def _strip_op_id_suffix(identifier: str) -> str:
    """HTP profiling events may append `:OpId_{N} ({unit})` to the op name
    (e.g. `add_node:OpId_17 (cycles)`). Strip this suffix to recover the
    base op name used as the lookup key. Mirrors
    `Serializer::LookupOnnxSources()` in qnn_profile_serializer.cc."""
    pos = identifier.find(":OpId_")
    return identifier[:pos] if pos != -1 else identifier


def _format_op_sources(sources: list[dict]) -> str:
    """Emit only op-typed entries with non-empty names, semicolon-separated,
    mirroring the QNN EP's existing CSV serialization in
    `Serializer::LookupOnnxSources()`. Reused for both `sources[]` (optimized)
    and `original_sources[]` (original), since both follow the same
    `TraceSourcePair` schema."""
    parts = []
    for s in sources or []:
        if s.get("type") != _TRACE_TYPE_OP:
            continue
        name = s.get("name", "")
        if name:
            parts.append(name)
    return ";".join(parts)


def build_lookups(merged_trace: dict) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Return (originals_lookup, optimized_lookup), each keyed by QNN op name.

    The optimized_lookup mirrors what the QNN EP's `Serializer::LookupOnnxSources()`
    returns at runtime: the `sources[]` chain from the op_mapping. It is needed
    when the input CSV is missing the `ONNX Source Ops` column (e.g. profiling
    CSVs from BASIC level or runs without framework op trace enabled), so the
    enrichment tool can synthesize that column from the merged trace alongside
    the new `Original ONNX Source Ops` column.
    """
    originals: dict[str, list[dict]] = {}
    optimized: dict[str, list[dict]] = {}
    for sg in merged_trace.get("subgraph_traces", []):
        for mapping in sg.get("op_mappings", []) or []:
            dst = mapping.get("dst_name")
            if not dst:
                continue
            # Store either chain only when present and non-empty; an empty list
            # yields an empty cell at lookup time, so there is nothing to record.
            if orig := mapping.get("original_sources"):
                originals[dst] = orig
            if src := mapping.get("sources"):
                optimized[dst] = src
    return originals, optimized


def enrich(profiling_csv: Path, merged_trace: dict, output_csv: Path) -> dict:
    """Read profiling_csv, append `Original ONNX Source Ops` column (and
    `ONNX Source Ops` if missing from the input). Skipped entirely if the
    input has no NODE-level events (BASIC profiling — nothing to enrich).
    `merged_trace` is the parsed JSON dict (loaded from disk in Mode A,
    computed in-memory in Mode B). Returns a stats dict.

    Raises ValueError if the input CSV is empty, missing the
    `Event Identifier` column, or already carries the `Original ONNX Source Ops`
    column (i.e. is itself an enricher output)."""
    originals_lookup, optimized_lookup = build_lookups(merged_trace)

    stats = {
        "rows_total": 0,
        "rows_with_identifier": 0,
        "rows_enriched_originals": 0,
        "rows_enriched_optimized": 0,
        "rows_unresolved_qnn_op": 0,
        "added_optimized_column": False,
        "filled_existing_onnx_column": False,
        "node_events_present": False,
    }

    # Read input fully so we can detect NODE-level events before deciding
    # the output column layout. Profiling CSVs are typically small (a few
    # thousand rows at most) so buffering is fine.
    with profiling_csv.open(newline="", encoding="utf-8") as fin:
        reader = csv.reader(fin)
        try:
            header = next(reader)
        except StopIteration as e:
            raise ValueError(f"empty CSV: {profiling_csv}") from e

        if _COL_EVENT_IDENTIFIER not in header:
            raise ValueError(
                f"input CSV is missing the `{_COL_EVENT_IDENTIFIER}` column "
                f"(found: {header}). Is this a QNN EP profiling CSV?"
            )
        identifier_idx = header.index(_COL_EVENT_IDENTIFIER)
        rows = list(reader)

    stats["rows_total"] = len(rows)
    has_node_events = any(identifier_idx < len(row) and row[identifier_idx] for row in rows)
    stats["node_events_present"] = has_node_events
    has_existing_onnx_col = _COL_ONNX_SOURCE_OPS in header
    onnx_source_ops_idx = header.index(_COL_ONNX_SOURCE_OPS) if has_existing_onnx_col else -1

    # When the `ONNX Source Ops` column is in the header, scan NODE rows to see
    # whether it carries any data. The AOT-no-sidecar case (framework op trace
    # is on but no sidecar exists next to the context model) emits the column
    # because column gating is session-stable, yet leaves every cell empty —
    # the merged trace passed to the enricher is then the authoritative source
    # of optimized-name annotations, so the existing empty cells should be
    # filled in instead of left next to a populated `Original ONNX Source Ops`.
    existing_onnx_col_has_data = False
    if has_existing_onnx_col:
        for row in rows:
            if (
                identifier_idx < len(row)
                and row[identifier_idx]
                and onnx_source_ops_idx < len(row)
                and row[onnx_source_ops_idx]
            ):
                existing_onnx_col_has_data = True
                break

    # Refuse to enrich a CSV that already carries the `Original ONNX Source Ops`
    # column — that means it is already an enricher output, and appending a
    # second identically-named column would silently produce a malformed CSV.
    if _COL_ORIGINAL_ONNX_SOURCE_OPS in header:
        raise ValueError(
            f"input CSV already has the `{_COL_ORIGINAL_ONNX_SOURCE_OPS}` column "
            f"({profiling_csv}); it looks like an enricher output. Enrich the original "
            f"QNN EP profiling CSV instead of a previously-enriched one."
        )

    # Output column layout:
    # - No NODE events (BASIC profiling, or HasNodeLevelProfiling()=false at
    #   runtime) -> nothing to enrich; copy the CSV verbatim.
    # - DETAILED/OPTRACE with `ONNX Source Ops` present and populated -> append
    #   only `Original ONNX Source Ops`; preserve the runtime-written values.
    # - DETAILED/OPTRACE with `ONNX Source Ops` missing -> synthesize both
    #   columns at the end from the merged trace.
    # - DETAILED/OPTRACE with `ONNX Source Ops` present but every NODE row's
    #   cell empty (AOT-no-sidecar) -> fill the existing column in place from
    #   the merged trace and append `Original ONNX Source Ops`.
    if not has_node_events:
        # Use shutil.copyfile so byte-level content (incl. line endings) is
        # preserved exactly. Routing through csv.writer would normalize line
        # endings to CRLF and silently change the output.
        shutil.copyfile(profiling_csv, output_csv)
        return stats

    fill_existing_onnx_col = has_existing_onnx_col and not existing_onnx_col_has_data
    if has_existing_onnx_col:
        new_columns = [_COL_ORIGINAL_ONNX_SOURCE_OPS]
        synthesize_optimized = False
    else:
        new_columns = [_COL_ONNX_SOURCE_OPS, _COL_ORIGINAL_ONNX_SOURCE_OPS]
        synthesize_optimized = True
    stats["added_optimized_column"] = synthesize_optimized
    stats["filled_existing_onnx_column"] = fill_existing_onnx_col

    with output_csv.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow([*header, *new_columns])

        for row in rows:
            if identifier_idx < len(row) and row[identifier_idx]:
                stats["rows_with_identifier"] += 1
                qnn_op = _strip_op_id_suffix(row[identifier_idx])
                originals = originals_lookup.get(qnn_op)
                optimized = optimized_lookup.get(qnn_op)

                # Either lookup may be unresolved if the merged trace doesn't
                # include this QNN op (e.g. trace from a different model run).
                # Count it once per row, not once per missing column.
                if originals is None and optimized is None:
                    stats["rows_unresolved_qnn_op"] += 1

                originals_str = _format_op_sources(originals) if originals else ""
                optimized_str = _format_op_sources(optimized) if optimized else ""
                if originals_str:
                    stats["rows_enriched_originals"] += 1
                if (synthesize_optimized or fill_existing_onnx_col) and optimized_str:
                    stats["rows_enriched_optimized"] += 1

                if fill_existing_onnx_col:
                    # Overwrite the all-empty existing cell at onnx_source_ops_idx.
                    # Copy the row first so the original `rows` list stays intact,
                    # and pad if the input row was shorter than the header.
                    row_out = list(row)
                    if onnx_source_ops_idx >= len(row_out):
                        row_out.extend([""] * (onnx_source_ops_idx - len(row_out) + 1))
                    row_out[onnx_source_ops_idx] = optimized_str
                    writer.writerow([*row_out, originals_str])
                else:
                    appended = [optimized_str, originals_str] if synthesize_optimized else [originals_str]
                    writer.writerow([*row, *appended])
            else:
                # Non-NODE rows (SESSION, GRAPH, etc.) get empty cells in any
                # new columns we add, and their existing ONNX Source Ops cell
                # (if any) was already empty — leave it untouched.
                empty_appended = ["", ""] if synthesize_optimized else [""]
                writer.writerow([*row, *empty_appended])

    return stats


def _resolve_merged_trace(args: argparse.Namespace) -> tuple[dict, str]:
    """Return (merged_trace_dict, mode_label).

    Mode A — load a pre-computed merged trace JSON from --merged-trace.
    Mode B — run the matcher in-memory on (--source-model, --optimized-model,
    --qnn-trace) and return the joined result dict.
    """
    if args.merged_trace is not None:
        try:
            return json.loads(args.merged_trace.read_text(encoding="utf-8")), "A (pre-computed merged trace)"
        except json.JSONDecodeError as e:
            sys.stderr.write(f"error: failed to parse {args.merged_trace}: {e}\n")
            sys.exit(2)

    # Mode B: import matcher as a library and compute in-memory.
    # The matcher script lives next to this file; running this script puts
    # its directory on sys.path automatically.
    if not _MATCHER_AVAILABLE:
        sys.stderr.write(
            f"error: Mode B requires the `onnx` package and "
            f"source_to_optimized_matcher.py in the same directory: "
            f"{_MATCHER_IMPORT_ERROR}\n"
        )
        sys.exit(2)

    source = onnx.load(str(args.source_model))
    optimized = onnx.load(str(args.optimized_model))
    matcher = Matcher(source, optimized)
    matches, tensor_matches, mstats, _, _ = matcher.run()

    try:
        qnn_trace = json.loads(args.qnn_trace.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        sys.stderr.write(f"error: failed to parse {args.qnn_trace}: {e}\n")
        sys.exit(2)
    merged, _ = join_qnn_trace(matches, tensor_matches, qnn_trace, matcher.src)

    if args.verbose:
        print(
            f"Matcher (inline): {mstats.matched}/{mstats.total_optimized} optimized "
            f"nodes, {mstats.matched_tensors}/{mstats.total_optimized_tensors} tensors"
        )

    return merged, "B (inline matcher)"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Append Original ONNX Source Ops column to a QNN EP profiling CSV.",
        allow_abbrev=False,
    )
    ap.add_argument(
        "--profiling-csv",
        required=True,
        type=Path,
        help="Profiling CSV from QNN EP (any qnn.profiling_level — BASIC is "
        "copied verbatim, DETAILED/OPTRACE is enriched).",
    )
    ap.add_argument("--output", required=True, type=Path, help="Output CSV path")

    # Mode A: pre-computed merged trace.
    ap.add_argument(
        "--merged-trace",
        type=Path,
        default=None,
        help="(Mode A) Merged framework op trace JSON, output of "
        "source_to_optimized_matcher.py --qnn-trace --joined-output. "
        "Mutually exclusive with the Mode B trio.",
    )

    # Mode B: raw inputs — matcher is run inline.
    ap.add_argument("--source-model", type=Path, default=None, help="(Mode B) Original user-provided ONNX model")
    ap.add_argument(
        "--optimized-model",
        type=Path,
        default=None,
        help="(Mode B) ORT-optimized ONNX model (saved via SessionOptions::optimized_model_filepath)",
    )
    ap.add_argument("--qnn-trace", type=Path, default=None, help="(Mode B) QNN EP qnn_op_trace.json")

    ap.add_argument("--verbose", action="store_true", help="Print row-level diagnostics")
    args = ap.parse_args()

    # Validate mode selection.
    mode_a = args.merged_trace is not None
    mode_b_inputs = [args.source_model, args.optimized_model, args.qnn_trace]
    mode_b_partial = any(x is not None for x in mode_b_inputs)
    mode_b_complete = all(x is not None for x in mode_b_inputs)

    if mode_a and mode_b_partial:
        sys.stderr.write(
            "error: --merged-trace (Mode A) is mutually exclusive with "
            "--source-model/--optimized-model/--qnn-trace (Mode B)\n"
        )
        return 2
    if not mode_a and not mode_b_complete:
        if mode_b_partial:
            sys.stderr.write("error: Mode B requires all three of --source-model, --optimized-model, --qnn-trace\n")
        else:
            sys.stderr.write(
                "error: provide either --merged-trace (Mode A) or "
                "--source-model + --optimized-model + --qnn-trace (Mode B)\n"
            )
        return 2

    # Validate input file existence.
    required_paths = [(args.profiling_csv, "profiling CSV")]
    if mode_a:
        required_paths.append((args.merged_trace, "merged trace"))
    else:
        required_paths.extend(
            [
                (args.source_model, "source model"),
                (args.optimized_model, "optimized model"),
                (args.qnn_trace, "QNN trace"),
            ]
        )
    for path, label in required_paths:
        if not path.is_file():
            sys.stderr.write(f"error: {label} not found: {path}\n")
            return 2

    if args.output.exists():
        sys.stderr.write(f"warning: overwriting existing output: {args.output}\n")

    merged_trace, mode_label = _resolve_merged_trace(args)
    if args.verbose:
        print(f"Mode: {mode_label}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        stats = enrich(args.profiling_csv, merged_trace, args.output)
    except ValueError as e:
        sys.stderr.write(f"error: {e}\n")
        return 2

    if not stats["node_events_present"]:
        print(
            f"No NODE-level events in input CSV ({stats['rows_total']} rows; "
            f"likely BASIC profiling or framework op trace not emitting per-node "
            f"data). Output is a verbatim copy — nothing to enrich."
        )
    elif stats["added_optimized_column"]:
        print(
            f"Enriched {stats['rows_enriched_originals']}/{stats['rows_with_identifier']} rows "
            f"with original ONNX sources, "
            f"{stats['rows_enriched_optimized']}/{stats['rows_with_identifier']} rows "
            f"with optimized ONNX sources "
            f"(input CSV missing `ONNX Source Ops` column — both columns added; "
            f"total rows: {stats['rows_total']}, "
            f"unresolved QNN ops: {stats['rows_unresolved_qnn_op']})"
        )
    else:
        print(
            f"Enriched {stats['rows_enriched_originals']}/{stats['rows_with_identifier']} rows "
            f"with original ONNX sources "
            f"(total rows: {stats['rows_total']}, "
            f"unresolved QNN ops: {stats['rows_unresolved_qnn_op']})"
        )
    if args.verbose and stats["rows_unresolved_qnn_op"]:
        print(
            f"  warning: {stats['rows_unresolved_qnn_op']} row(s) had a QNN op "
            f"name not present in the merged trace. Make sure the trace was "
            f"produced from the same model run as the profiling CSV."
        )
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
