#!/usr/bin/env python3
"""
compare_conv_json_graphs.py

Runs Conv integration tests with QNN_DUMP_JSON=1 and compares the dumped
QNN graph JSONs against the snapshot goldens to verify STRUCTURAL consistency
of the QNN EP's Conv translation.

Usage (run from the build directory):
  python3 <repo>/onnxruntime/test/providers/qnn/unit/tools/compare_conv_json_graphs.py

Options:
  --binary    Path to onnxruntime_provider_test (default: ./onnxruntime_provider_test)
  --golden-dir Path to conv_session goldens dir (default: auto-detected from repo)
  --filter    Glob pattern to restrict which snapshot names are checked
  --verbose   Print diffs on mismatch

# ---------------------------------------------------------------------------
# DESIGN: What this script compares, and what it intentionally ignores
# ---------------------------------------------------------------------------
#
# PURPOSE
#   Cross-verify that the snapshot tier (QnnUnit_SessionSnapshot_ConvTest)
#   and the integration tier (QnnHTPBackendTests / QnnCPUBackendTests) produce
#   QNN graphs with the same TRANSLATION STRUCTURE for each Conv spec.
#
# WHAT IS COMPARED (structural elements)
#   - Node op-type names (QNN_OP_CONV_2D, QNN_OP_CONV_3D, etc.)
#   - Number of nodes and tensors
#   - Tensor shapes (dims)
#   - Tensor data types
#   - Quantization structure:
#       * Per-tensor vs per-channel (count of scale entries)
#       * Presence/absence of quantization on each tensor
#       * Quantization definition type
#   - Graph connectivity (which canonical-position tensors connect to each node)
#
# WHAT IS INTENTIONALLY IGNORED (and why)
#   1. Tensor name strings
#      WHY: The snapshot builder (BuildConvQDQFn in conv_model_builders.h) and
#      the integration test builder (BuildQDQPerChannelConvTestCase in conv_test.cc)
#      use different naming conventions for intermediate QDQ tensors:
#        snapshot:    "qdq_input_q_out", "qdq_weights_q_out", ...
#        integration: "input_qdq_q_out", "weights_qdq_q_out", ...
#      These are equivalent tensors with different names. Aligning the names
#      would require coupling two independent test tiers, which is undesirable.
#      The snapshot tests verify correctness against their OWN golden files
#      (which use consistent naming), so naming regressions are caught there.
#
#   2. Quantization scale / offset VALUES
#      WHY: The two tiers use different test data generation methods:
#        snapshot:    GetFloatDataInRange (deterministic, linearly spaced values)
#        integration: mix of GetFloatDataInRange and RandomData (seed=2345)
#      Different input/weight values → different output ranges → different
#      scale/offset values, even for identical graph structures.
#      RISK ACCEPTED: a translation bug that produces wrong scale values but
#      happens to be within accuracy tolerance (0.4% of output range) would
#      not be caught here. This risk is mitigated by:
#        a) Snapshot tests compare the FULL golden including scale values
#           (regression within the snapshot tier is caught).
#        b) Accuracy tests (QnnUnit_SessionAccuracy_ConvTest) verify that
#           QNN EP output matches ORT CPU EP output numerically.
#
#   3. Weight / bias data arrays
#      WHY: Same reason as (2) — different test data → different weight values.
#      The shapes (dims) are still compared.
#
# COVERAGE SUMMARY
#   Translation structure bugs  → caught here (structural comparison)
#   Quantization value bugs     → caught by snapshot tests (vs golden)
#                                 + accuracy tests (vs CPU EP reference)
#   Numerical output bugs       → caught by accuracy tests
# ---------------------------------------------------------------------------
"""

import argparse
import copy
import json
import os
import pathlib
import shutil
import subprocess
import sys

# ---------------------------------------------------------------------------
# Mapping: snapshot_golden_name → (test_suite, integration_test_name)
#
# snapshot_golden_name is the filename stem of a file in:
#   goldens/builder/opbuilder/conv_session/<name>.json
#
# integration_test_name is the TEST_F name in conv_test.cc.
# ---------------------------------------------------------------------------

HTP = "QnnHTPBackendTests"
CPU = "QnnCPUBackendTests"

# (snapshot_name, test_suite, integration_test_name, json_ndim_hint)
# json_ndim_hint: expected number of dims for the "input" tensor in the JSON.
#   Used only for ConvU8U8S32_DynamicWeight_NoBias (one TEST_F dumps 2 JSONs).
#   None means no disambiguation needed.
MAPPING = [
    # ── directly matching names (35) ─────────────────────────────────────────
    ("Conv1DU8U8S32_AutoPadUpper",                       HTP, "Conv1DU8U8S32_AutoPadUpper",                       None),
    ("Conv3D_U16S8S32_PerChannel",                       HTP, "Conv3D_U16S8S32_PerChannel",                       None),
    ("Conv3D_U16S8S32_PerChannel2",                      HTP, "Conv3D_U16S8S32_PerChannel2",                      None),
    ("Conv3D_U8S8S32_PerChannel",                        HTP, "Conv3D_U8S8S32_PerChannel",                        None),
    ("Conv3D_U8S8S32_PerChannel2",                       HTP, "Conv3D_U8S8S32_PerChannel2",                       None),
    ("ConvDepthwiseU16S8S32_PerChannel",                 HTP, "ConvDepthwiseU16S8S32_PerChannel",                 None),
    ("ConvDepthwiseU8S8S32_PerChannel",                  HTP, "ConvDepthwiseU8S8S32_PerChannel",                  None),
    ("ConvS8S8S32_PerChannel_ReluClipFusion",            HTP, "ConvS8S8S32_PerChannel_ReluClipFusion",            None),
    ("ConvTranspose1DU8U8S32_AutoPadLower",              HTP, "ConvTranspose1DU8U8S32_AutoPadLower",              None),
    ("ConvTranspose3D_U16S8S32_PerChannel",              HTP, "ConvTranspose3D_U16S8S32_PerChannel",              None),
    ("ConvTranspose3D_U8S8S32_PerChannel",               HTP, "ConvTranspose3D_U8S8S32_PerChannel",              None),
    ("ConvTranspose3D_U8U8S32_DynamicWeight_NoBias",     HTP, "ConvTranspose3D_U8U8S32_DynamicWeight_NoBias",     None),
    ("ConvTransposeU16S8S32_PerChannel",                 HTP, "ConvTransposeU16S8S32_PerChannel",                 None),
    ("ConvTransposeU8S8S32_PerChannel",                  HTP, "ConvTransposeU8S8S32_PerChannel",                  None),
    ("ConvTransposeU8U8S32_DynamicWeight_NoBias",        HTP, "ConvTransposeU8U8S32_DynamicWeight_NoBias",        None),
    ("ConvTransposeU8U8S32_OutputShape",                 HTP, "ConvTransposeU8U8S32_OutputShape",                 None),
    ("ConvU16S4S32_PerChannel",                          HTP, "ConvU16S4S32_PerChannel",                          None),
    ("ConvU16S4S32_PerChannel_NegativeWeightQuantAxis",  HTP, "ConvU16S4S32_PerChannel_NegativeWeightQuantAxis",  None),
    ("ConvU16S4_PerChannel_NoBias",                      HTP, "ConvU16S4_PerChannel_NoBias",                      None),
    ("ConvU16S8S32_PerChannel",                          HTP, "ConvU16S8S32_PerChannel",                          None),
    ("ConvU16U8S32_DynamicBias",                         HTP, "ConvU16U8S32_DynamicBias",                         None),
    ("ConvU16U8S32_NoBias",                              HTP, "ConvU16U8S32_NoBias",                              None),
    ("ConvU16U8S32_StaticBias",                          HTP, "ConvU16U8S32_StaticBias",                          None),
    ("ConvU16U8_PerTensor_NoBias",                       HTP, "ConvU16U8_PerTensor_NoBias",                       None),
    ("ConvU8S8S32_PerChannel",                           HTP, "ConvU8S8S32_PerChannel",                           None),
    ("ConvU8U8S32_AutoPadValid",                         HTP, "ConvU8U8S32_AutoPadValid",                         None),
    ("ConvU8U8S32_BiasRequantization",                   HTP, "ConvU8U8S32_BiasRequantization",                   None),
    ("ConvU8U8S32_LargeInput_Dilations_Pads",            HTP, "ConvU8U8S32_LargeInput_Dilations_Pads",            None),
    ("ConvU8U8S32_RedundantClipQDQ",                     HTP, "ConvU8U8S32_RedundantClipQDQ",                     None),
    ("ConvU8U8S32_ReluClipFusion",                       HTP, "ConvU8U8S32_ReluClipFusion",                       None),
    ("ConvU8U8S32_bias_dynamic_input",                   HTP, "ConvU8U8S32_bias_dynamic_input",                   None),
    ("DepthwiseConvU16U8S32_DynamicBias",                HTP, "DepthwiseConvU16U8S32_DynamicBias",                None),
    ("DepthwiseConvU16U8S32_NoBias",                     HTP, "DepthwiseConvU16U8S32_NoBias",                     None),
    ("DepthwiseConvU16U8S32_StaticBias",                 HTP, "DepthwiseConvU16U8S32_StaticBias",                 None),
    # ── mismatched names (13) ────────────────────────────────────────────────
    ("Conv2D_f32_DynamicBias",             CPU, "Convf32_dynamic_bias",                          None),
    ("Conv2D_f32_StaticBias",              CPU, "Convf32_bias_initializer",                      None),
    ("Conv2D_f32_AutoPadSameUpper",        CPU, "Convf32_AutoPadUpper",                          None),
    ("ConvTranspose2D_f32_AutoPadSameUpper", CPU, "ConvTransposef32_AutoPadUpper",               None),
    ("Conv2D_f32_AutoPadSameLower",        CPU, "Convf32_AutoPadLower",                          None),
    ("ConvTranspose2D_f32_AutoPadSameLower", CPU, "ConvTransposef32_AutoPadLower",               None),
    ("ConvTranspose3D_f32_AutoPadSameLower", CPU, "ConvTranspose3D_f32_AutoPadLower",            None),
    ("Conv2D_f32_LargePads",               CPU, "Convf32_large_input1_pad_bias_initializer",     None),
    ("Conv2D_f32_LargeInput",              CPU, "Convf32_large_input2_nopad_bias_initializer",   None),
    ("Conv1D_f32_StaticWeights",           CPU, "Conv1Df32_StaticWeights_DefaultBias",           None),
    ("Conv1D_f32_DynamicWeights",          CPU, "Conv1Df32_DynamicWeights_DefaultBias",          None),
    ("ConvTranspose1D_f32_StaticWeights",  CPU, "ConvTranspose1Df32_StaticWeights_DefaultBias",  None),
    ("ConvTranspose1D_f32_DynamicWeights", CPU, "ConvTranspose1Df32_DynamicWeights_DefaultBias", None),
    # ── 1:N split: one TEST_F dumps both 2D and 3D JSONs ────────────────────
    # ndim_hint distinguishes the two JSON files by input tensor rank.
    ("ConvU8U8S32_DynamicWeight_NoBias",    HTP, "ConvU8U8S32_DynamicWeight_NoBias", 4),  # 2D
    ("Conv3D_U8U8S32_DynamicWeight_NoBias", HTP, "ConvU8U8S32_DynamicWeight_NoBias", 5),  # 3D
]


# ---------------------------------------------------------------------------
# JSON normalization
# ---------------------------------------------------------------------------

def normalize(graph: dict) -> dict:
    """Remove unstable 'id' fields so JSON diffs are stable across runs.
    Used for snapshot-vs-snapshot comparisons (same test data, exact match)."""
    g = graph.get("graph", {})
    for tensor in g.get("tensors", {}).values():
        tensor.pop("id", None)
    for node in g.get("nodes", {}).values():
        for param_group in node.get("tensor_params", {}).values():
            for param_tensor in param_group.values():
                if isinstance(param_tensor, dict):
                    param_tensor.pop("id", None)
    return graph


def _quant_structure(quant_params: dict) -> dict:
    """Return a structure-only representation of quant_params.
    Zeros out scale/offset values; keeps per-tensor vs per-channel distinction."""
    if not quant_params:
        return quant_params
    qp = copy.deepcopy(quant_params)
    if "scale_offset" in qp:
        qp["scale_offset"] = {"scale": 0.0, "offset": 0}
    if "scale_offsets" in qp:
        count = len(qp["scale_offsets"])
        qp["scale_offsets"] = [{"scale": 0.0, "offset": 0}] * count
    return qp


def normalize_structural(graph: dict) -> dict:
    """Structural normalization for cross-tier comparison (snapshot vs integration).

    Strips variable parts that differ due to different test data or builder
    naming conventions. See module docstring for full rationale.

    Removes:
      - 'id' fields (unstable across runs)
      - Tensor name strings (renamed to canonical t0, t1, ... by sorted position)
      - Scale/offset VALUES in quant_params (keeps per-channel count)
      - 'data' arrays in tensors and tensor_params (keeps 'dims')

    Keeps:
      - Node op types
      - Tensor shapes (dims) and data types
      - Quantization structure (per-tensor vs per-channel, count of scales)
      - Graph connectivity (via canonical tensor names)
    """
    graph = copy.deepcopy(graph)
    g = graph.get("graph", {})

    tensors = g.get("tensors", {})

    # Build canonical name mapping: sort tensors by (dims, data_type, quant_key)
    # so the mapping is stable regardless of original name strings.
    def tensor_sort_key(name_tensor):
        name, t = name_tensor
        dims = tuple(t.get("dims", []))
        dtype = t.get("data_type", 0)
        # quant presence as a secondary key
        qp = t.get("quant_params", {})
        quant_key = ("scale_offsets" if "scale_offsets" in qp
                     else "scale_offset" if "scale_offset" in qp
                     else "none",
                     len(qp.get("scale_offsets", [])))
        return (dims, dtype, quant_key, name)  # 'name' as stable tiebreaker

    sorted_tensors = sorted(tensors.items(), key=tensor_sort_key)
    name_to_canonical = {name: f"t{i}" for i, (name, _) in enumerate(sorted_tensors)}

    # Rebuild tensors with canonical names and stripped variable data.
    new_tensors = {}
    for name, tensor in sorted_tensors:
        t = copy.deepcopy(tensor)
        t.pop("id", None)
        t.pop("data", None)  # strip weight/bias data values, keep dims
        if "quant_params" in t:
            t["quant_params"] = _quant_structure(t["quant_params"])
        new_tensors[name_to_canonical[name]] = t
    g["tensors"] = new_tensors

    # Rebuild nodes: update tensor name references, strip variable data.
    nodes = g.get("nodes", {})
    new_nodes = {}
    for node_name, node in nodes.items():
        n = copy.deepcopy(node)
        n.pop("id", None)

        # Remap input/output tensor references to canonical names.
        for key in ("inputs", "outputs"):
            if key in n:
                n[key] = {k: name_to_canonical.get(v, v)
                          for k, v in n[key].items()}

        # Strip variable data from tensor_params (e.g. pad_amount, bias values).
        for pg in n.get("tensor_params", {}).values():
            for pt in pg.values():
                if isinstance(pt, dict):
                    pt.pop("id", None)
                    pt.pop("data", None)
                    if "quant_params" in pt:
                        pt["quant_params"] = _quant_structure(pt["quant_params"])

        new_nodes[node_name] = n
    g["nodes"] = new_nodes

    return graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_json_files(dump_dir: pathlib.Path) -> list[pathlib.Path]:
    """Return all .json files in dump_dir, excluding _tensor_log.json."""
    return [
        p for p in dump_dir.iterdir()
        if p.suffix == ".json" and "_tensor_log." not in p.name
    ]


def input_ndim(graph: dict) -> int | None:
    """Return the number of dims of the 'input' tensor, or None if not found."""
    tensors = graph.get("graph", {}).get("tensors", {})
    t = tensors.get("input")
    if t and "dims" in t:
        return len(t["dims"])
    return None


def pick_json_for_ndim(jsons: list[pathlib.Path], ndim: int) -> pathlib.Path | None:
    """Given multiple JSON files, return the one whose input tensor has `ndim` dims."""
    for p in jsons:
        try:
            g = json.loads(p.read_text())
            if input_ndim(g) == ndim:
                return p
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", default="./onnxruntime_provider_test",
                        help="Path to the test binary")
    parser.add_argument("--golden-dir", default=None,
                        help="Path to conv_session goldens dir (auto-detected if omitted)")
    parser.add_argument("--filter", default=None,
                        help="Only check snapshot names containing this substring")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full JSON diff on mismatch")
    args = parser.parse_args()

    binary = pathlib.Path(args.binary).resolve()
    if not binary.exists():
        sys.exit(f"Test binary not found: {binary}")

    # Auto-detect golden dir from the binary location.
    if args.golden_dir:
        golden_dir = pathlib.Path(args.golden_dir)
    else:
        # Binary is in <build_dir>/; source is sibling of build/ or 2 levels up.
        # Heuristic: walk up until we find onnxruntime/test/providers/qnn/unit.
        candidate = binary.parent
        for _ in range(6):
            unit_dir = candidate / "onnxruntime" / "test" / "providers" / "qnn" / "unit"
            if unit_dir.is_dir():
                golden_dir = unit_dir / "goldens" / "builder" / "opbuilder" / "conv_session"
                break
            candidate = candidate.parent
        else:
            sys.exit("Could not auto-detect golden dir. Pass --golden-dir explicitly.")

    if not golden_dir.is_dir():
        sys.exit(f"Golden dir not found: {golden_dir}")

    rows = MAPPING
    if args.filter:
        rows = [r for r in rows if args.filter in r[0]]
    if not rows:
        sys.exit("No matching entries after --filter.")

    # Group rows by (test_suite, integration_test_name) to avoid running the
    # same TEST_F multiple times (e.g., DynamicWeight 2D+3D share one TEST_F).
    from collections import defaultdict
    test_to_rows: dict[tuple, list] = defaultdict(list)
    for row in rows:
        snapshot_name, suite, test_name, ndim_hint = row
        test_to_rows[(suite, test_name)].append(row)

    results = {}  # snapshot_name → "PASS" | "FAIL: <reason>"
    build_dir = pathlib.Path.cwd()

    for (suite, test_name), group_rows in test_to_rows.items():
        dump_dir = build_dir / f"{suite}_{test_name}"
        # Clean up any leftover dump from previous runs.
        if dump_dir.exists():
            shutil.rmtree(dump_dir)

        gtest_filter = f"{suite}.{test_name}"
        env = os.environ.copy()
        env["QNN_DUMP_JSON"] = "1"

        print(f"Running: {gtest_filter} ...", end=" ", flush=True)
        result = subprocess.run(
            [str(binary), f"--gtest_filter={gtest_filter}"],
            env=env, capture_output=True, text=True,
            cwd=build_dir,
        )
        if result.returncode != 0:
            for row in group_rows:
                results[row[0]] = f"FAIL: test binary returned non-zero ({result.returncode})"
            print(f"FAIL (exit {result.returncode})")
            if args.verbose:
                print(result.stdout[-2000:])
                print(result.stderr[-2000:])
            continue
        print("OK")

        if not dump_dir.exists():
            for row in group_rows:
                results[row[0]] = f"FAIL: dump dir not created ({dump_dir})"
            continue

        json_files = find_json_files(dump_dir)
        if not json_files:
            for row in group_rows:
                results[row[0]] = "FAIL: no JSON files dumped"
            continue

        for snapshot_name, _, _, ndim_hint in group_rows:
            golden_path = golden_dir / f"{snapshot_name}.json"
            if not golden_path.exists():
                results[snapshot_name] = f"FAIL: golden not found ({golden_path})"
                continue

            # Pick the right JSON file.
            if ndim_hint is not None:
                chosen = pick_json_for_ndim(json_files, ndim_hint)
                if chosen is None:
                    results[snapshot_name] = (
                        f"FAIL: no JSON with input ndim={ndim_hint} in {dump_dir} "
                        f"(files: {[f.name for f in json_files]})"
                    )
                    continue
            elif len(json_files) == 1:
                chosen = json_files[0]
            else:
                results[snapshot_name] = (
                    f"FAIL: expected 1 JSON but found {len(json_files)} in {dump_dir}"
                )
                continue

            # Normalize and compare (structural comparison only).
            # See module docstring for what is and is not compared.
            try:
                actual = normalize_structural(json.loads(chosen.read_text()))
                expected = normalize_structural(json.loads(golden_path.read_text()))
            except Exception as e:
                results[snapshot_name] = f"FAIL: JSON parse error: {e}"
                continue

            actual_str = json.dumps(actual, indent=2, sort_keys=True)
            expected_str = json.dumps(expected, indent=2, sort_keys=True)

            if actual_str == expected_str:
                results[snapshot_name] = "PASS"
            else:
                results[snapshot_name] = "FAIL: JSON mismatch"
                if args.verbose:
                    import difflib
                    diff = "\n".join(difflib.unified_diff(
                        expected_str.splitlines(), actual_str.splitlines(),
                        fromfile="golden", tofile="integration_test", lineterm=""))
                    print(f"\n  Diff for {snapshot_name}:\n{diff}\n")

        # Clean up dump dir.
        shutil.rmtree(dump_dir, ignore_errors=True)

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    passed = [k for k, v in results.items() if v == "PASS"]
    failed = [(k, v) for k, v in results.items() if v != "PASS"]

    for name, reason in failed:
        print(f"  FAIL  {name}: {reason}")

    print(f"\nResult: {len(passed)}/{len(results)} PASS", end="")
    if failed:
        print(f", {len(failed)} FAIL")
        sys.exit(1)
    else:
        print()


if __name__ == "__main__":
    main()
