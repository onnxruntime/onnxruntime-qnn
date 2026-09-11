# Op Trace Matcher Tools

This directory contains two related offline tools that, together, deliver
end-to-end provenance for QNN-EP-compiled models —
`original ONNX -> optimized ONNX -> QNN op` — without any ORT-core or QNN-EP
code changes.

> **Dependency:** these tools consume the QNN EP framework-op-trace artifacts —
> the `qnn_op_trace.json` sidecar and the `ONNX Source Ops` profiling-CSV column
> produced by `Serializer::LookupOnnxSources()` / `Serializer::InitCsvFile()`.
> That feature ships with the QNN EP framework op tracing support
> (`qnn.enable_framework_op_trace`). The matcher and enricher consume those
> artifacts as plain files and do not link against the EP, but a `qnn_op_trace.json`
> from a build without framework op tracing will be absent.

| Tool | Purpose |
|------|---------|
| [`source_to_optimized_matcher.py`](#source_to_optimized_matcherpy) | Compute source ONNX -> optimized ONNX node and tensor mappings. Optionally extend a QNN EP `qnn_op_trace.json` with `original_sources[]`. |
| [`enrich_profiling_csv.py`](#enrich_profiling_csvpy) | Add original ONNX op names to a QNN EP profiling CSV by joining on the merged framework op trace. Adapts to the input: verbatim-copies BASIC profiling; appends one column when `ONNX Source Ops` is already present and populated; fills the empty column in place plus appends Original when the column is present but every NODE row's cell is empty (AOT-no-sidecar); synthesizes both columns when the column is missing. Mode A reads a pre-computed merged trace; Mode B invokes the matcher inline. |

---

## `source_to_optimized_matcher.py`

Offline tool that computes a structural correspondence between a user-supplied
ONNX model (`source.onnx`) and an ORT-optimized graph. The optimized side is
either an ONNX model (`optimized.onnx`, saved via
`SessionOptions::optimized_model_filepath`) or a QNN-Netron-schema JSON dumped
by the QNN EP's `dump_qnn_ep_input_graph` option (the exact graph the EP saw;
see the QNN EP workflow section below). Output JSON aligns with
the QNN EP `FrameworkOpTrace` schema's `original_sources` extension and can be
stitched together with the existing QNN EP op trace to produce end-to-end
provenance: original ONNX node -> optimized ONNX node -> QNN op.

Approach: zero ORT core changes; relies on the public
`optimized_model_filepath` session option to obtain the post-optimizer graph,
then applies layered structural matching offline.

### Feasibility Assessment

The tool produces both **node-level** and **tensor-level** mappings.

#### Node-level matching strategies

| Case | Strategy | Confidence |
|------|----------|-----------|
| Unchanged 1:1 nodes (same name + op_type) | Exact-name match | High |
| Renamed but topologically identical nodes | Exact I/O signature match | High |
| Optimized node consumes initializer with same name as source consumer | Initializer-name anchoring | High |
| Initializer renamed but data byte-identical | Initializer SHA-256 hash | High |
| Multiple source nodes share initializers with one optimized node | Initializer-name anchoring -> fusion | Medium |
| Source nodes folded into an upstream optimized node (Conv+BN, Conv+Relu) | Producer-preferred lineage extension | Medium |
| Common known patterns (MatMul+Add -> Gemm; Gelu / FastGelu / QuickGelu / BiasGelu / BiasSoftmax / SkipLayerNormalization / FusedMatMul fusions) | Pattern matcher (extensible table) | Medium |

#### Tensor-level matching strategies

| Case | Strategy | Confidence |
|------|----------|-----------|
| Tensor name preserved (model I/O, surviving intermediates) | Exact-name match | High |
| Initializer with same name in both graphs | Initializer-name match | High |
| Initializer renamed but byte-identical | Initializer SHA-256 hash | High |
| Output tensor of a 1:1 matched node | Node-output positional inheritance | High |
| Output tensor of a fused node (multiple source nodes) | Node-output positional, attributed to last source | Medium |

#### What this tool cannot do

| Case | Why | Consequence in output |
|------|-----|----------------------|
| Constant folding (subgraph -> initializer) | Source node identity is destroyed at folding | Reported in `removed_original_nodes` with `likely_reason: "constant_folded"` |
| Layout transposition (NHWC) when initializer data is repacked | Bytes change, names change, no anchor | Falls through to topology heuristic; flagged as low-confidence or unmatched |
| Novel fusion patterns not in the pattern table | Pattern unknown to matcher | Optimized node reported in `unmapped_optimized_nodes`; user can add the pattern |
| Multi-tenant initializer reuse (same weight consumed by multiple unrelated nodes) | Hash matches everyone | Multiple candidate matches reported as Medium-confidence with explanatory note |
| Tensors introduced purely by optimization (e.g., new Transpose I/O) | No source counterpart | Not present in `tensor_mappings` |

#### Honest caveats

1. **The matcher does not run ORT.** It operates on saved ONNX files. If your downstream consumer needs a runtime (in-process) mapping rather than an offline one over saved files, this tool is not the right choice.
2. **Pattern table is hand-curated and lives in one place.** The authoritative list is the `FUSION_PATTERNS` dict in `source_to_optimized_matcher.py`; adding a new ORT fusion means adding one entry there and nowhere else. The matcher will not crash on unknown patterns — it just leaves the optimized node in `unmapped_optimized_nodes`. Branching fusions (LayerNormalization, EmbedLayerNormalization, Attention, GroupQueryAttention) are deliberately absent because the walk-back is linear; the matcher reports partial matches for those via initializer anchoring instead.
3. **Fusion tensor attribution.** For fused nodes, the optimized output tensor is attributed to the **last** source node (deepest in topological order). This matches the semantics of most ORT fusions; the explanatory note in the JSON makes the heuristic explicit.
4. **No guarantees on attribute equivalence.** The matcher checks op_type and I/O structure but does not verify node attributes (kernel size, strides, etc.) are consistent between source and optimized.

#### QNN EP / QDQ-direct workflow

**Why QNN EP sees the QDQ source graph, not a QLinear-fused graph**

When QNN EP is registered, ORT applies Level 1 optimizations to the graph
before handing it to the EP. The Level 2 `QDQSelectorActionTransformer`
— which converts `DQ -> op -> Q` clusters into QLinear* fused ops — does not
apply to QNN EP's partition because QNN EP does not support QLinear*
operators; it handles QDQ natively. As a result, **the graph the QNN EP
receives is structurally very close to the user-supplied source ONNX**, with
only the following Level 1 changes:

| Level 1 transformation | Visible effect |
|---|---|
| `EnsureUniqueDQForNodeUnit` | Each DQ with multiple consumers gets its own copy, named with a `/duplicated` suffix |
| `TransposeOptimizer` | May add or rearrange Transpose nodes |
| `ConstantFolding` (DQ preserved) | Some constants folded, but DQ nodes kept intact |
| `DoubleQDQPairsRemover`, `QDQPropagationTransformer` | Minor QDQ structural cleanup |

**`-o 1` as an approximation of the EP-input graph**

Running `onnxruntime_perf_test -o 1 -u ep_approx.onnx` (basic optimization
only, no QNN EP) produces a file that is a close approximation of what QNN EP
sees.  The gap between `-o 1` output and the true EP-input is small — a
second pass of `MatMulAddFusion` and `QDQFinalCleanupTransformer` from Level
2 may make minor structural changes — but for typical QDQ models the
difference is negligible. The crucial Level 2 `QDQSelectorActionTransformer`
(QDQ->QLinear) is not in that gap: it doesn't apply to QNN EP's partition
regardless of the optimization level used.

**Obtaining the `--optimized-model` for this workflow**

`SessionOptions::SetOptimizedModelFilePath` (`-u` in perf_test) fails when
QNN EP is active because the serialization happens after the EP has compiled
its partition into opaque compiled nodes, and the ORT plugin EP API does not
expose a graph-serialization interface. The practical alternatives, best first:

1. **QNN EP `dump_qnn_ep_input_graph` (exact)** — the QNN EP can dump the
   ONNX graph it actually receives at compile time (after ORT Level 1
   optimizations, before partitioning) as a QNN-Netron-schema JSON. This is
   the true EP-input graph, not an approximation:

   ```bash
   onnxruntime_perf_test ... \
       -i "... dump_qnn_ep_input_graph|1 dump_qnn_ep_input_graph_dir|./"
   # writes <graph_name>.<n>_qnn_ep_input_graph.json
   ```

   `--optimized-model` accepts this JSON directly (detected by the `.json`
   extension) — no `onnx` load is needed for the optimized side:

   ```bash
   python source_to_optimized_matcher.py \
       --source-model    model.onnx \
       --optimized-model graph.0_qnn_ep_input_graph.json \
       --qnn-trace       qnn_op_trace.json \
       ...
   ```

   The dump carries node name/op_type/inputs/outputs and a tensor table that
   flags initializers (tensor `type == 4`). It does not carry initializer data
   bytes, so data-hash matching (renamed-weight detection) is inert for this
   input — name-based and topology-based matching are unaffected, which is what
   the QNN-EP-direct workflow needs.

   **Multiple dumps for the same graph name — which file to feed.** A single
   session can produce more than one dump per graph because ORT may invoke
   `GetCapability` more than once: once before its NHWC layout transform and
   once after on the rewritten graph; once per `If`/`Loop`/`Scan` subgraph;
   and once again on EPContext model loads. The atomic counter in the
   filename (`<graph_name>.<n>_…json`) gives each pass a distinct file, with
   an overwrite warning logged if the same `<n>` is ever about to be reused.

   For each unique `<graph_name>`, feed the file with the **highest `<n>`**
   as `--optimized-model`. That is the graph the QNN EP actually compiled,
   and its node names are the names that appear in `qnn_op_trace.json`'s
   `sources[]` — the join `source -> optimized -> QNN op` only closes if you
   feed the post-transform graph. Lower-numbered files are intermediate
   snapshots and useful for debugging the layout transformer itself, not for
   matcher input. Models with subgraphs need one matcher run per unique
   graph name, each pointed at that graph's highest-numbered dump.

2. **`-o 1 -u` without QNN EP** — a close approximation when the dump is not
   available. Run the model through ORT with only Level 1 optimizations (no
   QNN EP) and save:

   ```bash
   onnxruntime_perf_test.exe -o 1 -u ep_approx.onnx -r 0 model.onnx
   # (omit --plugin_ep_libs / --plugin_eps so QNN EP is not registered)
   ```

   This applies the same Level 1 passes that QNN EP sees, including
   `EnsureUniqueDQForNodeUnit` (which adds `/duplicated` DQ copies).
   The gap is a second pass of `MatMulAddFusion` and
   `QDQFinalCleanupTransformer` from Level 2 — both are negligible for
   typical QDQ models. Crucially, `QDQSelectorActionTransformer` (the
   QDQ->QLinear fusion) does **not** run in either case.

   ```bash
   python source_to_optimized_matcher.py \
       --source-model    model.onnx \
       --optimized-model ep_approx.onnx \
       --qnn-trace       qnn_op_trace.json \
       ...
   ```

3. **Pass the source ONNX as both arguments** — adequate for most QDQ models
   because the EP-input is structurally very close to the source (Level 1
   changes are mostly node copies and Transpose rearrangements, not renames):

   ```bash
   python source_to_optimized_matcher.py \
       --source-model    model.onnx \
       --optimized-model model.onnx \
       --qnn-trace       qnn_op_trace.json \
       ...
   ```

The matcher has an identity fallback in `join_qnn_trace`: if a node name
from `sources[]` is not in the matcher's `node_mappings` but is a node in
the source ONNX, it is passed through unchanged. This covers the common case
where source ≈ EP-input, without requiring a precise EP-input dump.

EP-synthesized node names (`_token_N` suffixes, `/duplicated` copies, bare
`DequantizeLinear` without a namespace) have no user-authored ONNX counterpart
and remain unresolved; this is correct behaviour.

If you use a file saved with CPU EP or full ORT optimization
(`SessionOptions::optimized_model_filepath` from a default session), it will
contain QLinear* fused nodes that do not match what QNN EP consumed; most
`original_sources[]` entries will be empty.

### Usage

#### Basic: produce the source->optimized mapping

```bash
# Step 1: Have ORT save the optimized model.
#         In your inference code:
#
#         session_options.optimized_model_filepath = "optimized.onnx"
#         # ... create session, run once
#
# Step 2: Run the matcher.
python source_to_optimized_matcher.py \
    --source-model path/to/source.onnx \
    --optimized-model path/to/optimized.onnx \
    --output mapping.json

# Optional: --verbose for per-strategy match counts and unmapped diagnostics.
```

#### Combined: extend an existing QNN EP `qnn_op_trace.json`

If the QNN EP's `qnn_op_trace.json`
is also available, pass it via `--qnn-trace` and the matcher will produce an
extended trace where each `op_mappings[]` and `tensor_mappings[]` entry gets
a new `original_sources[]` field, populated by joining on the optimized node
and tensor names. This yields the end-to-end `original ONNX -> QNN op` /
`original ONNX tensor -> QNN tensor` provenance directly.

```bash
python source_to_optimized_matcher.py \
    --source-model path/to/source.onnx \
    --optimized-model path/to/optimized.onnx \
    --output mapping.json \
    --qnn-trace path/to/qnn_op_trace.json \
    --joined-output path/to/qnn_op_trace.with_original_sources.json
```

If `--joined-output` is omitted, the joined file is written next to the input
QNN trace as `<stem>.with_original_sources.json`.

Both op-typed (`type="OP"`) and tensor-typed (`type="TENSOR"`) sources are resolved.
`original_sources` entries follow the QNN EP `TraceSourcePair` schema
(`{name, type}`), with an additional `op_type` field on op entries for human
readability. Source references with no matcher counterpart are left without
an `original_sources` field and reported in the warning counts.

### Output schema

`original_sources[]` (in `node_mappings`) and `original_tensors[]` (in
`tensor_mappings`) follow the QNN EP `TraceSourcePair` schema (`{name, type}`,
where `type` is the string `"OP"` or `"TENSOR"`). Op entries carry an extra
`op_type` field for human readability. For multi-source matches (fusions),
the chain is emitted in source-graph topological order with intermediate
tensors woven between consecutive ops, matching the QNN EP convention
(`op1, tensor_out1, op2, ..., opN`).

```json
{
  "source_model": "...",
  "optimized_model": "...",
  "summary": {
    "total_source_nodes": 7,
    "total_optimized_nodes": 5,
    "matched_optimized_nodes": 5,
    "unmatched_optimized_nodes": 0,
    "by_method": {
      "exact_name": 4,
      "initializer_name": 1
    },
    "total_optimized_tensors": 11,
    "matched_tensors": 11,
    "by_tensor_method": {
      "exact_name": 6,
      "initializer_name": 5
    }
  },
  "node_mappings": [
    {
      "optimized_node": "Gemm_fused_42",
      "optimized_op_type": "Gemm",
      "original_sources": [
        {"name": "matmul1", "type": "OP", "op_type": "MatMul"},
        {"name": "mm_out",  "type": "TENSOR"},
        {"name": "add1",    "type": "OP", "op_type": "Add"}
      ],
      "match_method": "initializer_name",
      "confidence": "medium",
      "notes": "Multiple source nodes share initializers — likely fusion"
    }
  ],
  "tensor_mappings": [
    {
      "optimized_tensor": "y",
      "original_tensors": [{"name": "y"}],
      "match_method": "exact_name",
      "confidence": "high"
    }
  ],
  "unmapped_optimized_nodes": [
    {"name": "...", "op_type": "...", "reason": "no_strategy_matched"}
  ],
  "removed_original_nodes": [
    {"name": "...", "op_type": "Constant", "likely_reason": "constant_folded"}
  ]
}
```

### Extending

To add a new fusion pattern, edit `FUSION_PATTERNS` in
`source_to_optimized_matcher.py`. Each entry maps an optimized op type to a
list of source op-type sequences. The matcher walks back from the optimized
node's input tensors looking for a chain of source nodes matching the
sequence.

---

## `enrich_profiling_csv.py`

Companion tool for the QNN EP's profiling output. Joins a profiling CSV
with the merged framework op trace to add **user-authored** (pre-optimization)
ONNX op names to each NODE-level row, completing the original ONNX ->
optimized ONNX -> QNN op chain in profiling output.

Behavior adapts to the shape of the input CSV — there are four branches
(BASIC verbatim copy, append one column, fill empty column + append one,
synthesize two columns); see [Behavior](#behavior) and
[Output schema](#output-schema-1) below for the exact rules. The QNN EP-side
column this tool complements is written by `Serializer::InitCsvFile()` in
`qnn_profile_serializer.cc`.

### Workflow

The tool accepts the merged trace either as a pre-computed JSON file
(**Mode A**) or as raw ingredients that it joins inline by invoking the
matcher as a library (**Mode B**). The two modes are mutually exclusive —
exactly one input style must be selected.

| | Mode A | Mode B |
|---|--------|--------|
| Inputs | `--merged-trace` | `--source-model` + `--optimized-model` + `--qnn-trace` |
| Intermediate file | Pre-existing `*.merged.json` (from the matcher) | None — matcher runs in-process |
| Best for | Iterating on enrichment, sharing the merged trace as a CI artifact, debugger workflows | One-shot profiling-only workflows |
| Dependencies | Pure stdlib | Requires `onnx` package + sibling-importable `source_to_optimized_matcher.py` |

```bash
# Mode A — pre-computed merged trace.
#
# Step 1: extend the QNN trace with original_sources via the matcher.
python source_to_optimized_matcher.py \
    --source-model source.onnx \
    --optimized-model optimized.onnx \
    --output mapping.json \
    --qnn-trace qnn_op_trace.json \
    --joined-output qnn_op_trace.merged.json
#
# Step 2: enrich the profiling CSV.
python enrich_profiling_csv.py \
    --profiling-csv qnn_profile.csv \
    --merged-trace qnn_op_trace.merged.json \
    --output qnn_profile.with_originals.csv

# Mode B — one-shot from raw ingredients (matcher runs inline).
python enrich_profiling_csv.py \
    --profiling-csv qnn_profile.csv \
    --source-model source.onnx \
    --optimized-model optimized.onnx \
    --qnn-trace qnn_op_trace.json \
    --output qnn_profile.with_originals.csv
```

### Behavior

- The `Event Identifier` column is the lookup key. The HTP `:OpId_{N} (unit)`
  suffix is stripped before lookup, mirroring `Serializer::LookupOnnxSources()`
  at `qnn_profile_serializer.cc`.
- Only op-typed entries (`type="OP"`) from `original_sources[]` are emitted in
  the new column, semicolon-separated, mirroring the existing
  `ONNX Source Ops` column format.
- **The tool detects whether the input has NODE-level events** (any row with a
  non-empty `Event Identifier`). If none — for example, a CSV from BASIC
  profiling or any case where `HasNodeLevelProfiling()` was false at run time —
  no columns are added and the file is copied byte-for-byte unchanged. There
  is nothing to enrich.
- **If NODE-level events are present and the input is missing the
  `ONNX Source Ops` column** (DETAILED/OPTRACE profiling with
  `qnn.enable_framework_op_trace=false` at run time), the tool synthesizes
  both `ONNX Source Ops` and `Original ONNX Source Ops` from the merged
  trace in a single pass. The synthesized `ONNX Source Ops` column matches
  what QNN EP would have written natively when run-time tracing was on.
- **If NODE-level events are present and the input has the `ONNX Source Ops`
  column but every NODE row's cell in that column is empty** (AOT Phase 2
  with framework op trace requested but no sidecar found next to the context
  model — see `Serializer::InitCsvFile()`), the tool fills the existing
  column in place from the merged trace and appends `Original ONNX Source
  Ops`. The detection is conservative: only an entirely-empty column is
  filled, so a CSV with mixed populated/empty cells preserves runtime data.
- Non-NODE rows (SESSION/GRAPH events, no Event Identifier) get empty
  cells in any new columns added.
- QNN op names not present in the merged trace get empty cells and
  are reported in the warning count — the tool does not fabricate provenance.
- Mode A and Mode B produce byte-identical output for the same inputs (the
  matcher logic and join logic are reused; only the input-resolution path
  differs).

### Output schema

The output CSV preserves all original columns. The new columns appended
depend on the input shape:

| Input | Columns appended | Existing column overwritten? |
|-------|------------------|------------------------------|
| BASIC profiling (no NODE-level events) | None — output is a verbatim byte-for-byte copy of the input | n/a |
| DETAILED/OPTRACE with `ONNX Source Ops` present and populated (framework op trace ran at run time) | `Original ONNX Source Ops` only | No — runtime values preserved |
| DETAILED/OPTRACE with `ONNX Source Ops` present but every NODE row's cell empty (AOT Phase 2 with no sidecar found) | `Original ONNX Source Ops` only | Yes — existing column filled in place from the merged trace |
| DETAILED/OPTRACE with `ONNX Source Ops` missing (run-time tracing off entirely) | `ONNX Source Ops`, then `Original ONNX Source Ops` | n/a |

Example (DETAILED + run-time tracing off — both columns synthesized):

```
Msg Timestamp,...,Event Identifier,ONNX Source Ops,Original ONNX Source Ops
...,qnn_conv_0:OpId_5 (cycles),conv1,conv1;bn1
...,qnn_fc_3:OpId_8 (cycles),flat1;Gemm_fused_42,flat1;matmul1;add1
```

---

## Requirements

- Python 3.9+ (PEP 604 union syntax in module-level annotations; `onnx` package floor).
- `onnx` (any recent version, `pip install onnx`) — required for the matcher and for Mode B of the enrichment tool. Mode A of the enrichment tool is pure stdlib and does not need `onnx`.

For convenience, this directory ships its own [`requirements.txt`](./requirements.txt) listing the runtime deps; install with `pip install -r qcom/tools/op_trace_matcher/requirements.txt`. The repo-wide `requirements-dev.txt` already covers these plus `pytest` for the unit tests.

No ORT runtime dependency — both tools operate on saved `.onnx` files and JSON/CSV artifacts only.

## Output safety

Both tools overwrite an existing output file (`--output`, and the matcher's
`--joined-output`), printing a warning to stderr when they do. The enricher
additionally refuses a `--profiling-csv` that already carries the
`Original ONNX Source Ops` column, since that indicates a previously-enriched
CSV — enrich the original QNN EP profiling CSV instead.
