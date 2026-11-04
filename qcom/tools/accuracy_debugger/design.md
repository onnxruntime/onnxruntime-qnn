# Accuracy Debugger: design notes

This document describes how the accuracy debugger works, how it differs from the legacy
`qairt-accuracy-debugger snooping_ort_qnn` and how to run its tests. For setup and usage, see
[README.md](README.md).

## Contents

- [Overview](#overview)
- [Modules](#modules)
- [Dependencies](#dependencies)
- [Tensor selection and matching](#tensor-selection-and-matching)
- [Layerwise subgraphs](#layerwise-subgraphs)
- [Framework op tracing](#framework-op-tracing)
- [CPU sessions](#cpu-sessions)
- [Output layout](#output-layout)
- [Report schemas](#report-schemas)
- [Changes from the legacy tool](#changes-from-the-legacy-tool)
- [Known limitations](#known-limitations)
- [Running the tests](#running-the-tests)

## Overview

`accuracy_debugger.py` finds where a quantized (QDQ) ONNX model loses accuracy when it runs on
ONNX Runtime's QNN Execution Provider. It runs the model on QNN EP (the **target**) and compares every
selected intermediate tensor against a **reference**. The reference is one of:

- the QDQ model itself on ORT's CPU EP (the default);
- a separate FP32 or QDQ reference model on ORT's CPU EP (`--reference_model`, oneshot only);
- a directory of pre-computed `.raw` tensors (`--golden_reference`).

It then writes per-tensor comparator scores (MSE, cosine, SNR, ...) to CSV and JSON reports, plus
optional plots.

| Algorithm | What it does |
|---|---|
| `oneshot` (default) | Runs the whole QDQ model once on the target with the selected activation Q/DQ outputs exposed, and compares them with the reference. Errors accumulate along the graph, as they do in deployment. |
| `layerwise` | Extracts each layer (op → its Q → DQ) as a subgraph, feeds it reference activations and runs it on the target in a fresh process. Each row shows the error that one layer introduces on its own. |

## Modules

| Module | Role |
|---|---|
| `accuracy_debugger.py` | Standalone entry point of the CLI. |
| `parser.py` | Command-line parser and post-parse validation. |
| `ort_qnn_snooper.py` | Oneshot and layerwise snooping: output selection, layer extraction and execution (a fresh `spawn` worker per layer), target-to-reference tensor matching and report rows. |
| `ort_qnn_inference_engine.py` | QNN HTP and CPU inference sessions: QNN EP registration, trace and graph dump options, host dequantization of captured Q/DQ pairs (F27) and the picklable `run_inference_job` worker entry point. |
| `reference.py` | ORT CPU reference runner, reference-output dumper and golden-reference loader. |
| `subgraph_extractor.py` | Layer-window subgraph extraction for layerwise snooping. |
| `op_trace.py` | Framework op trace lookup: fusion groups, tensor mapping and the trace columns. |
| `verification.py` | Verification of target outputs against reference outputs with a set of comparators. |
| `comparators.py` | Tensor comparators. |
| `reporting.py` | Activation statistics, CSV/JSON reports and comparator score plots. |
| `inputs.py` | Input-list parsing and loading of raw input tensors. |
| `onnx_utils.py` | ONNX model helpers shared by the reference runner, the inference engine and the subgraph extractor. |
| `common.py` | Shared constants, exceptions, pydantic bases and small helpers. |
| `log.py` | Logging helpers, including the logging setup of spawned workers. |

## Dependencies

Hard dependencies: `numpy`, `onnx>=1.18.0` (the first release whose `onnx.utils.Extractor` is
iterative, so large graphs do not hit Python's recursion limit), `onnxruntime>=1.24.2`, `pandas`,
`pydantic>=2` and `typing_extensions`. `onnxruntime-qnn` provides the QNN EP plugin and the HTP backend, and
needs `onnxruntime>=1.24.2`. `matplotlib` is optional, for `plots/<comparator>.png`.
[requirements.txt](requirements.txt) pins the versions that the tool was validated with.

The CPU-only parts (reference generation, report writing, tests that do not need QNN) also work with an
older CPU onnxruntime. The tool does not check the ORT version at runtime. On an ORT older than 1.24,
registering the QNN EP fails with `RuntimeError("Failed to register QNN EP library: ...")`.

The repository's top-level `onnxruntime/` source folder has no `__init__.py`, so it does not shadow an
installed `onnxruntime` package. It matters only when `onnxruntime` is not installed and the repository root is
on `sys.path`, for example with `python -m pytest` from the repository root: `import onnxruntime` then gives an
empty namespace package, and the snooper fails with a confusing ImportError.

## Tensor selection and matching

Oneshot exposes the selected activation Q/DQ outputs as graph outputs of the target model. Layerwise compares
the Q/DQ outputs that each extracted subgraph returns (see [Layerwise subgraphs](#layerwise-subgraphs)).

- Q/DQ outputs are selected and named by data edges, not by node positions (F30). A selected layer compares
  every Q/DQ output that follows it through Q/DQ nodes, on every output branch, such as both outputs of a
  `Split` and a Q → DQ → Q → DQ requantization chain. A row is named after the op that feeds its Q/DQ chain;
  a chain on a graph input is named after its own node.
- Weight and bias Q/DQ outputs are not compared (F26): Q/DQ outputs computed only from initializers and
  `Constant` nodes, either directly, through layout ops that rearrange weights, or through ops that ORT folds.
- With trace options, tensors that only a QNN HTP fusion produces and consumes are not exposed (F28, see
  [Framework op tracing](#framework-op-tracing)).

Each target tensor is matched to exactly one reference tensor, by the first strategy that applies:

1. The generated or supplied framework op trace. A tensor that the trace maps to more than one reference tensor
   is skipped with a warning.
2. Exact name.
3. Sanitised name.
4. For an FP32 reference, a reference tensor whose name is a prefix of the target name, followed by
   `_ / . :`.

When the reference is an FP32 model (`--reference_model model.fp32.onnx`), only DequantizeLinear outputs
are compared.

Tensor file names are sanitised: every character outside `[A-Za-z0-9_]` becomes `_`, and a leading `_` is
added unless the name starts with a letter or `_` (`/conv1/Conv` → `_conv1_Conv`).

## Layerwise subgraphs

Q/DQ, Transpose, Identity and Constant nodes are skipped (they do not change values). Every other node
whose output feeds Q/DQ nodes is extracted from its first input to its last Q/DQ output. Its rows are the
Q/DQ outputs that follow it through Q/DQ nodes, found by data edges on every output branch (F30). A layer whose
only Q/DQ outputs are weights and biases is skipped, as a layer without Q/DQ nodes is (F26). Each
subgraph runs in a fresh `spawn` worker process. This matches the legacy tool: QNN/HTP state cannot leak
from one layer into the next. With trace options, each QNN HTP fusion is one subgraph instead (see
[Framework op tracing](#framework-op-tracing)).

A subgraph that runs on QNN HTP consumes quantized tensors, as its layers do in the full model (F29):

- Weights, biases and other tensors that only initializers determine are computed in the subgraph by the
  model's own nodes, such as a weight's DQ and its quantized initializer, instead of being float inputs.
- Each input that a DQ produces is quantized again by an inserted `<input>_requant_QuantizeLinear` →
  `<input>_requant_DequantizeLinear` pair with that DQ's parameters. The pair passes DQ outputs through
  unchanged. Its QuantizeLinear runs on CPU, as the model's input QuantizeLinear does with QNN EP's default
  `offload_graph_io_quantization`.

QNN HTP therefore compiles each layer as it does in the full model: on ResNet-18, a `Conv2d` on ufxp8
activations with static weights instead of an fp32 `Conv2d` with weight inputs. Subgraphs forced onto CPU
with `--set_cpu_layers` keep float inputs, as before, because ORT CPU would fuse the inserted pair into
integer kernels.

Each subgraph input is taken from the first source that has it:

1. The model's input data, from the input list.
2. With `--set_cpu_layers` (mixed backend), the target outputs of the earlier subgraphs, on QNN HTP or CPU.
3. The reference tensors, by raw name and then by sanitised name.

## Framework op tracing

Use a QNN EP framework trace to resolve tensor names and show the compiled fusion groups. The QNN EP side is
described in
[QNN EP Framework Op Tracing](../../../docs/execution_providers/QNN-ExecutionProvider.md#qnn-ep-framework-op-tracing).

### Generated and supplied traces

`--enable_framework_op_trace` sets the EP option of the same name on the target session and places the trace
in the run's output directory. Each compilation writes `qnn_op_trace.json` into a new `op_trace/run_*`
directory: oneshot under the output directory, layerwise under each `subgraph_data/<layer>/`, including when
inference runs in a spawned worker. This prevents a previous run or layer from supplying stale mappings. The
Python engine result exposes the generated path as `op_trace_file`.

`--op_trace_file` reads a schema version 2 trace, including single-SoC and multi-SoC files. An explicitly
supplied trace takes precedence over generated mappings. Tensor aliases are resolved first, followed by
graph-based Q/DQ correspondence and fusion boundary outputs. Quantized and dequantized tensors keep
their distinct identities for QDQ comparisons. Ambiguous trace matches are skipped with a warning;
missing, malformed or unsupported traces use the existing name heuristics. Without trace options the report
columns and mapping are unchanged.

Generated traces describe the actual debug compilation, while a supplied trace describes the compilation
that produced that file.

### Fusion protection

With trace options, the debugger also keeps QNN HTP fusions intact (F28). Before the reference run, it compiles
the unmodified QDQ model on QNN HTP with tracing but without running it (`fusion_probe/op_trace/run_*`), and
reads the fused node groups from that trace. Exposing a tensor that only a fusion produces and consumes stops
QNN EP from fusing the group, so:

- Oneshot compares a fusion's output Q/DQ tensors, not the tensors inside it. A `--set_intermediate_layers`
  layer inside a fusion selects the fusion's outputs, with a warning that names them.
- Layerwise extracts each fusion as one subgraph, from the input of its first layer to its outputs. The
  subgraph directory and the rows are named after the fusion's last layer. Any of its layers, by name or op
  type, selects the whole fusion in `--set_intermediate_layers` and `--set_cpu_layers`. Like every QNN HTP
  subgraph, it quantizes its inputs again (F29), so fusions that start at a DQ, such as `GeluFusion`, form.

If the probe fails or QNN EP writes no trace, a warning says so and the fusions are not protected. A fusion of
one ONNX node and the DQs that feed it, such as `DQQFusion`, hides no tensor and changes nothing.

### Trace columns

Trace-enabled CSV reports add `QNN_Op_Name`, `QNN_Op_Type`, `Fusion_Pattern`, `Original_ONNX_Nodes`, and
`CPU_Fallback_Reason`. JSON stores the same data as layer metadata using snake_case keys. `Layer Name`
always names the ONNX op that feeds the Q/DQ node; fused source nodes appear in `Original_ONNX_Nodes`.
Trace groups made only of EP-inserted ops (such as layout `Transpose_token_*`) are ignored. The QNN
columns describe **compiled** operations; CPU rejection reasons and any automatic ORT retry on CPU are
reported separately. A DQ output whose Q output is also captured is dequantized on CPU by the engine (F27)
and says so in `CPU_Fallback_Reason`. Group provenance that includes input tensors is not treated as an output
alias. When the EP renames a rejected node (e.g. `DequantizeLinear_token_N`), its reason is attributed through
the `*_qnn_ep_input_graph.json` dumps next to the trace, so add `--dump_qnn_ep_input_graph` to see it. Layers
forced onto CPU with `--set_cpu_layers` report only `Layer forced to CPU by set_cpu_layers.`; a supplied
trace never labels them.

### Offline name matching

When optimizer names differ, supply a merged trace containing `original_sources`. To produce the
matcher input, add `--dump_qnn_ep_input_graph` with `--enable_framework_op_trace`. Then run the adjacent
offline [matcher](../op_trace_matcher/README.md):

```bat
python ..\op_trace_matcher\source_to_optimized_matcher.py --source-model model.qdq.onnx --optimized-model RUN_DIR\op_trace\run_ID\GRAPH.N_qnn_ep_input_graph.json --qnn-trace RUN_DIR\op_trace\run_ID\qnn_op_trace.json --output mapping.json --joined-output qnn_op_trace.merged.json
```

`RUN_DIR` is the oneshot output directory (see [Output layout](#output-layout)). Select the highest numeric `N`
for each graph in that compilation directory. The debugger consumes the merged JSON directly; it does not
invoke the matcher automatically.

### EPContext models

The target must be an original QDQ ONNX model. Cached `com.microsoft::EPContext` models are not supported,
including models that retain Q/DQ boundary nodes. Exposing intermediates or extracting individual
layers can alter fusion; with trace options the debugger keeps the fusions that the probe finds (F28).

## CPU sessions

`--set_cpu_layers` (engine `use_cpu=True`) creates a regular ORT session with `CPUExecutionProvider`.
CPU inference does not load or register the QNN plugin and needs no QNN backend library. QNN trace,
EP input graph dump, and CPU fallback control options are ignored for CPU sessions. These sessions return
`op_trace_file=None`; QNN HTP sessions continue to generate traces when requested.

## Output layout

```text
<working_directory or ./working_directory>/<oneshot|layerwise>_snooping/<output_directory or YYYY-MM-DD_HH-MM-SS>/
  accuracy_debugger_user_info.log        rotating log (10 MiB x 40)
  reference_outputs/<sanitized>.raw      reference dump (float32)
  reference_outputs/profile_info.json    reference tensor profiles
  target_outputs/<sanitized>.raw         oneshot target dump (float32)
  oneshot.csv  oneshot.json   |  layerwise.csv  layerwise.json
  plots/<comparator>.png                 when matplotlib is installed
  elementwise_stats.log                  appended, only when some element exceeds the allowed steps
  subgraph_data/<sanitized layer>/extracted_model.onnx
  subgraph_data/<sanitized layer>/target_outputs/<sanitized>.raw
  op_trace/run_*/qnn_op_trace.json       oneshot trace, with --enable_framework_op_trace (graph dumps sit alongside)
  subgraph_data/<sanitized layer>/op_trace/run_*/qnn_op_trace.json
  fusion_probe/op_trace/run_*/qnn_op_trace.json   compile-only trace of the unmodified model, with either trace option
```

`reference_outputs/` is not written with `--golden_reference`. The golden loader reads every `*.raw` file
below the given directory, except under `target_outputs/` and `subgraph_data/` (F15, F24).

## Report schemas

**`oneshot.csv`**: `Layer Name, Layer Output, Layer Type, Layer Shape, Source(Min, Max, Median), Target(Min, Max, Median), <comparators...>`.
`Layer Name` and `Layer Type` describe the op that feeds the Q/DQ node.

**`layerwise.csv`**: `Layer Name, Layer Output, Layer Status, Layer Type, Layer Shape, Source(Min, Max, Median), Target(Min, Max, Median), <comparators...>, Exception`.
A row is appended, and the file flushed, as soon as each layer finishes.

- `Layer Status` is `SUCCESS`, `SKIP` (extraction failed), `INFERENCE_FAILURE` or `VERIFICATION_FAILURE`.
- Non-SUCCESS rows have `NaN` comparator cells, `()` shape and distributions, the subgraph end tensor as
  `Layer Output`, and the reason in `Exception`.

With trace options, the five trace columns follow `Target(Min, Max, Median)` in both reports.

Cells are Python-formatted: a shape looks like `(1, 64, 56, 56)` and a distribution like `(0.0, 1.5, 0.25)`.
Float32 values keep their shortest representation, so `0.1` is written as `0.1`, and the text is the same
under numpy 1.x and 2.x.

**`<algorithm>.json`**:

```json
{"header": {"header_version": {"major": 0, "minor": 1, "patch": 0},
            "artifact_type": "ONESHOT_SNOOPING_REPORT | LAYERWISE_SNOOPING_REPORT",
            "algorithm": "oneshot | layerwise",
            "version": {"major": 0, "minor": 1, "patch": 0}},
 "layers": [{"layer_name": "...", "layer_output": "...", "layer_status": "SUCCESS (layerwise only)",
             "layer_type": "...", "layer_shape": [1, 64, 56, 56],
             "source_min_max_median": [0.0, 1.0, 0.5], "target_min_max_median": [0.0, 1.0, 0.5],
             "comparators": {"mse": 0.01, "snr": "NaN"}, "exception": "(layerwise only)"}]}
```

NaN and ±inf scores are written as `"NaN"`; `null` never appears in `comparators`. The JSON is written from
the report data frame by pandas `to_json`, which rounds scores to 10 decimal places (its default
`double_precision`): an `mse` of `0.019589794799685478` in `oneshot.csv` is `0.0195897948` in `oneshot.json`.
The file is written with an indent of 4.

## Changes from the legacy tool

Compared with the legacy `qairt-accuracy-debugger snooping_ort_qnn`:

- The tool is invoked as `python accuracy_debugger.py <args>`, without a subcommand.
- `--input_list` replaces `--input_sample`; the former `--input_sample` alias is no longer accepted.
- `--enable_framework_op_trace`, `--dump_qnn_ep_input_graph` and `--op_trace_file` are new (see
  [Framework op tracing](#framework-op-tracing)).

The behaviour changes below are referenced by ID from code comments and tests. There is no F8.

| ID | Change |
|---|---|
| F1 | Layerwise subgraph directories are `subgraph_data/<sanitized node name>`. A colliding name gets `_<idx>`; an unnamed node becomes `node_<idx>`. Legacy used the raw node name, so `/conv1/Conv` escaped to the filesystem root. |
| F2 | Plot and log messages name the algorithm that actually ran (legacy always said oneshot). |
| F3 | JSON `layer_status` and `exception` are top-level layer keys, no longer under `comparators`. |
| F4 | Layerwise JSON header says `LAYERWISE_SNOOPING_REPORT` / `layerwise`. |
| F5 | The reference model is run as given. Legacy rewrote INT_MAX/65504 constants to -1. |
| F6 | Only the first input-list sample runs, with a warning when the list has more. No `batch_N/` directories. |
| F7 | Plots are `plots/<comparator>.png` (matplotlib, optional) instead of plotly `.html`. |
| F9 | Reference inference errors raise. Legacy turned them into `None` and crashed later. |
| F10 | Oneshot `--set_intermediate_layers` that matches nothing in either graph compares graph outputs only, with a warning. If it matches only in the QDQ graph, every reference tensor is dumped (legacy). |
| F11 | Elementwise diffs of integer tensors are computed in int64, so unsigned values no longer wrap around. |
| F12 | New subgraph extractor: deterministic input order, no recursion limit, `''` optional names ignored, initializer-backed inputs treated as constants, graph-input start tensors supported, >2 GB models via external data. |
| F13 | A layer whose outputs cannot be paired with any reference gives a `VERIFICATION_FAILURE` row. Legacy silently dropped it. |
| F14 | Golden references load with any ONNX dtype; dynamic dims are not reshaped. |
| F15 | The golden loader ignores `target_outputs/` and `subgraph_data/` subtrees. |
| F16 | Re-running in the same process logs to the new run's file (stale handlers are replaced). |
| F17 | The `QAIRT_LOG_LEVEL` environment override is gone. |
| F18 | `l1_error` and `mae` return Python floats (same CSV text); `topk` uses a stable sort, so `topk` scores can differ from legacy when values tie at the top-k boundary (common on quantised outputs). |
| F19 | The target dump's 255-byte file-name check uses the sanitised name. |
| F20 | Engine output order: requested outputs first, then the original graph outputs in their order. |
| F21 | Parser hygiene: empty comma items dropped, duplicate comparators removed, `--dump_elementwise_stats` steps must be ≥ 0, `--help` texts fixed. |
| F22 | A node with no first input gives a `SKIP` row with `Layer has no primary input.` |
| F23 | A layerwise run with no rows still writes a header-only `layerwise.csv`. |
| F24 | Golden directories with `batch_1/`, `batch_2/`, ...: only `batch_1/` is loaded (with a warning); on duplicate tensor names the first file wins. |
| F25 | `--output_directory ""` gives `<base>/<algorithm>_snooping` with no leaf (legacy kept, documented). |
| F26 | Q/DQ output selection skips weight and bias tensors: Q/DQ outputs computed only from initializers and `Constant` nodes, either directly, through layout ops that rearrange weights (`Reshape`, `Transpose`, `Concat`, ...), or through ops that ORT folds. Legacy exposed them as graph outputs and reported them. QNN EP rejects such DQ nodes (`QNN EP is configured to not take DQ nodes that generate a graph output.`, or `QNN EP does not support a standalone DQ op with per-channel quantization` for per-channel weights), so ORT ran them on CPU. Q/DQ outputs that QNN EP computes from weights, such as a `MatMul` of two weights, are still compared. The CPU reference still captures weight and bias tensors, so a oneshot `reference_outputs/` directory still works as a layerwise `--golden_reference`. A oneshot run that selects only weight and bias Q/DQ outputs (a weight-only QDQ model, or such a `--set_intermediate_layers` filter) compares graph outputs only, with a warning. Layerwise skips a layer whose only Q/DQ outputs are weights and biases, as it skips layers without Q/DQ nodes, and no longer ends a layer's subgraph at a weight DQ. Exposed activation DQs whose Q output is not captured still show the graph-output reason in the trace and `CPU_Fallback_Reason`: QNN EP leaves graph-output dequantization to the CPU EP by default, as it does for the model's own outputs. |
| F27 | QNN HTP sessions with CPU fallback allowed no longer return a DQ output whose Q input is also captured. The engine dequantizes the returned Q output on ORT CPU and keeps output names and order. With its default `offload_graph_io_quantization`, QNN EP leaves graph-output DQs to the CPU EP and maps each one to its Q tensor, so a captured Q/DQ pair named that QNN tensor twice. Compilation then lost an output (`Inconsistent output sizes`) and ORT silently reran the whole model on CPU, so every score compared CPU with CPU. On ResNet-18 this hit oneshot `--set_intermediate_layers` and 44 of 64 layerwise layers. The QNN graph and values match QNN EP's own graph-output dequantization. With trace options, these DQ rows report `Dequantized on ORT CPU from its captured Q output; QNN EP cannot return a Q output and its DQ output together.` This covers every DQ whose scale and zero point only initializers determine, including parameters that PyTorch exports share through `Identity` nodes: ORT folds such nodes before QNN EP partitions the graph, so their pairs failed as well. DQs whose scale or zero point is computed at runtime, for example from an input, stay on the session, as do all outputs of `--set_cpu_layers` sessions and of sessions with CPU fallback disabled (QNN EP then turns the offload off). |
| F28 | With trace options, output selection and layerwise extraction keep QNN HTP fusions intact. An extra compile-only QNN HTP session of the unmodified model, with tracing, lists the fused node groups. Oneshot no longer exposes tensors that only a fusion produces and consumes; it compares the fusion's outputs, and a `--set_intermediate_layers` layer inside a fusion selects them with a warning. Layerwise extracts a fusion as one subgraph, named after its last layer, instead of one subgraph per layer. Legacy exposed these tensors, so QNN EP could not fuse the group and ran its layers separately or on CPU. On a test model with `Rank6ToRank5Fusion`, the exposed tensors put the fusion's 10 nodes on CPU in oneshot, and its first `Reshape` subgraph ran entirely on CPU in layerwise; both now compile as the fusion on QNN HTP. Without trace options, fusions are not protected. |
| F29 | Layerwise subgraphs that run on QNN HTP consume quantized tensors, as their layers do in the full model, with and without trace options. A subgraph computes its weights, biases and other tensors that only initializers determine with the model's own nodes, and an inserted `QuantizeLinear` → `DequantizeLinear` pair with the producing DQ's parameters quantizes each input that a DQ produces again (see [Layerwise subgraphs](#layerwise-subgraphs)). Legacy fed these tensors as float inputs, so QNN HTP compiled the layers as float ops: on ResNet-18, `Conv`, `Add`, `GlobalAveragePool`, `Flatten` and `Gemm` compiled as fp32 QNN ops with weight inputs, and on a test model `GeluFusion` did not form and its `Erf` ran on CPU. Each layer now compiles as in the full model, for example a `Conv2d` on ufxp8 activations with static ufxp8 weights and an sfxp32 bias, so layerwise scores change. On ResNet-18, Conv and Gemm errors grow (`layer4.1.conv2` from 0.0039 to 0.030 MSE on its Q output) and some Add errors shrink; reports keep the same rows. Subgraph files now contain their layers' quantized weights, and only activations are fetched as inputs. `--set_cpu_layers` subgraphs are extracted as before. |
| F30 | Q/DQ outputs are selected and named by data edges instead of node positions, with and without trace options. Legacy assumed that a layer's Q and DQ are the next two nodes: layerwise and oneshot `--set_intermediate_layers` compared the Q and DQ at positions +1 and +2 of a selected layer, and every row was named after the node one (Q) or two (DQ) positions before its tensor. Where branches interleave in node order, both picked unrelated nodes. At each ResNet downsample block, `conv1` and `downsample.0` precede their four Q/DQ nodes: legacy layerwise skipped `conv1` and reported `conv1`'s Q output as the downsample Conv's only row, and oneshot named that tensor after the downsample Conv and three others after Q nodes. A selected layer now compares every Q/DQ output that follows it through Q/DQ nodes, on every output branch, such as both outputs of a `Split` and a Q → DQ → Q → DQ requantization chain. Its layerwise subgraph returns them all, plus any model graph output that its own nodes compute. A graph output of another branch inside the wider window, as in the multi-input/output model and YOLOv3-tiny, is left out, so it is not reported twice and its branch is not extracted. A row is named after the op that feeds its Q/DQ chain; a chain on a graph input is named after its own node. Oneshot without `--set_intermediate_layers` compares the same tensors as before. Across 9 test models, positional selection differed in 8 (all but EfficientNet-Lite): it missed or misattributed 6 Conv layers of ResNet-18, 8 of ResNet-50 and RegNet, 6 layers at the YOLOv3-tiny route, 16 layers of a multi-input/output model, and the second Q/DQ pair of 48 LeakyRelus with requantization chains. The ResNet-18 layerwise report has 64 rows instead of 55 and 14 oneshot rows have new names; the e2e tests correct the legacy references accordingly. The extra requantization pairs depend on F27 for parameters shared through `Identity` nodes. |

## Known limitations

- Without trace options, adding intermediate Q/DQ outputs as graph outputs can change how QNN fuses node
  units, as in legacy. With trace options, only the fusions that QNN EP lists in its trace are kept (F28).
  Fusions that the QNN HTP backend makes on the QNN graph are not in the trace and are not protected.
- A QNN HTP layerwise subgraph input stays float, as before F29, when the DQ that produces it has a scale or
  zero point computed at runtime, int32 or unknown data, or no zero point, data other than uint8 and a scale
  that is not an initializer. The debug log names each such input and the reason.
- Golden `.raw` files are assumed to be float32 dumps produced by this tool.
- The CLI has no flags for dynamic input dimensions (`onnx_symbols`) or for disabling the silent CPU
  fallback.
- A captured DQ whose scale or zero point is computed at runtime, for example from an input, cannot be
  dequantized on the host (F27).
  If its Q output is also captured, QNN EP fails with `Inconsistent output sizes` and ORT reruns the whole
  session on CPU. For any such retry the engine logs a warning, and layerwise `SUCCESS` rows carry the
  retry reason in `Exception` because their scores compare CPU with CPU.
- Worker processes append to the run log with a non-rotating handler. If the parent rotates the log at
  10 MiB while a worker is running, that one layer's worker lines end up in the rotated file.
- `rtol_atol` keeps the legacy formula, which divides by `len()` of the unflattened tensor, its first
  dimension, instead of by its number of elements. Its scores are only correct for 1-D tensors and are usually
  negative for activations such as `(1, 64, 56, 56)`. `adjusted_rtol_atol` flattens the tensors first.

## Running the tests

```bash
# from the accuracy_debugger directory
python -m pytest tests -q
```

`tests/conftest.py` adds the debugger directory to the import path so tests use the same local modules
as the script. Tests can also be run from the repository root with
`python -m pytest qcom/tools/accuracy_debugger/tests -q`.

Comparator parity tests use fixed reference values, so SciPy is not required. Engine CPU tests use regular
ORT sessions, including spawned inference workers, and do not require the QNN plugin.

The CPU end-to-end tests (`test_adbg_e2e_cpu.py`) run the target on ORT's CPU EP through test fixtures,
including a picklable CPU worker for spawn tests. Tests that need the real QNN EP carry the `qnn` marker and
are skipped unless `onnxruntime-qnn` is installed and a QNN device is available. The ResNet-18 QNN
end-to-end test (`test_adbg_e2e_qnn.py`) also needs `ADBG_E2E_ASSET_DIR` to point to a directory with
`r18.qdq.onnx`, `dog.raw` and the reference reports `oneshot_full.csv`, `oneshot_partial.csv`,
`layerwise_full.csv` and `layerwise_partial.csv`:

```bash
ADBG_E2E_ASSET_DIR=/path/to/r18 python -m pytest tests -m qnn -q
```

The test ignores the weight and bias rows of these reports (DequantizeLinear outputs of initializers, see F26).
