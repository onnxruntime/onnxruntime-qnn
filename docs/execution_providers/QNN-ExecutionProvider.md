# QNN Execution Provider

The QNN Execution Provider (QNN EP) for ONNX Runtime enables hardware accelerated execution on Qualcomm chipsets.
It uses the Qualcomm AI Runtime SDK (QAIRT SDK) to construct a QNN graph from an ONNX model which can
be executed by a supported accelerator backend library.
ONNX Runtime QNN EP can be used on Windows devices with Qualcomm Snapdragon SOC's.

- **Note**: QNN EP version < 2.0 is **NOT** included in this page, please refer to [this page](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html) for QNN EP version < 2.0.
- **Note**: QNN EP version < 2.0 will be deprecated and no longer maintained.

## Contents

- [Install Pre-requisites (Build from Source Only)](#install-pre-requisites-build-from-source-only)
- [Build (Windows)](#build-windows)
- [Pre-built Packages](#pre-built-packages)
- [Qualcomm AI Hub](#qualcomm-ai-hub)
- [Configuration Options](#configuration-options)
- [Flexible Context Binary (FCB) / multi-SoC EP context](#flexible-context-binary-fcb--multi-soc-ep-context)
- [Supported ONNX operators](#supported-onnx-operators)
- [Running a model with QNN EP's HTP backend (Python)](#running-a-model-with-qnn-eps-htp-backend-python)
- [Running a model with QNN EP's GPU backend](#running-a-model-with-qnn-eps-gpu-backend)
- [Running an LLM model with QNN EP's Genie backend](#running-an-llm-model-with-qnn-eps-genie-backend)
- [QNN context binary cache feature](#qnn-context-binary-cache-feature)
- [QNN EP Framework Op Tracing](#qnn-ep-framework-op-tracing)
- [QNN EP Input Graph Dump](#qnn-ep-input-graph-dump)
- [QNN EP Profiling](#qnn-ep-profiling)
- [QNN EP weight sharing](#qnn-ep-weight-sharing)
- [Usage](#usage)
- [Error handling](#error-handling)
- [Add new operator support in QNN EP](#add-new-operator-support-in-qnn-ep)
- [QNN User-Defined Operation (UDO)](#qnn-user-defined-operation-udo)
- [Mixed precision support](#mixed-precision-support)
- [LoRAv2 support](#lorav2-support)

## Install Pre-requisites (Build from Source Only)

If you build QNN EP from source, you should first
download the Qualcomm AI Runtime SDK (QAIRT SDK) from [https://qpm.qualcomm.com/#/main/tools/details/Qualcomm_AI_Runtime_SDK](https://qpm.qualcomm.com/#/main/tools/details/Qualcomm_AI_Runtime_SDK) and the ONNX Runtime SDK from [https://github.com/microsoft/onnxruntime/releases](https://github.com/microsoft/onnxruntime/releases).

### QNN Execution Provider Version Requirements

ONNX Runtime QNN EP has been built and tested with the following SDK version combinations on Windows:
| QNN EP Version | QAIRT SDK Version | ONNX Runtime Version |
|----------------|-------------------|----------------------|
| v2.5.0         | v2.49.40          | v1.26.0              |

> **Note**: ONNX Runtime QNN EP 2.5.0 was built and tested with ORT 1.26.0 but it is compatible with ORT >= 1.24.1

## Build (Windows)
For build instructions, please see the [BUILD page](./build.md).

## Pre-built Packages
- [Python package](https://pypi.org/project/onnxruntime-qnn/)
  - Requirements:
    - Windows ARM64 (for inferencing on local device with Qualcomm NPU)
    - Windows X64 (for quantizing models. see [Generating a quantized model](./QNN-ExecutionProvider.md#generating-a-quantized-model-x64-only))
    - Linux ARM64 (for inferencing on local Qualcomm-powered Linux devices)
    - Linux x64 (for generating a quantized model on Linux Host)
    - Python 3.11.x
    - Numpy 1.25.2 or >= 1.26.4
  - Install: `pip install onnxruntime-qnn`
- [NuGet package - `Qualcomm.ML.OnnxRuntime.QNN`](https://www.nuget.org/packages/Qualcomm.ML.OnnxRuntime.QNN)
  - A single package covering Windows ARM64 (ARM64X), ARM64EC, and x64 (`win-arm64`, `win-arm64ec`, `win-arm64x`, `win-x64` RuntimeIdentifiers) with appropriate fallbacks.
- Archives (`.zip` and `.tgz`)
  - **Note**: Ships the QNN EP shared library and headers for use outside of Python on Windows and Linux hosts and targets.
- Maven package (Android)
  - **Note**: The Maven package supports Android ARM64
  - Group ID / Artifact ID: `com.qualcomm.qti:onnxruntime-android-qnn`
  - This release is validated against the following dependency versions:
    | Dependency | Maven Coordinate | Version |
    |---|---|---|
    | ONNX Runtime Android | `com.microsoft.onnxruntime:onnxruntime-android` | `1.26.0` |
    | QNN Runtime | `com.qualcomm.qti:qnn-runtime` | `2.49.40` |

## Qualcomm AI Hub
Qualcomm AI Hub can be used to optimize and run models on Qualcomm hosted devices.
OnnxRuntime QNN Execution Provider is a supported runtime in [Qualcomm AI Hub](https://aihub.qualcomm.com/)

## Configuration Options
The QNN Execution Provider supports a number of configuration options. These provider options are specified as key-value string pairs.
### EP Provider Options

|`"backend_type"`|Description|
|---|-----|
|'cpu'|Enable CPU backend. Useful for integration testing. The CPU backend is a reference implementation of QNN operators.|
|'gpu'|Enable GPU backend.|
|'htp'|Enable HTP backend. Offloads compute to NPU. Default.|
|'saver'|Enable Saver backend.|
|'ir'|Enable IR backend.|

> **Note**: Execution on the Genie backend is automatically enabled based on the contents of the ONNX model passed to onnxruntime-qnn.

|`"backend_path"`|Description|
|---|-----|
|'libQnnCpu.so' or 'QnnCpu.dll'|Enable CPU backend. See `backend_type` 'cpu'. Not shipped in the onnxruntime-qnn release package; obtain the file directly from the QAIRT SDK.|
|'libQnnHtp.so' or 'QnnHtp.dll'|Enable HTP backend. See `backend_type` 'htp'.|
|'libQnnGpu.so' or 'QnnGpu.dll'|Enable GPU backend. See `backend_type` 'gpu'.|
|'libQnnSaver.so' or 'QnnSaver.dll'|Enable Saver backend. See `backend_type` 'saver'.|
|'libQnnIr.so' or 'QnnIr.dll'|Enable IR backend. See `backend_type` 'ir'.|

**Note:** `backend_path` is an alternative to `backend_type`. At most one of the two should be specified.
`backend_path` requires a platform-specific path (e.g., `libQnnCpu.so` vs. `QnnCpu.dll`) but also allows one to specify an arbitrary path.

### CPU backend availability (`ORT_QNN_ENABLE_CPU_BACKEND`)

The QNN CPU backend is a reference implementation intended for integration testing, not production inference. Its visibility to the runtime depends on the host architecture:

- **x86_64 hosts**: the CPU backend is always advertised (it doubles as the HTP emulator).
- **arm64 hosts**: the CPU backend is hidden by default. Set the `ORT_QNN_ENABLE_CPU_BACKEND` environment variable to a non-empty, non-`0` value (e.g., `1`) before the QNN EP is registered / `OrtEnv` is created to opt in.

This only affects automatic EP/backend selection. Explicit `backend_type='cpu'` or `backend_path='...QnnCpu...'` provider options continue to work on both architectures when the corresponding QNN CPU library is available on disk (see note above about packaging).

|`"genie_log_level"`|Description|
|---|---|
|'error'|Default. Error-level logging only.|
|'warn'|Warning and error messages.|
|'info'|Informational, warning, and error messages.|
|'verbose'|All messages including verbose output.|

> **Note**: `genie_log_level` only applies when using the [Genie LLM inference pathway](#running-an-llm-model-with-qnn-eps-genie-backend).

|`"profiling_level"`|Description|
|---|---|
|'off'|Default. Profiling disabled.|
|'basic'|Basic profiling enabled.|
|'detailed'|Detailed profiling enabled.|
|'optrace'|Optrace profiling enabled. Requires QAIRT 2.39 or later.|

|`"profiling_file_path"`|Description|
|---|---|
|File path (string)|Specify the CSV file path to dump the QNN profiling events.|

See [profiling-tools](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html) for more info on profiling
Alternatively to setting profiling_level at compile time, profiling can be enabled dynamically with ETW (Windows). See [tracing](https://onnxruntime.ai/docs/performance/tune-performance/logging_tracing.html) for more details

|`"rpc_control_latency"`|Description|
|---|---|
|Microseconds (string)|Allows client to set up RPC control latency in microseconds. Defaults to "0" (not set).|

|`"vtcm_mb"`|Description|
|---|---|
|Size in MB (string)|QNN VTCM size in MB. Defaults to "0" (not set). Must be > 0 to take effect.|

|`"enable_vtcm_backup_buffer_sharing"`|Description|
|---|---|
|'0'|Default. Disabled.|
|'1'|Enable VTCM backup buffer sharing across sessions. Requires QNN API version >= 2.26. Conflicts with `ep.context_embed_mode`.|

|`"htp_performance_mode"`|Description|
|---|---|
|'burst'|Burst performance mode.|
|'balanced'|Balanced performance mode.|
|'default'|Default performance mode.|
|'high_performance'|High performance mode.|
|'high_power_saver'|High power saver mode.|
|'low_balanced'|Low balanced mode.|
|'low_power_saver'|Low power saver mode.|
|'power_saver'|Power saver mode.|
|'extreme_power_saver'|Extreme power saver mode.|
|'sustained_high_performance'|Sustained high performance mode.|

|`"qnn_saver_path"`|Description|
|---|---|
|File path (string)|File path to the QNN Saver backend library ('QnnSaver.dll' or 'libQnnSaver.so'). Dumps QNN API calls to disk for replay/debugging.|

|`"qnn_context_priority"`|[Description](https://docs.qualcomm.com/doc/80-63442-10/topic/htp_yielding.html)|
|---|---|
|'low'|Low priority.|
|'normal_low'|Normal low priority.|
|'normal'|Normal priority. Default.|
|'normal_high'|Normal high priority.|
|'high'|High priority.|
|'high_plus'|High plus priority.|
|'critical'|Critical priority.|
|'critical_plus'|Critical plus priority.|

|`"htp_graph_finalization_optimization_mode"`|Description|
|---|---|
|'0'|Default optimization mode.|
|'1'|Faster preparation time, less optimal graph.|
|'2'|Longer preparation time, more optimal graph.|
|'3'|Longest preparation time, most optimal graph.|

|`"soc_model"`|Description|
|---|---|
|Model number (string)|The SoC model to target. Accepts a **numeric model ID** (e.g. `"69"`) or a **chip-family name string** (e.g. `"SM8750"`, case-insensitive) for a subset of well-known chips — note this list is not exhaustive and chips not recognised by name must use the numeric ID. Refer to the [QAIRT SDK documentation](https://docs.qualcomm.com/doc/80-63442-10/topic/QNN_general_overview.html#supported-snapdragon-devices) for valid numeric values. Defaults to `"0"` (unknown). Accepts a comma-separated list (e.g. `"43,69"`) to enable [Flexible Context Binary (FCB) / multi-SoC](#flexible-context-binary-fcb--multi-soc-ep-context) compilation, where one EPContext model is produced that targets every listed SoC.|

|`"htp_arch"`|Description|
|---|---|
|'0'|Default. No architecture specified.|
|'68'|HTP v68.|
|'69'|HTP v69.|
|'73'|HTP v73.|
|'75'|HTP v75.|
|'79'|HTP v79.|
|'81'|HTP v81.|

Refer to the [QAIRT SDK documentation](https://docs.qualcomm.com/doc/80-63442-10/topic/enum_QnnHtpDevice_8h_1a0ed976142af98a86143459dfd326f717.html) for the full list of valid values.

`htp_arch` accepts a comma-separated list of architectures (e.g. `"73,81"`). Supplying more than one value enables [Flexible Context Binary (FCB) / multi-SoC](#flexible-context-binary-fcb--multi-soc-ep-context) compilation, where one EPContext model is produced that targets every listed SoC.

|`"device_id"`|Description|
|---|---|
|Device ID (string)|The ID of the device to use when setting `htp_arch`. Defaults to "0" (for single device). Must be >= 0.|

|`"enable_htp_fp16_precision"`|Description [Example](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx/QNN_EP/mobilenetv2_classification)|
|---|---|
|'0'|Default. Disabled. Inference with fp32 precision if it's fp32 model.|
|'1'|Enable the float32 model to be inferenced with fp16 precision.|

|`"enable_htp_monolithic_lstm"`|Description|
|---|---|
|'0'|Default. LSTM is expanded into per-timestep cells at the ORT layer.|
|'1'|Run the monolithic LSTM kernel on HTP (single graph node).|

Warning: Enabling HTP Monolithic LSTM may improve session creation time, but this improvement may come with a regression in inference performance.

|`"enable_htp_spill_fill_buffer"`|Description|
|---|---|
|'0'|Default. Disabled.|
|'1'|Enable HTP spill fill buffer for context cache.|

|`"offload_graph_io_quantization"`|Description|
|---|---|
|'0'|Disabled. QNN EP will handle quantization and dequantization of graph I/O.|
|'1'|Default. Enabled. Offload quantization and dequantization of graph I/O to CPU EP.|

|`"enable_block_quant_weight_optimization"`|Description|
|---|---|
|`"0"`|Default. Disabled. Block-quantized models use the standard compatibility path.|
|`"1"`|Enabled. Uses an optimized path for block-quantized weights when supported. If the optimized path is not available, QNN EP falls back to the standard compatibility path.|

|`"enable_htp_shared_memory_allocator"`|Description|
|---|---|
|'0'|Default. Disabled.|
|'1'|Enable the QNN HTP shared memory allocator. Requires libcdsprpc.so/dll to be available. [Code example](https://github.com/microsoft/onnxruntime/blob/544bdd60730270f49f6a5baafdff54065f626776/onnxruntime/test/shared_lib/test_inference.cc#L2262-L2354)|

|`"enable_dx12_shared_memory_allocator"`|Description|
|---|---|
|'0'|Default. Disabled.|
|'1'|Enable the QNN DX12 shared memory allocator. Requires a D3D12 capable system, and only available on Windows.|

|`"extended_udma"`|Description|
|---|---|
|'0'|Default. Disabled.|
|'1'|Enable HTP extended UDMA mode for better performance on supported hardware.|

|`"op_packages"`|Description|
|---|---|
|Op package config (string)|Register custom op packages. Format: `<OpType>:<PackagePath>:<InterfaceSymbolName>[:<Target>]`. Multiple packages can be separated by commas.|

|`"dump_json_qnn_graph"`|Description|
|---|---|
|'0'|Default. Disabled.|
|'1'|Enable dumping QNN graphs to JSON format for debugging.|

|`"json_qnn_graph_dir"`|Description|
|---|---|
|Directory path (string)|Directory path for dumping QNN JSON graphs. Only effective when `dump_json_qnn_graph` is enabled.|

|`"dump_qnn_ep_input_graph"`|Description|
|---|---|
|'0'|Default. Disabled.|
|'1'|Dump the ONNX graph the QNN EP receives at compile time (after ORT Level 1 optimizations, before partitioning) as a JSON file. The output uses the same schema as `dump_json_qnn_graph`, so it opens in QNN Netron. Unlike `dump_json_qnn_graph` (which dumps the post-compile QNN graph), this captures the pre-compile ONNX graph. See `dump_qnn_ep_input_graph_dir` to control the output location.|

|`"dump_qnn_ep_input_graph_dir"`|Description|
|---|---|
|Directory path (string)|Directory path for the QNN EP input graph dump. Only effective when `dump_qnn_ep_input_graph` is `'1'`. Defaults to the current working directory if not set. The directory is created automatically if it does not exist. If the directory cannot be created or written to, the dump is disabled with a WARNING log. Output files use the pattern `<graph_name>.<n>_qnn_ep_input_graph.json`, where `<n>` is a per-EP counter that increments for each dumped graph. When the source graph has no name, `<graph_name>` is the literal `graph`.|

|`"dump_qnn_ir_dlc"`|Description|
|---|---|
|'0'|Default. Disabled.|
|'1'|Enable dumping QNN graphs to DLC (Deep Learning Container) format using IR backend.|

|`"dump_qnn_ir_dlc_dir"`|Description|
|---|---|
|Directory path (string)|Directory path for dumping QNN IR DLC files. Only effective when `dump_qnn_ir_dlc` is enabled.|

|`"enable_framework_op_trace"`|Description|
|---|---|
|'0'|Default. Disabled.|
|'1'|Enable framework op tracing. Writes a JSON file that maps each ONNX operator (or fused group) to its corresponding QNN operator(s), records unsupported operators, and includes summary statistics. See `framework_op_trace_dir` to control the output location. The trace is produced from the QNN graph composition path: it is emitted on a fresh JIT compile and on AOT context-binary generation (when `ep.context_enable=1` writes a new EPContext model). It is **not** emitted when loading an existing EPContext model (cached run), because no graph composition occurs in that path; in that case an INFO log records that no trace file is written.|

|`"framework_op_trace_dir"`|Description|
|---|---|
|Directory path (string)|Directory path for framework op trace JSON files. Only effective when `enable_framework_op_trace` is `'1'`. Defaults to the current working directory if not set. The directory is created automatically if it does not exist. If the directory cannot be created or written to (e.g. read-only mount, missing permissions), framework op tracing is disabled with a WARNING log and no trace file is produced. Trace files use the pattern `qnn_op_trace.json`.|

**Schema migration note:** The `qnn_op_trace.json` schema moves the `compilation_target` object out of the document root and into a new `soc_traces` array. Single-SoC sessions emit exactly one entry; multi-SoC FCB sessions emit one entry per `(htp_arch, soc_model)` pair. `subgraph_traces` (the ONNX-to-QNN op mapping) stays at the document root because op-builder output is SoC-agnostic. A new `summary.total_socs` field reports the entry count. A top-level `"schema_version"` field (currently `2`) is bumped on every breaking schema change so consumers can branch on it.

- Old (legacy): `j["compilation_target"]["htp_arch"]`
- New: `j["soc_traces"][0]["compilation_target"]["htp_arch"]`

Consumers that read the compilation target must update the JSON path. Consumers of `subgraph_traces`, `unsupported_nodes`, and `summary` are unaffected.

|`"qnn_ir_backend_path"`|Description|
|---|---|
|Backend path (string)|Path to the QNN IR backend library. Defaults to 'libQnnIr.so' or 'QnnIr.dll'. Only effective when `dump_qnn_ir_dlc` is enabled.|

|`"skip_backend_op_validation"`|Description|
|---|---|
|'0'|Default. The target backend (e.g. HTP) validates each op config during DLC dump.|
|'1'|Skip target-backend op validation during DLC dump and fall back to the serializer's generic op checks. Intended for device-less hosts (e.g. cloud x86_64 compilation): there the target backend validates against an arch-agnostic op table and over-rejects arch-specific (v73+) ops such as `ScatterElements(reduction=max)`, even though the op is valid on real hardware. Only effective when `dump_qnn_ir_dlc` is enabled. Op lowering still targets the intended backend; only the validation verdict changes.|

|`"skip_qnn_version_check"`|Description|
|---|---|
|'0'|Default. Version check enabled.|
|'1'|Skip QNN API interface version check to use other QNN library versions.|

|`"num_graph_prepare_threads"`|Description|
|---|---|
|Number (string)|The number of threads to use during model compilation. Must be at least 1 (a value of 1 disables the feature) and no more than the maximum supported concurrency threads as [reported by the hardware](https://en.cppreference.com/w/cpp/thread/thread/hardware_concurrency.html).<br><br>An invalid value will result in a warning and default behavior.<br><br>Defaults to 8 or the maximum supported threads, whichever is lower.<br><br><b>Will only take effect if the model is partitioned into 5+ subgraphs</b><br><br><b>Currently only supported on Windows ARM64 devices</b>|

For more information, see the [Parallel Graph Preparation](#parallel-graph-preparation) section below.

|`"htp_share_resource_optimization"`|Description|
|---|---|
|'1'|Enable HTP VTCM backup buffer sharing across sessions. Only `'1'` is a valid value. Supersedes `enable_vtcm_backup_buffer_sharing`. Requires QNN API version >= 2.26.|

**Note:** `htp_share_resource_optimization` and `enable_vtcm_backup_buffer_sharing` both enable the same underlying feature. Prefer `htp_share_resource_optimization`.

|`"disable_file_mapped_weights"`|Description|
|---|---|
|'0'|Default. File-mapped weights enabled (when supported).|
|'1'|Disable file-mapped weight loading. File-mapped weights are only available on Windows ARM64 with QNN API >= 2.32; this option is ignored on other platforms.|

|`"htp_bf16_enable"`|Description|
|---|---|
|'0'|Default. Disabled.|
|'1'|Enable BFloat16 precision on the HTP backend. Requires `soc_model` to be set to a value >= 88 (e.g., SM8750). An error is raised at session creation if `soc_model` is unset or below 88.|

|`"enable_htp_prepare_only"`|Description|
|---|---|
|'0'|Default. Disabled.|
|'1'|Compile the model and save the QNN context binary, but skip inference. `OnRunStart`, `OnRunEnd`, `CreateState`, and `SetDynamicOptions` are all no-ops. Useful for a compile-once/run-later workflow. Requires `ep.context_enable=1`; silently disabled with a warning if context cache is not enabled.|

|`"enable_htp_graph_splitting"`|Description|
|---|---|
|'0'|Default. Disabled.|
|'1'|Enable HTP graph splitting: the HTP backend splits the model graph into independently-prepareable sub-graphs, reducing context preparation time. Effective in both JIT (compile-and-run) and AOT (context binary generation) workflows, including on-device AOT. Has no effect when loading from an already-compiled context binary. Requires QAIRT SDK 2.49+ at runtime; enabling this with an older runtime will cause context creation to fail. The number of sub-graph partitions is controlled by the `GPE_KWAY_PARTITIONS` environment variable.|

|`"GPE_KWAY_PARTITIONS"`|Description|
|---|---|
| `8` | Default |
| `N` | Split into N sub-graphs (must be ≥ 2) |

|`"session.disable_cpu_ep_fallback"`|Description|
|---|---|
|'0'|Default. Unsupported operators fall back to the CPU EP.|
|'1'|Disable CPU EP fallback. Returns an error if any operator cannot be handled by QNN EP. Set via `session_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1")`.|

### Flexible Context Binary (FCB) / multi-SoC EP context

The Flexible Context Binary (FCB) feature packages one or more QNN context binaries into a single QNN DLC, so a single EPContext ONNX model can be deployed across multiple Snapdragon SoCs. It requires **QAIRT 2.48 or later (QNN API >= 2.37)**.

Multi-SoC mode is enabled automatically when `soc_model` and/or `htp_arch` is given a comma-separated list of more than one value. In that case:

- `soc_model` and `htp_arch` may each be given on their own; the other is then defaulted for every target SoC. If both are given as lists, they must contain the same number of values (one per target SoC), and `soc_model` takes priority if the two do not agree.
- The following HTP backend options may each be given `0`, `1`, or `N` values (comma-separated), where `N` matches the number of SoCs. With `0` values the default is applied to every SoC; with `1` value it is duplicated across all SoCs; with `N` values each is applied to the corresponding SoC:
  - `vtcm_mb`

Constraints:

- Supported only on **x86_64** hosts (offline preparation). On-device preparation will soon be supported in later QAIRT version.
- Requires the EP context cache to be enabled (`ep.context_enable=1`). `enable_htp_prepare_only` is turned on automatically (preparation only, no inference).
- Not supported together with shared EP contexts (`share_ep_contexts` / `stop_share_ep_contexts`).

**Example (Python):**
```python
ep_options = {
    "backend_type": "htp",
    "htp_arch": "73,81",          # two target SoCs -> multi-SoC mode
    "soc_model": "60,88",         # one SoC model per arch
    "vtcm_mb": "8",               # single value duplicated to every SoC
}
# ep.context_enable must be set to 1; enable_htp_prepare_only is enabled automatically.
```

#### Stripping an FCB for deployment (`deploy_multi_soc_ep_context.py`)
After an FCB is generated, you may want to **prune** the SoCs it targets — for example, to ship a smaller artifact that only covers the devices you actually support. The `deploy_multi_soc_ep_context.py` helper (shipped in the `onnxruntime-qnn` Python wheel) lists and removes the per-SoC HTP cache records inside the FCB, and keeps the accompanying ONNX model's compatibility metadata in sync.

The script works on both EP context model layouts, selected by which input you pass:

* **Non-embed mode** (`ep.context_embed_mode = "0"`, the default): the context binary is a standalone `*_qnn.bin` file alongside the ONNX model. Pass it via `--input_bin` (and `--input_onnx` for the metadata). The stripped binary is written to `--output_bin` and the updated model to `--output_onnx`; both inputs are left untouched. Pass `--inplace` to overwrite the inputs instead.
* **Embed mode** (`ep.context_embed_mode = "1"`): the context binary is embedded inside the EP context node of the ONNX model. Omit `--input_bin` and pass the model via `--input_onnx`; the tool extracts the embedded binary to a temporary file, edits it, and re-embeds the result into `--output_onnx`. The input model is left untouched. Pass `--inplace` to overwrite `--input_onnx` instead.

Run it as a module from an installed wheel:

```bash
python -m onnxruntime_qnn.deploy_multi_soc_ep_context --input_bin model_ctx_qnn.bin --list
```

|Argument|Description|
|---|---|
|`--input_bin`|Path to a standalone context binary / EP-context file (**non-embed mode**). Omit it for an embed-mode model, in which case the embedded binary is read from `--input_onnx`.|
|`--input_onnx`|Path to the input EP context ONNX model. **Required with `--remove_record`** so its `ep_compatibility_info.QNNExecutionProvider` metadata is kept in sync. In embed mode this is also the source of the embedded binary, so it is always required there.|
|`-l`, `--list`|List every HTP cache record with its DSP architecture, SoC model and (first-graph) VTCM size.|
|`-r`, `--remove_record`|Comma-separated HTP cache record name(s) to remove (e.g. `-r backend.metadata0,backend.metadata1`).|
|`--output_onnx`|**Required with `--remove_record`** (both modes) unless `--inplace` is given. Destination path for the updated ONNX model. In embed mode the stripped binary is re-embedded into it; in non-embed mode it carries the updated metadata. The input `--input_onnx` is left untouched.|
|`--output_bin`|**Required with `--remove_record` in non-embed mode** unless `--inplace` is given. Destination path for the stripped binary. The input `--input_bin` is left untouched.|
|`--temp_bin`|*Embed mode only, optional.* Path to stage the extracted binary while it is edited. When omitted, an auto-generated temporary file is used. The file is removed when the tool finishes either way.|
|`--inplace`|*Optional.* Overwrite the input model (and, in non-embed mode, the input binary) in place instead of writing to new paths. Mutually exclusive with `--output_onnx` / `--output_bin`.|

**Listing the packaged SoCs:**

```bash
# Non-embed mode (standalone context binary)
python -m onnxruntime_qnn.deploy_multi_soc_ep_context --input_bin model_ctx_qnn.bin --list

# Embed mode (context binary embedded in the ONNX model)
python -m onnxruntime_qnn.deploy_multi_soc_ep_context --input_onnx model_ctx.onnx --list
```

**Removing a SoC (non-embed mode)** — writes the pruned binary and updated model to new paths, leaving the inputs untouched:

```bash
python -m onnxruntime_qnn.deploy_multi_soc_ep_context \
    --input_bin model_ctx_qnn.bin \
    --input_onnx model_ctx.onnx \
    --output_bin model_ctx_qnn.pruned.bin \
    --output_onnx model_ctx.pruned.onnx \
    -r backend.metadata0
```

**Removing a SoC (embed mode)** — extracts, edits, and re-embeds the binary into a new model:

```bash
python -m onnxruntime_qnn.deploy_multi_soc_ep_context \
    --input_onnx model_ctx.onnx \
    --output_onnx model_ctx.pruned.onnx \
    -r backend.metadata0 \
    --list
```

**Removing a SoC in place** — overwrite the inputs instead of writing new files (works in both modes):

```bash
# Non-embed mode: overwrites both model_ctx_qnn.bin and model_ctx.onnx
python -m onnxruntime_qnn.deploy_multi_soc_ep_context \
    --input_bin model_ctx_qnn.bin \
    --input_onnx model_ctx.onnx \
    --inplace \
    -r backend.metadata0

# Embed mode: overwrites model_ctx.onnx
python -m onnxruntime_qnn.deploy_multi_soc_ep_context \
    --input_onnx model_ctx.onnx \
    --inplace \
    -r backend.metadata0
```

> **Note**: `deploy_multi_soc_ep_context.py` depends on the `libDlModelToolsPy` extension from the QAIRT SDK. The `onnxruntime-qnn` wheel bundles it next to the script, so `python -m onnxruntime_qnn.deploy_multi_soc_ep_context` works out of the box. When running the script directly from a source checkout instead, the SDK's `lib/python/qti/aisw/dlc_utils/<platform>` directory must be on `PYTHONPATH`.

> **Note**: Only **Python 3.12** is supported. The `onnxruntime-qnn` wheel bundles the `libDlModelToolsPy` extension (compiled for Python 3.12) only into the 3.12 wheel, and the script imports it directly. The script is thus not shipped in the wheels other than Python 3.12 one.

### Run Options

Run options can be set dynamically at runtime using the ORT Run API. These options allow you to configure QNN EP behavior on a per-inference basis.

|Option Key|Description|
|---|---|
|`"qnn.perf_mode"`|HTP performance mode for this inference run. Valid values: 'burst', 'balanced', 'default', 'high_performance', 'high_power_saver', 'low_balanced', 'low_power_saver', 'power_saver', 'extreme_power_saver', 'sustained_high_performance'. Overrides the EP provider option `htp_performance_mode` for this run.|
|`"qnn.rpc_control_latency"`|RPC control latency in microseconds for this inference run. Overrides the EP provider option `rpc_control_latency` for this run.|
|`"qnn.lora_config"`|LoRAv2 config file path. Format: `<graph name>;<adapter binary section path>`. See **LoRAv2 support** section for details.|
|`"kvcache_rewind"`|Genie KV-cache reset. Set to `"0"` to reset the KV cache before the next inference (useful at the start of a new conversation). Set to `"1"` (default) to skip the reset. Only applies when using the [Genie LLM inference pathway](#running-an-llm-model-with-qnn-eps-genie-backend).|
|`"ep.dynamic.workload_type"`|EP workload type. `'Default'` resets to the default context priority. `'Efficient'` sets context priority to `low`, reducing resource usage at the cost of throughput. HTP backend only. Set via the `SetDynamicOptions` API.|

**Example usage (Python):**
```python
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

# ORT QNN EP Version
print(qnn_ep.__version__)

# Register QNN EP library
ep_lib_path = qnn_ep.get_library_path()
ep_registration_name = "QNNExecutionProvider"
ort.register_execution_provider_library(ep_registration_name, ep_lib_path)

# Select QNN EP device
all_ep_devices = ort.get_ep_devices()
selected_ep_devices = [ep_device for ep_device in all_ep_devices if ep_device.ep_name == ep_registration_name]

# Configure and create session
ep_options = {'backend_path': qnn_ep.get_qnn_htp_path()}
session_options = ort.SessionOptions()
session_options.add_provider_for_devices(selected_ep_devices, ep_options)
session = ort.InferenceSession("model.onnx", sess_options=session_options)

# Set run options for this specific inference
run_options = ort.RunOptions()
run_options.add_run_config_entry("qnn.perf_mode", "burst")
run_options.add_run_config_entry("qnn.rpc_control_latency", "100")

result = session.run(None, {"input": input_data}, run_options)

# Clean up
del session
ort.unregister_execution_provider_library(ep_registration_name)
```

## Supported ONNX operators

|Operator|Notes|
|---|---|
|ai.onnx:Abs||
|ai.onnx:Add||
|ai.onnx:And||
|ai.onnx:ArgMax||
|ai.onnx:ArgMin||
|ai.onnx:Asin||
|ai.onnx:Atan||
|ai.onnx:Attention|Opsets 23–24. 3D \[B,S,n·hs\] and 4D \[B,n,S,hs\] inputs. HTP: full decomposition for MHA, GQA/MQA, is\_causal, attn\_mask, softcap, KV cache, qk\_matmul\_output modes 0–3. GPU: native QNN\_OP\_GROUP\_QUERY\_ATTENTION when is\_causal=1 and no softcap/attn\_mask/qk\_output (covers MHA and GQA/MQA); decomposition otherwise. Static shapes only.|
|ai.onnx:AveragePool||
|ai.onnx:Bernoulli|CPU and HTP backends supported.|
|ai.onnx:BatchNormalization|fp16 supported since 1.18.0|
|ai.onnx:Cast||
|ai.onnx:CastLike||
|ai.onnx:Ceil||
|ai.onnx:Clip|fp16 supported since 1.18.0|
|ai.onnx:Concat||
|ai.onnx:Conv|3d supported since 1.18.0|
|ai.onnx:ConvInteger|Supported exclusively via DynamicQuantizeLinear → ConvInteger fusion pattern|
|ai.onnx:ConvTranspose|3d supported since 1.18.0|
|ai.onnx:Cos||
|ai.onnx:CumSum||
|ai.onnx:DepthToSpace||
|ai.onnx:DequantizeLinear||
|ai.onnx:DynamicQuantizeLinear|Supported exclusively via ConvInteger and MatMulInteger fusion patterns|
|ai.onnx:Div||
|ai.onnx:Einsum||
|ai.onnx:Elu||
|ai.onnx:Equal||
|ai.onnx:Exp||
|ai.onnx:Expand||
|ai.onnx:Flatten||
|ai.onnx:Floor||
|ai.onnx:Gather|Only supports positive indices|
|ai.onnx:GatherElements||
|ai.onnx:GatherND||
|ai.onnx:Gelu||
|ai.onnx:Gemm||
|ai.onnx:GlobalAveragePool||
|ai.onnx:GlobalMaxPool||
|ai.onnx:Greater||
|ai.onnx:GreaterOrEqual||
|ai.onnx:GridSample||
|ai.onnx:GRU|layout=0 (batch-first) only; fixed-length sequences; default activations (sigmoid/tanh); no clip; no initial hidden/cell state from ONNX|
|ai.onnx:GroupNormalization||
|ai.onnx:HardSigmoid||
|ai.onnx:HardSwish||
|ai.onnx:If|Single output per branch with identical shape+dtype; cond must be scalar bool; both branches are always executed (lowered to Where/Select)|
|ai.onnx:Identity||
|ai.onnx:InstanceNormalization||
|ai.onnx:Inverse||
|ai.onnx:IsInf|float32 and float16 inputs only; CPU backend only|
|ai.onnx:IsNaN||
|ai.onnx:LRN||
|ai.onnx:LSTM||
|ai.onnx:LayerNormalization||
|ai.onnx:LeakyRelu||
|ai.onnx:Less||
|ai.onnx:LessOrEqual||
|ai.onnx:Log||
|ai.onnx:LogSoftmax||
|ai.onnx:LpNormalization|p == 2|
|ai.onnx:LpPool|p=1 or p=2 only; rank-5 (3D pooling) requires GPU backend or HTP with bf16_mode=1|
|ai.onnx:MatMul|Supported input data types on HTP backend: (uint8, uint8), (uint8, uint16), (uint16, uint8)|
|ai.onnx:MatMulInteger|Supported exclusively via DynamicQuantizeLinear → MatMulInteger fusion pattern|
|ai.onnx:Max||
|ai.onnx:MaxPool||
|ai.onnx:MaxRoiPool|rois must be a constant initializer|
|ai.onnx:Mean||
|ai.onnx:Min||
|ai.onnx:Mod||
|ai.onnx:Mul||
|ai.onnx:Neg||
|ai.onnx:NonMaxSuppression|max_output_boxes_per_class, iou_threshold, and score_threshold must be static constants; max_output_boxes_per_class must be > 0; center_point_box must be 0 (diagonal corners); GPU backend not supported|
|ai.onnx:NonZero||
|ai.onnx:Not||
|ai.onnx:Or||
|ai.onnx:OneHot|depth and values must be constant initializers; indices must be int32, int64, or uint32|
|ai.onnx:PRelu|fp16, int32 supported since 1.18.0|
|ai.onnx:Pad||
|ai.onnx:Pow||
|ai.onnx:QLinearConv|All scale/zero-point inputs must be initializers; per-tensor activation/output quant; per-channel weight quant (1-D w_scale/w_zp) supported; input types int8 or uint8; optional int32 bias; 1D, 2D, and 3D convolution|
|ai.onnx:QLinearMatMul|Per-tensor (scalar) scale/zero-point only; scale must be float32, float16, or bfloat16; quantized input/output types int8 or uint8|
|ai.onnx:QuantizeLinear||
|ai.onnx:RandomNormalLike||
|ai.onnx:RandomUniformLike||
|ai.onnx:Range|All inputs (start, limit, delta) must be constant initializers; pre-computed host-side into a static tensor|
|ai.onnx:Reciprocal||
|ai.onnx:ReduceL2||
|ai.onnx:ReduceLogSumExp|Decomposed into ReduceMax->Sub->Exp->ReduceSum->Log->Add. Quantized input not supported.|
|ai.onnx:ReduceMax||
|ai.onnx:ReduceMean||
|ai.onnx:ReduceMin||
|ai.onnx:ReduceProd||
|ai.onnx:ReduceSum||
|ai.onnx:Relu||
|ai.onnx:RMSNormalization||
|ai.onnx:Reshape||
|ai.onnx:Resize|tf_half_pixel_for_nn coordinate_transformation_mode: only mode "nearest" with nearest_mode "floor" on rank-4 input is supported on HTP; other combinations fall back to CPU|
|ai.onnx:RoiAlign||
|ai.onnx:Round||
|ai.onnx:RotaryEmbedding|HTP backend only|
|ai.onnx:STFT||
|ai.onnx:ScatterElements||
|ai.onnx:ScatterND||
|ai.onnx:Selu||
|ai.onnx:Shape|start/end attributes supported (ONNX opset 15+); output is int64 (upcast from QNN int32); empty-slice (end <= start) not supported|
|ai.onnx:Sigmoid||
|ai.onnx:Sign||
|ai.onnx:Sin||
|ai.onnx:Slice||
|ai.onnx:Softmax||
|ai.onnx:Softplus||
|ai.onnx:SpaceToDepth||
|ai.onnx:Split||
|ai.onnx:Sqrt||
|ai.onnx:Squeeze||
|ai.onnx:Sub||
|ai.onnx:Sum||
|ai.onnx:Tan||
|ai.onnx:Tanh||
|ai.onnx:ThresholdedRelu||
|ai.onnx:Tile||
|ai.onnx:TopK||
|ai.onnx:Transpose||
|ai.onnx:Unsqueeze||
|ai.onnx:Upsample||
|ai.onnx:Where||
|ai.onnx:Xor||
|com.microsoft:DequantizeLinear|Provides 16-bit integer dequantization support|
|com.microsoft:FusedMatMul||
|com.microsoft:Gelu||
|com.microsoft:GatherBlockQuantized|GPU backend only; bits=4; block_size must be a power-of-2 ≥ 16; quantize_axis=1; symmetric quantization only (no zero points); requires QAIRT SDK ≥ 2.48|
|com.microsoft:GroupQueryAttention|GPU backend only; requires QAIRT SDK ≥ 2.48 (QNN opset 2.12); rotary_interleaved=0; no k_quant_type/v_quant_type|
|com.microsoft:MatMulNBits||
|com.microsoft:QuantizeLinear|Provides 16-bit integer quantization support|
|com.microsoft:RMSNormalization||
|com.microsoft:QuickGelu||
|com.microsoft:RotaryEmbedding|HTP backend only|
|com.microsoft:SimplifiedLayerNormalization||

Supported data types vary by operator and QNN backend. Refer to the [QAIRT SDK documentation](https://docs.qualcomm.com/doc/80-63442-10/topic/operations.html) for more information.

## Supported operator fusions

QNN EP recognizes the following multi-op patterns and fuses them into a single QNN operation or a more efficient subgraph. Fusions are attempted before individual op builders run.

### DQ/Q and quantization fusions

| Pattern | Fused QNN op | Notes |
|---|---|---|
| `DequantizeLinear → QuantizeLinear` | `QNN_OP_CONVERT` | Eliminates redundant re-quantization between compatible quantization schemes. Scalar scale/zero-point required. |
| `DynamicQuantizeLinear → DequantizeLinear` | `QNN_OP_TRANSPOSE` (identity) | Fuses a quantize+dequantize round-trip into a pass-through when all three DQL outputs are consumed exclusively by a single DQ node and the DQ output is float32. |
| `(non-DQ node) → Cast(→float) → QuantizeLinear` | `QNN_OP_CONVERT` | Fuses a cast-to-float followed by quantization when there is no preceding DQ node. |
| `DynamicQuantizeLinear → ConvInteger → Cast → Mul → [Add]` | `QNN_OP_CONV_2D` / `QNN_OP_DEPTH_WISE_CONV_2D` | ConvInteger is supported exclusively via this fusion. Dynamic quantization + integer convolution pattern. Constant int8/uint8 weights required. |
| `DynamicQuantizeLinear → MatMulInteger → Cast → Mul → [Add]` | `QNN_OP_MAT_MUL` | MatMulInteger is supported exclusively via this fusion. Dynamic quantization + integer matmul pattern. Constant rank-2 int8/uint8 weights required. |

### Low-Power Block Quantization (LPBQ) fusions

| Pattern | Fused QNN op | Notes |
|---|---|---|
| `Scale_DQL → [W_QL →] W_DQL → Act_DQL → Gemm → QL` | `QNN_OP_FULLY_CONNECTED` | Per-block integer scales with per-channel float scales. HTP/NPU backend only. |
| `Scale_DQL → [W_QL →] W_DQL → MatMul` | `QNN_OP_MAT_MUL` | Per-block integer scales with per-channel float scales. int4/int8 weights. HTP/NPU backend only. |

### Operator decomposition fusions

| Pattern | Fused QNN op | Notes |
|---|---|---|
| `[Div/Mul(√2)] → Erf → [Mul(0.5) →] Add(1) → Mul → [Mul(0.5)]` | `QNN_OP_GELU` | Matches two common Gelu decomposition variants (ErfAdd and ErfMul). Surrounding DQ nodes are also handled. |
| `Mul(x,x) → Mul(x²,x) → Mul(0.044715) → Add(x) → Mul(√(2/π)) → Tanh → Add(1) → Mul(x) → Mul(0.5)` | `QNN_OP_GELU` | Tanh-based GELU approximation. SingleNode (non-QDQ) only. The QNN GELU op uses the exact-erf definition; max substitution error is ~4.7e-4 over [-10, 10]. |
| `HardSigmoid(α=1/6, β=0.5) → Mul` (shared input) | `QNN_OP_ELEMENT_WISE_NEURON` (HardSwish) | Both inputs to Mul must originate from the same source tensor. |
| `ReduceMean → Sub → Pow(2) → ReduceMean → Add(ε) → Sqrt → Div → Mul(γ) → Add(β)` | `QNN_OP_LAYER_NORM` | Matches the manual LayerNorm decomposition. Gamma and beta must be constants. |
| `Mul(scalar constant) → Softmax` | `QNN_OP_SOFTMAX` | The scalar multiplier is folded into the beta parameter of QNN's Softmax. |
| `ReduceL2 → Add(ε) → Div(x, .)` | `QNN_OP_L2_NORM` | Avoids fp16 overflow in the ReduceL2→Sqrt decomposition by using HTP's native L2Norm kernel. Epsilon must be a constant scalar. |
| `Reshape(ND→2D) → Gemm` | `QNN_OP_FULLY_CONNECTED` | Input Reshape must not be shared; Gemm: transA=0, transB=0, alpha=1, beta=1; weight must be a constant; input rank ≤ 4. CPU and HTP backends only. |
| `Reshape(ND→2D) → Gemm → Reshape(2D→MD)` | `QNN_OP_FULLY_CONNECTED` + `QNN_OP_RESHAPE` | 3-node variant; quantized weights supported. |
| `Reshape(ND→2D) → Gemm → Reshape(2D→MD) → Reshape(MD→PD)` | `QNN_OP_FULLY_CONNECTED` + `QNN_OP_RESHAPE` | 4-node variant; two consecutive output Reshapes. |

### Layout and reshape fusions

| Pattern | Fused QNN op | Notes |
|---|---|---|
| `Transpose → Reshape → Transpose → Reshape → Transpose` | `QNN_OP_CHANNEL_SHUFFLE` | 5-node pattern. Head and tail transposes must cancel; middle transpose permutes only the channel dimension. |
| `[Transpose →] Reshape(4D→6D) → Transpose(DCR/CRD) → Reshape(6D→4D) [→ Transpose]` | `QNN_OP_SPACE_TO_DEPTH` | Matches both NCHW and NHWC layout variants and both DCR and CRD modes. Static dimensions required. |
| `Reshape(4D→6D) → Einsum(transpose-equivalent) → Reshape(6D→4D)` | `QNN_OP_DEPTH_TO_SPACE + QNN_OP_TRANSPOSE` | Matches Einsum used as a rank-6 permutation with perm `[0,5,1,3,2,4]` (DCR DepthToSpace). |
| `Reshape(5D→6D) → Transpose → Reshape(6D→5D)` (unit dim) | `QNN_OP_RESHAPE + QNN_OP_TRANSPOSE` | Unit dimension must appear at the same index in the rank-6 intermediate. Does not apply to SpaceToDepth decompositions. |
| `Gather(rank-5, axis=4) → Transpose → Reshape` | Multi-node QNN subgraph | Constant rank-2 indices (row-major or column-major). |
| `Transpose → Reshape → Transpose` | `QNN_OP_RESHAPE` | Fires when the combined transformation reduces to a pure reshape (dimension-merging only, no reordering). Intermediate QDQ nodes are permitted. |
| `Reshape → [Q → DQ]? → Transpose` | `QNN_OP_TRANSPOSE` or noop `QNN_OP_RESHAPE` | Fires when Reshape is equivalent to a permutation (same rank, non-1 dims in same relative order). Composes the reshape-implied perm with the Transpose perm; emits identity Reshape if composed perm is identity, or a single Transpose otherwise. |

### Miscellaneous fusions

| Pattern | Fused QNN op | Notes |
|---|---|---|
| `[DQ inputs →] Custom UDO op [→ Q outputs]` | Custom QNN UDO | Strips surrounding DQ/Q nodes and passes quantization parameters directly into the user-defined operator. |

## Running a model with QNN EP's HTP backend (Python)
<p align="center"><img width="100%" src="../images/qnn_ep_quant_workflow.png" alt="Offline workflow for quantizing an ONNX model for use on QNN EP"/></p>

The QNN HTP backend supports both floating-point models and quantized models. Models with 32-bit floating-point activations and weights can be quantized to use a lower integer precision (e.g., 8-bit or 16-bit integers) to get better performance.

This section provides instructions for quantizing a model and then running the quantized model on QNN EP's HTP backend using Python APIs. Please refer to the [quantization page](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) for a broader overview of quantization concepts.

### Model requirements
QNN EP does not support models with dynamic shapes (e.g., a dynamic batch size). Dynamic shapes must be fixed to a specific value. Refer to the documentation for [making dynamic input shapes fixed](https://onnxruntime.ai/docs/tutorials/mobile/helpers/make-dynamic-shape-fixed.html) for more information.

Additionally, QNN EP supports a subset of ONNX operators (e.g., Loops and Ifs are not supported). Refer to the [list of supported ONNX operators](./QNN-ExecutionProvider.md#supported-onnx-operators).

### Generating a quantized model (x64 only)
The ONNX Runtime python package provides utilities for quantizing ONNX models via the `onnxruntime.quantization` import. The quantization utilities are currently only supported on x86_64 due to issues installing the `onnx` package on ARM64.
Therefore, it is recommended to either use an x64 machine to quantize models or, alternatively, use a separate x64 python installation on Windows ARM64 machines.

Install the ONNX Runtime x64 python package. (please note, you must use x64 package for quantizing the model. use the arm64 package for inferencing and utilizing the HTP/NPU)
```shell
python -m pip install onnxruntime
```

Quantization for QNN EP requires the use of calibration input data. Using a calibration dataset that is representative of typical model inputs is crucial in generating an accurate quantized model.

The following snippet defines a sample `DataReader` class that generates random float32 input data. Note that using random input data will most likely produce an inaccurate quantized model.
Refer to the [implementation of a Resnet data reader](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/resnet50_data_reader.py) for one example of how to create a `CalibrationDataReader` that provides input from image files on disk.

```python
# data_reader.py

import numpy as np
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader


class DataReader(CalibrationDataReader):
    def __init__(self, model_path: str):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        inputs = session.get_inputs()

        self.data_list = []

        # Generate 10 random float32 inputs
        # TODO: Load valid calibration input data for your model
        for _ in range(10):
            input_data = {inp.name : np.random.random(inp.shape).astype(np.float32) for inp in inputs}
            self.data_list.append(input_data)

        self.datasize = len(self.data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                self.data_list
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None

```

The following snippet pre-processes the original model and then quantizes the pre-processed model to use `uint16` activations and `uint8` weights.
Although the quantization utilities expose the `uint8`, `int8`, `uint16`, and `int16` quantization data types, QNN operators typically support the `uint8` and `uint16` data types.
Refer to the [QAIRT SDK operator documentation](https://docs.qualcomm.com/doc/80-63442-10/topic/HtpOpDefSupplement.html) for the data type requirements of each QNN operator.

```python
# quantize_model.py

import data_reader
import numpy as np
import onnx
from onnxruntime.quantization import QuantType, quantize
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model

if __name__ == "__main__":
    input_model_path = "model.onnx"  # TODO: Replace with your actual model
    output_model_path = "model.qdq.onnx"  # Name of final quantized model
    my_data_reader = data_reader.DataReader(input_model_path)

    # Pre-process the original float32 model.
    preproc_model_path = "model.preproc.onnx"
    model_changed = qnn_preprocess_model(input_model_path, preproc_model_path)
    model_to_quantize = preproc_model_path if model_changed else input_model_path

    # Generate a suitable quantization configuration for this model.
    # Note that we're choosing to use uint16 activations and uint8 weights.
    qnn_config = get_qnn_qdq_config(model_to_quantize,
                                    my_data_reader,
                                    activation_type=QuantType.QUInt16,  # uint16 activations
                                    weight_type=QuantType.QUInt8)       # uint8 weights

    # Quantize the model.
    quantize(model_to_quantize, output_model_path, qnn_config)
```

Running `python quantize_model.py` will generate a quantized model called `model.qdq.onnx` that can be run on Windows ARM64 devices via ONNX Runtime's QNN EP.

Refer to the following pages for more information on usage of the quantization utilities:
- [Quantization example for mobilenet on CPU EP](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/quantization/image_classification/cpu)
- [quantization/execution_providers/qnn/preprocess.py](https://github.com/microsoft/onnxruntime/blob/23996bbbbe0406a5c8edbf6b7dbd71e5780d3f4b/onnxruntime/python/tools/quantization/execution_providers/qnn/preprocess.py#L16)
- [quantization/execution_providers/qnn/quant_config.py](https://github.com/microsoft/onnxruntime/blob/23996bbbbe0406a5c8edbf6b7dbd71e5780d3f4b/onnxruntime/python/tools/quantization/execution_providers/qnn/quant_config.py#L20-L27)

### Running a quantized model on Windows ARM64 (onnxruntime-qnn version >= 1.18.0)
Install the ONNX Runtime ARM64 python package for QNN EP (requires Python 3.11.x and Numpy 1.25.2 or >= 1.26.4):
```shell
python -m pip install onnxruntime-qnn
```

The following Python snippet creates an ONNX Runtime session with QNN EP and runs the quantized model `model.qdq.onnx` on the HTP backend.

```python
# run_qdq_model.py

import onnxruntime as ort
import onnxruntime_qnn as qnn_ep
import numpy as np

# Register QNN EP library
# The registration name is used as the EP name for QNN EP
ep_lib_path = qnn_ep.get_library_path()
ep_registration_name = "QNNExecutionProvider"
ort.register_execution_provider_library(ep_registration_name, ep_lib_path)

# Select OrtEpDevice(s) matching the registration name (which is the EP name for QNN EP)
all_ep_devices = ort.get_ep_devices()
selected_ep_devices = [ep_device for ep_device in all_ep_devices if ep_device.ep_name == ep_registration_name]

if len(selected_ep_devices) == 0:
    raise RuntimeError("QNN EP device not found")

# Create session options
options = ort.SessionOptions()

# (Optional) Enable configuration that raises an exception if the model can't be
# run entirely on the QNN HTP backend.
options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

# Configure QNN EP options
ep_options = {"backend_path": qnn_ep.get_qnn_htp_path()}  # Use HTP backend

# Add QNN EP to session
options.add_provider_for_devices(selected_ep_devices, ep_options)

# Create an ONNX Runtime session.
# TODO: Provide the path to your ONNX model
session = ort.InferenceSession("model.qdq.onnx", sess_options=options)

# Run the model with your input.
# TODO: Use numpy to load your actual input from a file or generate random input.
input0 = np.ones((1,3,224,224), dtype=np.float32)
result = session.run(None, {"input": input0})

# Print output.
print(result)

# Clean up
del session

# Unregister the library after all sessions using it have been released
ort.unregister_execution_provider_library(ep_registration_name)
```

Running `python run_qdq_model.py` will execute the quantized `model.qdq.onnx` model on the QNN HTP backend.

Notice that the session has been optionally configured to raise an exception if the entire model cannot be executed on the QNN HTP backend. This is useful for verifying that the quantized model is fully supported by QNN EP.
Available session configurations include:
- [session.disable_cpu_ep_fallback](https://github.com/microsoft/onnxruntime/blob/a4cfdc1c28ac95ec6fd0667e856b6a6b8dd1020c/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h#L229): Disables fallback of unsupported operators to the CPU EP.
- [ep.context_enable](https://github.com/microsoft/onnxruntime/blob/a4cfdc1c28ac95ec6fd0667e856b6a6b8dd1020c/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h#L243): [Enable QNN context cache](./QNN-ExecutionProvider.md#qnn-context-binary-cache-feature) feature to dump a cached version of the model in order to decrease session creation time.

The above snippet only specifies the `backend_path` provider option. Refer to the [Configuration options section](./QNN-ExecutionProvider.md#configuration-options) for a list of all available QNN EP provider options.

## Running a model with QNN EP's GPU backend

The QNN GPU backend can run models with 32-bit/16-bit floating-point activations and weights as such without prior quantization. A 16-bit floating-point model generally can run inference faster on the GPU compared to its 32-bit version. To help reduce the size of large models, quantizing weights to `uint8`, while keeping activations in float is also supported.

Other than the quantized model requirement mentioned in the above HTP backend section, all other requirements are valid for the GPU backend also. So is the model inference sample code except for the portion where you specify the backend.

```python
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

# Register QNN EP library
# The registration name is used as the EP name for QNN EP
ep_lib_path = qnn_ep.get_library_path()
ep_registration_name = "QNNExecutionProvider"
ort.register_execution_provider_library(ep_registration_name, ep_lib_path)

# Select OrtEpDevice(s) matching the registration name (which is the EP name for QNN EP) and the device type (GPU)
all_ep_devices = ort.get_ep_devices()
selected_gpu_ep_devices = [ep_device for ep_device in all_ep_devices if ep_device.ep_name == ep_registration_name \
                            and ep_device.device.type == ort.OrtHardwareDeviceType.GPU]

if len(selected_gpu_ep_devices) == 0:
    raise RuntimeError("QNN GPU EP device not found")

# Once we filtered the ep devices, we don't need to specify the backend path
ep_options = {}

# Add QNN EP to session
options = ort.SessionOptions()
options.add_provider_for_devices(selected_gpu_ep_devices, ep_options)

# Create an ONNX Runtime session.
# TODO: Provide the path to your ONNX model
session = ort.InferenceSession("model.onnx", sess_options=options)

# Use session for inference
# ...

# Clean up
del session

# Unregister the library after all sessions using it have been released
ort.unregister_execution_provider_library(ep_registration_name)
```

## Running an LLM model with QNN EP's Genie backend

The QNN EP includes a Genie execution pathway for accelerated LLM inference on Windows ARM64 devices.
Genie is Qualcomm's generative AI framework (part of the QAIRT SDK) that the QNN EP can delegate token-to-logit inference
for auto-regressive LLM generation to using pre-compiled DLC (Deep Learning Container) models.

For complete dialog functionalities, including tokenization, sampling, autoregressive looping, etc., a higher-level
framework such as onnxruntime-genai should be used.

### Requirements

- **Platform**: Windows ARM64 only
- **QAIRT SDK**: Version ≥ 2.45.40 (Genie API ≥ 1.17)
- **Model**: An ONNX model containing an `EPContext` node with `ep_context_type="dlc"` pointing to a Genie-prepared DLC file

### How it works

Instead of using the standard QNN graph pathway, the EP automatically detects ONNX models that
contain an `EPContext` node referencing a Genie-compatible DLC file. At session creation and inference time, the EP:

1. Loads the Genie backend library (`Genie.dll`)
2. Creates a Genie DLC handle from the DLC path embedded in the ONNX model
3. Searches the DLC's parent directory for a backend extension override JSON file (`tmp*.json`)
4. Creates a Genie node configured with that DLC and executes token-to-logit inference on each `session.run()` call

The input tensor contains token IDs and the output tensor contains logits over the vocabulary.

### KV-cache management

The Genie node maintains an internal KV-cache across successive `session.run()` calls, enabling
efficient auto-regressive generation. Use the `kvcache_rewind` run option (see [Run Options](#run-options))
to reset the KV-cache at the start of a new conversation.

### Python example

```python
import numpy as np
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

# Register the QNN EP plugin library
ep_lib_path = qnn_ep.get_library_path()
ep_registration_name = "QNNExecutionProvider"
ort.register_execution_provider_library(ep_registration_name, ep_lib_path)

# Select QNN EP device
all_ep_devices = ort.get_ep_devices()
selected_ep_devices = [d for d in all_ep_devices if d.ep_name == ep_registration_name]

# Set Genie-specific EP options
ep_options = {"genie_log_level": "error"}

sess_options = ort.SessionOptions()
sess_options.add_provider_for_devices(selected_ep_devices, ep_options)

# Load the ONNX model that wraps the Genie DLC
sess = ort.InferenceSession("model.onnx", sess_options=sess_options)

# Reset the KV-cache at the start of a new conversation
run_options = ort.RunOptions()
run_options.add_run_config_entry("kvcache_rewind", "0")

# Run one auto-regressive step: input = token IDs, output = logits
input_tokens = np.array([[1, 128]], dtype=np.int32)
logits = sess.run(None, {"input_ids": input_tokens}, run_options)[0]

# Subsequent steps reuse the KV-cache (no rewind needed)
next_tokens = np.array([[1, 128]], dtype=np.int32)
logits = sess.run(None, {"input_ids": next_tokens})[0]

del sess
ort.unregister_execution_provider_library(ep_registration_name)
```

For an out-of-the-box example, see [`qcom/samples/test_genie.py`](../../qcom/samples/test_genie.py).

## QNN EP Input Graph Dump

Setting `dump_qnn_ep_input_graph` to `'1'` writes the ONNX graph the QNN EP
receives at compile time — after ORT Level 1 optimizations, before partitioning
and EP compile — to a JSON file (one per graph, named
`<graph_name>.<n>_qnn_ep_input_graph.json`, where `<n>` is a per-EP counter
that disambiguates collisions when two graph names sanitize to the same
string). Use `dump_qnn_ep_input_graph_dir` to
choose the output directory (defaults to the current working directory).

This is useful for inspecting exactly what the EP sees: that graph is **not** the
user's source `.onnx` (ORT has run constant folding, QDQ duplication via
`EnsureUniqueDQForNodeUnit`, Transpose optimization, etc.), and it is otherwise
hard to observe — `SessionOptions::optimized_model_filepath` refuses to serialize
a graph once an EP has compiled part of it into opaque compiled nodes ("Unable to
serialize model as it contains compiled nodes").

### Why the dump is QNN-Netron JSON, not an ONNX model

Netron is the open-source neural-net viewer that already understands the
JSON shape produced by QNN's existing `dump_json_qnn_graph`
(the post-compile QNN graph). The schema is not formally specified,
but it is stable and observable in this repository at
`onnxruntime/core/providers/qnn/builder/qnn_utils.cc` (`QnnJSONGraph`). The
EP-input dump deliberately reuses that schema rather than emitting a real
`.onnx`.

The QNN EP ships as a **standalone ABI plugin** that
links only the public ONNX Runtime C/C++ API, Abseil, and nlohmann/json — it does
**not** link protobuf or the ONNX proto definitions. Constructing or serializing
an `onnx::ModelProto` therefore is not possible from the EP without adding a
protobuf dependency to every shipped `onnxruntime_providers_qnn` binary, which
would break the plugin's minimal-dependency contract.

Reusing the QNN-Netron JSON schema instead means:

- **No new dependencies** — nlohmann/json is already used by the EP.
- **Opens directly in QNN Netron** — same schema Netron already understands.
- **One graph-dump format** across the EP: `dump_json_qnn_graph` for the
  post-compile QNN graph, `dump_qnn_ep_input_graph` for the pre-compile ONNX graph.
- **Consumable by tooling** — the offline `qcom/tools/op_trace_matcher/source_to_optimized_matcher.py`
  reads this JSON directly as its `--optimized-model`, recovering
  `original ONNX -> EP-input -> QNN op` provenance.

The schema carries node name/op_type/input_names/output_names and a tensor table
flagging each tensor's role (graph input / output / intermediate / initializer),
which is everything downstream consumers need.

**Known limitation:** the dump does **not** include initializer tensor data
bytes. Only the tensor name, dtype, shape, and role are recorded. Tools
consuming the dump (for example `source_to_optimized_matcher.py`'s
`--optimized-model` input) must therefore rely on initializer **names** or
graph **topology** — not data hashes — when correlating tensors across the
source and the EP-input graph. The matcher uses a name-derived sentinel hash
for initializers from this input so name-based and topology-based matching
work unchanged.

### Multiple dumps per graph name

A single inference session can produce **more than one** dump for the same
graph name, because ORT may invoke `GetCapability` more than once:

- **Layout transformation 2-pass cycle.** When the graph contains
  layout-sensitive ops (Conv, Resize, Pool, …), ORT runs a 1st pass to
  collect EP claims, applies an NCHW->NHWC rewrite for nodes the EP
  accepted, and runs a 2nd pass on the transformed graph. The two dumps
  differ — the 2nd has Conv in `kMSInternalNHWCDomain` and includes
  inserted Transpose ops.
- **Control-flow subgraphs.** Each `If` / `Loop` / `Scan` subgraph is a
  separate `OrtGraph*` and gets its own `GetCapability` invocation, so
  models with control flow legitimately produce multiple dumps with
  different graph names.
- **EPContext model loads.** Loading a previously-cached EPContext model
  triggers another `GetCapability` pass on that context graph.

Filenames append a per-EP atomic counter (`<graph_name>.<n>_…json`) so each
pass gets a distinct file, and an explicit overwrite warning is logged if
the same name is ever about to be rewritten.

**Rule for the offline matcher (`source_to_optimized_matcher.py`):** for
each unique `<graph_name>` in the dump directory, feed the file with the
**highest `<n>`** as `--optimized-model`. That is the graph the EP actually
compiled, and its node names are exactly the names that appear in
`qnn_op_trace.json`'s `sources[]`. Lower-numbered files are intermediate
snapshots (useful for debugging the layout transformer itself, not for
matcher input).

If a model has subgraphs (different graph names), pick the highest-numbered
file *per unique graph name* and run the matcher per graph.

## QNN context binary cache feature
There's a QNN context which contains QNN graphs after converting, compiling, finalizing the model. QNN can serialize the context into binary file, so that user can use it for futher inference directly (without the QDQ model) to improve the model loading cost.
The QNN Execution Provider supports a number of session options to configure this.

### Dump QNN context binary
1. Create session option, set "ep.context_enable" to "1" to enable QNN context dump. The key "ep.context_enable" is defined as kOrtSessionOptionEpContextEnable in [onnxruntime_session_options_config_keys.h](https://github.com/microsoft/onnxruntime/blob/8931854528b1b2a3f320d012c78d37186fbbdab8/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h#L239-L252).
2. Create the session with the QDQ model using session options created in step 1, and use HTP backend
A Onnx model with QNN context binary will be created once the session is created/initialized. No need to run the session.
The QNN context binary generation can be done on the QualComm device which has HTP using Arm64 build. It can also be done on x64 machine using x64 build (not able to run it since there's no HTP device).

The generated Onnx model which has QNN context binary can be deployed to production/real device to run inference. This Onnx model is treated as a normal model by QNN Execution Provider. Inference code keeps same as inference with QDQ model on HTP backend.

[Code example](https://github.com/microsoft/onnxruntime-inference-examples/blob/733ce6f3e8dd2ede8b67a8465684bca2f62a4a33/c_cxx/QNN_EP/mobilenetv2_classification/main.cpp#L90-L97)
```
#include "onnxruntime_session_options_config_keys.h"

// C++
Ort::SessionOptions so;
so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");

// C
const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
OrtSessionOptions* session_options;
CheckStatus(g_ort, g_ort->CreateSessionOptions(&session_options));
g_ort->AddSessionConfigEntry(session_options, kOrtSessionOptionEpContextEnable, "1");
```

```python
# Python
import onnxruntime

options = onnxruntime.SessionOptions()
options.add_session_config_entry("ep.context_enable", "1")
```

### Configure the context binary file path
The generated Onnx model with QNN context binary is default to [input_QDQ_model_name]_ctx.onnx in case user does not specify the path. User can to set the path in the session option with the key "ep.context_file_path". Example code below:

```
// C++
so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, "./model_a_ctx.onnx");

// C
g_ort->AddSessionConfigEntry(session_options, kOrtSessionOptionEpContextFilePath, "./model_a_ctx.onnx");
```

```python
# Python
options.add_session_config_entry("ep.context_file_path", "./model_a_ctx.onnx")
```

### Enable the embed mode
The QNN context binary content is not embedded in the generated Onnx model by default. A bin file will be generated separately. The file name looks like [input_model_file_name]_QNN_[hash_id].bin. The name is provided by Ort and tracked in the generated Onnx model. It will cause problems if any changes are made to the bin file. This bin file needs to sit together with the generated Onnx file. User can enable it by setting "ep.context_embed_mode" to "1". In that case the content of the context binary is embedded inside the Onnx model.

```
// C++
so.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "1");

// C
g_ort->AddSessionConfigEntry(session_options, kOrtSessionOptionEpContextEmbedMode, "1");
```

```python
# Python
options.add_session_config_entry("ep.context_embed_mode", "1")
```

### At-rest encryption of the context binary (ORT API v28+)

By default the QNN context binary is written to / read from disk in plaintext (as an external
file when `ep.context_embed_mode` is `"0"`, or embedded in the EPContext model when `"1"`). ORT
API v28 adds a write/read callback pair so an application can encrypt the binary before it is
persisted and decrypt it before QNN consumes it, without either version of ONNX Runtime having
any built-in cipher.

The callback-based encryption/decryption path described here applies only when
`ep.context_embed_mode` is `"0"`. When `ep.context_embed_mode` is `"1"` (embedded), the context
binary is stored within the EPContext model and applications should use ONNX Runtime model
protection mechanisms (for example, `SetOutputModelWriteFunc`) to protect the generated ONNX
model. If no callback is registered while `ep.context_embed_mode` is `"0"`, the EP uses the
legacy plaintext behavior.

- **Write callback** (`SetEpContextDataWriteFunc`) is set on `Ort::ModelCompilationOptions`
  during a compile session. The EP hands the plaintext context bytes to the callback instead of
  writing them to disk; the callback is responsible for encrypting and persisting them itself.
- **Read callback** (`SetEpContextDataReadFunc`) is set on `Ort::SessionOptions` for a later
  inference session. The EP asks the callback for the plaintext bytes (by name) instead of
  reading the on-disk file directly; the callback decrypts and returns them.
- Registering **no callback** is fully backward compatible — the EP falls back to the original
  plaintext disk read/write, unchanged.
- File-mapped weights and this feature are mutually exclusive for a given session: registering a
  read callback disables file mapping for that session, since file mapping requires reading the
  on-disk bytes directly.

**Scope — what is (and isn't) covered:**

- Only the standard EPContext artifact is encrypted: the external `_qnn.bin` (or the
  `EP_CACHE_CONTEXT`-referenced buffer) and the context-binary-list buffers used by multi-SoC /
  Flexible Context Binary compilation.
- The QNN IR-serializer `.dlc` artifact (written when the QNN IR serializer backend is used) and
  the DLC consumed by the [Genie LLM inference pathway](#running-an-llm-model-with-qnn-eps-genie-backend)
  node's `DlcConfig_create` are written to and read from disk directly and are **not** covered by
  these callbacks. An application combining Genie or the IR serializer backend with an encrypting
  write callback will still get a plaintext `.dlc` on disk.
- The cipher itself is entirely the application's responsibility — ORT and the QNN EP only provide
  the callback plumbing.

## Parallel Graph Preparation
For general context, please read the [QNN context binary cache feature](#qnn-context-binary-cache-feature) section above.

The process to prepare a model to run on the HTP backend is to do the following for each subgraph:

1. Compilation/composition into a QNN Graph
2. Finalization into a HTP-friendly binary
3. Input/Ouput tensor setup

For a typical model, this is done sequentially. However, for larger models, such as LLMs, one model may contain multiple subgraphs of substantial size. If done sequentially, the total time to prepare a model may seem like an eternity, thus resulting in the need for parallelization.

Parallel graph preparation was implemented to parallelize the prepare process as much as possible, saving a considerable amount of time. The QNN API currently only supports asynchronous graph finalization. As such, parallelized prepare is as follows:
1. Create and start a thread pool
2. For each subgraph, compose the QNN graph
3. For each subgraph, submit a job to finalize the QNN graph into the thread pool
4. Wait for all jobs to finish
5. For each subgraph, setup the I/O tensors

### General Usage
To utilize this feature, set the value of `num_graph_prepare_threads` to the desired number of threads as shown in the example code below:
```python
# Python on Windows on Snapdragon device
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

# General setup
# ...

# Configure EP options with optrace profiling
ep_options = {
    "backend_type": "htp",
    "htp_performance_mode": "burst",
    "num_graph_prepare_threads": "8" # Use 8 threads to graph preparation
}

sess_options = ort.SessionOptions()

# Enable context bin generation (not required)
sess_options.add_session_config_entry("ep.context_embed_mode", "0")
sess_options.add_session_config_entry("ep.context_enable", "1")

sess_options.add_provider_for_devices(selected_ep_devices, ep_options)

session = ort.InferenceSession("model.onnx", sess_options=sess_options)
```


### Important Considerations
#### Feature Disabled if Number of Subgraphs is Less Than 5
While graph composition is responsible for the majority of the preparation time, asynchronously finalizing the subgraphs cuts the total time down by a considerable amount, depending on the graph. For smaller models or models with only a few subgraphs, the overhead of setting up for parallel graph preparation will negate any possible performance gains and may actually result in worse performance.

#### Feature Disabled if `num_graph_prepare_threads` is 1
This defeats the purpose of the feature, and enabling the feature will only add additional overhead from thread pool creation.

#### Number of Supported Concurrency Threads by Hardware
The range of `num_graph_prepare_threads` is restricted to the maximum supported as reported by the hardware via [`std::thread::hardware_concurrency`](https://en.cppreference.com/w/cpp/thread/thread/hardware_concurrency.html).

If `std::thread::hardware_concurrency` fails, then the allowable range will be restricted to 4.

#### Default Behavior
If `num_graph_prepare_threads` is never set or set to a value outside of the allowable range, then the default number of threads will be 8.

If `std::thread::hardware_concurrency` returns a value less than 8, then the default number of threads will be the lower value rather than 8. This applies when `num_graph_prepare_threads` is set to an invalid value.

## QNN EP Framework Op Tracing

Framework op tracing records the mapping between ONNX operators and the QNN operators they compile to. It is useful for debugging accuracy issues, understanding node-group fusions, and diagnosing why certain operators fall back to CPU EP.

Enable tracing by setting `enable_framework_op_trace` to `'1'` and optionally `framework_op_trace_dir` to choose the output directory.

### Output File Naming

The trace is written as `{framework_op_trace_dir}/qnn_op_trace.json`.

### Sidecar for Pre-compiled Context Models

When loading a pre-compiled context model with `profiling_level=detailed` or `optrace`, the EP looks for a sidecar trace file to annotate profiling output with ONNX source op names. The sidecar path is `qnn_op_trace.json` in the same directory as the context model.

Because AOT context-binary generation (`ep.context_enable=1`) also writes `qnn_op_trace.json`, no manual rename is needed. The only prerequisite is that the Phase 1 (context-generation) run must set `framework_op_trace_dir` to the directory where the context model is saved (i.e. the parent directory of `ep.context_file_path`), so the sidecar lands next to the context model. If the two directories differ, copy `qnn_op_trace.json` next to the context model before the Phase 2 run.

## QNN EP Profiling
Profiling data is available with the HTP backend. Enabling QNN profiling will generate a user-readable .csv file that will contain information from initialization, execution, and de-initialization.

If onnxruntime is compiled with a more recent QAIRT SDK (2.39 or later), then a _qnn.log file will also be generated alongside the .csv file. This .log file is parsable by [qnn-profile-viewer](https://docs.qualcomm.com/doc/80-63442-10/topic/general_tools.html#qnn-profile-viewer), which is provided in the SDK.

### General Usage
To utilize QNN profiling, simply set the EP option profiling_level to basic, detailed, or optrace. Additionally, the EP option profiling_file_path must also be set to the output .csv filepath you would like to write data to:
```python
# Python on Windows on Snapdragon device
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep
import numpy as np

# Register QNN EP library
ep_lib_path = qnn_ep.get_library_path()
ep_registration_name = "QNNExecutionProvider"
ort.register_execution_provider_library(ep_registration_name, ep_lib_path)

# Select QNN EP device
all_ep_devices = ort.get_ep_devices()
selected_ep_devices = [ep_device for ep_device in all_ep_devices if ep_device.ep_name == ep_registration_name]

# Configure EP options with profiling
ep_options = {
    "backend_type": "htp",
    "htp_performance_mode": "burst",
    "device_id": "0",
    "htp_graph_finalization_optimization_mode": "3",
    "soc_model": "60",
    "htp_arch": "73",
    "vtcm_mb": "8",
    "profiling_level": "basic",
    "profiling_file_path": "output.csv"
}

sess_options = ort.SessionOptions()
sess_options.add_provider_for_devices(selected_ep_devices, ep_options)

session = ort.InferenceSession("model.onnx", sess_options=sess_options)

input0 = np.ones((1,2,3,4), dtype=np.float32)
result = session.run(None, {"input": input0})

# Clean up
del session
ort.unregister_execution_provider_library(ep_registration_name)
```

With the example above, a file "output.csv" will be generated containing the profiling data. Additionally, if using QAIRT 2.39 SDK or later, another file "output_qnn.log" will be generated.

"output_qnn.log" can then be parsed with the appropriate qnn-profile-viewer binary:
```console
> qnn-profile-viewer.exe --input_log .\output_qnn.log --output output_2.csv
```

The above will output basic information, such as the profiling data for the fastest and slowest execution as well as the average case. A .csv file can also be generated this way, too, though the information will likely not differ from the "output.csv".

Additionally, if the profiling_level is set to "detailed" or "optrace", additional data will be shown per-network-layer.

> **Combining profiling with framework op tracing:** When `profiling_level` is `detailed` or `optrace` **and** `enable_framework_op_trace` is `'1'`, the profiling CSV gains an extra `ONNX Source Ops` column. For each per-layer `NODE` event row, this column lists the originating ONNX operator name(s) (semicolon-separated for fused groups). This makes it easy to correlate QNN-level hardware profiling data back to the original ONNX model operators without manual lookup.
>
> At `profiling_level=basic` the `ONNX Source Ops` column is **not** added because basic profiling does not emit per-layer `NODE` events.

### Optrace-Level Profiling
[Optrace-level profiling](https://docs.qualcomm.com/doc/80-63442-10/topic/htp_backend.html#qnn-htp-profiling) generates a profiling .log file that contains [Qualcomm Hexagon Tensor Processor Analaysis Summary (QHAS)](https://docs.qualcomm.com/doc/80-63442-10/topic/htp_backend.html#qnn-htp-analysis-summary-qhas-) data. This data can be used to generate chrometraces and provide a web browser-friendly UI to visualize data.

**This feature is only available with the QAIRT 2.39 SDK and later.**

### Optrace Setup
To utilize this feature, a context binary must be generated prior to execution:
```python
# Python on Windows on Snapdragon device
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep
import numpy as np

# Register QNN EP library
ep_lib_path = qnn_ep.get_library_path()
ep_registration_name = "QNNExecutionProvider"
ort.register_execution_provider_library(ep_registration_name, ep_lib_path)

# Select QNN EP device
all_ep_devices = ort.get_ep_devices()
selected_ep_devices = [ep_device for ep_device in all_ep_devices if ep_device.ep_name == ep_registration_name]

# Configure EP options with optrace profiling
ep_options = {
    "backend_type": "htp",
    "htp_performance_mode": "burst",
    "device_id": "0",
    "htp_graph_finalization_optimization_mode": "3",
    "soc_model": "60",
    "htp_arch": "73",
    "vtcm_mb": "8",
    "profiling_level": "optrace",  # Set profiling_level to optrace
    "profiling_file_path": "optrace.csv"
}

sess_options = ort.SessionOptions()

# Enable context bin generation
sess_options.add_session_config_entry("ep.context_embed_mode", "0")
sess_options.add_session_config_entry("ep.context_enable", "1")

sess_options.add_provider_for_devices(selected_ep_devices, ep_options)

session = ort.InferenceSession("model.onnx", sess_options=sess_options)
```

Upon successful session creation, three files will be generated:
- model_ctx.onnx
- model_qnn.bin
- QNNExecutionProvider_QNN_\<number\>_schematic.bin

model_ctx.onnx is an onnx model with a node that points to the model_qnn.bin context binary, which will be used by the HTP backend for execution. The _schematic.bin file will be used by qnn-profile-viewer to generate QHAS data.

### Generating QHAS Data
Previously for general profiling data, a session was created and executed with "model.onnx". However, now there is a new _ctx.onnx model that utilizes a newly generated context binary. As such, a new inference session must be created with the new _ctx.onnx model:
```python
# Continuing from Optrace Setup:
sess_options = ort.SessionOptions()
sess_options.add_session_config_entry("ep.context_enable", "0")
sess_options.add_provider_for_devices(selected_ep_devices, ep_options)

optrace_session = ort.InferenceSession("model_ctx.onnx", sess_options=sess_options)

input0 = np.ones((1,2,3,4), dtype=np.float32)
result = optrace_session.run(None, {"input": input0})

# Clean up
del optrace_session
ort.unregister_execution_provider_library(ep_registration_name)
```

As before under "General Usage", a .csv file (optrace.csv) and a _qnn.log file (optrace_qnn.log) are generated.

qnn-profile-viewer can be used with different parameters and files to parse all the data written to optrace_qnn.log:
```console
> qnn-profile-viewer.exe --config .\config.json --reader .\QnnHtpOptraceProfilingReader.dll --input_log .\optrace_qnn.log  --schematic .\QNNExecutionProvider_QNN_12345_schematic.bin --output optrace.json
```

Please note:
- Three new files are used:
  - config.json: Please refer to the "Post Process (Chrometrace Generation)" section [on this page](https://docs.qualcomm.com/doc/80-63442-10/topic/htp_backend.html#qnn-htp-optrace-profiling).
  - QnnHtpOptraceProfilingReader.dll: Provided as part of the QAIRT SDK. The corresponding file for Linux is libQnnHtpOptraceProfilingReader.so.
  - QNNExecutionProvider_QNN_12345_schematic.bin: The name will vary. This file must be the same one generated alongside the context binary under "Optrace Setup".
- The output file is now a .json file containing chrometrace data. This .json file can be opened with either [Perfetto Trace Vizualizer](https://ui.perfetto.dev/) or with chrome://tracing.

After running qnn-profile-viewer, you should see a handful of .json files generated with the same prefix as the --output filename parameter. You should also see an .html file generated as well. This .html file can be opened by Chrome to view the chrometrace in a more user-friendly GUI.

### Additional References
For more information how to interpret QHAS data, please refer to [this page](https://docs.qualcomm.com/doc/80-63442-10/topic/htp_backend.html#qnn-htp-analysis-summary-qhas-).

For more information on the data collected with optrace profiling, please refer to [this page](https://docs.qualcomm.com/doc/80-63442-10/topic/htp_backend.html#qnn-htp-optrace-profiling).

## QNN EP weight sharing
Refers to the [EPContext design doc](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html#epcontext-with-weight-sharing)

Note: QNN EP requires **Linux x86_64** or **Windows x86_64** platform.

Additionally, if user creates the QNN context binary (`qnn_ctx.bin`) with weight sharing using the QNN toolchain (`qnn-context-binary-generator`), they can use a script to generate the wrapper Onnx model from the context:  [gen_qnn_ctx_onnx_model.py](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/qnn/gen_qnn_ctx_onnx_model.py). The script creates multiple `model_x_ctx.onnx` files, each containing an `EPContext` node that references the shared `qnn_ctx.bin` file. Each `EPContext` node specifies a unique node name, referring to different Qnn graph from the QNN context.


## Usage

### Using QNN EP as a Plugin EP

QNN EP can be used as a plugin execution provider. This is the recommended approach as it follows the standardized plugin EP pattern described in the [EP Context Design documentation](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html).
> **Note**: QNN EP requires the ep registartion name as "QNNExecutionProvider", and the ep_device name is the same as the registration name.

### C++
C API details are [here](https://onnxruntime.ai/docs/get-started/with-c.html).

```cpp
#include "onnxruntime_cxx_api.h"

const char* lib_registration_name = "QNNExecutionProvider";
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};

// Register QNN EP library with ONNX Runtime
env.RegisterExecutionProviderLibrary(
    lib_registration_name,           // Registration name
    ORT_TSTR("onnxruntime_providers_qnn.dll")  // Path to QNN EP library
);

{
    // Get available EP devices
    std::vector<Ort::ConstEpDevice> ep_devices = env.GetEpDevices();

    // Find the QNN EP device
    std::array<Ort::ConstEpDevice, 1> selected_ep_devices = { nullptr };
    for (Ort::ConstEpDevice ep_device : ep_devices) {
        if (std::strcmp(ep_device.GetName(), lib_registration_name) == 0) {
            selected_ep_devices[0] = ep_device;
            break;
        }
    }

    if (selected_ep_devices[0] == nullptr) {
        throw std::runtime_error("QNN EP device not found");
    }

    // Configure QNN EP options
    Ort::KeyValuePairs ep_options;
    ep_options.Append("backend_type", "htp");

    // Create session with QNN EP
    Ort::SessionOptions session_options;
    session_options.AppendExecutionProvider_V2(env, selected_ep_devices, ep_options);

    Ort::Session session(env, ORT_TSTR("model.onnx"), session_options);

    // Run model ...
}

// Unregister the library after all sessions using it have been released
env.UnregisterExecutionProviderLibrary(lib_registration_name);
```

### Python
```python
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

# Register QNN EP library
# The registration name is used as the EP name for QNN EP
ep_lib_path = qnn_ep.get_library_path()
ep_registration_name = "QNNExecutionProvider"
ort.register_execution_provider_library(ep_registration_name, ep_lib_path)

# Select OrtEpDevice(s) matching the registration name (which is the EP name for QNN EP)
all_ep_devices = ort.get_ep_devices()
selected_ep_devices = [ep_device for ep_device in all_ep_devices if ep_device.ep_name == ep_registration_name]

if len(selected_ep_devices) == 0:
    raise RuntimeError("QNN EP device not found")

# Configure QNN EP options
ep_options = {'backend_path': qnn_ep.get_qnn_htp_path()}  # Use HTP backend

# Create session options and add QNN EP
session_options = ort.SessionOptions()
session_options.add_provider_for_devices(selected_ep_devices, ep_options)

# Create a session with QNN EP using HTP (NPU) backend
sess = ort.InferenceSession(model_path, sess_options=session_options)

# Use session for inference
# ...

# Clean up
del sess

# Unregister the library after all sessions using it have been released
ort.unregister_execution_provider_library(ep_registration_name)
```

### Inference example

[Image classification with Mobilenetv2 in CPP using QNN Execution Provider with QNN CPU & HTP Backend](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx/QNN_EP/mobilenetv2_classification)


## Error handling
### HTP SubSystem Restart - [SSR](https://docs.qualcomm.com/doc/80-63442-10/topic/htp_backend.html#subsystem-restart-ssr-)
When an unrecoverable SSR (NPU crash) is detected, QNN EP returns `ORT_DEVICE_RESET` on ONNX Runtime 1.28.0+, or `ORT_ENGINE_ERROR` (`StatusCode::ENGINE_ERROR`) on older ONNX Runtime versions. Either way, the upper level framework/application should recreate the ONNX Runtime session if this error is detected during session run.


## Add new operator support in QNN EP
To enable new operator support in EP, areas to visit:
- QDQ script support this Op? [code example](https://github.com/microsoft/onnxruntime/pull/14867/files#diff-b1ea073c326fef46054382117c256f106d39bd7c34539d44c6e6d9e9eacc059c)
- Onnxruntime QDQ node unit support this Op? [code example](https://github.com/microsoft/onnxruntime/pull/14867/files#diff-ce0281aaf63e03ecadd592240e41f18742bf8eb095b3725c0e55e589c890946f)
- Is it layout sensitive operator?
  - Registered in LayoutTransformer?
    [code example](https://github.com/microsoft/onnxruntime/blob/6d464748ba7fed2275ecba3a7406298cabc93438/onnxruntime/core/optimizer/transpose_optimizer/transpose_optimizer.cc#L2168)
  - NHWC op schema registered?
    Example error message: <lambda_acc29b18d21b7c13448c4952cd957a60>::operator ()] Model face_det_qdq failed to load:Fatal error: com.ms.internal.nhwc:BatchNormalization(9) is not a registered function/op
    [Example PR](https://github.com/microsoft/onnxruntime/pull/15278)

### Example PRs to enable new operators:
- Non-layout sensitive operator. [Enable Hardsigmoid for QNN EP using SDK support direct support](https://github.com/microsoft/onnxruntime/pull/20956)

- Layout sensitive operator. [Add InstanceNormalization operator to QNN EP](https://github.com/microsoft/onnxruntime/pull/14867)


## QNN User-Defined Operation (UDO)

A **User-Defined Operation (UDO)** allows developers to extend the Qualcomm® Neural Network (QNN) runtimes with custom operators. UDO enables execution of operations that are not natively supported in the default QNN op set, while maintaining compatibility with model conversion, compilation, and runtime execution.

### Overview

A UDO lets you define and register custom operations—describing their inputs, outputs, parameters, data types, and backend behavior—so they can run on:

- CPU
- HTP

Once registered, UDOs integrate transparently into model conversion and runtime execution.

#### Limitations

**HTP UDO is not supported on Windows.** `qnn-op-package-generator` and the Hexagon toolchain only target HTP on Linux; the `x86_64-windows-msvc` host platform supports the CPU backend only. On Windows, CPU UDO must be generated with the `--gen_cmakelists` flag (see Step 2), which produces a `CMakeLists.txt`-based build instead of the Linux Makefile flow.

**The shipped UDO unit test runs on Linux x86_64 only.** `qnn-op-package-generator` requires Python 3.10, while the Windows CI environment ships with Python 3.11+, so the end-to-end UDO build/run loop is currently gated on Linux x86_64. Windows-side UDO support has separate CI coverage via `ParseOpPackages` parsing tests in `qnn_basic_test.cc`.

### UDO Workflow

#### **Step 1: Create a UDO Configuration File**
The configuration defines:
- Operation name
- Inputs / outputs
- Parameter definitions
- Supported data types
- Backend information

Schema references are available in the QNN SDK. You can also
see [MyAddOpPackageCpu.xml](../../onnxruntime/test/providers/qnn/udo/MyAddOpPackageCpu.xml) for CPU backend
and [MyAddOpPackageHtp.xml](../../onnxruntime/test/providers/qnn/udo/MyAddOpPackageHtp.xml) for HTP backend.

#### **Step 2: Generate the UDO Package**

Use the QNN Op Package Generator:

```bash
qnn-op-package-generator -p <path/to/op.xml> -o <output_dir>
```

The generator creates:
- Package scaffolding
- Interface provider
- Backend-specific template code

**Windows (CPU only):** the generator must be invoked from a `Developer PowerShell for VS 2022` shell with `python` and the `--gen_cmakelists` flag, which emits a `CMakeLists.txt` instead of a `Makefile`:

```powershell
python qnn-op-package-generator -p <path\to\op.xml> -o <output_dir> --gen_cmakelists
```

HTP UDO generation is not available on Windows (see Limitations).

#### **Step 3: Implement Custom Operation**

Fill in the generated skeleton:

Your custom logic goes into `src/ops/*.cpp`.
see [MyAddCPU.cpp](../../onnxruntime/test/providers/qnn/udo/MyAddCPU.cpp) for CPU backend
and [MyAddHTP.cpp](../../onnxruntime/test/providers/qnn/udo/MyAddHTP.cpp) for HTP backend.

#### **Step 4: Compile the UDO Package**

Compile against the QNN SDK and backend toolchains to produce:

- `lib<OpPackage>.so` — Implementation library
- Registration library used by QNN runtime

Note: To enable UDO compilation on CPU, you must prepare clang++:
```
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-21.1.8/LLVM-21.1.8-Linux-X64.tar.xz
tar -xvf LLVM-21.1.8-Linux-X64.tar.xz
export PATH=$(realpath ./LLVM-21.1.8-Linux-X64/bin/):$PATH
```
Note: To enable UDO compilation on HTP, you must prepare the Hexagon SDK:
```
wget https://softwarecenter.qualcomm.com/api/download/software/sdks/Hexagon_SDK/Linux/Debian/6.5.0.0/Hexagon_SDK_Linux.zip
unzip Hexagon_SDK_Linux.zip
export HEXAGON_SDK_ROOT=$(realpath Hexagon_SDK/6.5.0.0)
```

After Step 2 generated `<output_dir>/MyAddOpPackage/`, drop the implementation from Step 3 into `src/ops/MyAdd.cpp` and run `make` from inside the package directory.

**CPU backend (`all_x86` target):**

```bash
export QNN_SDK_ROOT=<path-to-qnn-sdk>
make -C <output_dir>/MyAddOpPackage all_x86
# Output: <output_dir>/MyAddOpPackage/libs/x86_64-linux-clang/libMyAddOpPackage.so
```

**HTP backend (`htp_x86` target):**

The HTP `htp_x86` target needs a Makefile that links the LLVM C++ runtime statically; use the
[HTP_Makefile](../../onnxruntime/test/providers/qnn/udo/HTP_Makefile) shipped with this repo
in place of the one produced by the generator.

```bash
export QNN_SDK_ROOT=<path-to-qnn-sdk>
export HEXAGON_SDK_ROOT=<path-to-hexagon-sdk>/6.5.0.0
cp onnxruntime/test/providers/qnn/udo/HTP_Makefile <output_dir>/MyAddOpPackage/Makefile
make -C <output_dir>/MyAddOpPackage htp_x86
# Output: <output_dir>/MyAddOpPackage/build/x86_64-linux-clang/libQnnMyAddOpPackage.so
```

#### **Step 5: Execute the Model with UDO**

```
./onnx_test_runner -v -e qnn -j 1 -i "backend_path|./libQnnCpu.so op_packages|<op_type>:<op_package_path>:<interface_symbol_name>[:<target>],<op_type2>:<op_package_path2>:<interface_symbol_nam2e>[:<target2>]" <models>
```

For the whole pipeline, refer [udo unit test](../../cmake/onnxruntime_unittests_udo.cmake)

### UDO References

- https://docs.qualcomm.com/doc/80-63442-10/topic/tutorial1.html
- https://docs.qualcomm.com/doc/80-63442-10/topic/op_package_gen_example.html


## Mixed precision support
The following figure demonstrates an example of mixed precision model.
<p align="center"><img width="100%" src="../images/quantization_mixed_precision_1.png" alt="mixed precision model"/></p>
A mixed precision QDQ model consists of regions with different activation/weight quantization data types. The boundary between regions converts between activation quantization data types (e.g., uint8 to uint16) using a DQ to Q sequence.

The ability to specify regions with different quantization data types enables exploring the tradeoffs between accuracy and latency. A higher integer precision may improve accuracy at the expense of latency, so selectively promoting certain regions to a higher precision can aid in achieving a desirable balance in key metrics.

The following figure shows a model with a region that has been promoted to 16-bit from the default 8-bit activation type.
<p align="center"><img width="60%" src="../images/quantization_mixed_precision_2.png" alt="mixed precision layers"/></p>

This model is quantized to uint8 precision, but tensor "Op4_out" is quantized to 16-bit. This can be achieved by specifying the following initial tensor quantization overrides:

```
# Op4_out could be an inaccurate tensor that should be upgraded to 16bit
initial_overrides = {"Op4_out": [{"quant_type": QuantType.QUInt16}]}

qnn_config = get_qnn_qdq_config(
    float_model_path,
    data_reader,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QUInt8,
    init_overrides=initial_overrides,  # These initial overrides will be "fixed"
)
```

The above snippet generates the following "fixed" overrides (get via qnn_config.extra_options["TensorQuantOverrides"]):

```
overrides = {
  “Op2_out”: [{“quant_type”: QUInt8, “convert”: {“quant_type”: QUInt16, “recv_nodes”: {“Op4”}}}],
  “Op3_out”: [{“quant_type”: QUInt8, “convert”: {“quant_type”: QUInt16, “recv_nodes”: {“Op5”}}}],
  “Op4_out”: [{“quant_type”: QUInt16}],
  “Op5_out”: [{“quant_type”: QUInt16, “convert”: {“quant_type”: QUInt8, “recv_nodes”: {“Op6”}}}]
}
```

After the override, the model works like this:

- Op2’s output is consumed by Op4, Op7, and Op8. Op4 consumes the converted u16 type, while Op7 and Op8 consume the original u8 type.
- Op3’s output is converted from u8 to u16. Op5 consumes the converted u16 type.
- Op4’s output is just u16 (not converted).
- Op5’s output is converted from u16 to u8. Op6 consumes the u8 type.

## LoRAv2 support
Currently, only pre-compiled models with EPContext nodes are supported. The  example script for reference [gen_qnn_ctx_onnx_model.py](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/qnn/gen_qnn_ctx_onnx_model.py). After applying the model LoRAv2 using the [QAIRT SDK](https://docs.qualcomm.com/doc/80-63442-10/topic/general_tutorials.html#lora-low-rank-adaptation), a main qnn context binary and several adapter binary sections will be generated. We use the LoRAv2 config and place it into RunOptions for inference.

* The format of the LoRAv2 config:
  * graph name: QNN graph in QNN pre-built context binary.
  * adapter binary section path: binary section generated by qnn-context-binary-generator
```
<graph name>;<adapter binary section path>
```
