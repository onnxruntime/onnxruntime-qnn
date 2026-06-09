# ONNX Runtime QNN Execution Provider v2.2.0
This release delivers operator coverage improvements, multi-NPU device selection, and build fixes.

**ONNX Runtime Compatibility:** >= 1.24.1 (compiled with v1.24.4)<br>
**QAIRT SDK Compatibility:** 2.46.0

```
pip install onnxruntime==1.24.4
pip install onnxruntime-qnn==2.2.0
```

## Bug Fixes

- QNN EP: Fixed `GlobalMaxPool`/`GlobalAveragePool` falsely claiming rank-3 support; unified the 3D→4D reshape path with windowed pool ops. ([#201](https://github.com/onnxruntime/onnxruntime-qnn/pull/201))
- QNN EP: Restored Genie builds against QAIRT SDKs older than 2.45.0 by keying conditional compilation off the Genie API version (`GenieDlc.h` breaking change). ([#225](https://github.com/onnxruntime/onnxruntime-qnn/pull/225))
- QNN EP: Fixed GCC 13 build failures: corrected `memory_order_acq_rel` on `std::atomic::store()` to `memory_order_release`, and suppressed a false-positive `-Wmaybe-uninitialized` in `TestInputDef`. ([#228](https://github.com/onnxruntime/onnxruntime-qnn/pull/228))
- QNN EP: Fixed HNRD Model Compatibility checks incorrectly running on x86 platforms where they don't apply. ([#319](https://github.com/onnxruntime/onnxruntime-qnn/pull/319))

## Improvements

- QNN EP: Relaxed QDQ BatchNormalization selector to accept BN nodes with 2 dequantized inputs (instead of requiring 3), matching the common pattern where `bias`/`mean`/`variance` stay as float initializers. Reduces CPU fallback and graph fragmentation. ([#209](https://github.com/onnxruntime/onnxruntime-qnn/pull/209))
- QNN EP: NPU device selection now supports HTP cores with non-zero device IDs. ([#215](https://github.com/onnxruntime/onnxruntime-qnn/pull/215))

## Known Issues

- **WoS AMD64 — Python 3.11 installer issue causes inference failure** — On Windows on Snapdragon, `ep.get_library_path()` returns the `amd64` folder path instead of `arm64ec`, causing inference to fail in the AMD64 Python 3.11 environment, due to a known issue with the installer. As a workaround, manually construct the path to the `arm64ec` library. This issue affects Python 3.11 only.

### Platform Support

| Package | Windows ARM64 | Windows x64 | Linux ARM64 |
|---|---|---|---|
| Python Wheel | Inference | AOT compilation + Inference | Inference |
| NuGet | Inference | — | — |
| ZIP | Inference | — | — |
| tgz | — | — | Inference |

**Full Changelog:** [rel-2.1.0...rel/ort-qnn-ep/2.2.0](https://github.com/onnxruntime/onnxruntime-qnn/compare/rel-2.1.0...rel/ort-qnn-ep/2.2.0)

## Contributors

This release includes contributions from:

[Arnav Deshpande](https://github.com/qti-arnadesh), [Ashwath Shankarnarayan](https://github.com/qti-ashwshan), [Badri Narayanan](https://github.com/qti-mbadnara), [Calvin Nguyen](https://github.com/quic-calvnguy), [Cheng-Hsin Weng](https://github.com/qti-chenweng), [Chun-Chih Teng](https://github.com/qti-chuteng), [Hua-Yu Chou](https://github.com/huaychou), [Hung-Jui Wang](https://github.com/qti-hungjuiw), [Jeff Kilpatrick](https:/github.com/qti-jkilpatrick), [Kuan-Yu Lin](https://github.com/kuanyul-qti), [Kyle Romero](https://github.com/qti-kromero), [Matthew Sinclair](https://github.com/qti-mattsinc), [Mike Hsu](https://github.com/quic-muchhsu), [Min Fong Hong](https://github.com/minfhong-qti), [Shubham Patel](https://github.com/qti-shubham), [Tirupathi Reddy T](https://github.com/tirupath-qti), [Yathindra Kota](https://github.com/quic-ykota), [Yuduo Wu](https://github.com/qti-yuduo), [Yu-Hung Chuang](https://github.com/yuhuchua-qti)

---

---



# ONNX Runtime QNN Execution Provider v2.1.1
This is a patch release of the QNN Execution Provider, containing bug fixes and packaging updates.

**ONNX Runtime Compatibility:** >= 1.24.1 (compiled with v1.24.4)<br>
**QAIRT SDK Compatibility:** 2.45.41

```
pip install onnxruntime==1.24.4
pip install onnxruntime-qnn==2.1.1
```

## Bug Fixes

- QNN EP: Fixed a per-tensor, per-inference memory leak in `OrtTensorTypeAndShapeInfo` during `ExecuteGraph` on the ABI path. ([#326](https://github.com/onnxruntime/onnxruntime-qnn/pull/326))
- QNN EP: Fixed `TryGetMaxSpillFillSize` reading all EP contexts instead of only main contexts, which caused `QNN_CONTEXT_ERROR_INVALID_CONFIG` on multi-split weight-shared models. ([#328](https://github.com/onnxruntime/onnxruntime-qnn/pull/328))

## Improvements

- QNN EP: Switched `onnxruntime_providers_qnn.dll` to static MSVC runtime linkage, eliminating the runtime dependency on `MSVCP140.dll` and `VCRUNTIME140.dll`. ([#241](https://github.com/onnxruntime/onnxruntime-qnn/pull/241))
- QNN EP: Reduced peak memory in model compatibility validation from ~200 MB to ~50 MB by removing the context blob version from compatibility checks, avoiding the fake context binary creation and preparation library load. ([#366](https://github.com/onnxruntime/onnxruntime-qnn/pull/366))

## Packaging

- Linux ARM64 Python wheels — promoted from preview (v2.1.0) to officially supported. As with Windows, wheels are published for Python 3.11 through Python 3.14.
- Linux ARM64 `.tgz` archive — new distribution shipping the QNN EP shared library and headers for use outside of Python.

### Platform Support

| Package | Windows ARM64 | Windows x64 | Linux ARM64 |
|---|---|---|---|
| Python Wheel | Inference | AOT compilation + Inference | Inference |
| NuGet | Inference | — | — |
| ZIP | Inference | — | — |
| tgz | — | — | Inference |

**Full Changelog:** [rel-2.1.0...rel-2.1.1](https://github.com/onnxruntime/onnxruntime-qnn/compare/rel-2.1.0...rel-2.1.1)

## Contributors

This release includes contributions from:

[Arnav Deshpande](https://github.com/qti-arnadesh), [Ashwath Shankarnarayan](https://github.com/qti-ashwshan), [Badri Narayanan](https://github.com/qti-mbadnara), [Calvin Nguyen](https://github.com/quic-calvnguy), [Cheng-Hsin Weng](https://github.com/qti-chenweng), [Chun-Chih Teng](https://github.com/qti-chuteng), [Hua-Yu Chou](https://github.com/huaychou), [Hung-Jui Wang](https://github.com/qti-hungjuiw), [Jeff Kilpatrick](https:/github.com/qti-jkilpatrick), [Kuan-Yu Lin](https://github.com/kuanyul-qti), [Kyle Romero](https://github.com/qti-kromero), [Matthew Sinclair](https://github.com/qti-mattsinc), [Mike Hsu](https://github.com/quic-muchhsu), [Min Fong Hong](https://github.com/minfhong-qti), [Samrat Dutta](https://github.com/samrdutt-design), [Shubham Patel](https://github.com/qti-shubham), [Tirupathi Reddy T](https://github.com/tirupath-qti), [Yathindra Kota](https://github.com/quic-ykota), [Yuduo Wu](https://github.com/qti-yuduo), [Yu-Hung Chuang](https://github.com/yuhuchua-qti)

---

---



# ONNX Runtime QNN Execution Provider v2.1.0

**ONNX Runtime Compatibility:** >= 1.24.1 (compiled with v1.24.4)
**QAIRT SDK Compatibility:** 2.45.40

```
pip install onnxruntime
pip install onnxruntime-qnn==2.1.0
```

---

## Highlights

QNN EP 2.1.0 adds Genie backend support for on-device LLM inference, 64-bit uDMA (user-DMA) for next-gen hardware, seven new operators, four graph-level fusions, and significant memory and startup performance improvements for large model deployments on ARM64 Windows.

---

## New Features

### Genie Backend Support
Added Genie backend integration for ONNX Runtime, enabling on-device LLM inference through the Qualcomm Genie runtime. Genie operates on pre-compiled context binaries and provides optimized execution for generative AI workloads.

### 64-bit Extended uDMA (user-DMA) Mode
Enabled 64-bit Extended uDMA (user-DMA) for v81+ hardware via the `extended_udma` provider option. Allows far-mapping of weights and spill/fill buffers, expanding addressable memory for large models on next-generation Qualcomm devices.

### File-Mapped Weights on ARM64 Windows
Enabled Windows file mapping of weights and context binaries on ARM64. Multiple ORT sessions can share a single loaded context binary, eliminating per-session heap allocation. Significantly improves memory efficiency and initialization time for LLM-scale deployments. Automatically disabled when `context_embed_mode=1`.

### HTP Weight Sharing on ARM64
Enabled HTP weight sharing on ARM64. A warning is logged on unsupported platforms.

### Parallelized Graph Prepare
Introduced `QnnJobThreadPool` to parallelize graph finalization during `ComposeGraph`. Reduces ORT session creation time for models with many subgraphs.

### Runtime Version Query
Added `__version__` attribute to the QNN EP Python package:
```python
import onnxruntime_qnn as qnn_ep
print(qnn_ep.__version__)
```

### GetHardwareDeviceIncompatibilityDetails API
Implemented the `GetHardwareDeviceIncompatibilityDetails` API for the QNN ABI EP, enabling callers to retrieve structured details about hardware incompatibilities at runtime.

### Offline x64 Compilation with MEMHANDLE IO Type
Extended offline compilation support to x64 platforms for graphs using MEMHANDLE IO type (previously ARM64-only).

---

## New Operator Support (7 new ops)

| Operator | Backends | Notes |
|---|---|---|
| **Softplus** | CPU, HTP | U8/U16 quantized on HTP; ranks > 4 on CPU only |
| **RotaryEmbedding** | HTP | QNN_OP_ROPE via new RopeOpBuilder |
| **Tan** | CPU, HTP | Standard trigonometric op |
| **IsNaN** | CPU, HTP | Boolean output |
| **GroupNormalization** | HTP | Group normalization |
| **SimplifiedLayerNormalization** | HTP | RMSNorm-style normalization |
| **RoiAlign** | HTP | Region of interest alignment |

### Improved Operators

- **SpaceToDepth** — Added DCR (Depth-Column-Row) mode attribute.
- **Resize** — Added cubic interpolation mode.

For a complete list of operators supported by QNN and ORT QNN EP, refer to the [ORT QNN EP operator list](docs/execution_providers/QNN-ExecutionProvider.md#supported-onnx-operators). For detailed QNN operator definitions and backend-specific constraints, see the [QNN Master Operator Definition](https://docs.qualcomm.com/doc/80-63442-10/topic/MasterOpDef.html).

---

## Graph Optimizations & Fusions

### Reshape-Transpose-Reshape → SpaceToDepth Fusion
Folds Reshape-Transpose-Reshape sequences (including surrounding pre/post transpose ops) into a single SpaceToDepth op. Resolves context binary failures caused by 6D transpose/reshape graphs and brings parity with QAIRT converter behavior.

### LayerNorm Decomposed Pattern Fusion
Detects decomposed LayerNorm subgraphs in ONNX models and fuses them into a single `QNN_OP_LAYER_NORM` operator.

### Gemm Decomposition for Dynamic Bias
Extended Gemm handling for cases where the bias tensor is an intermediate (NATIVE) tensor. Decomposes Gemm into FC + ElementWiseAdd, fixing compatibility with models where ORT's MatMulAddFusion produces Gemm ops with dynamic bias (e.g., CLIP-style text projection layers).

### 6D Reshape-Einsum-Reshape Pattern
Added handling for 6D Reshape-Einsum-Reshape patterns that previously caused compilation failures.

---

## Bug Fixes

Various correctness and stability fixes across the EP.

---

## Infrastructure & Packaging

- **Unified x64 + ARM64EC wheel** — Single Python wheel supports both Windows x64 and Windows ARM64 targets

---

### Platform Support

| Package | Windows ARM64 | Windows x64 |
|---|---|---|
| Python Wheel | Inference | AOT compilation + Inference |
| NuGet | Inference | — |
| ZIP | Inference | — |

---

## Known Issues

- **SpaceToDepth FP32 accuracy** — FP32 validation tests for the SpaceToDepth operator have been disabled due to known accuracy issues. A fix is targeted for the next release.

---

## Contributors

This release includes contributions from:

[Arnav Deshpande](https://github.com/qti-arnadesh), [Ashwath Shankarnarayan](https://github.com/qti-ashwshan), [Badri Narayanan](https://github.com/qti-mbadnara), [Calvin Nguyen](https://github.com/quic-calvnguy), [Cheng-Hsin Weng](https://github.com/qti-chenweng), [Chun-Chih Teng](https://github.com/qti-chuteng), [Hua-Yu Chou](https://github.com/huaychou), [Hung-Jui Wang](https://github.com/qti-hungjuiw), [Jeff Kilpatrick](https:/github.com/qti-jkilpatrick), [Kuan-Yu Lin](https://github.com/kuanyul-qti), [Kyle Romero](https://github.com/qti-kromero), [Matthew Sinclair](https://github.com/qti-mattsinc), [Mike Hsu](https://github.com/quic-muchhsu), [Min Fong Hong](https://github.com/minfhong-qti), [Samrat Dutta](https://github.com/samrdutt-design), [Shubham Patel](https://github.com/qti-shubham), [Tirupathi Reddy T](https://github.com/tirupath-qti), [Yathindra Kota](https://github.com/quic-ykota), [Yuduo Wu](https://github.com/qti-yuduo), [Yu-Hung Chuang](https://github.com/yuhuchua-qti)

---
---

# ONNX Runtime QNN Execution Provider v2.0.0 (Preview)

**v2.0.0 is the first Plugin QNN EP (Preview) release** — a standalone package that brings Qualcomm hardware acceleration to any standard ONNX Runtime installation, with no custom ORT build required.

```
pip install onnxruntime
pip install onnxruntime-qnn==2.0.0
```

---

## What is the Plugin QNN EP?

Starting with v2.0.0, the QNN Execution Provider ships as a **standalone shared library** (`onnxruntime_providers_qnn.dll`) built on the [Execution Provider ABI](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/) introduced in ONNX Runtime 1.24.1. The plugin is registered at runtime and only links against the standard ONNX Runtime shared library — no internal ORT dependencies, no custom builds.

This replaces the previous built-in EP model (distributed as `onnxruntime-qnn` without a version pin) with a decoupled plugin that can be versioned and released independently.

---

## Highlights

### Feature Parity with ORT 1.24 Built-in QNN EP

v2.0.0 delivers the same capabilities as the built-in QNN EP in ONNX Runtime 1.24, re-implemented on the new plugin architecture:

- **97 ONNX operators supported** — [full operator list](docs/execution_providers/QNN-ExecutionProvider.md#supported-onnx-operators)
- **HTP, CPU, and GPU backend support** via [Qualcomm AI Runtime SDK (QAIRT)](https://qpm.qualcomm.com/#/main/tools/details/Qualcomm_AI_Runtime_SDK)
- **Context binary caching** with cross-version compatibility verification
- **Mixed precision inference** — automatic FP32-to-FP16 conversion on HTP
- **QDQ quantization** — per-tensor, per-channel, and low power block quantization
- **Graph optimizations** — operator fusions (Gelu, LPBQ MatMul/Gemm, HardSigmoid, channel shuffle, and more)
- **Weight sharing** across inference sessions
- **HTP performance tuning** — burst/balanced/power-saver modes, RPC latency control
- **Profiling support** — ETW, QNN profiling, and Perfetto trace integration

As a standalone plugin, the QNN EP is no longer tied to ONNX Runtime core release timelines. This enables faster iteration on Qualcomm-specific features and optimizations.

---

## Migration

Migrating from the built-in QNN EP (`onnxruntime-qnn`) to the Plugin QNN EP (`onnxruntime-qnn==2.0.0`) requires changes to both installation and session setup.

| | Built-in QNN EP | Plugin QNN EP (v2.0.0) |
|---|---|---|
| Install | `pip install onnxruntime-qnn` | `pip install onnxruntime` + `pip install onnxruntime-qnn==2.0.0` |
| ORT build | Custom build with QNN | Standard ORT release |
| EP registration | Automatic | Explicit plugin registration via `register_execution_provider_library` |

**Client impact and migration guide:**
- [Plugin EP Usage Guide](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/usage.html) — covers the new registration API and session setup
- [C++ example](docs/execution_providers/QNN-ExecutionProvider.md#c)
- [Python example and more](docs/execution_providers/QNN-ExecutionProvider.md#qnn-execution-provider)

---

## Dependencies

| Component | Version | Notes |
|---|---|---|
| **QAIRT SDK** | 2.42.0 | Other QAIRT SDK versions may also be used |
| **ONNX Runtime** | 1.24.1+ | Compatible with any ORT version supporting the EP ABI |
| **OGA** | 0.13 | [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai) |

### Platform Support

| Package | Windows ARM64 | Windows x64 |
|---|---|---|
| Python Wheel | Inference | AOT compilation |
| NuGet | Inference | — |
| ZIP | Inference | — |

---

## Resources

| Topic | Link |
|---|---|
| QNN EP documentation | [QNN-ExecutionProvider.md](docs/execution_providers/QNN-ExecutionProvider.md) |
| Build from source | [Build Guide](docs/execution_providers/build.md) |
| Development guide | [Development Guide](docs/execution_providers/development.md) |
| Plugin EP overview | [Plugin EP Libraries](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/) |
| Plugin EP usage | [Plugin EP Usage](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/usage.html) |

---

## Contributors

This release includes contributions from the Qualcomm engineering teams.