# ONNX Runtime QNN Execution Provider v2.4.0

**ONNX Runtime Compatibility:** >= 1.24.1 (compiled with v1.26.0)<br>
**QAIRT SDK Compatibility:** 2.48.40

```
pip install onnxruntime==1.26.0
pip install onnxruntime-qnn==2.4.0
```

## Packaging

### New in 2.4.0

- **ONNX Runtime** — Compiled with v1.26.0. ([#371](https://github.com/onnxruntime/onnxruntime-qnn/pull/371))
- **ONNX opset library** — Upleveled to v1.21.0, picking up opset 26. ([#379](https://github.com/onnxruntime/onnxruntime-qnn/pull/379))
- **Artifacts** — Windows ARM64 (ARM64x), Windows x86_64, Windows ARM64EC Zip, Linux x86_64 tgz artifacts are now shipped as part of each Release.

### Platform Support

| Package | Windows ARM64 | Windows ARM64 (ARM64x) | Windows x86_64 | Linux ARM64 | Linux x86_64 | Android ARM64 |
|---|---|---|---|---|---|---|
| Python Wheel | Inference | — | AOT compilation | Inference | AOT compilation | — |
| NuGet | - | Inference | — | — | — | — |
| ZIP | Inference | Inference | AOT compilation | — | AOT compilation | — |
| tgz | — | — | — | Inference | — | — |
| Maven | — | — | — | — | — | Inference |

## New Operators and Fusions

- **GRU** — CPU (FP32) and HTP (QDQ/FP16). Time steps always unrolled. Default activations only; `layout=1` and dynamic sequence lengths not supported. ([#284](https://github.com/onnxruntime/onnxruntime-qnn/pull/284))
- **IsInf** — FP32/FP16 inputs; boolean output. ([#403](https://github.com/onnxruntime/onnxruntime-qnn/pull/403))
- **LpPool** — p ∈ {1, 2}; ranks 3/4/5. CPU FP32; HTP FP32/FP16/BF16 (V81+); GPU FP32/FP16. QDQ not supported. ([#437](https://github.com/onnxruntime/onnxruntime-qnn/pull/437))
- **OneHot** — `depth` and `values` must be constant initializers. HTP supports FP16/BF16 (V81+) and QDQ. Negative indices fall back to CPU EP. ([#466](https://github.com/onnxruntime/onnxruntime-qnn/pull/466))
- **Selu** — CPU, HTP, and GPU. FP32/FP16. ([#404](https://github.com/onnxruntime/onnxruntime-qnn/pull/404))
- **Xor** ([#402](https://github.com/onnxruntime/onnxruntime-qnn/pull/402))
- **Gather block quantize (GPU)** — Added quantization support for gather blocks in LLMs, enabling execution on Qualcomm GPU through EpContextBinary. ([#356](https://github.com/onnxruntime/onnxruntime-qnn/pull/356))
- **GroupQueryAttention (GPU)** — Support `com.Microsoft.GroupQueryAttention` by mapping onto `QNN_OP_GROUP_QUERY_ATTENTION`, enabling GQA nodes in LLMs to run on GPU for better performance. ([#424](https://github.com/onnxruntime/onnxruntime-qnn/pull/424))
- **DynamicQuantizeLinear → DequantizeLinear** — Fuses DQL+DQ round-trips into an identity Transpose, eliminating CPU fallback in models with incomplete quantization conversion. ([#490](https://github.com/onnxruntime/onnxruntime-qnn/pull/490))

For the full list of supported operators, see [Supported ONNX Operators](execution_providers/QNN-ExecutionProvider.md#supported-onnx-operators) and for supported fusions, see [Supported Operator Fusions](execution_providers/QNN-ExecutionProvider.md#supported-operator-fusions).

## Improvements

- **DX12 shared memory (GPU)** — Added DX12 shared memory allocator via `enable_dx12_shared_memory_allocator` provider option, leveraging `QNN_MEM_TYPE_DX12` mem-handles. Eliminates CPU↔GPU copies of KV cache tensors for improved LLM inferencing speed. ([#213](https://github.com/onnxruntime/onnxruntime-qnn/pull/213))
- **MatMul / Gemm** — Block-quantized (`BW_FLOAT_BLOCK`) weight support on HTP. INT4/UINT4/INT8/UINT8 weights. Requires QAIRT >= 2.47. ([#476](https://github.com/onnxruntime/onnxruntime-qnn/pull/476), [#477](https://github.com/onnxruntime/onnxruntime-qnn/pull/477))
- **LayerNormalization** — Decomposes into `LN → Mul + Add` when scale/bias rank is not aligned with the normalization axes. ([#417](https://github.com/onnxruntime/onnxruntime-qnn/pull/417))
- **Resize** — Routes rank-4 linear `pytorch_half_pixel` Resize to `ResizeBilinear`. ([#393](https://github.com/onnxruntime/onnxruntime-qnn/pull/393))
- **RoiAlign** — Accepts `coordinate_transformation_mode=half_pixel` and `sampling_ratio=0`. ([#389](https://github.com/onnxruntime/onnxruntime-qnn/pull/389))
- **SimplifiedLayerNormalization** — Added to the RMSNorm QDQ node-group selector, enabling QDQ execution on HTP. ([#338](https://github.com/onnxruntime/onnxruntime-qnn/pull/338))
- **DLC validation** — `backendValidateOpConfig` now routes to the target backend (HTP/CPU/GPU) when `dump_qnn_ir_dlc=1`, instead of always using QnnIr. ([#438](https://github.com/onnxruntime/onnxruntime-qnn/pull/438))
- **Framework op tracing** — New `qnn.enable_framework_op_trace` / `qnn.framework_op_trace_dir` session options for debugging graph partitioning. ([#345](https://github.com/onnxruntime/onnxruntime-qnn/pull/345))
- **QNN CPU backend** — No longer advertised by default; set `ORT_QNN_ENABLE_CPU_BACKEND` to re-enable. ([#500](https://github.com/onnxruntime/onnxruntime-qnn/pull/500))
- **HNRD path warning** — Logs a one-shot warning when the HNRD fallback path is active. ([#453](https://github.com/onnxruntime/onnxruntime-qnn/pull/453))
- **ETW profiling** — No longer escalates via ETW when `profiling_level` is OFF. ([#446](https://github.com/onnxruntime/onnxruntime-qnn/pull/446))
- **`prepare_only` validation** — Errors at session creation if `prepare_only=1` is set without `ep.context_enable=1`. ([#514](https://github.com/onnxruntime/onnxruntime-qnn/pull/514))

## Bug Fixes

- **BatchNormalization** — Run in FP when input is u8/u16. ([#434](https://github.com/onnxruntime/onnxruntime-qnn/pull/434))
- **BatchNormalization** — Allow quantized-input / no-output-Q BN as a float island on HTP; fixes graph capture failures in NASNet-style models. ([#497](https://github.com/onnxruntime/onnxruntime-qnn/pull/497))
- **Conv + Relu** — Insert explicit Relu when fused output encoding has `zp > 0`. ([#478](https://github.com/onnxruntime/onnxruntime-qnn/pull/478))
- **GlobalAveragePool** — Remove unconditional `count_pad_for_edges = true`; aligns DLC output with the native converter. ([#460](https://github.com/onnxruntime/onnxruntime-qnn/pull/460))
- **HardSigmoid** — Fix output encodings to full [0, 1] range required by HTP. ([#473](https://github.com/onnxruntime/onnxruntime-qnn/pull/473))
- **RandomNormalLike / RandomUniformLike** — Fix "Zero tensor size!" crash caused by missing `QnnTensorWrapper` for the shape-source `x` input. ([#415](https://github.com/onnxruntime/onnxruntime-qnn/pull/415))
- **GetGenieCapability** — Fix crash from dangling `session_options_` reference on Windows ARM64 LLM model load. ([#513](https://github.com/onnxruntime/onnxruntime-qnn/pull/513))
- **Linux ARM64 NPU detection** — QNN EP now registers an NPU device on Linux ARM64 by probing `/dev/fastrpc-cdsp*`, fixing "No QNN EP devices found". ([#557](https://github.com/onnxruntime/onnxruntime-qnn/pull/557))
- **offload_graph_io_quantization** — Fix crash on EPContext load by serializing the tensor-name rename map into the EPContext node at compile time. Backward compatible with previously generated context binaries. ([#544](https://github.com/onnxruntime/onnxruntime-qnn/pull/544))
- **Quantized Gelu ↔ Erf fusion** — Support `Div`/`Mul` as the parent of `Erf` (optimizers can rewrite `Div(sqrt(2))` as `Mul(1/sqrt(2))`), canonicalize duplicated intermediate root tensor names produced by ORT, validate against outer quantized I/O when wrapped by DQ/Q, and reject fusion when intermediate QDQGroups are present to avoid accuracy loss. ([#309](https://github.com/onnxruntime/onnxruntime-qnn/pull/309))

**Full Changelog:** [rel-2.3.0...rel-2.4.0](https://github.com/onnxruntime/onnxruntime-qnn/compare/rel-2.3.0...rel-2.4.0)

## Contributors

This release includes contributions from:

[Ashwath Shankarnarayan](https://github.com/qti-ashwshan), [Badri Narayanan](https://github.com/qti-mbadnara), [Calvin Nguyen](https://github.com/quic-calvnguy), [Cheng-Hsin Weng](https://github.com/qti-chenweng), [Chun-Chih Teng](https://github.com/qti-chuteng), [Hua-Yu Chou](https://github.com/huaychou), [Hung-Jui Wang](https://github.com/qti-hungjuiw), [Kuan-Yu Lin](https://github.com/kuanyul-qti), [Kyle Romero](https://github.com/qti-kromero), [Matthew Sinclair](https://github.com/qti-mattsinc), [Mike Hsu](https://github.com/quic-muchhsu), [Min Fong Hong](https://github.com/minfhong-qti), [Mu-Chien Hsu](https://github.com/quic-muchhsu), [Nischay Mamidi](https://github.com/qti-niscmami), [Sachin Jangid](https://github.com/sachjang-qti), [Simon Janezic](https://github.com/simejanko), [Shubham Patel](https://github.com/qti-shubham),  [Tirupathi Reddy T](https://github.com/tirupath-qti), [Vineeth Jatoth](https://github.com/vjatoth-qti), [Yathindra Kota](https://github.com/yath1), [Yuduo Wu](https://github.com/qti-yuduo), [Yu-Hung Chuang](https://github.com/yuhuchua-qti)

---

---

# ONNX Runtime QNN Execution Provider v2.3.0

**ONNX Runtime Compatibility:** >= 1.24.1 (compiled with v1.24.4)<br>
**QAIRT SDK Compatibility:** 2.47.0

```
pip install onnxruntime==1.24.4
pip install onnxruntime-qnn==2.3.0
```

## Packaging

### New in 2.3.0

- **NuGet** — ARM64 (ARM64X) package support added. Previously ARM64-only.
- **Linux x86_64 Python wheels** — New **preview** wheels for Ubuntu 22.04 (`manylinux_2_35_x86_64`), Python 3.11–3.14. Requires GLIBC >= 2.35 due to QAIRT library dependencies.
- **Maven (Android)** — New Android ARM64 package. Group ID / Artifact ID: `com.qualcomm.qti:onnxruntime-android-qnn`.

For instructions on building wheels across different architectures, see the [Build Guide](execution_providers/build.md).

### Platform Support

| Package | Windows ARM64 | Windows x64 | Linux ARM64 | Linux x86_64 | Android ARM64 |
|---|---|---|---|---|---|
| Python Wheel | Inference | AOT compilation | Inference | AOT compilation | — |
| NuGet | Inference | — | — | — | — |
| ZIP | Inference | — | — | — | — |
| tgz | — | — | Inference | — | — |
| Maven | — | — | — | — | Inference |

## New Operators and Fusions

- NonZero ([#217](https://github.com/onnxruntime/onnxruntime-qnn/pull/217))
- RandomNormalLike ([#266](https://github.com/onnxruntime/onnxruntime-qnn/pull/266))
- Identity ([#268](https://github.com/onnxruntime/onnxruntime-qnn/pull/268))
- **Gelu Pattern 3** — New `Erf*0.5 + 0.5` decomposition variant; fixes models previously not fused. ([#236](https://github.com/onnxruntime/onnxruntime-qnn/pull/236))
- **DynamicQuantizeLinear + MatMulInteger** — Fuses `DQL → MatMulInteger → Cast → Mul → [Add]` into a float QNN MatMul. ([#367](https://github.com/onnxruntime/onnxruntime-qnn/pull/367))
- **DynamicQuantizeLinear + ConvInteger** — Fuses `DQL → ConvInteger → Cast → Mul → [Add]` into a float QNN Conv2d. ([#364](https://github.com/onnxruntime/onnxruntime-qnn/pull/364))

For the full list of supported operators, see [Supported ONNX Operators](execution_providers/QNN-ExecutionProvider.md#supported-onnx-operators) and for supported fusions, see [Supported Operator Fusions](execution_providers/QNN-ExecutionProvider.md#supported-operator-fusions).

## Improvements

- Added `htp_share_resource_optimization` and `ep.enable_htp_prepare_only` provider options. See [Configuration Options](execution_providers/QNN-ExecutionProvider.md#configuration-options). ([#107](https://github.com/onnxruntime/onnxruntime-qnn/pull/107), [#347](https://github.com/onnxruntime/onnxruntime-qnn/pull/347))
- Added int32 input support for ScatterElements (QAIRT 2.45+). ([#247](https://github.com/onnxruntime/onnxruntime-qnn/pull/247))
- GatherND now uses shared index-normalization primitives for consistency with ScatterND/ScatterElements. ([#336](https://github.com/onnxruntime/onnxruntime-qnn/pull/336))
- Gemm with `beta=0.0` now maps to QNN FullyConnected without bias instead of falling back to CPU. ([#375](https://github.com/onnxruntime/onnxruntime-qnn/pull/375))
- RoiAlign now accepts `coordinate_transformation_mode=half_pixel` and `sampling_ratio=0`. ([#389](https://github.com/onnxruntime/onnxruntime-qnn/pull/389))
- MatMulNBits extended to HTP with 2-bit and 4-bit support (block sizes 32/64).([#288](https://github.com/onnxruntime/onnxruntime-qnn/pull/288))
- Graph verification in tests migrated to the public `GetEpGraphAssignmentInfo` API. ([#346](https://github.com/onnxruntime/onnxruntime-qnn/pull/346))
- Conv now supports block-quantized weights on HTP via the `BW_FLOAT_BLOCK` kernel, including int2 support. ([#429](https://github.com/onnxruntime/onnxruntime-qnn/pull/429))
- Static MSVC runtime linkage enabled for Windows x86_64 builds. ([#432](https://github.com/onnxruntime/onnxruntime-qnn/pull/432))

## Bug Fixes

- BatchNormalization: incorrect QNN offset handling for `QNN_DATATYPE_UFIXED_POINT_16` scale inputs. ([#135](https://github.com/onnxruntime/onnxruntime-qnn/pull/135))
- ThresholdedRelu: stale `add → relu → sign → mul` pattern replaced with QAIRT-aligned `Greater → Select`. ([#221](https://github.com/onnxruntime/onnxruntime-qnn/pull/221))
- Graph composition failure when `offload_graph_io_quantization=1` and a graph input fans out to multiple QDQ pairs. ([#295](https://github.com/onnxruntime/onnxruntime-qnn/pull/295))
- Softmax `axis ≠ rank-1` falling back to CPU due to missing upstream tensor wrappers at validation time. ([#304](https://github.com/onnxruntime/onnxruntime-qnn/pull/304))
- ScatterND/ScatterElements silent CPU fallback for negative or INT_64 indices. ([#311](https://github.com/onnxruntime/onnxruntime-qnn/pull/311), [#317](https://github.com/onnxruntime/onnxruntime-qnn/pull/317))
- Build failure on Ubuntu 24.04 / GCC 13 due to false-positive `-Wmaybe-uninitialized`. ([#387](https://github.com/onnxruntime/onnxruntime-qnn/pull/387))
- QNN EP failure on devices where DXCore cannot discover the NPU. ([#12](https://github.com/onnxruntime/onnxruntime-qnn/pull/12))
- ORT Core version floor raised to `>= 1.24.2`, preventing accidental downgrade. ([#448](https://github.com/onnxruntime/onnxruntime-qnn/pull/448))

**Full Changelog:** [rel/ort-qnn-ep/2.2.0...rel-2.3.0](https://github.com/onnxruntime/onnxruntime-qnn/compare/rel/ort-qnn-ep/2.2.0...rel-2.3.0)

## Known Issues

- **WoS AMD64 — Python 3.11 installer issue** — `ep.get_library_path()` returns the `amd64` path instead of `arm64ec`; manually construct the path to the `arm64ec` library as a workaround. Ongoing.

## Contributors

This release includes contributions from:

[Ashwath Shankarnarayan](https://github.com/qti-ashwshan), [Badri Narayanan](https://github.com/qti-mbadnara), [Chun-Chih Teng](https://github.com/qti-chuteng), [Hua-Yu Chou](https://github.com/huaychou), [Hung-Jui Wang](https://github.com/qti-hungjuiw), [Jaykumar Luhar](https://github.com/qti-luharj), [Kuan-Yu Lin](https://github.com/kuanyul-qti), [Min Fong Hong](https://github.com/minfhong-qti), [Mu-Chien Hsu](https://github.com/quic-muchhsu), [Shubham Patel](https://github.com/qti-shubham), [Tirupathi Reddy T](https://github.com/tirupath-qti), [Xia Han](https://github.com/xiha0704), [Yathindra Kota](https://github.com/quic-ykota), [Yu-Hung Chuang](https://github.com/yuhuchua-qti), [Yuduo Wu](https://github.com/qti-yuduo)

---

---

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