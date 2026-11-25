# Persistent GQA compile probe (experimental)

This opt-in experiment replaces omitted GQA past key/value inputs with two
backend-managed FP16 tensors. It checks compilation only. It does not implement
KV initialization, reset, readback, or sharing between separate sessions.

`enable_htp_persistent_buffer=1` requires HTP, omitted past key/value inputs,
and named internal present key/value outputs with rank-4 FP16 metadata. Remove
KV graph IO while retaining the three GQA node outputs and their cache shapes.
The default option is off.

The prototype uses tensor type **14** via a local cast because the SDK headers
used for this experiment do not publish `QNN_TENSOR_TYPE_PERSISTENT`. This value
and the preparation contract require confirmation against the selected backend.
The standard enum-to-string helper prints `Unsupported type`; the numeric QNN
JSON dump records the actual tensor type.

## Run

The script requires Python `onnx`, a plugin-capable `onnxruntime` with
`ModelCompiler`, a built QNN EP, and a matching backend package. It disables
CPU fallback and explicitly assigns GQA to HTP.

On Windows, copy `onnxruntime_providers_qnn.dll` and
`onnxruntime_providers_shared.dll` into a separate directory without SDK DLLs.
Keep the matching backend DLLs and skels together in the backend directory to
avoid mixing library versions. Use fresh output directories for reruns.

```powershell
python qcom/tools/persistent_gqa_compile_probe.py --self-check

$probeEp = 'build/probe_ep/onnxruntime_providers_qnn.dll'
$probeBackend = 'path/to/backend/QnnHtp.dll'
python qcom/tools/persistent_gqa_compile_probe.py --ep-library $probeEp --backend $probeBackend --mode baseline --cross-device-prepare --output-dir build/probe_baseline
python qcom/tools/persistent_gqa_compile_probe.py --ep-library $probeEp --backend $probeBackend --mode default --cross-device-prepare --output-dir build/probe_default
python qcom/tools/persistent_gqa_compile_probe.py --ep-library $probeEp --backend $probeBackend --mode persistent --cross-device-prepare --output-dir build/probe_persistent
```

The fixture uses packed FP16 QKV `[1,1,512]`, 8 query heads, 4 KV heads,
head size 32, and cache `[1,4,128,32]`. Full attention is used without RoPE.
Sequence lengths are constants; the fixture is not an inference workload.
The default target architecture is v73, with one core.

- `baseline`: explicit KV IO, option off; three graph inputs and outputs.
- `default`: omitted KV IO, option off; one graph input and output.
- `persistent`: omitted KV IO, option on; two type-14 cache inputs inside QNN.

Successful runs verify EPContext export, graph IO counts, and numeric persistent
tensor types/shapes/dtypes in the JSON dump, then write a result JSON.
A native backend crash can terminate the process before a result JSON is written.

## Current evidence

Native ARM64 Release build and model structural checks pass. Standalone
ModelCompiler experiments with CPU fallback disabled produced:

| Backend | Mode | Result |
|---|---|---|
| Standard | baseline | Compile, finalize, and context export pass |
| Prototype | baseline | Compile, finalize, and context export pass |
| Prototype | default | Compile, finalize, and context export pass; no type-14 tensors |
| Prototype | persistent | Validation, tensor creation, and graphAddNode pass; graphFinalize crashes |
| Standard | persistent | graphAddNode fails with 6007 (`QNN_GRAPH_ERROR_UNCONNECTED_NODE`) |

Cross-device preparation was needed for both baselines; without it, local-device
op validation returned 4005 (`QNN_BACKEND_ERROR_OP_PACKAGE_NOT_FOUND`). Op
validation was not bypassed in the final experiments.

The prototype persistent graph dump confirms two type-14 FP16 caches in QNN
input slots 5/6. Present key/value remain internal NATIVE outputs. A first-chance
debugger reproduction locates the access violation (`0xc0000005`) at
`QnnHtpPrepare.dll+0x1984`, called from graph finalization. Matching prototype
HTP/Prepare module paths were verified. The root cause is unresolved without
the backend's private source/symbols and confirmed preparation recipe.

These are compile-probe results, not unit-test or inference results. Removing
optional KV IO with the option off does not demonstrate state retention.

Next: resolve finalization, establish initialization/update/reset semantics,
then compare consecutive decode outputs against an explicit-KV reference.
Separate prefill/decode sessions also require an explicit state handoff design.
