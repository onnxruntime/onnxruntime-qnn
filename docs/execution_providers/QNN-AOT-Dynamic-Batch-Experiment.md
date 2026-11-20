# AOT Batch Multiplier Experiment

This experiment reuses a static QNN context compiled at base batch `B` for runtime batch `N=M*B`. It targets one batch-preserving partition fully assigned to HTP, with symbolic ONNX I/O and unchanged ORT shape validation.

`SetGraphInputOutputInfo()` resolves unknown axis 0 from the matching context tensor and remembers which I/O declarations were dynamic, including persisted name overrides. `BindAndExecuteGraph()` validates positive integral and consistent multipliers, rank, type, fixed dimensions, and checked sizes. Each Run uses independent QNN dimension arrays, actual buffer sizes, and output batch dimensions scaled from each output's own base shape. Fixed I/O and compiled descriptors remain unchanged. The backend must report `QNN_PROPERTY_GRAPH_SUPPORT_BATCH_MULTIPLE` support.

## Reproduce

1. Compile a static model at base batch `B=1` with CPU fallback disabled. Keep the resulting context binary.
2. Change only the compiled ONNX I/O batch declarations:

```python
import onnx

model = onnx.load("compiled_b1.onnx")
assert len(model.graph.node) == 1
assert model.graph.node[0].op_type == "EPContext"
for value in list(model.graph.input) + list(model.graph.output):
    batch = value.type.tensor_type.shape.dim[0]
    assert batch.dim_value == 1
    batch.dim_param = "N"  # Use a shared symbol only for batch-preserving I/O.
onnx.save(model, "compiled_dynamic.onnx")
```

3. Load with the experimental EP and HTP backend. Do not apply a free-dimension override or a multiplier config when loading the dynamic artifact.
4. Run `N=1,2,128,1` on one session and compare every output batch with reference data. The static artifact must still reject `N=2`.
5. Repeat with `B=2`; accept `N=2/4/128`, reject `N=3`, and verify that incorrect fixed dimensions still fail ORT validation.

Use `onnxruntime_perf_test --plugin_ep_libs "QNN|<ep-library>" --plugin_eps QNN --plugin_ep_options "backend_path|<htp-library> htp_arch|<arch> soc_model|<soc>" -C "session.disable_cpu_ep_fallback|1"`. Add `--compile_ep_context --compile_only --compile_model_path compiled_b1.onnx` for compilation, or `-I -m times -r 8 --data_shape "input:[1,3,8,8][2,3,8,8][128,3,8,8][1,3,8,8]"` for the probe's batch-switching run. Use `onnxruntime_plugin_ep_onnx_test` with reference tensors for accuracy.

## Validation

Runtime validation used Linux x86_64, ORT 1.30.0, QAIRT 2.52.0, and the HTP host backend. The FP16 Conv probe had input `[N,3,8,8]` and output `[N,2,6,6]`, fully offloaded as one EPContext, with CPU fallback disabled.

| Case | Observed result |
| --- | --- |
| `B=1`, runtime `N=1/2/128/1` | Passed eight requests on one session. |
| `B=2`, runtime `N=2/4/128/2` | Passed eight requests on one session. |
| Reference outputs, both base contexts | All six data sets passed, repeated twice with `1e-5` tolerances and varying values across batches. |
| Static `N=2`; dynamic incorrect fixed dimensions | ORT rejected the mismatches. |
| `B=2`, runtime `N=3` | EP rejected the nonintegral multiplier. |
| Context binaries | SHA-256 unchanged after postprocessing and execution. |

Lint and 26 tests passed: nine model component tests and 17 CPU Conv regression tests. Coverage includes independent input/output base dimensions, NCHW/NHWC separation, name overrides, unchanged declarations/descriptors, actual buffer binding, varying Runs, and invalid ranks, types, fixed dimensions, ratios, and byte sizes.

Build component tests with `onnxruntime_QNN_ENABLE_INTERNAL_TEST_SYMBOLS=ON`, then run `onnxruntime_provider_test --gtest_filter='QnnUnit_ModelTest.SetGraphInputOutputInfo*:QnnUnit_ModelTest.SetupTensors*:QnnUnit_ModelTest.PrepareRuntimeTensor*:QnnUnit_ModelTest.BindTensor*'`.

Scope: dimension 0 must be the batch axis in both interfaces; scaling output axis 0 must match the model/backend contract. JIT dynamic compilation, other unknown axes, automatic dynamic export, and general partition/axis mapping remain outside this experiment. The runtime extension has not yet been rebuilt or validated on ARM64 hardware; earlier device validation covered only the loader and base-batch inference.
