# AOT Dynamic-Batch Declaration Experiment

This experiment separates ORT's dynamic I/O declaration from the concrete base shape in a QNN context. It targets one batch-preserving partition fully assigned to HTP. It does not enable batch multiplier execution.

`QnnModel::SetGraphInputOutputInfo()` resolves an unknown I/O dimension 0 from the matching context tensor, using persisted name overrides when present. Other unknown dimensions and JIT dynamic shapes remain unsupported. ORT graph metadata and shape validation are unchanged; runtime buffers and outputs still use the compiled base size.

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

3. Load and run both artifacts with the same HTP backend, the experimental EP, and actual input batch `N=1`. Compare outputs and repeat Run on the same session. Do not apply a free-dimension override when loading the dynamic artifact.
4. Feed `N=2`: the static artifact should fail ORT shape validation; the dynamic artifact should reach the EP's existing buffer-size rejection. This distinguishes the two validation layers without claiming multiplier support.
5. Feed an incorrect fixed dimension: ORT must still reject it.

Use `onnxruntime_perf_test --plugin_ep_libs "QNN|<ep-library>" --plugin_eps QNN --plugin_ep_options "backend_path|<htp-library> htp_arch|<arch> soc_model|<soc>" -C "session.disable_cpu_ep_fallback|1"`. Add `--compile_ep_context --compile_only --compile_model_path compiled_b1.onnx` for compilation, `-n` for load-only checks, or `-I -m times -r 1` for execution with generated base-batch input.

## Validation

Verified on Linux x86_64 with ORT 1.30.0 and the HTP host backend. The probe was an FP16 Conv with input `[N,3,8,8]` and output `[N,2,6,6]`, fully offloaded as one EPContext. Lint and five component tests passed.

| Case | Observed result |
| --- | --- |
| Static and dynamic artifacts, `N=B=1` | Both matched the reference output and passed repeated inference. |
| Static artifact, `N=2` | ORT rejected batch dimension 0. |
| Dynamic artifact, `N=2/128` | Reached EP buffer-size validation and was rejected there. |
| Dynamic artifact, incorrect fixed spatial dimension | ORT rejected the fixed dimension. |

The component tests cover independent input/output base dimensions, NCHW/NHWC metadata separation, persisted tensor-name overrides, unchanged ORT declarations, and rejection of missing contexts/tensors, rank mismatch, zero base batch, and other unknown dimensions.

Build component tests with `onnxruntime_QNN_ENABLE_INTERNAL_TEST_SYMBOLS=ON`, then run `onnxruntime_provider_test --gtest_filter='QnnUnit_ModelTest.SetGraphInputOutputInfo*:QnnUnit_ModelTest.SetupTensors*'`.

Limit: dimension 0 must be the batch axis in both interfaces. Explicit axis mapping and actual runtime tensor dimensions, output allocation, and memory binding are needed for general batch multiplier support. Host compilation and session creation do not prove HTP device execution.
