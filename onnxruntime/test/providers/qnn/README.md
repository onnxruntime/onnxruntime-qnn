# ONNX Runtime QNN Execution Provider Tests
## Overview
1. The `onnxruntime/test/providers/qnn` directory contains integration tests for the Qualcomm Neural Network (QNN) execution provider.
2. Most testcases run an ONNX model through the QNN-EP, then verifies the inference result against the one on CPU-EP

## Directory structure

Tests are organized **tier-first**: each testing tier is a top-level sibling
directory, with shared infrastructure factored out alongside them.

```
test/providers/qnn/
├── *.cc                # Op-level accuracy tests (QnnHTPBackendTests, QnnCPUBackendTests, …)
│                       #   — legacy integration tier; being migrated into the tiers below
├── test_infra/         # Cross-tier test infrastructure (mocks, stub backends, golden utils,
│                       #   shared op specs). Never depends on any tier directory.
├── component/          # Function/component-level white-box tests — coverage build only;
│                       #   some tests use libQnnHtp.so for op validation (no session)
├── snapshot/           # Op-builder snapshot tests — diff the QNN JSON graph produced by
│                       #   the op builder against a stored golden (real backend, no finalize)
├── session_snapshot/   # Session-level snapshot tests — diff the QNN JSON graph after the
│                       #   full ORT session (optimizer + partition transforms) against a golden
├── accuracy/           # Per-op accuracy tests — route a model through a real session and
│                       #   compare inference results; share op specs with the snapshot tier
└── integration/        # Targeted pipeline integration tests — require a real backend and
                        #   a full session (GetCapability + Compile + Execute); exercise
                        #   specific EP internal code paths rather than op accuracy
```

**During the transition period**, new tests should follow this rule:

| Test type | Where to add |
|---|---|
| Pure function / op-builder logic, no session (white-box) | `component/` |
| Op-builder → QNN graph structure (JSON golden) | `snapshot/` |
| Post-session QNN graph structure (JSON golden) | `session_snapshot/` |
| Op-level correctness / inference accuracy vs CPU EP | `accuracy/` for a spec-shared per-op case, or the `qnn/` root |
| Targets a specific EP internal code path with a minimal inline model | `integration/` |

The long-term plan is to migrate op-level tests from the `qnn/` root into the
tiers above. (Aspirational; no fixed timeline.) Until then, both coexist.

## Building the Tests
The tests are built as part of the regular ONNX Runtime build. After a successful build you will have an executable named
- onnxruntime_provider_test.exe   (Windows)
- onnxruntime_provider_test      (Linux/macOS)

## Running the Tests
1. QNN supports several backends. You can use the standard Google‑Test syntax for filtering:
    - `onnxruntime_provider_test.exe --gtest_filter=QnnCPUBackendTests.*`
    - `onnxruntime_provider_test.exe --gtest_filter=QnnHTPBackendTests.*`
    - `onnxruntime_provider_test.exe --gtest_filter=QnnGPUBackendTests.*`
    - `onnxruntime_provider_test.exe --gtest_filter=QnnIRBackendTests.*`
2. Saving Test Artifacts
    - For debugging it is often helpful to keep the intermediate files that the tests generate. The following environment
    variables are recognized by the test binary:
        - `QNN_DUMP_ONNX`: Saves the input ONNX model used for the test
        - `QNN_DUMP_JSON`: Save json qnn graph with provider_option `dump_json_qnn_graph`
        - `QNN_DUMP_DLC`: Saves the compiled QNN DLC file by specifying the provider_option `backend_path` to `QnnIr.dll`
    - The artifacts will be saved to a directory named with `<TestSuite>_<TestName>`
        ```
        .
        ├── QnnCPUBackendTests_BatchNorm2D_fp32         # RunQnnModelTest
        │   ├── dumped_f32_model.onnx                   # float32 ONNX model
        │   ├── QNNExecutionProvider_QNN_XXXX_X_X.dlc
        │   └── QNNExecutionProvider_QNN_XXXX_X_X.json
        ├── QnnHTPBackendTests_BatchNorm_FP16           # TestFp16ModelAccuracy
        │   ├── dumped_f16_model.onnx                   # float16 ONNX model
        │   ├── dumped_f32_model.onnx                   # float32 ONNX model
        │   ├── QNNExecutionProvider_QNN_XXXX_X_X.dlc
        │   └── QNNExecutionProvider_QNN_XXXX_X_X.json
        └── QnnHTPBackendTests_BatchNorm2D_U8U8S32      # TestQDQModelAccuracy
            ├── dumped_f32_model.onnx                   # float32 ONNX model
            ├── dumped_qdq_model.onnx                   # QDQ ONNX model
            ├── QNNExecutionProvider_QNN_XXXX_X_X.dlc
            └── QNNExecutionProvider_QNN_XXXX_X_X.json

        # All artifact files are placed under the current working directory from which the test binary is invoked.
        ```
3. Verbose
    - `QNN_VERBOSE`: Sets the ONNX Runtime log level to `ORT_LOGGING_LEVEL_VERBOSE`

4. You can enable any combination of these environment variables, for example:
    - On Linux/macOS
        ```bash
        export QNN_DUMP_ONNX=1
        export QNN_DUMP_JSON=1
        export QNN_DUMP_DLC=1
        export QNN_VERBOSE=1
        ```
    - On Windows
        ```cmd
        set QNN_DUMP_ONNX=1
        set QNN_DUMP_JSON=1
        set QNN_DUMP_DLC=1
        set QNN_VERBOSE=1
        ```
        ```ps1
        $Env:QNN_DUMP_ONNX = "1"
        $Env:QNN_DUMP_JSON = "1"
        $Env:QNN_DUMP_DLC = "1"
        $Env:QNN_VERBOSE = "1"
        ```

# Note
- An issue on QNN backends can prevent the test artifacts from being successfully saved.
- The `onnxruntime_provider_test.exe` does not automatically delete the artifact directories, so you may want to prune them after a debugging session.
