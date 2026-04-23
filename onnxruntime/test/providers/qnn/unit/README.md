# QNN EP Unit Tests

## Why this directory exists

Tests in `onnxruntime/test/providers/qnn/` have historically been integration tests — they require a QNN SDK runtime, physical hardware or an emulator, and a fully compiled EP stack. This makes them expensive to run and impossible to execute in most developer and CI environments.

The `unit/` subdirectory introduces a separate testing tier: **function-level and component-level unit tests** that target the internal logic of the QNN EP. No on-device hardware is required — all tests run on a Linux x86-64 host. Tests that exercise op validation load QNN SDK libraries (e.g. `libQnnCpu.so`, `libQnnHtp.so`) locally on the host; those tests are automatically skipped if the SDK is unavailable.

## What problem this solves

Because the QNN EP ships as a dynamically loaded plugin (`MODULE` library), its internal symbols are not normally accessible to external test binaries. The existing integration tests work around this by testing only through the public EP interface.

This unit test infrastructure solves the problem by introducing a **coverage build mode** (`ENABLE_COVERAGE=ON`) that:

1. Rebuilds the EP as a `SHARED` library so the test binary can link against it directly.
2. Exports all symbols via a permissive version script.
3. Defines `QNN_EP_FUNCTION_LEVEL_UT=1`, which activates the test code in this directory.

All test code in this directory is guarded by `#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_FUNCTION_LEVEL_UT`, so it compiles to empty translation units in normal (non-coverage) builds and has no impact on production binaries.

## Current test suites

| File | Test suite | Source file covered |
|---|---|---|
| `qnn_def_test.cc` | `QnnUnit_DefTest` | `builder/qnn_def.cc` |
| `qnn_model_wrapper_test.cc` | `QnnUnit_ModelWrapperTest` | `builder/qnn_model_wrapper.cc` |

## Future direction

The infrastructure is designed to grow in two directions:

1. **Coverage gap filling** — for core components such as `qnn_def.cc`, `qnn_model_wrapper.cc`, `qnn_backend_manager.cc`, and `qnn_execution_provider.cc`, add targeted unit tests to cover paths that are difficult to reach through integration tests: error paths, edge cases, and internal branch logic.

2. **Op builder test migration** — op builders (`opbuilder/*.cc`) are currently covered by on-device integration tests, which are expensive to run and structurally limited in reaching component-level logic. The goal is to migrate these tests into this tier, using QNN CPU/HTP SDK on the Linux host for op validation — no device required. Coverage improvement is a natural outcome of this migration, but the primary driver is lower test cost and better component-level precision.

Coverage builds are intended to run in CI as a regression gate.

## Benefits

- **No on-device hardware required** — all tests run on a Linux x86-64 host. QNN SDK libraries (CPU/HTP) execute locally for op validation; no Qualcomm device is needed.
- **Fast feedback loop** — tests compile and run in seconds on any Linux x86-64 host.
- **Regression protection** — uncovered paths that later break are caught before integration.
- **Coverage-driven quality** — the infrastructure enables systematic identification and elimination of untested branches in core EP logic.

## Running the tests

```bash
# Full coverage build, test, and HTML report
python qcom/build_and_test.py --target-py-version 3.12 coverage_linux_x86_64

# Run only the unit tests after a coverage build
cd build/linux-x86_64/RelWithDebInfo
./onnxruntime_provider_test --gtest_filter="QnnUnit_*"
```

## Adding new unit tests

1. Add `TEST` or `TEST_F` cases to an existing `*_test.cc` file in this directory, or create a new one following the same pattern.
2. New test files must be wrapped in `#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_FUNCTION_LEVEL_UT` ... `#endif`.
3. Use the helpers in `qnn_unit_test_utils.h` for QNN SDK stubs and ORT API mock objects.
4. Verify with `--gtest_filter="QnnUnit_*"` after a coverage build.
