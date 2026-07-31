# QNN EP Unit Tests

## Why this directory exists

Tests in `onnxruntime/test/providers/qnn/` have historically been integration tests — they require a QNN SDK runtime, physical hardware, and a fully compiled EP stack. This makes them expensive to run and impossible to execute in most developer and CI environments.

The `unit/` subdirectory introduces a separate testing tier: **function-level and component-level unit tests** that target the internal logic of the QNN EP. No on-device hardware is required — all tests run on a Linux x86-64 host. Tests that exercise op validation load `libQnnHtp.so` locally on the host (validation only, not graph execution); those tests are automatically skipped if the SDK is unavailable.

## What problem this solves

Because the QNN EP ships as a dynamically loaded plugin (`MODULE` library), its internal symbols are not normally accessible to external test binaries. The existing integration tests work around this by testing only through the public EP interface.

This unit test infrastructure solves the problem by introducing a **coverage build mode** (`ENABLE_COVERAGE=ON`) that:

1. Rebuilds the EP as a `SHARED` library so the test binary can link against it directly.
2. Exports all symbols via a permissive version script.
3. Defines `QNN_EP_INTERNAL_SYMBOL_ACCESS=1`, which activates the test code in this directory.

All test code in this directory is guarded by `#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS`, so it compiles to empty translation units in normal (non-coverage) builds and has no impact on production binaries.

## Current test suites

| File | Test suite | Source file covered |
|---|---|---|
| `qnn_def_test.cc` | `QnnUnit_DefTest` | `builder/qnn_def.cc` |
| `qnn_model_wrapper_test.cc` | `QnnUnit_ModelWrapperTest` | `builder/qnn_model_wrapper.cc` |
| `qnn_quant_params_wrapper_test.cc` | `QnnUnit_QuantParamsWrapperTest` | `builder/qnn_quant_params_wrapper.cc` |
| `qnn_model_test.cc` | `QnnUnit_ModelTest` | `builder/qnn_model.cc` |
| `qnn_utils_test.cc` | `QnnUnit_UtilsTest` | `builder/qnn_utils.cc` |
| `qnn_ep_utils_test.cc` | `QnnUnit_EpUtilsTest` | `qnn_ep_utils.cc` |
| `ort_api_test.cc` | `QnnUnit_OrtApiTest` | `ort_api.cc` |
| `qnn_backend_manager_test.cc` | `QnnUnit_BackendManagerTest` (stub, no real lib) / `QnnUnit_BackendManagerHtpTest` (loads a real backend, skips when unavailable) | `builder/qnn_backend_manager.cc` |

## Benefits

- **No on-device hardware required** — all tests run on a Linux x86-64 host. The QNN HTP SDK library (`libQnnHtp.so`) executes locally for op validation; no Qualcomm device is needed.
- **Fast feedback loop** — tests compile and run in seconds on any Linux x86-64 host.
- **Regression protection** — uncovered paths that later break are caught before integration.
- **Coverage-driven quality** — the infrastructure enables systematic identification and elimination of untested branches in core EP logic.

## Running the tests

```bash
# Full coverage build, test, and HTML report
python qcom/build_and_test.py coverage_linux_x86_64

# Run only the unit tests after a coverage build
cd build/linux-x86_64/RelWithDebInfo
./onnxruntime_provider_test --gtest_filter="QnnUnit_*"
```

## Adding new unit tests

### Policy

Review standards for new tests in this directory.

**File structure**
- Add to an existing `*_test.cc` or create a new file following the same pattern (one source file under test → one test file).
- Wrap everything except `#include "gtest/gtest.h"` in `#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS` ... `#endif`. (Keep the gtest include outside the guard so the file always parses; the body becomes an empty TU in non-coverage builds.) Files without this guard will be compiled in non-coverage CI and fail to link against EP-internal symbols.
- Test suite name: `QnnUnit_<Component>Test`. Test name: `<Function>_<Scenario>_<ExpectedResult>` (e.g., `ValidateQnnNode_HtpBackend_Relu_Succeeds`).

**Minimal example**

A complete file showing the required layout. Use this as a starting template:

```cpp
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "test/providers/qnn/unit/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(QnnUnit_ModelWrapperTest, GetQnnBackendType_ReturnsHTP) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings, qnn::QnnBackendType::HTP);
  EXPECT_EQ(wrapper->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
```

For richer patterns (real backend handle, OrtApi stubs, error paths, override maps), read the existing `qnn_def_test.cc` and `qnn_model_wrapper_test.cc` in this directory.

**Mock strategy**

Pick the lowest-cost layer that lets you write the test. Cost increases top to bottom.

| Dependency under test | Approach |
|---|---|
| Pure logic / utility — touches neither QNN nor ORT | Direct call, no fixture |
| Needs `OrtApi` but no real graph/logger object | Declare an `OrtApi stub{}` locally and stub only the function pointers your test path exercises |
| Needs `QnnModelWrapper`, no real graph/logger | Use `QnnModelWrapperTestContext` from `qnn_unit_test_utils.h` (bundles `OrtApi` stub + passes `nullptr` graph/logger). Relies on the wrapper's test-only ctor overload |
| Needs the QNN backend interface but no real SDK | Zero-init `QNN_INTERFACE_VER_TYPE` and override the function pointers your test path exercises (lowest cost, fully controllable) |
| Needs a real `Qnn_BackendHandle_t` (e.g., `backendValidateOpConfig`) | Use `QnnRealHtpBackendContext`: `dlopen` `libQnnHtp.so` + `backendCreate`. **Does not create a QNN context/session** — the validation path does not need one. Use `GTEST_SKIP()` when the SDK is unavailable |
| Needs a real QNN context/session, graph operations | **No helper today.** Please raise it — we need a fixture-shared session (avoid rebuilding per test) before adding such tests |
| Needs a real `OrtGraph` or `Ort::Logger` object | **Not currently possible** — public ORT headers are insufficient and private ORT headers are forbidden. Redesign the test to remove this dependency |

**Fake graph infrastructure (`qnn_fake_ort_graph.h`)**

ORT's graph types (`OrtGraph`, `OrtNode`, `OrtValueInfo`, etc.) are opaque C handles — the EP
never dereferences these pointers directly, only passes them to `OrtApi` function pointers.
This means tests can substitute lightweight POD structs (`FakeNode`, `FakeValueInfo`,
`FakeGraph`, `FakeOrtValue`) and install stub lambdas that cast back to the fake type:

```
Test:  FakeNode dq{"dq", "DequantizeLinear", "", 13, {&input}, {}};
       passes dq.AsNode() [= reinterpret_cast<const OrtNode*>(&dq)] to EP code
         ↓
EP:    api.Node_GetOperatorType(node, &out)   // EP never dereferences `node`
         ↓
Stub:  reinterpret_cast<const FakeNode*>(node)->op_type  →  "DequantizeLinear"
```

The pointer round-trips safely because only the original type (`FakeNode*`) is used to
read memory — the opaque handle is just a passthrough token.

| Type | Acts as | Key detail |
|------|---------|------------|
| `FakeValueInfo` | `OrtValueInfo*`, `OrtTypeInfo*`, `OrtTensorTypeAndShapeInfo*` | One struct serves three handle roles |
| `FakeOrtValue` | `OrtValue*`, `OrtTensorTypeAndShapeInfo*` | First 3 fields layout-compatible with `FakeValueInfo` (verified by `static_assert`) |
| `FakeNode` | `OrtNode*` | Holds input/output `FakeValueInfo*` vectors |
| `FakeGraph` | `OrtGraph*` | Owns nodes; observes inputs/outputs/initializers |

`InstallFakeGraphApiStubs(api)` installs all stubs at once. Override individual stubs
afterward for test-specific behaviour (e.g., `api.ValueInfo_GetValueProducer = ...`).

For tests that use `Ort::ConstNode` wrappers (which call `Ort::GetApi()` internally),
wrap with `OrtGlobalApiOverride` from `qnn_unit_test_utils.h` to redirect the global API.

**Constraints — common traps to avoid**
- **No private ORT headers.** Only the public C API (`onnxruntime_c_api.h`) and the EP's own headers under `core/providers/qnn/` are allowed. ORT's internal source tree (`core/graph/...`, `core/framework/...`) and ORT's internal test infrastructure (`test/util/include/...`) are off-limits.
- **No new cmake include paths** pointing into private ORT source — the linter rejects these.
- **`OrtApi` stub lambdas are `noexcept`.** Never call `assert`, `abort`, or `ORT_ENFORCE` inside — they terminate the process instead of failing the test. Document invariants in a comment.
- **Reference parameters need stable lvalues.** Some EP types store constructor args as const references. Passing a literal (e.g., `nullptr`, a temporary) creates an object that dies right after the call — the stored reference then dangles. Always pass a named local or member variable, even when the intended value is `nullptr`.
- **Stub only what your test path exercises.** Pre-stubbing entire APIs (every `OrtApi` / QNN function pointer, default return values everywhere) hides which call site actually fired and makes failures harder to diagnose. If an unstubbed pointer is hit at runtime, treat it as a signal and add the stub case-by-case.

**Coverage target**
- Aim for **≥90% line coverage and 100% function coverage** on the source file under test.
- We understand that certain scenarios (e.g., platform-dependent logic, error-handling paths that need real failures from external systems) may be hard to fully cover. In such cases, a reasonable explanation in the PR description is sufficient.
- **Default verification**: read the overall coverage report produced by `coverage_linux_x86_64` (`build/linux-x86_64/RelWithDebInfo/coverage/index.html`). If your new test raises the target file's line / function numbers, you're done.
- **Unit-only verification (optional, for coverage-gap-filling work)**: if your task is to prove the unit tier alone covers the target — without integration tests inflating the numbers — re-run with the unit-only filter:

  ```bash
  bash qcom/scripts/linux/generate_coverage.sh \
    --build-dir=build/linux-x86_64 \
    --output-dir=build/linux-x86_64/RelWithDebInfo/coverage_unit_only \
    --test-filter='*QnnUnit_*'
  ```

**Verification checklist before review**
- [ ] `./onnxruntime_provider_test --gtest_filter="QnnUnit_*"` — all green
- [ ] Overall coverage (`coverage_linux_x86_64`) shows the target file's numbers improving
- [ ] Release build (`build_ort_linux_x86_64`) still passes — confirms test-only code is correctly guarded
- [ ] `python qcom/build_and_test.py lint` clean

### Agent prompt template

Copy-paste the block below when asking Claude Code (or another agent) to add a unit test. The template is self-contained — the agent does not need additional context beyond the source file path you provide.

````
You are adding a function-level unit test to onnxruntime/test/providers/qnn/unit/.

Target file: <path/to/source.cc>
New function(s) / behaviour to cover: <describe what the UT must exercise>

Policy (must follow):
1. Add the test to onnxruntime/test/providers/qnn/unit/<source_basename>_test.cc.
   Create the file if it does not exist; wrap the entire body in:
     #if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
     ...
     #endif
2. Test suite: QnnUnit_<Component>Test. Test name: <Func>_<Scenario>_<Expected>.
3. Pick the lowest-cost mocking layer that works:
   - Pure logic: direct call, no fixture.
   - Needs OrtApi: local OrtApi stub{} + stub only the function pointers used.
   - Needs QnnModelWrapper: QnnModelWrapperTestContext from qnn_unit_test_utils.h.
   - Needs QNN backend interface only: zero-init QNN_INTERFACE_VER_TYPE + override
     the function pointers used.
   - Needs a real Qnn_BackendHandle_t: QnnRealHtpBackendContext + GTEST_SKIP if !IsValid().
     (This only creates a backend handle, NOT a QNN context/session.)
   - Needs a real QNN context/session OR real OrtGraph/Ort::Logger object: STOP and
     report — there is no helper for these today.
4. Forbidden:
   - Including any private ORT header (only public C API + core/providers/qnn/* allowed).
   - Adding new cmake include paths.
   - assert / abort / ORT_ENFORCE inside noexcept OrtApi stub lambdas.
   - Passing a literal (e.g., nullptr, a temporary) to a constructor parameter taken by
     const reference; use a named local or member variable as a stable lvalue.
5. Stub only what your test path exercises. Do not pre-stub entire APIs (every OrtApi /
   QNN function pointer, default returns everywhere) — it hides which call site fired and
   makes failures harder to diagnose.

Verification (run before reporting done):
- python qcom/build_and_test.py coverage_linux_x86_64
- ./build/linux-x86_64/RelWithDebInfo/onnxruntime_provider_test --gtest_filter="QnnUnit_*"  (must be green)
- python qcom/build_and_test.py lint  (must be clean)

Report back: number of tests added, whether all tests pass, any limitations encountered.
````

> **Note for coverage-gap-filling tasks** — if your goal is explicitly to raise a target file's coverage (rather than testing newly added code), additionally run the unit-only filter described in **Coverage target** above, and include the unit-only line / function coverage of the target file (before vs. after) in your report. The remaining policy is identical.

## Future direction

The infrastructure is designed to grow in two directions:

1. **Coverage gap filling** — for core components not yet covered at the unit tier (e.g., `qnn_backend_manager.cc`, `qnn_execution_provider.cc`), add targeted unit tests to cover paths that are difficult to reach through integration tests: error paths, edge cases, and internal branch logic. (`qnn_def.cc` and `qnn_model_wrapper.cc` already meet the ≥90% line / 100% function target via the test suites listed above; small follow-up patches to fill remaining branches are still welcome.)

2. **Op builder test migration** — op builders (`opbuilder/*.cc`) are currently covered by on-device integration tests, which are expensive to run and structurally limited in reaching component-level logic. The goal is to migrate these tests into this tier, using QNN HTP SDK on the Linux host for op validation — no device required. Coverage improvement is a natural outcome of this migration, but the primary driver is lower test cost and better component-level precision.

Coverage builds run in CI on every PR. Strict regression gating (failing PRs that drop coverage) is being rolled out in stages.
