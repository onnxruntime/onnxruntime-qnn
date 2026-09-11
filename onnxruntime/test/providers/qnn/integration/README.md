# QNN EP Pipeline Integration Tests

## Why this directory exists

Tests in `onnxruntime/test/providers/qnn/` have historically been op-level accuracy
tests — they run a full ONNX model through the QNN EP and compare inference results
against the CPU EP. While valuable, they are coarse-grained: they exercise entire
operator pipelines rather than targeting specific internal code paths.

The `integration/` subdirectory introduces a separate tier: **targeted pipeline
integration tests** that exercise specific internal EP code paths (e.g. `ort_api.cc`,
`qnn_model_wrapper.cc`) by building minimal models inline via the ORT model editor API
and running them through a real backend. Tests skip automatically when the required
backend is unavailable.

## How this differs from the other test tiers

| Tier | Directory | Backend required | What it targets |
|---|---|---|---|
| Unit | `unit/` | Optional (`libQnnHtp.so` for op validation only, no session) | Pure function logic; no graph execution |
| **Pipeline integration** | **`integration/`** | Real backend + full session (compile + execute) | Specific EP internal code paths triggered during EP compilation |
| Op-level accuracy | `qnn/` (root) | Real backend + full session | Op correctness, inference accuracy vs CPU EP |

## Migration plan

The long-term goal is for **all** pipeline integration tests to live here. The existing
op-level tests in `qnn/` (root) will be migrated gradually as bandwidth allows.

During the transition period, both locations contain tests. This is intentional and
expected.

## Where to add new tests

**During the transition period, use this rule:**

| Test type | Where to add |
|---|---|
| Targets a specific EP internal code path (e.g. a function in `ort_api.cc`, `qnn_model_wrapper.cc`) — builds a minimal model to trigger the path | **`integration/`** ← here |
| Tests op-level correctness or inference accuracy (compares QNN EP output vs CPU EP) | `qnn/` (root) until that migration is complete |
| Pure function logic or op validation only (no session/inference) | `unit/` |

When in doubt: the dividing line is the **ORT session**, not whether a real
backend is loaded. If your test creates a full ORT session (compile + execute a
model through the EP), it belongs in `integration/`. A test that constructs EP
components directly and calls their methods — even if that loads a real
`libQnnHtp.so` — is a component test and belongs in `unit/` (see the `*HtpTest`
fixtures there).

## Platform availability

| Platform | Behavior |
|---|---|
| Linux x86-64 | HTP simulator (`libQnnHtp.so`) runs fully — compile + execute. **Primary CI platform.** |
| Linux AArch64 | Real HTP hardware — compile + execute. |
| Windows x86-64 | HTP cannot execute graphs. Tests using `MakeQnnHtpSessionOptions` return `false` → `GTEST_SKIP()`. |
| Windows ARM64 | Real HTP hardware — compile + execute. |

Tests skip automatically via `GTEST_SKIP()` when the required backend is unavailable —
no platform-specific `#ifdef` guards are needed inside test bodies.

## Adding new tests

### Shared utilities

Common helpers live in `qnn_test_utils.h` — include it from any integration test file
to reuse them instead of re-defining locally:

| Symbol | Purpose |
|---|---|
| `RegisteredQnnEp` | RAII helper that registers / unregisters the QNN EP plugin. |
| `MakeQnnHtpSessionOptions(ep, opts)` | Build `Ort::SessionOptions` targeting the QNN HTP backend; returns `false` when the device is unavailable so the fixture can `GTEST_SKIP()`. |
| `MakeValueInfo1D / 2D / 3D / 4D` | Build `Ort::ValueInfo` for a tensor of the given rank, element type, and dimensions. |

### File structure

Add to an existing `*_test.cc` or create a new file following the same pattern. The
required guard is `#if !defined(ORT_MINIMAL_BUILD) && defined(__linux__)` — unlike `unit/`,
these files do **not** need `QNN_EP_INTERNAL_SYMBOL_ACCESS` because they only use the
public ORT C++ API, not EP-internal symbols. The `defined(__linux__)` half excludes
Windows x86-64, where the HTP backend cannot execute graphs.

Minimal template:

```cpp
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD) && defined(__linux__)

#include <cstdint>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "onnxruntime_cxx_api.h"

#include "test/providers/qnn/integration/qnn_test_utils.h"

namespace onnxruntime {
namespace test {

TEST_F(QnnInteg_OrtApiTest, MyFunction_MyScenario_ExpectedResult) {
  // Build a minimal inline model, create a session, run it.
  // The fixture (QnnInteg_OrtApiTest) already sets up the HTP backend and
  // calls GTEST_SKIP() if it is unavailable.
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && defined(__linux__)
```

### Verification checklist before review

- [ ] `./onnxruntime_provider_test --gtest_filter="QnnInteg_*"` — all green
- [ ] `python qcom/build_and_test.py lint` — clean

## Test suite naming

Tests in this directory use the `QnnInteg_` prefix:

```
QnnInteg_<Component>Test.<Function>_<Scenario>_<ExpectedResult>
```

Example: `QnnInteg_OrtApiTest.QDQGroup_CoversGetQDQIODefs`

## Running the tests

```bash
# Run only pipeline integration tests
./onnxruntime_provider_test --gtest_filter="QnnInteg_*"

# Run all QNN tests (unit + integration + op-level)
./onnxruntime_provider_test --gtest_filter="Qnn*"
```

Tests that require a backend that is unavailable on the current platform are
automatically skipped via `GTEST_SKIP()` — no manual filtering needed.

## Current test suites

| File | Suite | EP internal code path covered |
|---|---|---|
| `ort_api_test.cc` | `QnnInteg_OrtApiTest` | `ort_api.cc`: `GetQDQIODefs`, `OrtNodeUnit` QDQ ctor, `OrtNodeAttrHelper` found-paths |
