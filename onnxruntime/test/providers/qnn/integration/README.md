# QNN EP Pipeline Integration Tests

## Why this directory exists

Tests in `onnxruntime/test/providers/qnn/` have historically been op-level accuracy
tests — they run a full ONNX model through the QNN EP and compare inference results
against the CPU EP. While valuable, they are coarse-grained: they exercise entire
operator pipelines rather than targeting specific internal code paths.

The `integration/` subdirectory introduces a separate tier: **targeted pipeline
integration tests** that exercise specific internal EP code paths (e.g. `ort_api.cc`,
`qnn_model_wrapper.cc`) by building minimal models inline via the ORT model editor API
and running them through a real backend. The tests are Linux-only (guarded by
`defined(__linux__)`); on the Linux CI host the QNN SDK is always present, so a missing
backend is treated as a configuration error (hard failure) rather than a skip.

## How this differs from the other test tiers

| Tier | Directory | Backend required | What it targets |
|---|---|---|---|
| Component | `component/` | Optional (`libQnnHtp.so` for op validation only, no session) | Pure function / op-builder logic; no graph execution |
| Op-builder snapshot | `snapshot/` | Real backend, no finalize | QNN JSON graph structure produced by the op builder |
| Session snapshot | `session_snapshot/` | Real backend + full session (compile only) | QNN JSON graph structure after ORT optimizer + partition transforms |
| Accuracy | `accuracy/` | Real backend + full session (compile + execute) | Per-op inference accuracy, spec-shared with the `snapshot/` tier — the **new home** for accuracy tests |
| **Pipeline integration** | **`integration/`** | Real backend + full session (compile + execute) | Specific EP internal code paths triggered during EP compilation |
| Op-level accuracy | `qnn/` (root) | Real backend + full session | Same accuracy check as the `accuracy/` tier, but **legacy** and not spec-shared — being migrated into `accuracy/` (see below) |

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
| Pure function / op-builder logic, no session (white-box) | `component/` |
| QNN graph structure (JSON golden), op-builder or post-session | `snapshot/` or `session_snapshot/` |

When in doubt: if your test needs a real backend but is not primarily an accuracy
check, it belongs in `integration/`.

## Platform availability

These tests are **Linux-only** — the file guard `defined(__linux__)` compiles them out
everywhere else, so the Windows row below describes *why they are excluded*, not a
runtime skip.

| Platform | Behavior |
|---|---|
| Linux x86-64 | HTP simulator (`libQnnHtp.so`) runs fully — compile + execute. **Primary CI platform.** |
| Linux AArch64 | Real HTP hardware — compile + execute. |
| Windows (any) | File compiled out by the `defined(__linux__)` guard — these tests are not built or run. |

On the Linux CI host the QNN SDK is always present, so the fixture treats a missing EP
plugin or HTP device as a **hard failure** (CI configuration error), not a skip. The
only automatic skips are a minimal build with no `OrtModelEditorApi`, and a per-test
failure of the QNN EP to compile its inline model — no platform-specific `#ifdef`
guards are needed inside test bodies.

## Adding new tests

### Shared utilities

Common helpers live in `qnn_test_utils.h` — include it from any integration test file
to reuse them instead of re-defining locally:

| Symbol | Purpose |
|---|---|
| `RegisteredQnnEp` | RAII helper that registers / unregisters the QNN EP plugin. |
| `MakeQnnHtpSessionOptions(ep, opts)` | Build `Ort::SessionOptions` targeting the QNN HTP backend; returns `false` when the device is unavailable. The fixture treats that as a hard failure (CI config error) on the Linux host. |
| `MakeValueInfo1D / 2D / 3D / 4D` | Build `Ort::ValueInfo` for a tensor of the given rank, element type, and dimensions. |

### File structure

Add to an existing `*_test.cc` or create a new file following the same pattern. The
required guard is `#if !defined(ORT_MINIMAL_BUILD) && defined(__linux__)` — unlike `component/`,
these files do **not** need `QNN_EP_INTERNAL_SYMBOL_ACCESS` because they only use the
public ORT C++ API, not EP-internal symbols. The `defined(__linux__)` half excludes
all non-Linux platforms (e.g. Windows), where the HTP backend cannot execute graphs.

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
  // The fixture (QnnInteg_OrtApiTest) already sets up the HTP backend; a missing
  // SDK on the Linux host is a hard failure (CI config error). Have the test body
  // GTEST_SKIP() only if the QNN EP cannot compile the model.
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

# Run all QNN tests (component + snapshot + accuracy + integration + op-level)
./onnxruntime_provider_test --gtest_filter="Qnn*"
```

These tests are Linux-only and are built only on Linux. On the Linux CI host a missing
QNN SDK is a hard failure (CI config error); individual tests `GTEST_SKIP()` only when
the QNN EP fails to compile their inline model.

## Current test suites

| File | Suite | EP internal code path covered |
|---|---|---|
| `ort_api_test.cc` | `QnnInteg_OrtApiTest` | `ort_api.cc`: `GetQDQIODefs` + QDQ `OrtNodeUnit` ctor, standalone DequantizeLinear / QuantizeLinear branches, and `OrtNodeAttrHelper` found-paths (int64/int64s/float/int32/int32-vec across several opsets) |
