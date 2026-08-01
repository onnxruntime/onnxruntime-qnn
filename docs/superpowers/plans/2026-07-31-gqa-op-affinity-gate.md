# GQA op_affinity Config Gate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `op_affinity` QNN EP provider option pointing at a JSON config file that gates `GroupQueryAttention` `IsOpSupported` per backend (HTP opt-in, GPU opt-out, CPU silent-fallback, other-accelerator/heterogeneous → session fails).

**Architecture:** A new EP-level `OpAffinityMap` class parses the JSON config and encapsulates the whole decision truth table in one `Evaluate()` method. The parsed map is owned by the EP and reaches the GQA op builder through the existing `ModelSettings` → `QnnModelWrapper` → `GetModelSettings()` channel (Approach A — same path `htp_bf16_enable` uses). The gate sits at the top of GQA `IsOpSupported`, before PR #556's `IsGpuBackend || IsNpuBackend` guard.

**Tech Stack:** C++17, ONNX Runtime QNN EP, nlohmann/json (already a dependency), GoogleTest.

**Base branch:** `origin/dev/chunghow/gqa-htp-github` (PR #556). Create feature branch `dev/chuteng/add-gqa-op-affinity-gate` from it. Do NOT modify #556's branch.

**Spec:** `docs/superpowers/specs/2026-07-31-gqa-op-affinity-gate-design.md` (authoritative truth table in §3).

---

## Codebase Facts (verified — no guesswork needed)

- `QnnBackendType` enum (`onnxruntime/core/providers/qnn/builder/qnn_def.h:116`): `CPU=0, GPU, DSP, HTP, HTP_FP16, SERIALIZER`.
- `std::string QnnBackendTypeToString(QnnBackendType)` (`qnn_def.h:147`, impl `qnn_def.cc:671`) → lowercase: `"cpu"`,`"gpu"`,`"dsp"`,`"htp"`,`"htp_fp16"`,`"ir"`.
- `bool IsNpuBackend(QnnBackendType)` and `bool IsGpuBackend(QnnBackendType)` declared `qnn_def.h:135,137`. `IsNpuBackend` is true for HTP and HTP_FP16.
- String options read via `GetSessionConfigEntryOrDefault(ort_api, session_options_, FormatEPConfigKey("<key>"), "<default>", out_string)` — see `qnn_execution_provider.cc:532`.
- `model_settings_` (EP member of type `qnn::ModelSettings`) populated in the EP constructor (`qnn_execution_provider.cc:919`); passed by value into `QnnModelWrapper` at `qnn_execution_provider.cc:1375`.
- `ModelSettings` struct: `onnxruntime/core/providers/qnn/builder/qnn_model_wrapper.h:37`.
- Op builders read settings via `qnn_model_wrapper.GetModelSettings()` (e.g. `lp_pool_op_builder.cc:87`).
- CMake **auto-globs** `core/providers/qnn/*.cc` (`cmake/onnxruntime_providers_qnn.cmake:6-9`) and test `providers/qnn/unit/*` (`cmake/onnxruntime_unittests.cmake:202`). **No CMake edits needed** for new files.
- `nlohmann/json.hpp` is includable (`#include "nlohmann/json.hpp"`); `qnn_model_wrapper.h` already includes it.
- Unit tests use guard `#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS` and live in `onnxruntime/test/providers/qnn/unit/`.
- **There is no `TrimWhitespace` helper** in the codebase — do not reference one. Inline-trim where needed.
- GQA op builder: `onnxruntime/core/providers/qnn/builder/opbuilder/group_query_attention_op_builder.cc`. On PR #556, `IsOpSupported` starts (after `ORT_UNUSED_PARAMETER(logger)`) with:
  ```cpp
  auto backend_type = qnn_model_wrapper.GetQnnBackendType();
  RETURN_IF_NOT(IsGpuBackend(backend_type) || IsNpuBackend(backend_type),
                "GroupQueryAttention is only supported with the GPU backend and HTP backend");
  ```

---

## File Structure

| File | Responsibility |
|------|----------------|
| `onnxruntime/core/providers/qnn/qnn_op_affinity_map.h` | `OpAffinityMap` class declaration: `FromConfigFile`, `IsConfigured`, `Evaluate`, `Decision` enum. |
| `onnxruntime/core/providers/qnn/qnn_op_affinity_map.cc` | JSON parse (throw-on-error) + `Evaluate` truth-table logic + backend-string normalization. |
| `onnxruntime/core/providers/qnn/builder/qnn_model_wrapper.h` | `ModelSettings` gains `const OpAffinityMap* op_affinity = nullptr;` + forward declaration. |
| `onnxruntime/core/providers/qnn/qnn_execution_provider.h` | EP member `qnn::OpAffinityMap op_affinity_map_;`. |
| `onnxruntime/core/providers/qnn/qnn_execution_provider.cc` | Read `op_affinity` option, build map (uncaught throw → session fails), set `model_settings_.op_affinity`. |
| `onnxruntime/core/providers/qnn/builder/opbuilder/group_query_attention_op_builder.cc` | Gate at top of `IsOpSupported`. |
| `onnxruntime/test/providers/qnn/unit/qnn_op_affinity_map_test.cc` | Host-side unit tests (parse + Evaluate, no device). |
| `onnxruntime/test/providers/qnn/group_query_attention_test.cc` | EP-assignment integration tests. |
| `docs/execution_providers/QNN-ExecutionProvider.md` | Option documentation. |

---

## Task 0: Branch Setup

**Files:** none (git only)

- [ ] **Step 1: Fetch base and create feature branch**

```bash
cd C:/Users/chuteng/ORT/AUTO/test/onnxruntime-qnn
git fetch origin dev/chunghow/gqa-htp-github
git checkout -b dev/chuteng/add-gqa-op-affinity-gate origin/dev/chunghow/gqa-htp-github
```

Expected: new branch created, working tree at #556's HEAD. Confirm GQA builder shows the `IsGpuBackend || IsNpuBackend` guard:

```bash
grep -n "only supported with the GPU backend and HTP backend" onnxruntime/core/providers/qnn/builder/opbuilder/group_query_attention_op_builder.cc
```
Expected: one match.

- [ ] **Step 2: Bring the design spec onto this branch**

```bash
git checkout dev/chuteng/add_range_op_builder -- docs/superpowers/specs/2026-07-31-gqa-op-affinity-gate-design.md
git add docs/superpowers/specs/2026-07-31-gqa-op-affinity-gate-design.md
git commit -m "docs: add op_affinity GQA gate design spec"
```
Expected: spec committed on the feature branch.

---

## Task 1: `OpAffinityMap` header (class skeleton)

**Files:**
- Create: `onnxruntime/core/providers/qnn/qnn_op_affinity_map.h`

- [ ] **Step 1: Write the header**

Create `onnxruntime/core/providers/qnn/qnn_op_affinity_map.h`:

```cpp
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>

#include "core/providers/qnn/builder/qnn_def.h"

namespace onnxruntime {
namespace qnn {

// Parses and represents the "op_affinity" provider option: a JSON config file mapping ONNX op types
// to the single backend allowed to claim them, e.g. { "op_type": { "GroupQueryAttention": "HTP" } }.
//
// Heterogeneous execution is NOT supported, so each op maps to at most ONE backend; a value that is
// an array of length > 1 is rejected at parse time.
//
// Consumed today only by the GroupQueryAttention op builder's IsOpSupported. The decision truth table
// (see the design spec) is fully encapsulated in Evaluate() so call sites only branch on its result.
class OpAffinityMap {
 public:
  // Result of Evaluate(). kError is returned (not thrown) for a runtime backend mismatch so the op
  // builder can convert it into a fail Status via RETURN_IF, matching the codebase's RETURN_IF_* idiom.
  enum class Decision : uint8_t { kProceed,
                                  kReject,
                                  kError };

  // Unconfigured filter -- the state when the "op_affinity" option is unset.
  OpAffinityMap() = default;

  // Parse a JSON config file into a map. Throws std::runtime_error on ANY problem: unopenable file,
  // malformed JSON, missing/!object "op_type", a value that is neither string nor array, an empty or
  // length>1 array, or an unknown backend name. The EP caller deliberately does NOT catch, so a bad
  // config fails session creation loudly.
  static OpAffinityMap FromConfigFile(const std::filesystem::path& config_file);

  // True when a config file was successfully loaded (even if it lists no ops). Distinguishes
  // "no option given" from "option given but this op absent".
  bool IsConfigured() const { return configured_; }

  // Encapsulates the whole truth table. Does not throw. See the design spec §3 for every cell.
  Decision Evaluate(const std::string& op_type, QnnBackendType session_backend) const;

 private:
  // op type -> the single backend allowed to claim it. Populated only from a config file.
  std::unordered_map<std::string, QnnBackendType> op_to_backend_;
  bool configured_ = false;
};

}  // namespace qnn
}  // namespace onnxruntime
```

- [ ] **Step 2: Commit**

```bash
git add onnxruntime/core/providers/qnn/qnn_op_affinity_map.h
git commit -m "feat(qnn): add OpAffinityMap header skeleton"
```

---

## Task 2: `Evaluate` truth-table logic

`Evaluate` is pure logic over the private `op_to_backend_`/`configured_` members. Because those members are populated only by `FromConfigFile` (Task 3) and there is no test-only setter (adding one would violate minimal-change), `Evaluate` is exercised end-to-end through the parse-driven unit tests in Task 5 — not by a standalone test here. This task implements and compiles `Evaluate`; Task 5 verifies its behavior across every truth-table cell.

**Files:**
- Create: `onnxruntime/core/providers/qnn/qnn_op_affinity_map.cc`

- [ ] **Step 1: Write `Evaluate` + a normalization helper (no parse yet)**

Create `onnxruntime/core/providers/qnn/qnn_op_affinity_map.cc`:

```cpp
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/qnn_op_affinity_map.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>

#include "nlohmann/json.hpp"

namespace onnxruntime {
namespace qnn {

namespace {

// Lowercase a copy for case-insensitive backend-name matching ("HTP" == "htp").
std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

// Map a (case-insensitive) backend name to the enum, using QnnBackendTypeToString as the single
// source of truth so parsing and matching cannot drift. Returns nullopt for an unknown name.
std::optional<QnnBackendType> BackendFromName(const std::string& raw_name) {
  const std::string name = ToLower(raw_name);
  for (uint8_t i = 0; i <= static_cast<uint8_t>(QnnBackendType::SERIALIZER); ++i) {
    const auto backend = static_cast<QnnBackendType>(i);
    if (name == QnnBackendTypeToString(backend)) {
      return backend;
    }
  }
  return std::nullopt;
}

// True if a pinned backend matches the session backend, treating HTP and HTP_FP16 as the same
// physical backend (a pin written as either name matches a session running the other).
bool BackendMatches(QnnBackendType pinned, QnnBackendType session_backend) {
  const bool pinned_is_htp = (pinned == QnnBackendType::HTP || pinned == QnnBackendType::HTP_FP16);
  const bool session_is_htp = (session_backend == QnnBackendType::HTP ||
                               session_backend == QnnBackendType::HTP_FP16);
  if (pinned_is_htp && session_is_htp) {
    return true;
  }
  return pinned == session_backend;
}

}  // namespace

OpAffinityMap::Decision OpAffinityMap::Evaluate(const std::string& op_type,
                                                QnnBackendType session_backend) const {
  const bool session_is_htp = (session_backend == QnnBackendType::HTP ||
                               session_backend == QnnBackendType::HTP_FP16);

  // No config file, or file given but this op not listed: fall back to the per-backend default --
  // HTP is opt-in (reject), every other backend is opt-out (proceed).
  const auto it = op_to_backend_.find(op_type);
  if (!configured_ || it == op_to_backend_.end()) {
    return session_is_htp ? Decision::kReject : Decision::kProceed;
  }

  const QnnBackendType pinned = it->second;
  if (BackendMatches(pinned, session_backend)) {
    return Decision::kProceed;
  }
  if (pinned == QnnBackendType::CPU) {
    return Decision::kReject;  // Legitimate "fall back to CPU EP" intent -- silent.
  }
  return Decision::kError;  // Pinned to another accelerator the session isn't running -> fail loudly.
}

}  // namespace qnn
}  // namespace onnxruntime
```

- [ ] **Step 2: Build the QNN provider to confirm it compiles**

Run (adjust build dir to your setup):
```bash
python tools/ci_build/build.py --build_dir build/Windows --config RelWithDebInfo --use_qnn --build --target onnxruntime_providers_qnn --parallel 2>&1 | tail -20
```
Expected: compiles clean (no test yet — Evaluate is exercised in Task 5).

> If you use the `ort-qnn-ep:build` skill, invoke it instead of the raw command above.

- [ ] **Step 3: Commit**

```bash
git add onnxruntime/core/providers/qnn/qnn_op_affinity_map.cc
git commit -m "feat(qnn): implement OpAffinityMap::Evaluate truth-table logic"
```

---

## Task 3: `FromConfigFile` JSON parser

**Files:**
- Modify: `onnxruntime/core/providers/qnn/qnn_op_affinity_map.cc`

- [ ] **Step 1: Add `FromConfigFile` above the `Evaluate` definition (inside the `qnn` namespace, after the anonymous namespace)**

Insert this function definition just before `OpAffinityMap::Evaluate`:

```cpp
OpAffinityMap OpAffinityMap::FromConfigFile(const std::filesystem::path& config_file) {
  std::ifstream ifs(config_file);
  if (!ifs) {
    throw std::runtime_error("op_affinity config file could not be opened: " + config_file.string());
  }

  // ignore_comments=true allows JSONC-style // comments; parse errors propagate as exceptions.
  const nlohmann::json j = nlohmann::json::parse(ifs, /*cb*/ nullptr, /*allow_exceptions*/ true,
                                                 /*ignore_comments*/ true);

  if (!j.contains("op_type") || !j.at("op_type").is_object()) {
    throw std::runtime_error("op_affinity config: top-level \"op_type\" object is required.");
  }

  OpAffinityMap result;
  for (const auto& [op_name, value] : j.at("op_type").items()) {
    // Extract exactly one backend name from either a string or a length-1 array.
    std::string backend_name;
    if (value.is_string()) {
      backend_name = value.get<std::string>();
    } else if (value.is_array()) {
      if (value.size() != 1) {
        throw std::runtime_error(
            "op_affinity config: op type '" + op_name +
            "' must map to exactly one backend; heterogeneous execution is not supported.");
      }
      if (!value.at(0).is_string()) {
        throw std::runtime_error("op_affinity config: backend for op type '" + op_name +
                                 "' must be a string.");
      }
      backend_name = value.at(0).get<std::string>();
    } else {
      throw std::runtime_error("op_affinity config: value for op type '" + op_name +
                               "' must be a string or a single-element array of strings.");
    }

    const std::optional<QnnBackendType> backend = BackendFromName(backend_name);
    if (!backend.has_value()) {
      throw std::runtime_error("op_affinity config: unknown backend '" + backend_name +
                               "' for op type '" + op_name + "'.");
    }
    result.op_to_backend_[op_name] = *backend;
  }

  result.configured_ = true;
  return result;
}
```

- [ ] **Step 2: Build to confirm it compiles**

```bash
python tools/ci_build/build.py --build_dir build/Windows --config RelWithDebInfo --use_qnn --build --target onnxruntime_providers_qnn --parallel 2>&1 | tail -20
```
Expected: compiles clean.

- [ ] **Step 3: Commit**

```bash
git add onnxruntime/core/providers/qnn/qnn_op_affinity_map.cc
git commit -m "feat(qnn): parse op_affinity JSON config in FromConfigFile"
```

---

## Task 4: Thread `OpAffinityMap` through `ModelSettings` + EP

**Files:**
- Modify: `onnxruntime/core/providers/qnn/builder/qnn_model_wrapper.h` (struct `ModelSettings`, ~line 37)
- Modify: `onnxruntime/core/providers/qnn/qnn_execution_provider.h` (EP members)
- Modify: `onnxruntime/core/providers/qnn/qnn_execution_provider.cc` (option parse ~919; wrapper build ~1375)

- [ ] **Step 1: Forward-declare `OpAffinityMap` and add the field to `ModelSettings`**

In `qnn_model_wrapper.h`, inside `namespace onnxruntime { namespace qnn {` but **above** `struct ModelSettings`, add a forward declaration, then the field. Change:

```cpp
struct ModelSettings {
  bool offload_graph_io_quantization = false;
  bool htp_shared_memory = false;
  bool htp_bf16_enable = false;
};
```
to:

```cpp
class OpAffinityMap;  // forward declaration; full type in qnn_op_affinity_map.h (avoids include cycle)

struct ModelSettings {
  bool offload_graph_io_quantization = false;
  bool htp_shared_memory = false;
  bool htp_bf16_enable = false;
  // Op-to-backend affinity, owned by the EP for the session lifetime. nullptr = option unset.
  // A pointer (not a value) because ModelSettings is copied into every QnnModelWrapper.
  const OpAffinityMap* op_affinity = nullptr;
};
```

- [ ] **Step 2: Add the EP member and include**

In `qnn_execution_provider.h`, add near the other includes:

```cpp
#include "core/providers/qnn/qnn_op_affinity_map.h"
```

And add a private member alongside other option members (search for `model_settings_` declaration to place nearby):

```cpp
qnn::OpAffinityMap op_affinity_map_;
```

- [ ] **Step 3: Parse the option and wire the pointer in the EP constructor**

In `qnn_execution_provider.cc`, right after the `model_settings_.htp_bf16_enable = ...` block (ends ~line 929, before the BF16 compatibility check at ~931), insert:

```cpp
  // op_affinity: JSON config gating which backend may claim specific op types (currently GQA only).
  // A parse failure is deliberately NOT caught -- a bad config fails session creation loudly.
  std::string op_affinity_path;
  GetSessionConfigEntryOrDefault(ort_api, session_options_,
                                 FormatEPConfigKey("op_affinity"), "", op_affinity_path);
  // Trim surrounding ASCII whitespace (no shared trim helper exists in this codebase).
  {
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    op_affinity_path.erase(op_affinity_path.begin(),
                           std::find_if(op_affinity_path.begin(), op_affinity_path.end(), not_space));
    op_affinity_path.erase(std::find_if(op_affinity_path.rbegin(), op_affinity_path.rend(), not_space).base(),
                           op_affinity_path.end());
  }
  if (!op_affinity_path.empty()) {
    op_affinity_map_ = qnn::OpAffinityMap::FromConfigFile(std::filesystem::path(op_affinity_path));
  }
  model_settings_.op_affinity = &op_affinity_map_;
```

Ensure these includes exist at the top of `qnn_execution_provider.cc` (add any missing):

```cpp
#include <algorithm>
#include <cctype>
#include <filesystem>
```

- [ ] **Step 4: Build the provider**

```bash
python tools/ci_build/build.py --build_dir build/Windows --config RelWithDebInfo --use_qnn --build --target onnxruntime_providers_qnn --parallel 2>&1 | tail -20
```
Expected: compiles clean.

- [ ] **Step 5: Commit**

```bash
git add onnxruntime/core/providers/qnn/builder/qnn_model_wrapper.h onnxruntime/core/providers/qnn/qnn_execution_provider.h onnxruntime/core/providers/qnn/qnn_execution_provider.cc
git commit -m "feat(qnn): thread OpAffinityMap through ModelSettings and parse op_affinity option"
```

---

## Task 5: Unit tests for `OpAffinityMap` (parse + Evaluate)

**Files:**
- Create: `onnxruntime/test/providers/qnn/unit/qnn_op_affinity_map_test.cc`

These are host-side, no device. They write temp config files to exercise `FromConfigFile`, then assert on `Evaluate`.

- [ ] **Step 1: Write the test file**

Create `onnxruntime/test/providers/qnn/unit/qnn_op_affinity_map_test.cc`:

```cpp
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Function-level unit tests for OpAffinityMap -- JSON parse paths and the Evaluate() truth table.
// Pure logic + temp-file I/O; no QNN backend, hardware, or emulator required.

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <filesystem>
#include <fstream>
#include <string>

#include "core/providers/qnn/qnn_op_affinity_map.h"
#include "core/providers/qnn/builder/qnn_def.h"

namespace onnxruntime {
namespace test {

using qnn::OpAffinityMap;
using qnn::QnnBackendType;
using Decision = qnn::OpAffinityMap::Decision;

namespace {

// Writes `contents` to a uniquely-named temp file and returns its path. Caller deletes it.
std::filesystem::path WriteTempConfig(const std::string& contents, const std::string& tag) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / ("op_affinity_" + tag + ".json");
  std::ofstream ofs(path);
  ofs << contents;
  ofs.close();
  return path;
}

}  // namespace

// ---------------- Parse: success ----------------

TEST(QnnUnit_OpAffinityMap, ParsesSingleString) {
  const auto path = WriteTempConfig(R"({ "op_type": { "GroupQueryAttention": "HTP" } })", "single");
  const OpAffinityMap map = OpAffinityMap::FromConfigFile(path);
  EXPECT_TRUE(map.IsConfigured());
  EXPECT_EQ(map.Evaluate("GroupQueryAttention", QnnBackendType::HTP), Decision::kProceed);
  std::filesystem::remove(path);
}

TEST(QnnUnit_OpAffinityMap, ParsesLengthOneArray) {
  const auto path = WriteTempConfig(R"({ "op_type": { "GroupQueryAttention": ["GPU"] } })", "arr1");
  const OpAffinityMap map = OpAffinityMap::FromConfigFile(path);
  EXPECT_EQ(map.Evaluate("GroupQueryAttention", QnnBackendType::GPU), Decision::kProceed);
  std::filesystem::remove(path);
}

TEST(QnnUnit_OpAffinityMap, BackendNameIsCaseInsensitive) {
  for (const char* spelling : {"htp", "HTP", "Htp"}) {
    const auto path = WriteTempConfig(
        std::string(R"({ "op_type": { "GroupQueryAttention": ")") + spelling + R"(" } })", "case");
    const OpAffinityMap map = OpAffinityMap::FromConfigFile(path);
    EXPECT_EQ(map.Evaluate("GroupQueryAttention", QnnBackendType::HTP), Decision::kProceed) << spelling;
    std::filesystem::remove(path);
  }
}

TEST(QnnUnit_OpAffinityMap, HtpAndHtpFp16AreAliases) {
  const auto path = WriteTempConfig(R"({ "op_type": { "GroupQueryAttention": "htp_fp16" } })", "alias");
  const OpAffinityMap map = OpAffinityMap::FromConfigFile(path);
  // Pinned to htp_fp16, session running HTP -> still matches.
  EXPECT_EQ(map.Evaluate("GroupQueryAttention", QnnBackendType::HTP), Decision::kProceed);
  std::filesystem::remove(path);
}

// ---------------- Parse: throw paths ----------------

TEST(QnnUnit_OpAffinityMap, ThrowsWhenFileMissing) {
  const std::filesystem::path missing =
      std::filesystem::temp_directory_path() / "op_affinity_does_not_exist_12345.json";
  EXPECT_THROW(OpAffinityMap::FromConfigFile(missing), std::runtime_error);
}

TEST(QnnUnit_OpAffinityMap, ThrowsOnBadJson) {
  const auto path = WriteTempConfig("{ not valid json ", "badjson");
  EXPECT_ANY_THROW(OpAffinityMap::FromConfigFile(path));
  std::filesystem::remove(path);
}

TEST(QnnUnit_OpAffinityMap, ThrowsWhenOpTypeMissing) {
  const auto path = WriteTempConfig(R"({ "something_else": {} })", "nooptype");
  EXPECT_THROW(OpAffinityMap::FromConfigFile(path), std::runtime_error);
  std::filesystem::remove(path);
}

TEST(QnnUnit_OpAffinityMap, ThrowsOnNumericValue) {
  const auto path = WriteTempConfig(R"({ "op_type": { "GroupQueryAttention": 3 } })", "numeric");
  EXPECT_THROW(OpAffinityMap::FromConfigFile(path), std::runtime_error);
  std::filesystem::remove(path);
}

TEST(QnnUnit_OpAffinityMap, ThrowsOnEmptyArray) {
  const auto path = WriteTempConfig(R"({ "op_type": { "GroupQueryAttention": [] } })", "emptyarr");
  EXPECT_THROW(OpAffinityMap::FromConfigFile(path), std::runtime_error);
  std::filesystem::remove(path);
}

TEST(QnnUnit_OpAffinityMap, ThrowsOnMultiElementArray) {
  const auto path =
      WriteTempConfig(R"({ "op_type": { "GroupQueryAttention": ["HTP", "GPU"] } })", "multiarr");
  EXPECT_THROW(OpAffinityMap::FromConfigFile(path), std::runtime_error);
  std::filesystem::remove(path);
}

TEST(QnnUnit_OpAffinityMap, ThrowsOnUnknownBackend) {
  const auto path = WriteTempConfig(R"({ "op_type": { "GroupQueryAttention": "NPU2" } })", "unknownbe");
  EXPECT_THROW(OpAffinityMap::FromConfigFile(path), std::runtime_error);
  std::filesystem::remove(path);
}

// ---------------- Evaluate: truth table ----------------

TEST(QnnUnit_OpAffinityMap, UnconfiguredHtpRejectsGpuProceeds) {
  const OpAffinityMap map;  // default = unconfigured
  EXPECT_FALSE(map.IsConfigured());
  EXPECT_EQ(map.Evaluate("GroupQueryAttention", QnnBackendType::HTP), Decision::kReject);
  EXPECT_EQ(map.Evaluate("GroupQueryAttention", QnnBackendType::HTP_FP16), Decision::kReject);
  EXPECT_EQ(map.Evaluate("GroupQueryAttention", QnnBackendType::GPU), Decision::kProceed);
  EXPECT_EQ(map.Evaluate("GroupQueryAttention", QnnBackendType::CPU), Decision::kProceed);
}

TEST(QnnUnit_OpAffinityMap, ConfiguredButOpAbsentUsesDefault) {
  const auto path = WriteTempConfig(R"({ "op_type": { "SomeOtherOp": "HTP" } })", "absent");
  const OpAffinityMap map = OpAffinityMap::FromConfigFile(path);
  EXPECT_EQ(map.Evaluate("GroupQueryAttention", QnnBackendType::HTP), Decision::kReject);
  EXPECT_EQ(map.Evaluate("GroupQueryAttention", QnnBackendType::GPU), Decision::kProceed);
  std::filesystem::remove(path);
}

TEST(QnnUnit_OpAffinityMap, PinHtpEvaluations) {
  const auto path = WriteTempConfig(R"({ "op_type": { "GroupQueryAttention": "HTP" } })", "pinhtp");
  const OpAffinityMap map = OpAffinityMap::FromConfigFile(path);
  EXPECT_EQ(map.Evaluate("GroupQueryAttention", QnnBackendType::HTP), Decision::kProceed);
  EXPECT_EQ(map.Evaluate("GroupQueryAttention", QnnBackendType::GPU), Decision::kError);
  std::filesystem::remove(path);
}

TEST(QnnUnit_OpAffinityMap, PinGpuEvaluations) {
  const auto path = WriteTempConfig(R"({ "op_type": { "GroupQueryAttention": "GPU" } })", "pingpu");
  const OpAffinityMap map = OpAffinityMap::FromConfigFile(path);
  EXPECT_EQ(map.Evaluate("GroupQueryAttention", QnnBackendType::GPU), Decision::kProceed);
  EXPECT_EQ(map.Evaluate("GroupQueryAttention", QnnBackendType::HTP), Decision::kError);
  std::filesystem::remove(path);
}

TEST(QnnUnit_OpAffinityMap, PinCpuRejectsOnAccelerators) {
  const auto path = WriteTempConfig(R"({ "op_type": { "GroupQueryAttention": "CPU" } })", "pincpu");
  const OpAffinityMap map = OpAffinityMap::FromConfigFile(path);
  EXPECT_EQ(map.Evaluate("GroupQueryAttention", QnnBackendType::HTP), Decision::kReject);
  EXPECT_EQ(map.Evaluate("GroupQueryAttention", QnnBackendType::GPU), Decision::kReject);
  std::filesystem::remove(path);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
```

- [ ] **Step 2: Build the QNN unit test binary**

```bash
python tools/ci_build/build.py --build_dir build/Windows --config RelWithDebInfo --use_qnn --build --target onnxruntime_test_all --parallel 2>&1 | tail -25
```
Expected: compiles clean.

- [ ] **Step 3: Run the unit tests**

```bash
build/Windows/RelWithDebInfo/onnxruntime_test_all --gtest_filter="QnnUnit_OpAffinityMap.*" 2>&1 | tail -40
```
Expected: all `QnnUnit_OpAffinityMap.*` tests PASS.

- [ ] **Step 4: Commit**

```bash
git add onnxruntime/test/providers/qnn/unit/qnn_op_affinity_map_test.cc
git commit -m "test(qnn): unit tests for OpAffinityMap parse + Evaluate truth table"
```

---

## Task 6: Gate GQA `IsOpSupported`

**Files:**
- Modify: `onnxruntime/core/providers/qnn/builder/opbuilder/group_query_attention_op_builder.cc` (`IsOpSupported`, the backend-guard region)

- [ ] **Step 1: Add the include**

At the top of `group_query_attention_op_builder.cc`, add among the existing includes:

```cpp
#include "core/providers/qnn/qnn_op_affinity_map.h"
```

- [ ] **Step 2: Insert the gate before the existing backend guard**

Find:

```cpp
  auto backend_type = qnn_model_wrapper.GetQnnBackendType();
  RETURN_IF_NOT(IsGpuBackend(backend_type) || IsNpuBackend(backend_type),
                "GroupQueryAttention is only supported with the GPU backend and HTP backend");
```

Replace with:

```cpp
  auto backend_type = qnn_model_wrapper.GetQnnBackendType();

  // op_affinity gate (opt-in on HTP, opt-out on GPU). See docs/superpowers/specs truth table.
  const qnn::OpAffinityMap* affinity = qnn_model_wrapper.GetModelSettings().op_affinity;
  if (affinity != nullptr && affinity->IsConfigured()) {
    const auto decision = affinity->Evaluate("GroupQueryAttention", backend_type);
    RETURN_IF(decision == qnn::OpAffinityMap::Decision::kError,
              "GroupQueryAttention op_affinity pins it to a backend this session is not running.");
    RETURN_IF_NOT(decision == qnn::OpAffinityMap::Decision::kProceed,
                  "GroupQueryAttention filtered off QNN by the op_affinity provider option.");
  } else {
    // Unconfigured: HTP is opt-in (rejected without a config), GPU is opt-out (proceeds).
    RETURN_IF_NOT(!IsNpuBackend(backend_type),
                  "GroupQueryAttention on HTP requires an op_affinity config pinning it to HTP.");
  }

  RETURN_IF_NOT(IsGpuBackend(backend_type) || IsNpuBackend(backend_type),
                "GroupQueryAttention is only supported with the GPU backend and HTP backend");
```

- [ ] **Step 3: Build the provider**

```bash
python tools/ci_build/build.py --build_dir build/Windows --config RelWithDebInfo --use_qnn --build --target onnxruntime_providers_qnn --parallel 2>&1 | tail -20
```
Expected: compiles clean.

- [ ] **Step 4: Commit**

```bash
git add onnxruntime/core/providers/qnn/builder/opbuilder/group_query_attention_op_builder.cc
git commit -m "feat(qnn): gate GroupQueryAttention IsOpSupported on op_affinity"
```

---

## Task 7: GQA EP-assignment integration tests

**Files:**
- Modify: `onnxruntime/test/providers/qnn/group_query_attention_test.cc`

Reuse PR #556's harness. Each test writes a temp config, creates a session with the `op_affinity` provider option, and asserts whether GQA is assigned to the QNN EP (or that session creation fails).

- [ ] **Step 1: Inspect #556's harness to reuse its model-builder + session helpers**

```bash
grep -n "provider_options\|GetQnnBackendType\|BuildGraph\|RunQnnModelTest\|GroupQueryAttention\|EPNodeAssignment\|op_affinity\|SessionOptions\|CreateSession" onnxruntime/test/providers/qnn/group_query_attention_test.cc | head -40
```
Read the existing test that builds a GQA model and runs it on HTP; mirror its model construction. Identify:
- the helper that builds the GQA `ModelTestBuilder` graph,
- how provider options are passed (map of string→string),
- how EP assignment / session-creation-failure is asserted.

- [ ] **Step 2: Add a temp-config helper and the integration tests**

Add near the top of the test file's anonymous namespace (adjust names to match #556's existing helpers — reuse its GQA graph builder rather than duplicating it):

```cpp
// Writes an op_affinity JSON config to a temp file; returns its path. Caller removes it.
static std::filesystem::path WriteOpAffinityConfig(const std::string& json, const std::string& tag) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / ("gqa_affinity_" + tag + ".json");
  std::ofstream ofs(path);
  ofs << json;
  ofs.close();
  return path;
}
```

Then add tests (place among the existing `QnnHTPBackendTests` / `QnnGPUBackendTests` GQA tests). Use #556's existing GQA-graph builder function — shown here as `BuildGqaTestCase(...)`; **replace with the actual builder name found in Step 1**:

```cpp
// HTP + no config -> GQA is opt-in, must NOT be assigned to QNN EP.
TEST_F(QnnHTPBackendTests, GQA_OpAffinity_HtpNoConfig_FallsBack) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  // No op_affinity option set.
  RunQnnModelTestExpectingEpFallback(/* build GQA graph via #556 helper */, provider_options);
}

// HTP + config pinning HTP -> GQA IS assigned to QNN EP.
TEST_F(QnnHTPBackendTests, GQA_OpAffinity_HtpPinHtp_Assigned) {
  const auto cfg = WriteOpAffinityConfig(R"({ "op_type": { "GroupQueryAttention": "HTP" } })", "htp_pin_htp");
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["op_affinity"] = cfg.string();
  RunQnnModelTestExpectingEpAssignment(/* build GQA graph */, provider_options);
  std::filesystem::remove(cfg);
}

// HTP + config pinning GPU -> Evaluate returns kError -> session creation fails.
TEST_F(QnnHTPBackendTests, GQA_OpAffinity_HtpPinGpu_SessionFails) {
  const auto cfg = WriteOpAffinityConfig(R"({ "op_type": { "GroupQueryAttention": "GPU" } })", "htp_pin_gpu");
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["op_affinity"] = cfg.string();
  RunQnnModelTestExpectingSessionCreationFailure(/* build GQA graph */, provider_options);
  std::filesystem::remove(cfg);
}

// HTP + config pinning CPU -> silent fallback (GQA not assigned, session OK).
TEST_F(QnnHTPBackendTests, GQA_OpAffinity_PinCpu_FallsBack) {
  const auto cfg = WriteOpAffinityConfig(R"({ "op_type": { "GroupQueryAttention": "CPU" } })", "pin_cpu");
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["op_affinity"] = cfg.string();
  RunQnnModelTestExpectingEpFallback(/* build GQA graph */, provider_options);
  std::filesystem::remove(cfg);
}

// HTP + nonexistent config path -> session creation fails.
TEST_F(QnnHTPBackendTests, GQA_OpAffinity_MissingFile_SessionFails) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["op_affinity"] = (std::filesystem::temp_directory_path() / "no_such_affinity.json").string();
  RunQnnModelTestExpectingSessionCreationFailure(/* build GQA graph */, provider_options);
}
```

> **Note:** The `RunQnnModelTestExpecting*` names above are placeholders for whatever assertion style #556's file already uses (e.g. it may pass an `ExpectedEPNodeAssignment::All` / `::None` argument to `RunQnnModelTest`, or check a status). In Step 1 you identified the real helper — use it. For the "session creation fails" cases, assert that session creation returns a non-OK status (the parse `throw` / `kError` path surfaces as a failed `InferenceSession` initialization).

- [ ] **Step 3: Add a GPU opt-out test (guards the non-HTP default path)**

```cpp
// GPU + no config -> GQA is opt-out, remains assigned to QNN EP (behavior unchanged from #556).
TEST_F(QnnGPUBackendTests, GQA_OpAffinity_GpuNoConfig_Assigned) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "gpu";
  RunQnnModelTestExpectingEpAssignment(/* build GQA graph */, provider_options);
}
```

- [ ] **Step 4: Build the test binary**

```bash
python tools/ci_build/build.py --build_dir build/Windows --config RelWithDebInfo --use_qnn --build --target onnxruntime_test_all --parallel 2>&1 | tail -25
```
Expected: compiles clean.

- [ ] **Step 5: Run the GQA affinity tests**

```bash
build/Windows/RelWithDebInfo/onnxruntime_test_all --gtest_filter="*GQA_OpAffinity*" 2>&1 | tail -40
```
Expected: all pass. (HTP tests require an HTP device/emulator; if running host-only, at minimum the `_SessionFails` and `_FallsBack` EP-assignment checks that don't execute the graph should pass. Note in the commit which required a device.)

- [ ] **Step 6: Commit**

```bash
git add onnxruntime/test/providers/qnn/group_query_attention_test.cc
git commit -m "test(qnn): EP-assignment tests for GQA op_affinity gate"
```

---

## Task 8: Documentation

**Files:**
- Modify: `docs/execution_providers/QNN-ExecutionProvider.md`

- [ ] **Step 1: Add the provider-option row and a details section**

Find the provider-options table and add a row:

```markdown
|`op_affinity`|Path to a JSON config file pinning specific ONNX op types to a backend. Currently gates `GroupQueryAttention` only. See "OP Affinity" below.|
```

Then add a section:

```markdown
### OP Affinity

The `op_affinity` option points at a JSON config file that pins ONNX op types to a single backend:

```json
{ "op_type": { "GroupQueryAttention": "HTP" } }
```

- Backend names are case-insensitive (`"HTP"` == `"htp"`); `htp` and `htp_fp16` are aliases.
- A value may be a string or a single-element array (`["HTP"]`). **Arrays of length > 1 are rejected** — heterogeneous execution (one op across multiple backends) is not supported.
- On the command line (e.g. `onnxruntime_perf_test`), pass it with the `key|value` form: `op_affinity|./affinity_config.json`.

Behavior for `GroupQueryAttention` (the only op gated today):

| Config state | HTP session | GPU session |
|---|---|---|
| No config file | not claimed (opt-in) | claimed (opt-out) |
| File given, GQA not listed | not claimed | claimed |
| GQA pinned to the running backend | claimed | claimed |
| GQA pinned to CPU | not claimed (falls back to CPU EP) | not claimed |
| GQA pinned to another accelerator | **session creation fails** | **session creation fails** |

An unopenable or malformed config file fails session creation.
```

- [ ] **Step 2: Commit**

```bash
git add docs/execution_providers/QNN-ExecutionProvider.md
git commit -m "docs(qnn): document op_affinity provider option"
```

---

## Task 9: Lint + final verification

**Files:** none (verification only)

- [ ] **Step 1: Run the linter (mandatory quality gate)**

```bash
python qcom/build_and_test.py lint_and_fix 2>&1 | tail -30
```
Expected: no outstanding style errors. Commit any auto-fixes:

```bash
git add -A && git commit -m "style: lint fixes for op_affinity" || echo "no lint changes"
```

> If the `ort-qnn-ep:lint` skill is available, invoke it instead.

- [ ] **Step 2: Full QNN unit-test sweep for regressions**

```bash
build/Windows/RelWithDebInfo/onnxruntime_test_all --gtest_filter="QnnUnit_*:*GQA*" 2>&1 | tail -40
```
Expected: all pass.

- [ ] **Step 3: Confirm the file set matches the spec (§9)**

```bash
git diff --name-only origin/dev/chunghow/gqa-htp-github
```
Expected: exactly the 9 files listed in the plan's File Structure (plus the two `docs/superpowers/` files).

---

## Self-Review Notes (author checklist — completed)

- **Spec coverage:** truth table (§3) → Task 2 `Evaluate` + Task 5 tests; config format/array/case-insensitive (§2) → Task 3 + Task 5; command-line `key|value` (§2) → Task 4 (no perf_test change) + Task 8 docs; ModelSettings plumbing (§5.4/5.5) → Task 4; GQA gate (§5.6) → Task 6; error table (§6) → Tasks 3/6 + tests 5/7; test plan (§7) → Tasks 5/7; docs (§8) → Task 8. All covered.
- **Placeholder scan:** the only intentional placeholders are the `RunQnnModelTestExpecting*` helper names and `/* build GQA graph */` in Task 7 — these are explicitly flagged to be replaced with #556's real harness names discovered in Task 7 Step 1, because the harness API is defined in #556's file which this plan builds on but does not reproduce.
- **Type consistency:** `OpAffinityMap`, `Decision{kProceed,kReject,kError}`, `FromConfigFile`, `IsConfigured`, `Evaluate`, and `ModelSettings::op_affinity` (pointer) are named identically across Tasks 1–7.
```
