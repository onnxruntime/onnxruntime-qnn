# Design: `op_affinity` Config Gate for GroupQueryAttention

- **Date**: 2026-07-31
- **Author**: chuteng
- **Status**: Approved (pending written-spec review)
- **Base PR**: [#556 — Support GQA on HTP backend](https://github.com/onnxruntime/onnxruntime-qnn/pull/556) (`origin/dev/chunghow/gqa-htp-github`)
- **Supersedes approach of**: [#599 — op_affinity exclude/include filter](https://github.com/onnxruntime/onnxruntime-qnn/pull/599) (this design replaces that model; #599 is **not** built upon)

## 1. Motivation

PR #556 widened `GroupQueryAttention` (GQA) `IsOpSupported` from GPU-only to
`IsGpuBackend || IsNpuBackend`, so after #556 GQA is claimed by QNN on **both** GPU and HTP by
default. We want an operator-to-backend affinity config that turns GQA on HTP into an **opt-in**
(and keeps GPU as opt-out), so a user can pin which backend claims GQA via a JSON config file.

This design adds an `op_affinity` provider option pointing at a JSON config file, parses it into an
`OpAffinityMap`, threads it to the GQA op builder via the existing `ModelSettings` channel, and
gates GQA's `IsOpSupported` on it.

**Non-goals**: This gate applies **only** to `GroupQueryAttention`. It is not a general per-op
filter (that was #599's model, deliberately not reused here). Heterogeneous execution (one op split
across multiple backends) is **not supported**.

## 2. Confirmed Requirements

| Item | Decision |
|------|----------|
| Positioning | New PR **replacing** #599's exclude/include model. Built on top of #556. |
| Config format | `{ "op_type": { "GroupQueryAttention": "HTP" } }` |
| Backend value | Single string, or length-1 array (`["HTP"]`). **Length >1 → error** (no heterogeneous execution). |
| Command line | `op_affinity\|.\affinity_config.json` (perf_test `key\|value` convention; EP receives the path string directly, no `@` prefix). |
| Scope | Applied only in GQA `IsOpSupported`. |
| Backend string | **Case-insensitive** (`"HTP"` == `"htp"`). `htp` ≡ `htp_fp16` alias. |
| Malformed/unreadable config | **Fail session creation** (loud, not degrade). |
| Branch | Base on `origin/dev/chunghow/gqa-htp-github` (#556); new branch `dev/chuteng/XXXX`. #556 itself untouched. |

## 3. Gate Truth Table (authoritative)

Session runs a single backend (HTP or GPU). Config values are case-insensitive.

| Config state | HTP session | GPU session |
|---|---|---|
| No config file given | **false** (opt-in) | proceed (opt-out) |
| File given, GQA not listed | **false** (HTP default) | proceed (GPU default) |
| GQA pinned to *current* backend | proceed | proceed |
| GQA pinned to CPU | **false** (silent fallback) | **false** (silent fallback) |
| GQA pinned to *another single* accelerator (not current, not CPU) | **error → session fails** | **error → session fails** |
| GQA value is length-\>1 array | **error** (no heterogeneous) | **error** (no heterogeneous) |

- **"proceed"** = pass the gate and continue into PR #556's existing `IsGpuBackend || IsNpuBackend`
  check and all subsequent GQA validation.
- **CPU is the only special "silent false"** — pinning to CPU expresses a legitimate "fall back to
  CPU EP" intent. Every other misconfiguration fails loudly.
- The "pinned set contains current backend AND another" row cannot occur: length >1 is rejected at
  parse time, so the pin is always exactly one backend.

## 4. Architecture

```
Command line  op_affinity|.\affinity_config.json
   │  (perf_test key|value → EP receives option value = ".\affinity_config.json")
   ▼
[1] EP option parse (qnn_execution_provider.cc)
   │  read "op_affinity" option → path string
   │  → OpAffinityMap::FromConfigFile(path)   ← unopenable / malformed / >1 array / unknown backend → throw → session fails
   ▼
[2] OpAffinityMap  (new: qnn_op_affinity_map.{h,cc})
   │  stores map<op_type, QnnBackendType> + configured_ flag
   │  API: FromConfigFile(), IsConfigured(), Evaluate(op_type, backend) -> {kProceed,kReject,kError}
   ▼
[3] ModelSettings (existing struct, +1 field: const OpAffinityMap*)  →  QnnModelWrapper  →  GetModelSettings()
   ▼
[4] GQA IsOpSupported (group_query_attention_op_builder.cc)
      gate before #556's backend guard; RETURN_IF_NOT on kProceed; kError → fail status
```

**Approach A** (chosen): reuse the existing `ModelSettings` → `QnnModelWrapper` → `GetModelSettings()`
channel — the same path `htp_bf16_enable` and `offload_graph_io_quantization` already use to reach op
builders. Zero new plumbing invented. (Approach B: central `GetSupportedNodes` filter — rejected, it
bypasses the op builder where the requirement says the check belongs. Approach C: parallel getter on
`QnnModelWrapper` — rejected, redundant with `ModelSettings`.)

## 5. Component Detail

### 5.1 `OpAffinityMap` (new file `onnxruntime/core/providers/qnn/qnn_op_affinity_map.{h,cc}`)

EP-level (sibling of `qnn_execution_provider.cc`, not in `builder/opbuilder/`), because it is a
session-level setting.

```cpp
class OpAffinityMap {
 public:
  enum class Decision : uint8_t { kProceed, kReject, kError };

  OpAffinityMap() = default;  // unconfigured (no config file given)

  // Single entry point. Reads + parses the JSON file. ANY parse error (unopenable, bad JSON,
  // wrong-typed value, empty array, length>1 array, unknown backend name) throws std::runtime_error.
  // The EP caller does NOT catch → session creation fails (loud).
  static OpAffinityMap FromConfigFile(const std::filesystem::path& path);

  bool IsConfigured() const { return configured_; }

  // Encapsulates the entire truth table. Does NOT throw — runtime backend mismatch returns kError so
  // the op builder can convert it to a fail status via RETURN_IF_*, matching the codebase idiom.
  Decision Evaluate(const std::string& op_type, QnnBackendType session_backend) const;

 private:
  std::unordered_map<std::string, QnnBackendType> op_to_backend_;  // ≤1 backend per op
  bool configured_ = false;
};
```

**Representation rationale**: parse normalizes `"HTP"`/`"htp"`/`["GPU"]` into a `QnnBackendType`
enum once; queries compare directly against `qnn_model_wrapper.GetQnnBackendType()` with no per-query
string work. CPU is a natural `QnnBackendType` enumerator.

### 5.2 Parse path (`FromConfigFile`)

1. Open file; failure → `throw`.
2. `nlohmann::json::parse` (existing dependency; JSONC comments allowed). Parse error propagates.
3. Require top-level `"op_type"` object; missing or non-object → `throw`.
4. For each `(op_name, value)`:
   - string → normalize + store.
   - array → length 0 → `throw`; length 1 → take element; length >1 → `throw` ("heterogeneous
     execution not supported").
   - other type → `throw`.
5. Backend-string normalization: lowercase, then match against `QnnBackendTypeToString` (single
   source of truth). Accepts `htp`/`htp_fp16`/`gpu`/`cpu`/`dsp`/`ir`; `htp` ≡ `htp_fp16`. Unknown →
   `throw`.
6. Set `configured_ = true`.

### 5.3 Decision logic (`Evaluate`)

```
if (!configured_)
    return (session_backend is NPU/HTP) ? kReject : kProceed   // no file: HTP opt-in, GPU opt-out

look up op_type in op_to_backend_:
  not found (file given but op not listed):
    return (session_backend is NPU/HTP) ? kReject : kProceed   // same as no-file default

  found, pinned = op_to_backend_[op_type]:
    if pinned matches session_backend (htp/htp_fp16 alias-aware) → kProceed
    if pinned == CPU                                             → kReject  (silent fallback)
    else (pinned to another accelerator)                        → kError   (→ session fails)
```

### 5.4 `ModelSettings` change (`builder/qnn_model_wrapper.h`)

```cpp
struct ModelSettings {
  bool offload_graph_io_quantization = false;
  bool htp_shared_memory = false;
  bool htp_bf16_enable = false;
  const OpAffinityMap* op_affinity = nullptr;   // new; nullptr = unconfigured
};
```

Stores a **pointer** (not a value): the `OpAffinityMap` is owned by the EP for the session lifetime;
`ModelSettings` is copied every time a `QnnModelWrapper` is constructed, so a pointer avoids deep-copying
the map. `nullptr` naturally expresses "unconfigured". Lifetime is safe: the EP member outlives every
`QnnModelWrapper`.

### 5.5 EP parse call site (`qnn_execution_provider.{h,cc}`)

- `.h`: add member `qnn::OpAffinityMap op_affinity_map_;`
- `.cc` (option-parse region):

```cpp
std::string op_affinity_path;
if (auto it = provider_options.find("op_affinity"); it != provider_options.end()) {
  op_affinity_path = utils::TrimWhitespace(it->second);
}
if (!op_affinity_path.empty()) {
  // Not caught: unopenable / bad JSON / >1 array / unknown backend → propagates → session fails (loud).
  op_affinity_map_ = qnn::OpAffinityMap::FromConfigFile(std::filesystem::path(op_affinity_path));
}
```

- Before constructing `QnnModelWrapper` (in `GetSupportedNodes`): set
  `model_settings_.op_affinity = &op_affinity_map_;`.
- `op_affinity|.\affinity_config.json` is split by perf_test's `key|value` mechanism, so
  `provider_options["op_affinity"] == ".\affinity_config.json"`. No perf_test change needed.

### 5.6 GQA gate (`group_query_attention_op_builder.cc`)

Inserted **before** #556's existing backend guard:

```cpp
auto backend_type = qnn_model_wrapper.GetQnnBackendType();

// op_affinity gate (opt-in on HTP, opt-out on GPU) — see design truth table.
const qnn::OpAffinityMap* affinity = qnn_model_wrapper.GetModelSettings().op_affinity;
if (affinity != nullptr && affinity->IsConfigured()) {
  const auto decision = affinity->Evaluate("GroupQueryAttention", backend_type);
  RETURN_IF(decision == qnn::OpAffinityMap::Decision::kError,
            "GroupQueryAttention op_affinity pins it to a backend this session is not running");
  RETURN_IF_NOT(decision == qnn::OpAffinityMap::Decision::kProceed,
                "GroupQueryAttention filtered off QNN by op_affinity");
} else {
  // Unconfigured: HTP defaults to reject (opt-in), GPU proceeds (opt-out).
  RETURN_IF_NOT(!IsNpuBackend(backend_type),
                "GroupQueryAttention on HTP requires an op_affinity config pinning it to HTP");
}

// —— existing PR #556 checks, unchanged ——
RETURN_IF_NOT(IsGpuBackend(backend_type) || IsNpuBackend(backend_type),
              "GroupQueryAttention is only supported with the GPU backend and HTP backend");
```

Gate placed before the backend guard so the opt-in message is more specific than the generic
"only GPU/HTP" one. CPU/other backends: the unconfigured branch's `!IsNpuBackend` is true for CPU →
does not block → falls to the existing guard → rejected by "only GPU/HTP". Correct.

## 6. Error Handling Summary

| # | Condition | Stage | Mechanism | Result |
|---|-----------|-------|-----------|--------|
| 1 | Config file unopenable | parse | `throw` | session fails |
| 2 | JSON syntax error | parse | nlohmann exception propagates | session fails |
| 3 | Missing `"op_type"` key or non-object | parse | `throw` | session fails |
| 4 | Backend value not string/array | parse | `throw` | session fails |
| 5 | Backend array length 0 | parse | `throw` | session fails |
| 6 | Backend array length >1 (heterogeneous) | parse | `throw` (explicit msg) | session fails |
| 7 | Unknown backend string | parse | `throw` | session fails |
| 8 | GQA pinned to another single accelerator | runtime gate | `kError` → `RETURN_IF` → fail status | session fails |
| 9 | GQA pinned to CPU | runtime gate | `kReject` → gate returns false | **silent** fallback |
| 10 | HTP unconfigured / file given but GQA not listed | runtime gate | `!IsNpuBackend` → false | GQA not claimed (opt-in) |
| 11 | GPU unconfigured / file given but GQA not listed | runtime gate | proceed | GQA runs existing checks (opt-out) |

Only #9 (CPU) and #10 (HTP opt-in default) are silent; all other misconfigurations fail loudly.

## 7. Test Plan

**(A) `OpAffinityMap` unit tests** (new `unit/qnn_op_affinity_map_test.cc`, host-side, no device):
- Parse: valid single string; valid length-1 array; case-insensitive (`htp`/`HTP`/`Htp`); `htp`≡`htp_fp16`.
- Parse throw paths: missing file, bad JSON, missing `op_type`, numeric value, empty array,
  length>1 array, unknown backend name.
- `Evaluate` (every truth-table cell): unconfigured × {HTP→kReject, GPU→kProceed, CPU→kProceed};
  file-without-GQA × {HTP→kReject, GPU→kProceed}; pin-HTP × {HTP→kProceed, GPU→kError};
  pin-GPU × {GPU→kProceed, HTP→kError}; pin-CPU × {HTP→kReject, GPU→kReject}.

**(B) GQA EP-assignment integration tests** (added to existing `group_query_attention_test.cc`,
reusing #556's harness):
- HTP + no config → GQA falls back (EP does not claim).
- HTP + config pin HTP → GQA claimed (runs #556 HTP path).
- GPU + no config → GQA claimed (opt-out preserved).
- HTP + config pin GPU → session creation fails (#8).
- config pin CPU → GQA silent fallback (#9).
- config file not found → session creation fails (#1).

Tests write a temp config file and clean it up, following the project's existing temp-file test idiom.

## 8. Documentation

`docs/execution_providers/QNN-ExecutionProvider.md`: add an `op_affinity` row to the provider-option
table plus a section covering the JSON format, a truth-table summary, the `key|path` command-line
usage, and the error behavior.

## 9. Files Touched

| File | Change |
|------|--------|
| `onnxruntime/core/providers/qnn/qnn_op_affinity_map.h` | new — `OpAffinityMap` class |
| `onnxruntime/core/providers/qnn/qnn_op_affinity_map.cc` | new — parse + `Evaluate` |
| `onnxruntime/core/providers/qnn/builder/qnn_model_wrapper.h` | `ModelSettings` +1 field |
| `onnxruntime/core/providers/qnn/qnn_execution_provider.h` | EP member `op_affinity_map_` |
| `onnxruntime/core/providers/qnn/qnn_execution_provider.cc` | parse option + set `ModelSettings` pointer |
| `onnxruntime/core/providers/qnn/builder/opbuilder/group_query_attention_op_builder.cc` | gate in `IsOpSupported` |
| `onnxruntime/test/providers/qnn/unit/qnn_op_affinity_map_test.cc` | new — unit tests |
| `onnxruntime/test/providers/qnn/group_query_attention_test.cc` | integration tests |
| `docs/execution_providers/QNN-ExecutionProvider.md` | option docs |
