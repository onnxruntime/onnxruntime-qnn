# GQA HTP Test Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract `RunGQATest`'s three repeated setup blocks into named file-local helpers and align section banners with sibling QNN test files, without changing behavior.

**Architecture:** `RunGQATest` in `group_query_attention_test.cc` hand-rolls QNN-vs-CPU comparison (it can't use the shared `RunQnnModelTest` framework because GQA needs in-place present→past KV-cache aliasing on a shared-memory allocator). We keep that loop but split it into three `static` helpers: shared-memory feed creation, present→past output aliasing/collection, and cross-EP output comparison. The locally-defined `FeedCopy` struct is lifted to file scope so helpers can name it.

**Tech Stack:** C++17, GoogleTest, ONNX Runtime QNN EP test utilities (`qnn_test_utils.h`).

**Verification note:** All 22 GQA cases are `DISABLED_` (need QAIRT >= 2.49; CI SDK is 2.48.40), so there is NO runtime verification. Each task is verified by (a) incremental compile of `onnxruntime_provider_test` and (b) diff review confirming moved code is byte-for-byte equivalent to the original. This is stated in every commit.

**Build/verify command (used in every task):**
```bash
cmake --build "build/windows-arm64/Release" --config Release --target onnxruntime_provider_test --parallel
```
Expected: compiles clean, no errors.

---

### Task 1: Lift `FeedCopy` to file scope and extract `MakeSharedMemoryFeeds`

**Files:**
- Modify: `onnxruntime/test/providers/qnn/group_query_attention_test.cc`

- [ ] **Step 1: Lift the `FeedCopy` struct to file scope**

Currently `FeedCopy` is declared inside `RunGQATest` (lines ~220-223). Move it to just above `RunGQATest` (before line 125, after `BuildGQATestCase`'s closing brace), so helpers can reference it:

```cpp
// Holds a shared-memory allocation plus the Ort::Value view over it, so GQA feeds
// (and in-place present/past KV buffers) live on the QNN host-accessible allocator.
struct GQAFeedCopy {
  Ort::MemoryAllocation allocation;
  Ort::Value value{nullptr};
};
```

Then delete the in-function `struct FeedCopy { ... };` (lines ~220-223) and rename its uses in `RunGQATest` from `FeedCopy` to `GQAFeedCopy`.

- [ ] **Step 2: Add the `MakeSharedMemoryFeeds` helper**

Insert immediately after the `GQAFeedCopy` struct. This is the current loop at lines 242-266 moved verbatim, wrapped in a function that returns the two produced values:

```cpp
// Copies each graph input from helper.feeds_ into a tensor backed by `allocator` /
// `memory_info`, returning the feed copies (index-aligned with input_names) and a
// name->index map. Verbatim extraction of the per-input copy loop from RunGQATest.
static void MakeSharedMemoryFeeds(const std::vector<std::string>& input_names,
                                  const std::unordered_map<std::string, Ort::Value>& source_feeds,
                                  Ort::Allocator& allocator,
                                  const Ort::MemoryInfo& memory_info,
                                  std::vector<GQAFeedCopy>& qnn_feeds,
                                  std::unordered_map<std::string, size_t>& input_name_to_index) {
  qnn_feeds.reserve(input_names.size());
  for (const auto& input_name : input_names) {
    const Ort::Value& source_value = source_feeds.at(input_name);
    const auto tensor_info = source_value.GetTensorTypeAndShapeInfo();
    const auto shape = tensor_info.GetShape();
    const size_t num_bytes = source_value.GetTensorSizeInBytes();
    const auto* source_data = reinterpret_cast<const std::byte*>(source_value.GetTensorRawData());

    GQAFeedCopy feed_copy{allocator.GetAllocation(num_bytes)};
    ASSERT_NE(feed_copy.allocation.get(), nullptr);
    memcpy(feed_copy.allocation.get(), source_data, num_bytes);

    feed_copy.value = Ort::Value::CreateTensor(memory_info,
                                               feed_copy.allocation.get(),
                                               feed_copy.allocation.size(),
                                               shape.data(),
                                               shape.size(),
                                               tensor_info.GetElementType());

    input_name_to_index.emplace(input_name, qnn_feeds.size());
    qnn_feeds.push_back(std::move(feed_copy));
  }
}
```

Note: `ASSERT_NE` expands to a `return;` on failure, which is valid because the helper returns `void` (same early-exit semantics as the original inline `ASSERT_NE`).

- [ ] **Step 3: Replace the inline loop in `RunGQATest` with a call**

Replace current lines 242-266 with:

```cpp
  std::vector<GQAFeedCopy> qnn_feeds;
  std::unordered_map<std::string, size_t> input_name_to_index;
  ASSERT_NO_FATAL_FAILURE(MakeSharedMemoryFeeds(input_names, helper.feeds_, allocator, memory_info,
                                                qnn_feeds, input_name_to_index));
```

`ASSERT_NO_FATAL_FAILURE` propagates the helper's `ASSERT_NE` failure as a fatal failure in the caller, preserving original abort-on-null-allocation behavior.

- [ ] **Step 4: Compile**

Run: `cmake --build "build/windows-arm64/Release" --config Release --target onnxruntime_provider_test --parallel`
Expected: compiles clean.

- [ ] **Step 5: Diff review**

Run: `git --no-pager diff -- onnxruntime/test/providers/qnn/group_query_attention_test.cc`
Confirm: the copy-loop body inside `MakeSharedMemoryFeeds` is identical to the deleted lines except `FeedCopy`→`GQAFeedCopy` and `helper.feeds_`→`source_feeds`. No logic changed.

- [ ] **Step 6: Commit**

```bash
git add onnxruntime/test/providers/qnn/group_query_attention_test.cc
git commit -m "Refactor GQA test: extract MakeSharedMemoryFeeds helper

Lift FeedCopy to file scope as GQAFeedCopy and move RunGQATest's per-input
shared-memory copy loop into MakeSharedMemoryFeeds. Behavior-preserving;
verified by compile + diff (tests are DISABLED, no runtime check).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Extract `AliasPresentToPastAndRun`

**Files:**
- Modify: `onnxruntime/test/providers/qnn/group_query_attention_test.cc`

This covers the output-buffer aliasing (lines 274-284), the `Ort::GetApi().Run` call (286-294), and the output collection (296-312). These share `qnn_feeds` / `input_name_to_index` and must move together.

- [ ] **Step 1: Add the helper**

Insert after `MakeSharedMemoryFeeds`. Verbatim extraction of lines 274-312:

```cpp
// Aliases present_key/present_value outputs onto the past_key/past_value shared-memory
// buffers (in-place KV cache), runs the QNN session, then collects output views. Outputs
// that are not aliased are owned by owned_qnn_outputs. Verbatim extraction from RunGQATest.
static void AliasPresentToPastAndRun(Ort::Session& qnn_session,
                                     const std::vector<std::string>& input_names,
                                     const std::vector<std::string>& output_names,
                                     const std::vector<const char*>& input_names_cstr,
                                     const std::vector<const char*>& output_names_cstr,
                                     std::vector<GQAFeedCopy>& qnn_feeds,
                                     const std::unordered_map<std::string, size_t>& input_name_to_index,
                                     std::vector<Ort::Value>& owned_qnn_outputs,
                                     std::vector<const Ort::Value*>& qnn_outputs) {
  std::vector<const OrtValue*> qnn_input_values;
  qnn_input_values.reserve(qnn_feeds.size());
  for (const auto& qnn_feed : qnn_feeds) {
    qnn_input_values.push_back(qnn_feed.value);
  }

  std::vector<OrtValue*> qnn_output_values(output_names.size(), nullptr);
  const auto past_key_input = input_name_to_index.find("past_key");
  const auto past_value_input = input_name_to_index.find("past_value");
  for (size_t i = 0; i < output_names.size(); i++) {
    // Make present_key and present_value use the same buffer as past_key and past_value.
    if (output_names[i] == "present_key" && past_key_input != input_name_to_index.end()) {
      qnn_output_values[i] = qnn_feeds[past_key_input->second].value;
    } else if (output_names[i] == "present_value" && past_value_input != input_name_to_index.end()) {
      qnn_output_values[i] = qnn_feeds[past_value_input->second].value;
    }
  }

  Ort::RunOptions qnn_run_options;
  ASSERT_ORTSTATUS_OK(Ort::GetApi().Run(qnn_session,
                                        qnn_run_options,
                                        input_names_cstr.data(),
                                        qnn_input_values.data(),
                                        qnn_input_values.size(),
                                        output_names_cstr.data(),
                                        output_names_cstr.size(),
                                        qnn_output_values.data()));

  owned_qnn_outputs.reserve(output_names.size());
  qnn_outputs.reserve(output_names.size());
  for (size_t i = 0; i < output_names.size(); i++) {
    if (output_names[i] == "present_key" && past_key_input != input_name_to_index.end()) {
      ASSERT_EQ(qnn_output_values[i], static_cast<OrtValue*>(qnn_feeds[past_key_input->second].value));
      qnn_outputs.push_back(&qnn_feeds[past_key_input->second].value);
    } else if (output_names[i] == "present_value" && past_value_input != input_name_to_index.end()) {
      ASSERT_EQ(qnn_output_values[i], static_cast<OrtValue*>(qnn_feeds[past_value_input->second].value));
      qnn_outputs.push_back(&qnn_feeds[past_value_input->second].value);
    } else {
      ASSERT_NE(qnn_output_values[i], nullptr);
      owned_qnn_outputs.emplace_back(qnn_output_values[i]);
      qnn_outputs.push_back(&owned_qnn_outputs.back());
    }
  }
}
```

Note: `input_names` is unused inside the body but kept in the signature for symmetry with the caller's naming; if the compiler warns about unused params (this repo builds `-Werror` style), drop the `input_names` parameter. Verify at Step 3.

- [ ] **Step 2: Replace lines 268-312 in `RunGQATest` with a call**

```cpp
  std::vector<Ort::Value> owned_qnn_outputs;
  std::vector<const Ort::Value*> qnn_outputs;
  ASSERT_NO_FATAL_FAILURE(AliasPresentToPastAndRun(qnn_session, input_names, output_names,
                                                   input_names_cstr, output_names_cstr,
                                                   qnn_feeds, input_name_to_index,
                                                   owned_qnn_outputs, qnn_outputs));
```

- [ ] **Step 3: Compile**

Run: `cmake --build "build/windows-arm64/Release" --config Release --target onnxruntime_provider_test --parallel`
Expected: compiles clean. If an unused-parameter error fires on `input_names`, remove that parameter from both the signature and the call, then rebuild.

- [ ] **Step 4: Diff review**

Run: `git --no-pager diff -- onnxruntime/test/providers/qnn/group_query_attention_test.cc`
Confirm the moved block is identical to original lines 268-312 (modulo the `qnn_outputs`/`owned_qnn_outputs` now being caller-declared out-params).

- [ ] **Step 5: Commit**

```bash
git add onnxruntime/test/providers/qnn/group_query_attention_test.cc
git commit -m "Refactor GQA test: extract AliasPresentToPastAndRun helper

Move present->past KV-cache output aliasing, the QNN Run call, and output
collection out of RunGQATest. Behavior-preserving; verified by compile +
diff (tests are DISABLED, no runtime check).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Extract `CompareQnnVsCpuOutputs`

**Files:**
- Modify: `onnxruntime/test/providers/qnn/group_query_attention_test.cc`

Covers the CPU run + comparison (current lines 314-324).

- [ ] **Step 1: Add the helper**

Insert after `AliasPresentToPastAndRun`:

```cpp
// Runs the model on the CPU EP and compares each QNN output against it. Verbatim
// extraction of RunGQATest's final compare block.
static void CompareQnnVsCpuOutputs(Ort::Session& cpu_session,
                                   const std::unordered_map<std::string, Ort::Value>& cpu_feeds,
                                   const std::vector<std::string>& output_names,
                                   const std::vector<const Ort::Value*>& qnn_outputs,
                                   float fp32_abs_err) {
  Ort::RunOptions cpu_run_options;
  std::vector<Ort::Value> cpu_outputs;
  // The CPU EP can do GQA without buffer sharing, so we can just use RunWithEP
  RunWithEP(cpu_session, cpu_run_options, cpu_feeds, cpu_outputs);

  // Check QNN outputs against CPU
  ASSERT_EQ(cpu_outputs.size(), output_names.size());
  ASSERT_EQ(qnn_outputs.size(), output_names.size());
  for (size_t i = 0; i < output_names.size(); i++) {
    VerifyOutput(output_names[i], cpu_outputs[i], *qnn_outputs[i], ElementwiseAbsoluteVerifier{fp32_abs_err});
  }
}
```

- [ ] **Step 2: Replace lines 314-324 in `RunGQATest` with a call**

```cpp
  ASSERT_NO_FATAL_FAILURE(CompareQnnVsCpuOutputs(cpu_session, helper.feeds_, output_names,
                                                 qnn_outputs, fp32_abs_err));
```

The body of `RunGQATest` now ends here (the closing brace at old line 325 remains).

- [ ] **Step 3: Compile**

Run: `cmake --build "build/windows-arm64/Release" --config Release --target onnxruntime_provider_test --parallel`
Expected: compiles clean.

- [ ] **Step 4: Diff review**

Run: `git --no-pager diff -- onnxruntime/test/providers/qnn/group_query_attention_test.cc`
Confirm the moved block equals original lines 314-324 (with `helper.feeds_`→`cpu_feeds`).

- [ ] **Step 5: Commit**

```bash
git add onnxruntime/test/providers/qnn/group_query_attention_test.cc
git commit -m "Refactor GQA test: extract CompareQnnVsCpuOutputs helper

RunGQATest is now a clear sequence: build model, create sessions, make
shared-memory feeds, alias+run, compare. Behavior-preserving; verified by
compile + diff (tests are DISABLED, no runtime check).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Align section banners with sibling test files

**Files:**
- Modify: `onnxruntime/test/providers/qnn/group_query_attention_test.cc`

Sibling files (e.g. `conv_test.cc`, `lstm_test.cc`) group the file into banner-delimited sections. GQA has only one banner today (`// === op_affinity ... ===`). Add matching banners; text-only, no code movement.

- [ ] **Step 1: Add a banner above the shared helpers**

Immediately before the `GQAFeedCopy` struct (added in Task 1), insert:

```cpp
// === Shared model builder + backend-agnostic driver (used by HTP now, GPU after rebase) ===
```

- [ ] **Step 2: Add a banner above the HTP-specific drivers**

Immediately after the `#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)` line (current line 332) and its existing block comment, before `RunHTPPackedGQATest`, insert:

```cpp
// === HTP compact drivers (packed / unpacked QKV) ===
```

- [ ] **Step 3: Add a banner above the inference test cases**

Immediately before `TEST_F(QnnHTPBackendTests, DISABLED_GroupQueryAttention_Basic_FP32)` (current line 505), insert:

```cpp
// === HTP inference tests (QNN vs CPU) ===
```

- [ ] **Step 4: Compile**

Run: `cmake --build "build/windows-arm64/Release" --config Release --target onnxruntime_provider_test --parallel`
Expected: compiles clean (comments only).

- [ ] **Step 5: Commit**

```bash
git add onnxruntime/test/providers/qnn/group_query_attention_test.cc
git commit -m "Refactor GQA test: add section banners to match sibling files

Group the file into shared-driver / HTP-drivers / HTP-tests / op_affinity
sections with // === ... === banners, matching conv_test.cc and lstm_test.cc.
Comments only.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Extract duplicated logic → Tasks 1-3 (MakeSharedMemoryFeeds, AliasPresentToPastAndRun, CompareQnnVsCpuOutputs). ✓
- Align sibling style (banners) → Task 4. ✓
- 23-param BuildGQATestCase stays as-is → no task touches it. ✓
- In-place KV-cache logic unchanged → moved verbatim, not rewritten. ✓
- Compile + diff verification (no runtime) → every task Steps compile+diff. ✓

**Placeholder scan:** No TBD/TODO; all helper bodies are full code. ✓

**Type consistency:** `GQAFeedCopy` defined in Task 1 and used in Tasks 1-2. `MakeSharedMemoryFeeds`, `AliasPresentToPastAndRun`, `CompareQnnVsCpuOutputs` signatures are consistent between definition and call site. `qnn_feeds` / `input_name_to_index` / `owned_qnn_outputs` / `qnn_outputs` are declared in `RunGQATest` before the calls that fill them. ✓

**Known risk:** `driver naming alignment` from the spec is minimal (banners only); the spec called driver renaming "trivial where consistent" — current `RunHTPPackedGQATest`/`RunHTPUnpackedGQATest` names are already clear, so no rename is forced (avoids churn on 22 call sites). This is a deliberate YAGNI call.
