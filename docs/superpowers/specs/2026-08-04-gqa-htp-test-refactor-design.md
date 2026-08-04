# GQA HTP Test Refactor — Design

Date: 2026-08-04
File: `onnxruntime/test/providers/qnn/group_query_attention_test.cc`
Branch: `dev/chunghow/gqa-htp-github` (PR #556)

## Goal

Refactor the GQA HTP test file to (1) align with sibling QNN test-file conventions and
(2) extract duplicated setup logic — without changing test behavior.

## Hard constraints

1. **No execution-based verification available.** All 22 `QnnHTPBackendTests.GroupQueryAttention_*`
   cases are currently `DISABLED_` (GQA on HTP needs QAIRT >= 2.49; CI SDK is 2.48.40). The refactor
   can only be validated by (a) the test target compiling and (b) line-by-line diff showing semantic
   equivalence. Therefore: **behavior-preserving changes only.**
2. **`RunGQATest`'s hand-written QNN-vs-CPU compare loop cannot move to the shared framework.**
   GQA needs present→past in-place KV-cache aliasing plus a shared-memory allocator, which
   `RunQnnModelTest` / `TestQDQModelAccuracy` (in `qnn_test_utils.h`) do not support. This is a real
   limitation, not a style gap. The loop stays; it only gets split into named helpers.

## Scope (conservative)

### A. Extract duplicated logic (in-file, `static` helpers)

`RunGQATest` (currently ~200 lines, lines 125–325) contains three extractable blocks. Each is moved
verbatim (logic unchanged) into a named file-local `static` helper, leaving `RunGQATest` as a clear
three-step flow:

- `MakeSharedMemoryFeeds(...)` — copy `helper.feeds_` into shared-memory allocator tensors
  (current lines ~242–266).
- `AliasPresentToPast(...)` — bind `present_key`/`present_value` outputs to the `past_key`/`past_value`
  buffers for in-place KV cache (current lines ~274–312, which today has two near-duplicate
  past_key/past_value branch blocks).
- `CompareQnnVsCpuOutputs(...)` — cross-EP output comparison (current lines ~320+).

### B. Align with sibling style

- **Naming:** siblings use `RunHtpQDQ<Op>OpTest`. GQA's `RunHTPPackedGQATest` /
  `RunHTPUnpackedGQATest` are already close; adjust only for consistency where trivial.
- **Section comments:** siblings use `// === <section> ===` banners. GQA already has one
  (`// === op_affinity ... ===`); add matching banners for the helper region, packed-driver region,
  and unpacked-driver region.

## Explicitly NOT doing (YAGNI + risk control)

- **`BuildGQATestCase`'s 23 parameters stay as-is.** Sibling `lstm_test.cc` uses the same long
  parameter list + `std::optional<std::reference_wrapper<TestInputDef<...>>>` idiom. Wrapping GQA's
  params in a struct would *diverge* from sibling style, contradicting goal (1). (Confirmed with user.)
- No change to the in-place KV-cache comparison logic.
- No change to the 22 test-case bodies (they only get updated call sites if a helper is renamed).
- No changes to the shared framework; no cross-file changes.

## Verification

- Incremental build of `onnxruntime_provider_test` compiles clean.
- Per-section diff review confirms semantic equivalence.
- Tests remain DISABLED, so no runtime verification is possible — this limitation is noted in the
  commit message.

## Out of scope / follow-up

- Re-enabling the tests (separate task, gated on CI upgrading to QAIRT >= 2.49).
- Any move toward the shared `RunQnnModelTest` framework (would require framework support for
  in-place KV-cache aliasing — a much larger change).
