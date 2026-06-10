# LWD: ORT QNN-EP Conv Block Quantization (BQ and LPBQ) Support

---

## General Information

| Field | Value |
|-------|-------|
| **Feature Lead (FR owner)** | TBD |
| **FR Number** | TBD |
| **FR Description** | Enable 4-bit block-quantized weight Conv on QNN HTP via two kernels: BQ (BW_FLOAT_BLOCK weight + FP16 activation) and LPBQ (BLOCKWISE_EXPANSION weight + INT16 symmetric activation) |
| **JIRA (Epic) Number** | TBD |

| POC Type | Team |
|----------|------|
| MANDATORY | QNN EP Dev Lead |
| MANDATORY | QNN QA |
| OPTIONAL SIGN-OFF | QAIRT HTP Kernel Team (confirm Conv BQ/LPBQ kernel constraints) |
| INFORMATIONAL | ONNX Runtime Core Team (NodeUnit block_size gap, informational only) |

---

## 1. Feature Overview

### Problem Statement

- **Deploy** ONNX models with 4-bit block-quantized Conv weights on Snapdragon HTP. QNN EP currently supports only per-tensor and per-channel (INT8/INT4 axis) quantization for Conv weights; the BQ and LPBQ QNN kernels are unreachable.
- **Use** both QNN HTP BQ kernel (BW_FLOAT_BLOCK weight + FP16 activation) and LPBQ kernel (BLOCKWISE_EXPANSION weight + INT16 symmetric activation), which offer distinct power/accuracy trade-offs for model compression.
- **Unblock** workflows that produce ONNX Conv models with block-quantized weights (e.g., Olive GPTQ-style compression applied to convolutional layers).

### Execution Paths: JIT vs AOT

| Path | Behavior |
|------|----------|
| **JIT** | Full pattern recognition during `ComposeGraph()`. BQ/LPBQ Conv node groups are detected and translated to QNN Conv2d with the appropriate weight encoding. Inference runs on HTP. |
| **AOT Phase 1 (Context Generation)** | Same as JIT — pattern recognition occurs during `Compile()`. QNN context binary captures the BQ/LPBQ Conv2d graph. Context `.bin` is generated. |
| **AOT Phase 2 — Pure EPContext** | No `GetSupportedNodes()` or `ComposeGraph()` called. QNN graph is restored from context binary via `contextCreateFromBinary()`. Feature has no behavior; BQ/LPBQ Conv runs via the cached binary. |
| **AOT Phase 2 — Mixed EPContext** | Same as pure EPContext for the EPContext subgraph. Non-EPContext nodes (e.g., Conv not supported due to wrong quant format) fall back to CPU EP. |
| **Hybrid AOT (QAIRT-native CtxBin)** | ORT never calls `GetSupportedNodes()` or `ComposeGraph()`. No pattern recognition occurs in ORT. QAIRT tools may independently produce BQ/LPBQ Conv in the CtxBin; ORT loads it transparently. |

### Target Platforms

| Platform | HTP Available | CPU Applicable | File I/O Notes | Test Execution |
|----------|--------------|----------------|----------------|----------------|
| Linux x86_64 | Yes (simulator) | N/A (BQ/LPBQ HTP-only) | Standard POSIX | Local |
| Linux aarch64 | Yes (hardware) | N/A | Standard POSIX | QDC / on-device |
| Android aarch64 | Yes (hardware) | N/A | CWD may be non-writable | Archive build → push |
| Windows x86_64 | No | N/A | N/A | Not applicable |
| Windows ARM64 | Yes (hardware) | N/A | Backslash paths, file-mapped weights available | Local |

Cross-platform AOT (e.g., Linux x86_64 → Android aarch64): context generation runs on host with HTP backend (via simulator or `htp_arch`/`soc_model` config); context binary is deployed to target. This feature introduces no new file output, so cross-platform path handling is not a concern.

### Relationship to ORT Core Infrastructure

| ORT Core Feature | Location | What It Provides | Gap |
|------------------|----------|-----------------|-----|
| `NodeUnitIODef::QuantParam` | `onnxruntime/core/framework/node_unit.h` | Exposes `axis` and `scale`/`zero_point` node args to EPs | Does **not** expose `block_size`; EPs cannot read block quantization granularity through NodeUnit |
| QDQ Conv NodeUnit formation | `onnxruntime/core/optimizer/qdq_transformer/` | Groups `DQ → Conv → Q` into a QDQ NodeUnit | Block-size DQ weight nodes for Conv are not recognized; no QDQ NodeUnit is formed for this pattern |
| `DQ MatMulNBits Fusion` | `onnxruntime/core/optimizer/dq_matmulnbits_fusion.cc` | Fuses `DQ(INT4, block_size) → MatMul` into `MatMulNBits` | Conv equivalent does not exist; block-quant Conv passes through unmodified |
| QDQ propagation | `qdq_propagation.cc` | Propagates `block_size` attribute through Q/DQ pairs | Works correctly; block_size is preserved on DQ nodes reaching QNN EP |

**Conclusion:** ORT core does not expose `block_size` through NodeUnit, and does not form a QDQ NodeUnit for block-quant Conv. However, QNN EP's existing EP-internal node group (IQnnNodeGroup) infrastructure can directly access ONNX node attributes (bypassing NodeUnit) to read `block_size`. No ORT core changes are required.

### Goal

1. **Accurate** — BQ path maps ONNX per-block float scales directly to `QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK`; LPBQ path maps two-level scales to `QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION` without precision loss.
2. **Reusable** — leverages existing QNN EP infrastructure: `QnnQuantParamsWrapper` (BW_FLOAT_BLOCK and BLOCKWISE_EXPANSION constructors), `LowPowerBlockQuantizeData()`, and the IQnnNodeGroup pattern from `lpbqmatmul_fusion`.
3. **Self-contained** — no ORT core changes; all new code lives under `onnxruntime/core/providers/qnn/`.
4. **Phased** — Phase 1 delivers BQ (BW_FLOAT_BLOCK + FP16); Phase 2 delivers LPBQ (BLOCKWISE_EXPANSION + INT16). Each phase is independently shippable.
5. **Compatible** — existing per-tensor and per-channel Conv handling is unchanged; BQ/LPBQ fusion only activates when the specific node patterns are present.

### High-Level Approach

**Phase 1 (BQ) — JIT/AOT-P1:**
1. `OrtConvNodeGroupSelector` already claims the `DQ(INT4, block_size) → Conv ← DQ(INT16)` group during `GetCapability()` (INT4 weight and INT16 activation are already accepted). No selector change is needed.
2. In `conv_op_builder.cc`, detect `block_size` by reading the attribute directly from the weight DQ ONNX node (bypassing `QuantParam`). When present, route to BW_FLOAT_BLOCK encoding construction instead of the existing per-channel path.
3. Read per-block float scales from the weight DQ initializer; construct `QnnQuantParamsWrapper` with `QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK`.
4. For the INT16 activation: insert a QNN `Convert` op (INT16 → FP16) before the Conv node; build QNN Conv2d with FP16 input/output.

**Phase 2 (LPBQ) — JIT/AOT-P1:**
1. Detect the LPBQ Conv pattern: `Scale_DQL(INT8 per-block int scales, per-channel float scales) → Weight_DQL(INT4 weight) → Conv ← DQ(INT16 symmetric activation) → Q(INT16 output)` using a new `LPBQConvFusion::TryFusion()`, mirroring `lpbqmatmul_fusion`.
2. Extract per-channel float scales and per-block int scales; adapt `LowPowerBlockQuantizeData()` for Conv OIHW weight layout.
3. Construct `QnnQuantParamsWrapper` with `QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION`.
4. Build QNN Conv2d with LPBQ weight encoding and INT16 symmetric input/output tensors.

**AOT Phase 2 (both BQ and LPBQ):** No behavior — QNN context binary is loaded directly via `contextCreateFromBinary()`. Feature pattern matching is a no-op.

### Constraints

- **No ORT core changes.** All code lives under `onnxruntime/core/providers/qnn/`.
- **HTP backend only.** `QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK` and `QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION` are not supported on CPU or GPU backends. BQ/LPBQ Conv fusion is gated on `IsNpuBackend()`.
- **`block_size` must be a power of 2 and ≥ 16** (QNN HTP requirement). For INT4 with HTP, `block_size` must additionally be a multiple of 8.
- **Activation quantization:** BQ path requires per-tensor symmetric INT16 (or FP16) activation; LPBQ path requires per-tensor symmetric INT16 activation (zero_point = 0). Asymmetric or per-channel activation is not supported by the BQ/LPBQ Conv kernels.
- **Weight layout transposition:** ONNX Conv weight is OIHW `[OC, IC, H, W]`; QNN expects HWCN `[H, W, IC, OC]`. Block quantization axis adjustment is required: ONNX axis=0 (OC) maps to QNN axis=3 after transposition.
- **Depthwise Conv:** out of scope for this LWD. Depthwise Conv has `group = IC = OC` and different weight layouts; a separate feasibility study is needed.
- **AOT Phase 2 constraint:** Pattern recognition does not occur; this is by design and consistent with all existing QNN EP fusion features.

### QNN-EP Documentation Impact

- `docs/QNN-ExecutionProvider.md`: New section describing BQ and LPBQ Conv support, HTP-only constraint, required ONNX graph patterns, and block_size constraints.

---

## 2. Scoping

### Proposal 1: BQ via `conv_op_builder.cc` extension + LPBQ via new `lpbqconv_fusion` (Recommended)

**BQ (Phase 1):** `OrtConvNodeGroupSelector` already accepts INT4 weight and INT16 activation in `GetCapability()` — no selector change is needed. The BQ logic lives entirely in `conv_op_builder.cc`: when the weight DQ node carries a `block_size` attribute, the op builder reads the attribute directly (bypassing `QuantParam`), constructs `QnnQuantParamsWrapper(BW_FLOAT_BLOCK)`, and inserts the INT16→FP16 Cast. No new fusion class is required.

**LPBQ (Phase 2):** The two-level DQL pattern (`Scale_DQL → Weight_DQL → Conv`) spans multiple NodeUnits and cannot be handled inside the op builder alone. A new `LowPowerBlockQuantizedConvFusion` IQnnNodeGroup is needed, mirroring `LowPowerBlockQuantizedMatMulFusion`.

**Scope:**
- New files: `lpbqconv_fusion.h/cc` only
- Modified files: `conv_op_builder.cc` (BQ path), `qnn_node_group.cc` (register LPBQ TryFusion)
- Reused: `QnnQuantParamsWrapper` BW_FLOAT_BLOCK/BLOCKWISE_EXPANSION constructors, `LowPowerBlockQuantizeData()`

**Pros:**
- BQ path follows the natural op-builder extension pattern; no extra class or file.
- LPBQ follows the established IQnnNodeGroup pattern for multi-NodeUnit fusions.
- No ORT core changes; each phase is independently shippable.

**Cons:**
- `conv_op_builder.cc` gains a new branch guarded by `block_size` presence; slightly increases its complexity.

---

### Proposal 2: Both BQ and LPBQ via new IQnnNodeGroup classes

Create `bqconv_fusion.h/cc` (Phase 1) and `lpbqconv_fusion.h/cc` (Phase 2) as separate `IQnnNodeGroup` subclasses, handling both pattern detection and QNN graph building outside `conv_op_builder.cc`.

**Pros:**
- BQ and LPBQ are uniformly structured as node group fusions; `conv_op_builder.cc` stays unchanged.

**Cons:**
- A new `bqconv_fusion` class adds a file and indirection for logic that can be handled in-line in the existing op builder.
- The standard Conv QDQ NodeUnit is already formed by `OrtConvNodeGroupSelector`; intercepting it via a separate fusion class means duplicating the Conv build logic that already exists in the op builder.

---

**Decision:** Adopt **Proposal 1**. BQ logic belongs in `conv_op_builder.cc` because the node group is already a standard Conv QDQ NodeUnit — no multi-NodeUnit fusion is needed. LPBQ requires a new fusion class because its Scale_DQL sits outside the standard NodeUnit boundary. This keeps each component in its natural layer.

---

## 3. Use Cases

### Use Case 1: BQ Conv Inference (JIT — Happy Path)

**Applicable Paths:** JIT, AOT-P1

| Field | Description |
|-------|-------------|
| **Title** | BQ Conv JIT inference on HTP |
| **Description** | User runs ORT inference on an HTP device with a model containing a Conv node whose weight is 4-bit block-quantized (standard ONNX block-quant DQ with `block_size`) and whose activation is 16-bit quantized (INT16 symmetric QDQ). |
| **Actor** | ORT user / developer deploying a compressed model |
| **Precondition** | ONNX model contains the pattern `DQ(INT4, block_size) → Conv ← DQ(INT16, per-tensor symmetric)`. QNN EP is configured with HTP backend. |
| **Successful Postcondition** | QNN EP recognizes the BQ Conv pattern, builds QNN Conv2d with `QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK` weight and FP16 input/output (with an INT16→FP16 Cast node preceding Conv), and produces correct inference results on HTP. |
| **What the actor sees and does** | 1. User creates ORT `InferenceSession` with QNN EP (HTP backend). 2. Session initialization completes without error. 3. `session.Run()` produces float outputs matching expected accuracy. |
| **How it works** | Inside `conv_op_builder.cc::ProcessConv2D3DInputs()`, the weight DQ node is inspected for a `block_size` attribute. When found, the builder reads per-block float scales from the initializer, constructs `QnnQuantParamsWrapper(BW_FLOAT_BLOCK)`, inserts a QNN Convert(INT16→FP16) node for the activation tensor, and emits QNN Conv2d. |
| **Points to note** | The QNN Conv2d node receives FP16 activation (after Convert insertion) even though ONNX model has INT16 QDQ activation. The Convert node is a QNN-internal node not visible in the ONNX graph. `block_size` must be a power-of-2 multiple of 8 for INT4 on HTP. |
| **Requirements** | Phase 1: `conv_op_builder.cc` BQ branch, `QnnQuantParamsWrapper` BW_FLOAT_BLOCK, Convert insertion logic |

---

### Use Case 2: LPBQ Conv Inference (JIT — Happy Path)

**Applicable Paths:** JIT, AOT-P1

| Field | Description |
|-------|-------------|
| **Title** | LPBQ Conv JIT inference on HTP |
| **Description** | User runs ORT inference on a model whose Conv weight uses the two-level LPBQ quantization scheme (per-channel float × per-block int scales). Activation is INT16 symmetric. |
| **Actor** | ORT user / developer deploying an LPBQ-quantized model |
| **Precondition** | ONNX model contains the LPBQ pattern: `Scale_DQL(INT8 per-block scales, per-channel float scale) → Weight_DQL(INT4 weight, per-block float scale from Scale_DQL) → Conv ← DQ(INT16 symmetric activation)`. QNN EP HTP backend configured. |
| **Successful Postcondition** | QNN EP recognizes the two-level LPBQ scale pattern, builds QNN Conv2d with `QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION` weight and INT16 symmetric activation, and produces correct inference results on HTP. |
| **What the actor sees and does** | 1. User creates ORT `InferenceSession` with QNN EP (HTP backend). 2. Session initialization completes. 3. `session.Run()` produces correct results. |
| **How it works** | `LowPowerBlockQuantizedConvFusion::TryFusion()` detects the nested DQL pattern (per-channel float scales dequantizing per-block int scales → per-block float scales used as Conv weight DQ scale). Extracts per-channel float scales and per-block int scales, adapts `LowPowerBlockQuantizeData()` for Conv OIHW weight layout, constructs `QnnQuantParamsWrapper(BLOCKWISE_EXPANSION)`. INT16 activation is passed directly to QNN as quantized tensor (no Convert needed). |
| **Points to note** | The Scale_DQL output is a derived (non-initializer) tensor; the fusion must verify that both the Scale_DQL data and the Weight_DQL data are constant initializers. LPBQ does not support dynamic shapes. LPBQ activation must be symmetric (zero_point = 0). |
| **Requirements** | Phase 2: `lpbqconv_fusion.h/cc`, `LowPowerBlockQuantizeData()` adaptation for Conv, `QnnQuantParamsWrapper` BLOCKWISE_EXPANSION |

---

### Use Case 3: AOT Context Generation and Loading (BQ and LPBQ)

**Applicable Paths:** AOT-P1 (context generation), AOT-P2 (context loading)

| Field | Description |
|-------|-------------|
| **Title** | AOT round-trip: BQ/LPBQ Conv context generation and loading |
| **Description** | User generates a QNN context binary on host (Phase 1) containing BQ or LPBQ Conv nodes, then loads the binary on-device (Phase 2). |
| **Actor** | ORT developer / deployment pipeline |
| **Precondition** | AOT-P1: ONNX model with BQ/LPBQ Conv pattern, `ep.context_enable=1`, HTP backend. AOT-P2: EPContext ONNX model referencing the context `.bin`. |
| **Successful Postcondition** | AOT-P1: `session.Run()` generates context `.bin` containing compiled BQ/LPBQ Conv2d. AOT-P2: `session.Run()` loads context binary and runs inference correctly, with no pattern recognition or BQ/LPBQ fusion code executed. |
| **What the actor sees and does** | P1: Run `InferenceSession` with `ep.context_enable=1`; a `.onnx` + `.bin` file pair is generated. P2: Run `InferenceSession` on the EPContext `.onnx`; inference runs. |
| **How it works** | AOT-P1 follows JIT path for pattern recognition; the resulting QNN graph (with BQ/LPBQ Conv) is serialized to context binary. AOT-P2 calls `contextCreateFromBinary()` directly; the BQ branch in `conv_op_builder.cc` and `LowPowerBlockQuantizedConvFusion` are never called. |
| **Points to note** | Cross-platform AOT (e.g., Linux x86_64 generate → Android aarch64 deploy) is transparent to this feature since no new file output is introduced. AOT-P2 has zero overhead from this feature. |
| **Requirements** | No AOT-specific code change; AOT-P1 reuses JIT path; AOT-P2 is a no-op for this feature. |

---

### Use Case 4: Unsupported Quantization Format — CPU Fallback

**Applicable Paths:** JIT, AOT-P1

| Field | Description |
|-------|-------------|
| **Title** | BQ/LPBQ Conv not matched — falls back to CPU EP |
| **Description** | The Conv node has block-quantized weights but the pattern does not match (e.g., wrong `block_size`, non-symmetric activation, depthwise Conv, or CPU backend configured). QNN EP declines the node; it falls back to CPU EP. |
| **Actor** | ORT user with a partially compatible model |
| **Precondition** | Conv node has `block_size` but: (a) `block_size` is not a valid multiple, (b) activation zero_point ≠ 0, (c) backend is not HTP, or (d) the LPBQ two-level pattern is missing Scale_DQL. |
| **Successful Postcondition** | The BQ branch in `conv_op_builder.cc` and `LowPowerBlockQuantizedConvFusion::TryFusion()` both decline. Conv node falls to CPU EP (or triggers a warning). Model runs correctly at lower efficiency. |
| **What the actor sees and does** | `InferenceSession` creation succeeds. A warning log line is emitted noting the unsupported quantization format. Inference produces correct results via CPU EP. |
| **How it works** | In `conv_op_builder.cc`, the BQ branch validates `block_size` constraints and returns an error status (falling back to the default per-channel path, which then falls to CPU for INT4+block_size). `LowPowerBlockQuantizedConvFusion::TryFusion()` validates all LPBQ constraints and returns `nullptr` on failure. |
| **Points to note** | The user should be informed via log message which constraint failed to aid debugging. |
| **Requirements** | Validation in `TryFusion()` with informative log messages |

---

### Use Case 5: Non-Block-Quant Conv — Existing Path Unchanged

**Applicable Paths:** JIT, AOT-P1, AOT-P2

| Field | Description |
|-------|-------------|
| **Title** | Standard per-channel INT8 or INT4 Conv — existing behavior unchanged |
| **Description** | The ONNX model contains a Conv with standard per-channel quantized weight (no `block_size` attribute) and INT8/INT4 activation. This is the existing supported case. |
| **Actor** | Any ORT user with an existing QDQ INT8/INT4 Conv model |
| **Precondition** | Conv weight DQ node has no `block_size` attribute. |
| **Successful Postcondition** | The BQ branch in `conv_op_builder.cc` is not triggered (no `block_size`). `LowPowerBlockQuantizedConvFusion::TryFusion()` returns `nullptr`. Existing Conv QDQ handling processes the node unchanged. |
| **What the actor sees and does** | No behavioral change from today. |
| **How it works** | The BQ branch in `conv_op_builder.cc` is guarded by `block_size` presence; absent `block_size` takes the existing per-channel/per-tensor path. `LowPowerBlockQuantizedConvFusion::TryFusion()` requires the Scale_DQL pattern and returns early without it. |
| **Points to note** | Zero regression risk for existing Conv models. |
| **Requirements** | Early-exit guard in both new TryFusion methods |

---

### Use Case 6: Mixed EPContext Model — Phase 2 Loads BQ/LPBQ Conv from Binary

**Applicable Paths:** AOT-P2 (Mixed EPContext)

| Field | Description |
|-------|-------------|
| **Title** | Mixed EPContext model with BQ/LPBQ Conv nodes |
| **Description** | An EPContext model produced by AOT-P1 contains some EPContext nodes (including compiled BQ/LPBQ Conv subgraphs) and some non-EPContext nodes (e.g., unsupported ops that fell back to CPU during context generation). |
| **Actor** | ORT deployment pipeline on target device |
| **Precondition** | EPContext ONNX model with both EPContext nodes (QNN compiled subgraph containing BQ/LPBQ Conv) and non-EPContext nodes. |
| **Successful Postcondition** | `PartitionCtxModel()` claims only EPContext nodes; non-EPContext nodes go to CPU EP. `CompileContextModel()` → `contextCreateFromBinary()` restores BQ/LPBQ Conv from binary. No new pattern recognition; inference runs correctly. |
| **What the actor sees and does** | `InferenceSession` initializes without error. Mixed model inference produces correct results. |
| **How it works** | `GraphHasEpContextNode()` detects any EPContext node → whole model treated as context model. `PartitionCtxModel()` claims only EPContext subgraph. BQ branch and `LowPowerBlockQuantizedConvFusion` are never triggered. |
| **Points to note** | This UC confirms that BQ/LPBQ fusion code path has zero impact during AOT-P2 and mixed model loading. |
| **Requirements** | No new code; existing AOT-P2 path is sufficient |

---

## 4. Requirements

### 4.1 API

No new session options are introduced. Feature activation is automatic:
- **BQ path** is triggered inside `conv_op_builder.cc` when the Conv QDQ NodeUnit's weight DQ node carries a `block_size` attribute, the weight type is INT4/UINT4, and the backend is HTP. The op builder reads `block_size` directly from the DQ node's attribute map.
- **LPBQ path** is triggered when `LowPowerBlockQuantizedConvFusion::TryFusion()` detects the two-level DQL pattern (Scale_DQL feeding per-block float scales into Weight_DQL feeding the Conv weight), on an HTP backend.

Since BQ is detected inside the op builder and LPBQ is detected at the node-group level, both checks are independently scoped and cannot conflict.

### 4.2 Data Model

No new persistent data structures. The following existing types are reused:

- **`QnnQuantParamsWrapper`** (`qnn_quant_params_wrapper.h`): already has constructors for `QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK` (used by HTP MatMulNBits) and `QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION` (used by LPBQ MatMul/Gemm). New Conv-specific usage does not require new constructors; it requires correct block dimension mapping from ONNX OIHW to QNN HWCN layout.

- **Key data flow for BQ weight encoding:**
  - ONNX scale tensor shape: `[OC, num_blocks]` where `num_blocks = ceil(IC * H * W / block_size)`
  - QNN BW_FLOAT_BLOCK `blockSize` array (HWCN): `[block_h, block_w, block_c, 1]` encoding the block dimensions after transposition
  - Per-block float scales: flattened from `[OC, num_blocks]`

- **Key data flow for LPBQ weight encoding:**
  - `per_channel_float_scales`: 1D tensor `[OC]` from Scale_DQL's per-channel quant params
  - `per_block_int_scales`: 2D tensor `[OC, num_blocks_per_channel]` from Scale_DQL's input data
  - QNN `Qnn_BlockwiseExpansion_t`: `axis=3` (OC in HWCN), `scaleOffsets[OC]`, `blocksScale8[OC*num_blocks]`

### 4.3 Implementation Details

#### ORT Core Boundary

The following ORT core types are consumed but not modified:
- `onnxruntime/core/framework/node_unit.h` (`OrtNodeUnit`, `NodeUnitIODef`) — read-only access
- `onnxruntime/core/graph/graph_viewer.h` (`GraphViewer`) — read-only
- `onnxruntime/core/providers/qnn/ort_api.h` — wrapper headers, not modified

The following ORT core files are not modified:
- `onnxruntime/core/framework/node_unit.h/.cc`
- `onnxruntime/core/optimizer/qdq_transformer/`
- `onnxruntime/core/session/inference_session.cc`

#### Files to Create (QNN EP only)

| File | Purpose |
|------|---------|
| `onnxruntime/core/providers/qnn/builder/qnn_node_group/lpbqconv_fusion.h` | `LowPowerBlockQuantizedConvFusion` class declaration (Phase 2) |
| `onnxruntime/core/providers/qnn/builder/qnn_node_group/lpbqconv_fusion.cc` | `TryFusion()`, `IsSupported()`, `AddToModelBuilder()` implementation (Phase 2) |

#### Files to Modify (QNN EP only)

| File | Change |
|------|--------|
| `onnxruntime/core/providers/qnn/builder/opbuilder/conv_op_builder.cc` | **(Phase 1 — BQ)** In `ProcessConv2D3DInputs()`: detect `block_size` on weight DQ node; branch to BW_FLOAT_BLOCK encoding and INT16→FP16 Cast insertion |
| `onnxruntime/core/providers/qnn/builder/qnn_node_group/qnn_node_group.cc` | **(Phase 2)** Register `LowPowerBlockQuantizedConvFusion::TryFusion` in the `"Conv"` entry of the fusion map |
| `CMakeLists.txt` (QNN EP section) | Add `lpbqconv_fusion.cc` to source list |

#### Data Flow

```
JIT / AOT Phase 1 — BQ path (Phase 1):

[OrtConvNodeGroupSelector claims DQ(INT4,block_size)→Conv←DQ(INT16)→Q(INT16)]
       │
       ▼
conv_op_builder.cc :: ProcessConv2D3DInputs()
       │
       ├─ Read weight DQ node attrs directly: block_size present?
       │     NO  → existing per-tensor / per-channel path (unchanged)
       │     YES ↓
       ├─ ★ Read [OC, num_blocks] float scales from weight DQ initializer
       ├─ ★ Remap block dims: ONNX OIHW → QNN HWCN (blockSize[H,W,IC,1], axis=3)
       ├─ ★ Construct QnnQuantParamsWrapper(BW_FLOAT_BLOCK)
       ├─ ★ Insert QNN Convert(INT16→FP16) tensor for activation
       └─ Build QNN Conv2d(weight: BW_FLOAT_BLOCK, input: FP16, output: FP16)

JIT / AOT Phase 1 — LPBQ path (Phase 2):

[Scale_DQL, Weight_DQL, Conv are separate SingleNode NodeUnits]
       │
       ▼
LowPowerBlockQuantizedConvFusion::TryFusion()
       ├─ Detect Scale_DQL → Weight_DQL → Conv ← DQ(INT16 act) pattern
       ├─ ★ Extract per-channel float scales (Scale_DQL quant params)
       ├─ ★ Extract per-block int scales (Scale_DQL data initializer)
       ├─ ★ Adapt LowPowerBlockQuantizeData() for Conv OIHW layout
       ├─ ★ Construct QnnQuantParamsWrapper(BLOCKWISE_EXPANSION, axis=3 HWCN)
       └─ Build QNN Conv2d(weight: BLOCKWISE_EXPANSION, input: INT16, output: INT16)

AOT Phase 2 / Hybrid AOT:

[contextCreateFromBinary()]
       │
       ▼
[QNN Conv2d restored from binary — no feature behavior]
```

### 4.4 Documentation

- `docs/QNN-ExecutionProvider.md`: Add "4-bit Block-Quantized Conv (BQ and LPBQ)" subsection under the existing quantization section. Describe the two ONNX input patterns, HTP-only constraint, `block_size` requirements, and activation type requirements.

---

## 5. Test Plan

### 5.1 Dev Test Plan (L0)

All tests run via `onnxruntime_provider_test` with `--gtest_filter=Qnn*`.

| UC# | Scenario | Backend | Platform | Prerequisites | Test Name |
|-----|----------|---------|----------|---------------|-----------|
| UC1 | BQ Conv — INT4 weight block_size=32, INT16 act, per-tensor | HTP | Linux x86_64 (sim) | QAIRT SDK ≥ 2.45 | `QnnHTPBackendTests.ConvBQ_Int4BlockSize32Int16Act` |
| UC1 | BQ Conv — UINT4 weight, symmetric, FP16 act (direct) | HTP | Linux x86_64 (sim) | QAIRT SDK ≥ 2.45 | `QnnHTPBackendTests.ConvBQ_Uint4BlockSize64Fp16Act` |
| UC1 | BQ Conv — 3D Conv (1D kernel), block_size=16 | HTP | Linux x86_64 (sim) | QAIRT SDK ≥ 2.45 | `QnnHTPBackendTests.ConvBQ_1DKernelInt4BlockSize16` |
| UC2 | LPBQ Conv — INT4 two-level scale, INT16 symmetric act | HTP | Linux x86_64 (sim) | QAIRT SDK ≥ 2.45 | `QnnHTPBackendTests.ConvLPBQ_Int4TwoLevelScaleInt16Act` |
| UC2 | LPBQ Conv — varied block sizes per channel | HTP | Linux x86_64 (sim) | QAIRT SDK ≥ 2.45 | `QnnHTPBackendTests.ConvLPBQ_VariedBlockSizesPerChannel` |
| UC3 | AOT-P1: BQ Conv context generation succeeds | HTP | Linux x86_64 (sim) | `ep.context_enable=1` | `QnnHTPBackendTests.ConvBQ_AotContextGeneration` |
| UC3 | AOT-P2: Load BQ Conv context, inference correct | HTP | Linux x86_64 (sim) | Pre-generated context fixture | `QnnHTPBackendTests.ConvBQ_AotContextLoading` |
| UC4 | Fallback: block_size not power-of-2 → CPU EP | HTP | Linux x86_64 (sim) | — | `QnnHTPBackendTests.ConvBQ_InvalidBlockSizeFallback` |
| UC4 | Fallback: asymmetric INT16 act (zp≠0) → CPU EP | HTP | Linux x86_64 (sim) | — | `QnnHTPBackendTests.ConvBQ_AsymmetricActFallback` |
| UC5 | Standard per-channel INT8 Conv unchanged | HTP | Linux x86_64 (sim) | — | `QnnHTPBackendTests.ConvBQ_ExistingPerChannelInt8Unaffected` |
| UC6 | Mixed EPContext: BQ Conv in EPContext, other node CPU | HTP | Linux x86_64 (sim) | Mixed model fixture | `QnnHTPBackendTests.ConvBQ_MixedEPContextModel` |
| UC1 | BQ Conv accuracy: output diff < 1e-3 vs FP32 reference | HTP (hardware) | Linux aarch64 | On-device hardware | `QnnHTPBackendTests.ConvBQ_AccuracyOnDevice` |
| UC2 | LPBQ Conv accuracy: output diff < 1e-3 vs FP32 reference | HTP (hardware) | Linux aarch64 | On-device hardware | `QnnHTPBackendTests.ConvLPBQ_AccuracyOnDevice` |

### 5.2 Dev/QA Verification

**Build command:**
```bash
./build.sh --config RelWithDebInfo \
  --use_qnn \
  --qnn_home /path/to/qairt/2.45.x/qairt/2.45.x \
  --build_shared_lib \
  --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=ON \
  --skip_tests --parallel
```

**JIT path verification (BQ):**
```bash
./build/Linux/RelWithDebInfo/onnxruntime_provider_test \
  --gtest_filter="QnnHTPBackendTests.ConvBQ_Int4BlockSize32Int16Act"
```
Expected: Test passes; log shows `BQConvFusion` matched and built QNN Conv2d with `QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK`.

**JIT path verification (LPBQ):**
```bash
./build/Linux/RelWithDebInfo/onnxruntime_provider_test \
  --gtest_filter="QnnHTPBackendTests.ConvLPBQ_Int4TwoLevelScaleInt16Act"
```
Expected: Test passes; log shows `LPBQConvFusion` matched and built QNN Conv2d with `QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION`.

**AOT path verification (BQ):**
```bash
# Phase 1: generate context
./build/Linux/RelWithDebInfo/onnxruntime_provider_test \
  --gtest_filter="QnnHTPBackendTests.ConvBQ_AotContextGeneration"
# Verify: .onnx + .bin files produced; .bin size > 0

# Phase 2: load context
./build/Linux/RelWithDebInfo/onnxruntime_provider_test \
  --gtest_filter="QnnHTPBackendTests.ConvBQ_AotContextLoading"
# Verify: inference runs; no BQConvFusion log lines (fusion not triggered)
```

**Fallback verification:**
```bash
./build/Linux/RelWithDebInfo/onnxruntime_provider_test \
  --gtest_filter="QnnHTPBackendTests.ConvBQ_InvalidBlockSizeFallback"
```
Expected: Conv assigned to CPU EP; warning log message indicates unsupported `block_size`.

**Platform coverage matrix:**

| Verification Step | Linux x86_64 | Linux aarch64 | Android aarch64 | Windows ARM64 |
|-------------------|:------------:|:-------------:|:---------------:|:-------------:|
| BQ JIT (sim) | ✓ | — | — | — |
| LPBQ JIT (sim) | ✓ | — | — | — |
| AOT round-trip (sim) | ✓ | — | — | — |
| BQ JIT (hardware) | — | ✓ | ✓ | ✓ |
| LPBQ JIT (hardware) | — | ✓ | ✓ | ✓ |
| AOT round-trip (hardware) | — | ✓ | ✓ | ✓ |
| Fallback / regression | ✓ | — | — | — |

### 5.3 QA Test Plan (L2 & L4)

| Level | Test | Scope |
|-------|------|-------|
| L2 | BQ Conv end-to-end model inference on Android aarch64 | Real HTP hardware, FP32 accuracy baseline comparison |
| L2 | LPBQ Conv end-to-end model inference on Android aarch64 | Real HTP hardware, INT16 accuracy comparison |
| L2 | AOT cross-platform: Linux x86_64 generate → Android aarch64 deploy | BQ Conv context round-trip |
| L2 | AOT cross-platform: Linux x86_64 generate → Windows ARM64 deploy | BQ Conv context round-trip |
| L2 | Regression: existing INT8/INT4 per-channel Conv models unaffected | All platforms |
| L4 | Full model smoke test with BQ-quantized ResNet/MobileNet variant on HTP | Production-scale model |
| L4 | Full model smoke test with LPBQ-quantized model on HTP | Production-scale model |

### Compatibility

- **No ORT core changes** — no risk of breaking other EPs or upstream merge conflicts.
- **Existing per-tensor/per-channel Conv handling unchanged** — the BQ branch in `conv_op_builder.cc` is entered only when `block_size` is present; `LowPowerBlockQuantizedConvFusion::TryFusion()` requires the Scale_DQL pattern and declines otherwise.
- **Context cache compatibility** — no change to context binary format; AOT-P2 and Hybrid AOT paths are unaffected.
- **Mixed EPContext models** — Phase 2 (context loading) has identical behavior to pure EPContext; no regression for mixed models.
- **All target platforms** — new fusions are HTP-only (`IsNpuBackend()` gate); no effect on CPU-only or non-HTP deployments.
- **File-mapped weights (Windows HTP)**, **ETW tracing (Windows)**, **weight sharing** — no interaction with these platform-specific features.

---

## 6. Effort and Timeline Estimate

### Component Milestones

| Milestone | Component | Effort Estimate | JIRA |
|-----------|-----------|-----------------|------|
| #0 | Survey and Feature Design (LWD) | 1 week | TBD |
| #1 | **Phase 1 — BQ in `conv_op_builder.cc`**: detect `block_size` on weight DQ node, build BW_FLOAT_BLOCK encoding with OIHW→HWCN block-dim remapping, insert INT16→FP16 Convert for activation | 1 week | TBD |
| #2 | **Phase 1 — Unit tests** for BQ Conv: 7 L0 test cases, AOT round-trip test, fallback test | 0.5 weeks | TBD |
| #3 | **Phase 2 — `LowPowerBlockQuantizedConvFusion`** (`lpbqconv_fusion.h/cc`): two-level DQL pattern detection, `LowPowerBlockQuantizeData()` adaptation for OIHW layout, BLOCKWISE_EXPANSION encoding, INT16 symmetric activation | 1.5 weeks | TBD |
| #4 | **Phase 2 — Unit tests** for LPBQ Conv: 5 L0 test cases, varied block size test, accuracy test | 0.5 weeks | TBD |
| #5 | **Integration: ConvTranspose** — extend Phase 1 BQ branch and Phase 2 LPBQ fusion to cover ConvTranspose (weight axis flip, IOHW→HWCO layout) | 1 week | TBD |
| #6 | **Platform integration tests**: on-device HTP tests (Linux aarch64 / Android), cross-platform AOT round-trips | 0.5 weeks | TBD |
| #7 | Documentation update (`QNN-ExecutionProvider.md`) | 0.5 days | TBD |
| **Total** | | **~5.5 weeks** | |

### Execution Phases

| Phase | Milestones | Description |
|-------|-----------|-------------|
| Phase 1 — BQ | #0, #1, #2 | BQ (BW_FLOAT_BLOCK + FP16) via `conv_op_builder.cc` extension |
| Phase 2 — LPBQ | #3, #4 | LPBQ (BLOCKWISE_EXPANSION + INT16) via new `LowPowerBlockQuantizedConvFusion` |
| Phase 3 — ConvTranspose + QA | #5, #6, #7 | Extend to ConvTranspose; platform/QA validation |

### Notes

- **Most complex milestone: #3 (LPBQ weight layout).** The `LowPowerBlockQuantizeData()` utility was written for MatMul's [K, N] weight layout; adapting it to Conv's OIHW→HWCN transposition with per-channel block quantization requires careful index arithmetic. This is the primary engineering risk.
- **#1 (BQ) is simpler** because the Conv node group is already formed by `OrtConvNodeGroupSelector`; the op builder only needs a new branch rather than a new class.
- **#3 (LPBQ) builds on #1** — block-dim remapping utilities from the BQ branch are reusable.
- **Future scope (not in this estimate):** Depthwise BQ/LPBQ Conv (requires separate feasibility study on QNN kernel availability), extension to GPU backend.

---

*Document generated: 2026-05-21. Requires review by QNN EP Dev Lead and QNN QA before implementation.*
