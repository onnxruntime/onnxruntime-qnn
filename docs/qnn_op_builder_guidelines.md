# QNN EP Op Builder — Developer Guidelines

A checklist for implementing an op builder that translates a new ONNX operator to QNN ops. (For collapsing a **multi-op ONNX subgraph** into one QNN op, you want a node-group fusion instead — see [qnn_node_group_fusion_guidelines.md](qnn_node_group_fusion_guidelines.md). That's a different mechanism.)

All file paths are relative to the repo root.

## 0. Decide What You Actually Need to Build (do this first)

- **Check for an equivalent QNN op and reuse the existing builder.** Look in the `GetQnnOpType` map ([base_op_builder.h:131-232](../onnxruntime/core/providers/qnn/builder/opbuilder/base_op_builder.h#L131-L232)) — it's the single record of ONNX↔QNN equivalences (e.g. `SimplifiedLayerNormalization`→`QNN_OP_RMS_NORM`, `LeakyRelu`→`QNN_OP_PRELU`, `Upsample`→`QNN_OP_RESIZE`).
- Decision order: **reuse an existing builder → if new, do a 1:1 translation → decompose into multiple QNN nodes if no single equivalent exists → open the QDQ unit only when quant handling demands it.**
- If it's a clean 1:1 op with no attributes, don't write a class at all — register it with `CreateSimpleOpBuilder`.
- **One ONNX node → one-or-more QNN nodes is an op builder (this doc). Multiple ONNX nodes → one QNN op is a *fusion*** ([qnn_node_group_fusion_guidelines.md](qnn_node_group_fusion_guidelines.md)). Decomposition (§8) stays in op-builder territory — it is the op-builder fan-*out*, the mirror image of a fusion's fan-*in*.

## 0.5. What a Builder Operates On — `OrtNodeUnit`

- Every builder operates on an **`OrtNodeUnit`**, which is **either a single ONNX node or a fused QDQ group** (`Type::SingleNode | QDQGroup`, [ort_api.h:263-266](../onnxruntime/core/providers/qnn/ort_api.h#L263-L266)).
- `IsOpSupported` is invoked **once per NodeUnit that is not claimed by a fusion**, via `QnnNodeUnitWrapper::IsSupported` ([qnn_node_group.cc:49-58](../onnxruntime/core/providers/qnn/builder/qnn_node_group/qnn_node_group.cc#L49-L58)) — the single group type that delegates to op builders. A NodeUnit absorbed into an `IQnnNodeGroup` fusion bypasses op builders entirely (see [qnn_node_group_fusion_guidelines.md](qnn_node_group_fusion_guidelines.md) §4).
- The real build is symmetric: in Phase 2 / Compile, `QnnNodeUnitWrapper::AddToModelBuilder` calls your `AddToModelBuilder(..., do_op_validation=false)` ([qnn_node_group.cc:61-66](../onnxruntime/core/providers/qnn/builder/qnn_node_group/qnn_node_group.cc#L61-L66)).
- Primary accessors: `OpType()`, `Name()`, `Domain()`, `SinceVersion()`, `Index()`, `GetNode()` ([ort_api.h:277-284](../onnxruntime/core/providers/qnn/ort_api.h#L277-L284)).
- `node_unit.Inputs()` / `Outputs()` return `OrtNodeUnitIODef`s whose `quant_param` **already carries scale/zero-point/axis merged from the surrounding DQ/Q nodes** — you normally don't touch raw quant nodes. Reach into them only via `GetDQNodes()` / `GetQNodes()` ([ort_api.h:286-287](../onnxruntime/core/providers/qnn/ort_api.h#L286-L287)).

## 1. File & Setup

- Create `onnxruntime/core/providers/qnn/builder/opbuilder/<opname>_op_builder.cc` — lowercase, single `.cc`, no header.
- Put the builder class **and** the `CreateXxxOpBuilder` free function in that one file.
- No CMake edit needed — the directory is auto-globbed; just re-run configure.

## 2. Class Definition

- Inherit from `BaseOpBuilder`; pass a unique builder-type string to the constructor.
- Add `ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MyOpBuilder)` and mark overrides `ORT_MUST_USE_RESULT`.
- **Never override `AddToModelBuilder` — it is `final`.** Override only the hooks below.
- **Builders are stateless singletons** — one instance is shared across every op type it's registered for and across all graphs/threads (`GetOpBuilder` returns from a function-local `static`, [op_builder_factory.cc:135](../onnxruntime/core/providers/qnn/builder/op_builder_factory.cc#L135); one `SimpleOpBuilder` serves ~49 op types). **Never store per-node state in members** — all hooks are `const`.

## 3. Which Hooks to Override

- `ProcessInputs` — inputs need massaging, constant-input checks, or some inputs become QNN params.
- `ProcessAttributesAndOutputs` — op has attributes that become QNN param tensors.
- `OverrideOutputQuantParam` — quantized op needs output qparams forced (e.g. equal to input).
- `GetSupportedOutputDataType` — QNN output dtype differs from ONNX (e.g. int64→int32).
- `IsOpSupported` — **only** for layout-sensitive ops or cheap early rejections.
- `CheckCpu/Htp/GpuDataTypes` — per-backend datatype allow-lists.

## 4. Reusing a Builder for an Equivalent Op

- Register the new ONNX op type to the **same** `CreateXxx` function — the factory dedupes by builder-type string, so all calls collapse to one shared instance ([op_builder_factory.h:32-48](../onnxruntime/core/providers/qnn/builder/op_builder_factory.h#L32-L48)).
- Add a `GetQnnOpType` map entry for the new op type.
- Keep the builder op-type-agnostic: emit `GetQnnOpType(node_unit.OpType())` and branch only where behavior genuinely differs — prefer branching on **output/input count** over op type where possible (e.g. [rmsnormalization_op_builder.cc:42-46](../onnxruntime/core/providers/qnn/builder/opbuilder/rmsnormalization_op_builder.cc#L42-L46)).
- Precedents: Conv/ConvTranspose, Gather/GatherElements, Reshape/Flatten/Squeeze/Unsqueeze, ~50 ops on `SimpleOpBuilder`.

## 5. Factory Registration (all required — easy to miss)

- Declare `CreateMyOpBuilder` in `op_builder_factory.h` (keep alphabetical).
- Call it in the `OpBuilderRegistrations` constructor in `op_builder_factory.cc` (keep alphabetical).
- Define the function in your `.cc`: `op_registrations.AddOpBuilder(op_type, std::make_unique<MyOpBuilder>())`.
- **Add the ONNX→QNN mapping to `GetQnnOpType`** — if your builder (or the base `ProcessAttributesAndOutputs`) calls `GetQnnOpType(node_unit.OpType())`, a missing entry **throws** (`ORT_CXX_API_THROW`, [base_op_builder.h:228-230](../onnxruntime/core/providers/qnn/builder/opbuilder/base_op_builder.h#L228-L230)) — it does not fall back. Builders that *decompose* and pass explicit QNN op-type strings to `CreateQnnNode` (e.g. LayerNorm, Einsum, Mean) don't call it and need no map entry.
- If QNN needs a fixed input/output count where ONNX is variadic, add it to `GetInputOutputCountQnnRequired` ([base_op_builder.h:263-268](../onnxruntime/core/providers/qnn/builder/opbuilder/base_op_builder.h#L263-L268)). This is the mechanism that **drops extra ONNX outputs** (MaxPool indices, LayerNorm mean/var) and trailing optional inputs: base `ProcessInputs`/`ProcessOutputs` only iterate `GetInput/OutputCountQnnRequired(node_unit)` ([base_op_builder.cc:134,273](../onnxruntime/core/providers/qnn/builder/opbuilder/base_op_builder.cc#L134)). A `0` entry means "use the node's actual count".

## 6. IsOpSupported & Graceful Fallback

- To make an op fall back to CPU: **return a non-OK `Ort::Status`. Never throw. Never log ERROR.**
- Use the macros: `RETURN_IF`, `RETURN_IF_NOT`, `RETURN_IF_ERROR`, `MAKE_EP_FAIL`.
- For most ops the base `IsOpSupported` suffices — it dry-runs the build and lets QNN's validation API reject.
- Override it to reject up front: bad rank/shape, unsupported attribute values, dynamic-vs-constant input mismatches, backend restrictions.

## 7. Inputs, Initializers & Params

- An ONNX input that QNN expects as a **param** (e.g. Tile `repeats`, Reshape `shape`) must be *skipped* in `ProcessInputs` and turned into a `QnnParamWrapper` in `ProcessAttributesAndOutputs` — otherwise QNN gets a spurious extra graph input.
- Unpack constant inputs with `GetConstantTensor` + `UnpackInitializerData`.
- **Optional / missing inputs:** an absent IODef has an empty `name` and `Exists()` returns false ([ort_api.h:257](../onnxruntime/core/providers/qnn/ort_api.h#L257)). Always check `inputs[i].Exists()` before processing an optional input (e.g. RMSNorm scale, [rmsnormalization_op_builder.cc:50](../onnxruntime/core/providers/qnn/builder/opbuilder/rmsnormalization_op_builder.cc#L50)). The base pipeline already skips empty names in `ProcessDataTypes` and `ProcessInt64Tensors`.
- **int64 is mostly handled for you.** `AddToModelBuilder` runs `ProcessInt64Tensors`, which auto-inserts an int64→int32 `Cast` for every int64 *dynamic input tensor* ([base_op_builder.cc:50,200-236](../onnxruntime/core/providers/qnn/builder/opbuilder/base_op_builder.cc#L200-L236)); `ProcessOutputs` auto-casts int64/uint64 *graph outputs* to int32/uint32 ([base_op_builder.cc:288-296](../onnxruntime/core/providers/qnn/builder/opbuilder/base_op_builder.cc#L288-L296)). You only hand-downcast with `SafeInt<...>` when pulling **initializer bytes into a QNN param tensor** (e.g. Tile `repeats`→`multiples`, [tile_op_builder.cc:67-74](../onnxruntime/core/providers/qnn/builder/opbuilder/tile_op_builder.cc#L67-L74)).
- For each param: build the `QnnParamWrapper`, `push_back(GetParamTensorName())`, then `AddParamWrapper`.
- Set the `Qnn_Scalar_t` `dataType` and its matching union field together — a mismatch compiles but corrupts the value.
- **Don't hand-roll outputs** — call the base `ProcessOutputs(...)`. It handles graph-output detection, int64/uint64→int32/uint32 cast insertion, the qparam-override callback, and node emission. Note: base `ProcessAttributesAndOutputs` **early-returns OK if `input_names` is empty** ([base_op_builder.cc:243-245](../onnxruntime/core/providers/qnn/builder/opbuilder/base_op_builder.cc#L243-L245)) — a node that processes zero inputs silently emits no QNN node.

## 8. Decomposition — One ONNX Op → Multiple QNN Nodes

- **When needed:** QNN has no single equivalent op, or its equivalent has shape/axis/dtype constraints the ONNX op doesn't (e.g. `LayerNormalization` → LayerNorm → Mul → Add when scale/bias must be externalized: [layernormalization_op_builder.cc:354-704](../onnxruntime/core/providers/qnn/builder/opbuilder/layernormalization_op_builder.cc#L354-L704)). Note: "decomposition" here is the op-builder fan-*out* (1 ONNX op → many QNN nodes); the fusion doc uses "decomposition" for the opposite — the multi-node ONNX *pattern* a fusion collapses back into one QNN op. Don't confuse the two.
- Name intermediates with `utils::UniqueNameGenerator().New(node_unit, "_suffix")` so they never collide.
- Register each intermediate as a `QNN_TENSOR_TYPE_NATIVE` tensor (`QnnTensorWrapper` + `AddTensorWrapper`) **before** the node that produces it; only the final tensor gets a graph-output type.
- Chain `CreateQnnNode` calls by threading a `current` cursor from each node's output into the next node's input; pass the same `do_op_validation` flag to every call.
- For pure layout/shape ops use the convenience wrappers `AddTransposeNode` / `AddReshapeNode` (e.g. Softmax transpose-wrapping for non-last-axis: [softmax_op_builder.cc:118-142](../onnxruntime/core/providers/qnn/builder/opbuilder/softmax_op_builder.cc#L118-L142)); use `CreateQnnNode` directly for everything else.
- **Caveat:** inserted Reshape/Transpose on per-channel-quantized *dynamic* tensors is rejected — honor this when decomposing quantized graphs.

## 9. Attributes

- Use `OrtNodeAttrHelper node_helper(node_unit); node_helper.Get(name, default)`.
- Defaults must exactly match the ONNX spec defaults.
- For `axis`, use the shared `ProcessAxisAttribute` — it normalizes negatives and range-checks.

## 10. Layout (NCHW → NHWC)

- QNN spatial ops (Conv, Pool, etc.) are NHWC-native; ONNX is NCHW.
- Layout-sensitive builders must branch on the domain in `IsOpSupported`: only call QNN validation when `node_unit.Domain() == kMSInternalNHWCDomain`; for the pre-transform form do manual checks and return OK.
- After transform, channel is the **last** dimension.
- Reorder ONNX pads to QNN order with `ReArrangePads` — forgetting this gives silently wrong results.

## 10.5. The `do_op_validation` Pass

- These two executions map to the EP's two phases: `do_op_validation=true` runs during **Phase 1 / GetCapability** (via `QnnNodeUnitWrapper::IsSupported` → `IsOpSupported`), and `do_op_validation=false` runs during **Phase 2 / Compile** (via `QnnNodeUnitWrapper::AddToModelBuilder`, [qnn_node_group.cc:61-66](../onnxruntime/core/providers/qnn/builder/qnn_node_group/qnn_node_group.cc#L61-L66)). This is the same Phase 1 / Phase 2 lifecycle the fusion doc describes ([qnn_node_group_fusion_guidelines.md](qnn_node_group_fusion_guidelines.md) §1).
- `IsOpSupported` calls `AddToModelBuilder(..., do_op_validation=true)` ([base_op_builder.cc:32](../onnxruntime/core/providers/qnn/builder/opbuilder/base_op_builder.cc#L32)) — **the same build path runs**, but `CreateQnnNode` calls QNN's *validation* API instead of materializing the node.
- **Your hooks run in BOTH passes** (support-check and real build). Guard one-time / irreversible work with `if (do_op_validation)` (e.g. Tile's constant-input check, [tile_op_builder.cc:45-48](../onnxruntime/core/providers/qnn/builder/opbuilder/tile_op_builder.cc#L45-L48)).
- The int64 *output* cast is intentionally suppressed during validation (`... && !do_op_validation`, [base_op_builder.cc:314](../onnxruntime/core/providers/qnn/builder/opbuilder/base_op_builder.cc#L314)).
- When decomposing, **thread the same `do_op_validation`** into every `CreateQnnNode` / `AddTransposeNode` call.

## 11. Quantization / QDQ (HTP)

- HTP = quantized (u8/i8/u16/i16/sfixed32); CPU = float-only; GPU = fp16/fp32. Query via `GetQnnBackendType()` + `IsNpuBackend`/`IsCpuBackend`/`IsGpuBackend`.
- `IsOpSupported` runs `ProcessDataTypes` **first** ([base_op_builder.cc:31,56-91](../onnxruntime/core/providers/qnn/builder/opbuilder/base_op_builder.cc#L56-L91)) — it collects each IODef's `qnn_data_type` (skipping optional IODefs) and dispatches by backend to `CheckCpu/Htp/GpuDataTypes`. You override those to plug in an allow-list; you don't call the driver yourself.
- Enforce per-backend datatype combos by overriding `CheckHtpDataTypes` (etc.).
- **Bias synthesis:** quantized Conv/Gemm/MatMul without a bias input can synthesize an all-zero `SFIXED_POINT_32` bias via `AddZeroBiasInput` ([base_op_builder.cc:142-198](../onnxruntime/core/providers/qnn/builder/opbuilder/base_op_builder.cc#L142-L198)) — bias scale = product of input scales, per-channel if input[1] is per-channel. Constraint: input[0] must be per-tensor, input[1] per-tensor or per-channel.
- For movement/elementwise ops, force output qparams equal to input via `OverrideOutputQuantParam` + `SetOutputQParamEqualToInputIfNearlyEqual` so HTP can fuse (omitting it is correct but slower). Note: base only calls `OverrideOutputQuantParam` when the output is actually quantized (`if (output_info.quant_param.IsQuantized())`, [base_op_builder.cc:280-283](../onnxruntime/core/providers/qnn/builder/opbuilder/base_op_builder.cc#L280-L283)) — it won't fire on float outputs.
- **Keep the QDQ NodeUnit fused by default** — `node_unit.Inputs()/Outputs()` already carry the surrounding DQ/Q quant params; use those.
- **"Open" the unit via `node_unit.GetDQNodes()` / `GetQNodes()` only when you need:**
  - constness of a quantized param behind a DQ (e.g. `DQ(const_initializer)`: [batchnormalization_op_builder.cc:457-462](../onnxruntime/core/providers/qnn/builder/opbuilder/batchnormalization_op_builder.cc#L457-L462));
  - standalone Q/DQ as conversion ops — reject per-channel standalone Q/DQ unless the input is constant ([simple_op_builder.cc:88-117](../onnxruntime/core/providers/qnn/builder/opbuilder/simple_op_builder.cc#L88-L117));
  - constant-folding a standalone (`SingleNode`) Q/DQ on a constant input into a STATIC tensor;
  - per-channel / LPBQ weight-encoding decisions in Conv/MatMul/Gemm.

## 12. Error Handling & Logging

- Return `Ort::Status` (`Ort::Status()` means OK); never throw from builder code.
- `ORT_CXX_LOG(logger, LEVEL, msg.c_str())`: VERBOSE for per-node traces, WARNING for unexpected fallbacks/qparam overrides; a failed `IsOpSupported` just returns non-OK, no ERROR log.

## 12.5. Advanced Patterns (needed for non-trivial ops)

The sections above cover the "simple op" path. Conv, MatMulNBits, RNNs, and any quantized weight op rely on the following recurring patterns that the basic pipeline doesn't make obvious.

### The base-class graph-output Cast contract (you depend on this even if you don't write it)

When a **graph output**'s QNN dtype is narrower than the ONNX dtype — int64→int32, or anything your `GetSupportedOutputDataType` override changes — the base `ProcessOutputs` automatically appends a trailing `QNN_OP_CAST` to restore the ONNX-visible dtype ([base_op_builder.cc:285-357](../onnxruntime/core/providers/qnn/builder/opbuilder/base_op_builder.cc#L285-L357), collected in `cast_node_info_vec`). *Internal* tensors keep the narrowed type. Consequences:
- Override `GetSupportedOutputDataType` to narrow a dtype and you get the restoring cast for free — **only for graph outputs**.
- This is suppressed during the validation pass (`&& !do_op_validation`).
- If you hand-roll outputs (decomposition), you must replicate this cast yourself for narrowed graph outputs (see TopK/NonZero for the pattern).

### "The QNN validator lies" — reject in `IsOpSupported` to keep CPU fallback working

The QNN op-config validation API sometimes **accepts** a node that later fails at `graphFinalize`. If you let such a node be claimed during partitioning, the whole partition's Compile aborts (no fallback at that stage). The fix is to **preemptively reject the unsupported case in `IsOpSupported`** so it cleanly falls back to CPU. Canonical example with the full explanation in-comment: [isinf_op_builder.cc:39-47](../onnxruntime/core/providers/qnn/builder/opbuilder/isinf_op_builder.cc#L39-L47) ("validator accepts the node... graphFinalize then fails with error 1002"). The same defensive rejection recurs in nonzero, gemm, matmul, scatternd, rotary, and stft builders. **Rule of thumb:** if you know a config the backend can't actually finalize, reject it up front rather than trusting the validator.

### Negative / int64 indices silently fall back to CPU — normalize them

QNN drops a node to CPU on a single negative static index. For index ops (Gather/GatherND/ScatterND), host-normalize negative/int64 ONNX indices to non-negative INT32 with the shared helper `NormalizeIndicesBytes` / `AddNormalizedIndicesTensor` ([normalize_indices_utils.h:30,37](../onnxruntime/core/providers/qnn/builder/opbuilder/normalize_indices_utils.h#L30-L37)) — it uses an int64 accumulator to avoid int32 wraparound and inserts a runtime Cast for *dynamic* int64 indices.
- **Footgun:** Gather uses a **deterministic** rename for the rewritten indices ([gather_op_builder.cc:73-179](../onnxruntime/core/providers/qnn/builder/opbuilder/gather_op_builder.cc#L73-L179)), **not** `UniqueNameGenerator` — the unique generator is stateful/global and would emit a different name on the second (Compile) pass, duplicating the tensor. Use `UniqueNameGenerator` only when the tensor is created unconditionally on every pass.

### Quantization beyond the basics

The basic doc covers `AddZeroBiasInput` and output-qparam override. Quantized **weight** ops need more — study Conv/MatMul/Gemm/MatMulNBits and BatchNorm:
- **Block-quant: runtime BQ-vs-LPBQ fork.** Detect `quant_param.IsLPBQ()` and branch between LPBQ (`BLOCKWISE_EXPANSION`, keeps INT16 activation) and a hand-built `BW_FLOAT_BLOCK` (requires FP16 activation); compute `block_size = IC / num_blocks` and emit a layout-specific `block_size_arr`.
- **Unsigned→signed weight conversion.** `BW_FLOAT_BLOCK` / `SFIXED_POINT_4` accept only signed data — use `utils::TransformUnsignedToSignedFixedPoint` (MSB-XOR) and compensate the offset (`offset = (1<<(bits-1)) - onnx_zp`). Sign errors here corrupt weights silently.
- **Requantizing existing static data** (distinct from synthesizing a zero bias): `utils::RequantizeBiasTensor` + `CheckBiasScaleMatch` (re-scale a static bias when `scale != input_scale*weight_scale`); `RequantizePerTensorStatic` (re-scale an input scale initializer so both operands of a fused Multiply share a dtype, e.g. A16W8 LayerNorm).
- **`utils::InsertConvertOp` for 16-bit asymmetry / mixed precision** — insert a Convert when, e.g., a UFIXED16 weight must become SFIXED16; the `32768 = 2^(16-1)` symmetric check gates whether it's needed. Recurs in matmul/conv/instancenorm.
- **Deriving fresh quant params from fused float math** — when you fold constants (BatchNorm folds gamma/beta/mean/var in `double`), track running min/max and mint scale/zp via `utils::GetQuantParams` (force symmetric for signed/16-bit); fall back to float execution if a per-channel scale would overflow a per-tensor requant.
- **Per-channel quant axis must follow layout changes** — after a transpose/unsqueeze, remap the quant axis with `QnnQuantParamsWrapper::HandleTranspose` (inverse-perm remap) / `HandleUnsqueeze`. This is the only correct way to move a per-channel axis; it handles `AXIS_SCALE_OFFSET`, `BW_AXIS_SCALE_OFFSET`, and the LPBQ `BLOCKWISE_EXPANSION` axis.
- **16-bit activations are a hazard class** — expect special casing (disabling FullyConnected for dual-dynamic-uint16, the `32768` symmetric workarounds, validation skips for UFIXED16 Tanh/Sigmoid).

### High-value shared helpers the basics don't mention

- **`QnnModelWrapper::GetTensorInfo(name, TensorInfo&)`** → `TensorInfo{shape, qnn_data_type, quant_param, is_initializer, initializer_tensor}` ([qnn_model_wrapper.h:29-35](../onnxruntime/core/providers/qnn/builder/qnn_model_wrapper.h#L29-L35)) — the one-stop "open a tensor and inspect everything" call every advanced builder uses.
- **Folded-constant registry** — `IsFoldedConstant` / `IsEffectivelyConstantInput` / `MarkTensorAsFoldedConstant` ([qnn_model_wrapper.h:209-219](../onnxruntime/core/providers/qnn/builder/qnn_model_wrapper.h#L209-L219)) lets a chain of folded Q/DQ keep folding.
- **Graph-building helpers on `QnnModelWrapper`**: `AddReshapeNode`, `AddTransposeNode`, `AddNchwToHwcnTranspose`, `AddCastNode`, `AddNoopReshapeNode`; plus `GetQnnTensorWrapper` / `IsQnnTensorWrapperExist` to read back a wrapper's dtype and decide whether a runtime Cast is needed.
- **`UniqueNameGenerator`** (`utils::`) — `New(node_unit, "_suffix")`; **stateful/global** (see the Gather footgun above).
- **`ReinterpretAsSpan<U,T>`** — sanctioned typed view over initializer bytes.
- **Layout/transpose**: `InvertPerm`, `PermuteShape`, `NchwShapeToHwcn`, `TransposeFromNchwToHwcn`/`TransposeFromCnhwToHwcn` (these **unpack INT4→INT8 in the same pass**), `TwoDimensionTranspose`.
- **Sub-byte unpack with built-in bug workarounds**: `UnpackInt4ToInt8` (applies the *mandatory* top-nibble mask for a QNN INT4 accuracy bug), `UnpackInt2ToInt8`. Note the 3rd arg of `UnpackInitializerData(..., unpack_sub_byte_to_8_bit)` — MatMulNBits passes `false` to keep data packed.
- **`GetQnnTensorDataSizeInBytes`** — correctly sizes sub-byte dtypes (prefer over `GetElementSizeByType` for packed data).
- **Op substitution via the `GetQnnOpType` map** — non-1:1 entries *enable* rewrites without a custom builder: Expand→Multiply, Sum→Add, Clip→ReluMinMax, Gemm→FullyConnected, LeakyRelu→Prelu, Squeeze/Unsqueeze/Flatten→Reshape.

### Reference builders for advanced work

- `matmulnbits_op_builder.cc` — n-bit weight bit-unpack + sign transform + offset negation + transpose.
- `conv_op_builder.cc` — NCHW→HWCN weight repack with quant-axis sync + bias requant + version-windowed workarounds.
- `resize_op_builder.cc` — backend × mode × rank divergence with `#if QNN_API_VERSION` blocks.
- `batchnormalization_op_builder.cc` — fold constants in `double` and mint fresh quant params.
- `inverse_op_builder.cc` / `rotary_embedding_op_builder.cc` — large pure-math decompositions (incl. emulating a missing QNN Stack).
- `pad_op_builder.cc` / `reduce_op_builder.cc` — opset × domain gating tables.

## 13. Tests

- Add `onnxruntime/test/providers/qnn/<opname>_test.cc` (guard with `#if !defined(ORT_MINIMAL_BUILD)`).
- CPU/float path: `BuildOpTestCase` + `RunQnnModelTest`.
- HTP/QDQ path: `TestQDQModelAccuracy` — cover **both** uint8 and uint16 (uint16 with `use_contrib_qdq=true`).
- Always include a negative test asserting `ExpectedEPNodeAssignment::None` for configs your `IsOpSupported` rejects.
- Run `onnxruntime_test_all --gtest_filter="*QNN*MyOp*"` (run the full `*QNN*` suite if you changed `base_op_builder.*`).

## 14. Common Pitfalls

- Missing `GetQnnOpType` entry → **throws** (only if the builder actually calls it; decomposing builders that pass explicit QNN op strings don't need one).
- Overriding `AddToModelBuilder` (it's `final`).
- Treating an attribute-as-input as a graph tensor → spurious extra QNN graph input.
- int64 initializer data not downcast when building a **param** tensor (dynamic int64 tensor in/out is cast for you by the base pipeline).
- Running QNN validation on the NCHW form (must branch on `kMSInternalNHWCDomain`).
- Forgetting `ReArrangePads` → silently wrong results.
- Variadic ONNX vs fixed QNN arity not handled.
- `Qnn_Scalar_t` `dataType`/union-field mismatch — compiles, corrupts the param.
- Writing a new builder when an equivalent QNN op already exists (check the map first).

## Reference Builders to Copy From

- `tile_op_builder.cc` — inputs + params + qparam override (best starting point).
- `argmax_argmin_op_builder.cc` — attributes → params.
- `rmsnormalization_op_builder.cc` — one builder reused across equivalent ops.
- `layernormalization_op_builder.cc` / `softmax_op_builder.cc` — decomposition into multiple QNN nodes.
- `conv_op_builder.cc` / `pool_op_builder.cc` — layout transforms + per-channel quant.
- `batchnormalization_op_builder.cc` — per-backend datatype allow-lists + `GetDQNodes()` reach-through.

## Related Tooling

- `/qnn_ep_op_builder_codegen <manifest>` — scaffold a builder from a YAML manifest, or `--from-onnx <OpName>` to generate a draft manifest first.
