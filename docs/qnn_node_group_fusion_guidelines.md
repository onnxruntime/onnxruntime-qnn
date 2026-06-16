# QNN EP Node-Group Fusion — Developer Guidelines

A checklist for implementing a new **node-group fusion** that collapses a multi-op ONNX subgraph into one (or a few) QNN ops. These live in `onnxruntime/core/providers/qnn/builder/qnn_node_group/`. (For single-op ONNX→QNN translation, see [qnn_op_builder_guidelines.md](qnn_op_builder_guidelines.md) — that's a different mechanism.)

All file paths are relative to the repo root.

## 0. Decide Whether You Need a Fusion (do this first)

- A fusion is for **multi-op patterns** that map to a single QNN op (or a small rewrite), e.g. `Erf`-based GELU → `QNN_OP_GELU`, `ReduceMean`-based LayerNorm → `QNN_OP_LAYER_NORM`, `x * HardSigmoid(x)` → HardSwish.
- If your op is a single ONNX node, it's an **op builder**, not a fusion.
- Before writing new code, check the existing inventory (§7) — your pattern may already be covered or share a starting op with an existing fusion.

## 1. The `IQnnNodeGroup` Interface / Base Contract

Defined in [qnn_node_group.h:23-55](../onnxruntime/core/providers/qnn/builder/qnn_node_group/qnn_node_group.h#L23-L55). Subclass `IQnnNodeGroup` and implement five pure-virtual methods:

| Method | Contract |
|---|---|
| `Ort::Status IsSupported(QnnModelWrapper&, const Ort::Logger&) const` | Return OK iff the fused op can be emitted on this backend. Called during `GetCapability`. |
| `Ort::Status AddToModelBuilder(QnnModelWrapper&, const Ort::Logger&) const` | Emit the fused QNN op(s). Called during `Compile`. |
| `gsl::span<const OrtNodeUnit* const> GetNodeUnits() const` | All NodeUnits this group claims. Order is **not** required to be topological — it's used only to register claimed NodeUnits and enumerate nodes for partitioning. Any `node_units_[0]`/`.back()` dependence is the fusion's own private convention. |
| `const OrtNodeUnit* GetTargetNodeUnit() const` | The topological anchor — the convergence point of all input paths. Drives the group's slot in the emission sort (§2). |
| `std::string_view Type() const` | Human-readable type string. **Load-bearing:** used as a JSON key in the framework op trace (`fusion_count`) — renaming is a breaking change. |

- The interface does **not** declare `TryFusion` — that is a *static* factory on each subclass, referenced only by the registry (§2).
- A single concrete NodeUnit is wrapped as an `IQnnNodeGroup` via the built-in `QnnNodeUnitWrapper` ([qnn_node_group.cc:44-77](../onnxruntime/core/providers/qnn/builder/qnn_node_group/qnn_node_group.cc#L44-L77)) — it is the only group type that delegates to op-builders.

### Lifecycle (matching runs in both `GetCapability` and `Compile`)

`GetQnnNodeGroups` is the single entry point, called from:
1. **`GetCapability` / partitioning** ([qnn_execution_provider.cc:1253-1268](../onnxruntime/core/providers/qnn/qnn_execution_provider.cc#L1253-L1268)) — builds all groups, calls `IsSupported` ([:1259](../onnxruntime/core/providers/qnn/qnn_execution_provider.cc#L1259)); if OK, every node in every claimed NodeUnit is added to `supported_nodes`.
2. **`Compile` / graph composition** ([qnn_model.cc:288-294](../onnxruntime/core/providers/qnn/builder/qnn_model.cc#L288-L294)) — **rebuilds a fresh set of groups** against a different `QnnModelWrapper`, then calls `AddToModelBuilder` in topological order ([:294](../onnxruntime/core/providers/qnn/builder/qnn_model.cc#L294)).

**Consequences a developer MUST know:**
- `TryFusion` runs in both phases; `IsSupported`/`AddToModelBuilder` must stay consistent with what it matched. This is why the standard pattern funnels both through one shared `CreateOrValidateOnQnn` helper (§5).
- `IsSupported` is invoked **only in Phase 1**; Phase 2 calls **only** `AddToModelBuilder` and trusts the partition decision.
- The Phase-1 group object is discarded after `GetCapability`; Phase 2 builds entirely distinct objects against a different graph/`QnnModelWrapper`. **Never cache `OrtNode*` / `OrtNodeUnit*` across the two passes.**

## 2. Discovery / Registration

The registry is a static map from **starting op type** → an ordered list of fusion factory functions ([qnn_node_group.cc:82-104](../onnxruntime/core/providers/qnn/builder/qnn_node_group/qnn_node_group.cc#L82-L104)):

```cpp
static std::unordered_map<std::string, std::vector<FusionFunc>> fusions = {
    {"DequantizeLinear", {DQQFusion::TryFusion}},
    {"Gemm", {LowPowerBlockQuantizedGemmFusion::TryFusion, ReshapeGemmFusionGroup::TryFusion4,
              ReshapeGemmFusionGroup::TryFusion3, ReshapeGemmFusionGroup::TryFusion2}},
    {"Transpose", {ChannelShuffleFusion::TryFusion, TransposeReshapeTransposeFusion::TryFusion}}};
```

**To register a new fusion:**
- Add the `#include` near the top ([qnn_node_group.cc:12-35](../onnxruntime/core/providers/qnn/builder/qnn_node_group/qnn_node_group.cc#L12-L35)).
- Add an entry `{"StartOp", {MyFusion::TryFusion}}` (or insert into an existing op's vector).

**Ordering / priority** — `TryQnnFusions` ([qnn_node_group.cc:133-160](../onnxruntime/core/providers/qnn/builder/qnn_node_group/qnn_node_group.cc#L133-L160)) tries each factory **in vector order and returns the first non-null result**. For shared starting ops, **the more specific fusion must come first** (e.g. `TryFusion4` before `TryFusion3` before `TryFusion2`).

**Starting-NodeUnit gate** — `TryQnnFusions` only fires for `SingleNode`-type NodeUnits, with a hardcoded exception list (`Gather`/`MatMul`/`Erf`/`Reshape`) for ops that may start from a QDQ group ([qnn_node_group.cc:141-147](../onnxruntime/core/providers/qnn/builder/qnn_node_group/qnn_node_group.cc#L141-L147)). **If your starting op can appear as a QDQ group, you must add it to this list** or your `TryFusion` is never called.

**Double-claim prevention is two-layered:**
1. The dispatcher skips a starting NodeUnit already in `node_unit_to_qnn_node_group` *before* trying any fusion ([qnn_node_group.cc:205-207](../onnxruntime/core/providers/qnn/builder/qnn_node_group/qnn_node_group.cc#L205-L207)).
2. The traversal helpers (§3) refuse any *neighbor* already in that map.

After a successful fusion, **every** member NodeUnit is inserted into the map. Note the first loop visits each NodeUnit only at the NodeUnit's own representative node (`node_unit->GetNode()`, [qnn_node_group.cc:199](../onnxruntime/core/providers/qnn/builder/qnn_node_group/qnn_node_group.cc#L199)) — distinct from the group's `GetTargetNodeUnit()`, which is used in the *second* loop to decide sort order ([:235](../onnxruntime/core/providers/qnn/builder/qnn_node_group/qnn_node_group.cc#L235)). Leftover NodeUnits become `QnnNodeUnitWrapper` single-op groups.

**Dynamic registration (UDOs)** — UDOs register at runtime via `registerUDO(node_type, op_package)` ([qnn_node_group.cc:106-121](../onnxruntime/core/providers/qnn/builder/qnn_node_group/qnn_node_group.cc#L106-L121)), `std::bind`-ing extra args onto `UDOQDQFusion::TryFusion` into the same map.

## 3. The `TryFusion` / Pattern-Matching Convention

`FusionFunc` is the registry's function type ([qnn_node_group.cc:82-86](../onnxruntime/core/providers/qnn/builder/qnn_node_group/qnn_node_group.cc#L82-L86)). Every `TryFusion` must match it exactly:

```cpp
std::unique_ptr<IQnnNodeGroup> TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& starting_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Ort::Logger& logger);
```

- `node_to_node_unit` — keyed on `OrtNode*`; resolves any node reached by graph traversal to its owning NodeUnit.
- `node_unit_to_qnn_node_group` — keyed on `OrtNodeUnit*`; the set of **already-claimed** NodeUnits (value = the claiming group). A NodeUnit absent from this map is still available. Thread it into every helper call.

Canonical structure:

1. **Guard the trigger op first** — the map key is the only enforcement (e.g. [dq_q_fusion.cc:39-41](../onnxruntime/core/providers/qnn/builder/qnn_node_group/dq_q_fusion.cc#L39-L41) re-checks op type + `SingleNode`).
2. **Walk the graph via NodeUnit helpers** (never raw `OrtNode` edges).
3. **Verify structural / constant constraints**; return `nullptr` on any mismatch.
4. **Validate on QNN** before committing (§5).
5. **Construct and return** the `unique_ptr<IQnnNodeGroup>`.

### `SingleNode` vs `QDQGroup` and how fusions see quant params

A `NodeUnit` is either `SingleNode` (one node) or `QDQGroup` (`DQ*→target→Q*`). `GetNode()` returns the target op in both cases; the wrapping quantizers are reached via `GetDQNodes()` / `GetQNodes()` (as `GetChildNodeUnitAllowQdq` does at [utils.cc:143-147](../onnxruntime/core/providers/qnn/builder/qnn_node_group/utils.cc#L143-L147)). Because most helpers reject `QDQGroup` neighbors, fusions normally operate on standalone float nodes; only the four gate-listed starting ops (§2) may anchor a fusion from a `QDQGroup`.

### Traversal helpers ([utils.h](../onnxruntime/core/providers/qnn/builder/qnn_node_group/utils.h) / [utils.cc](../onnxruntime/core/providers/qnn/builder/qnn_node_group/utils.cc))

Every helper takes both maps and refuses a neighbor already in `qnn_node_group_map` (the front-line double-claim guard) and a neighbor that produces a graph output. **Beyond that, the guarantees differ — do not assume uniform checks:**

- **Single fan-out** (parent has exactly one consumer) is enforced **only by child helpers** (`GetOnlyChildOfType`, `GetOnlyChildOfOutput`, `GetChildNodeUnitAllowQdq`). The **parent walkers do NOT** check it ([utils.cc:271-383,451-499](../onnxruntime/core/providers/qnn/builder/qnn_node_group/utils.cc#L271-L383)).
- **`SingleNode` enforcement**: only `GetOnlyChildOfType`, `GetParentOfType`, `GetParentOfInputByName`. `GetParentOfInput` and `GetOnlyChildOfOutput` may return a `QDQGroup`; `GetChildNodeUnitAllowQdq` deliberately allows QDQ.
- **Op-type matching**: done by `GetOnlyChildOfType` / `GetParentOfType` (against a list). `GetParentOfInput` / `GetOnlyChildOfOutput` match on a specific IODef name only — **you must check the returned NodeUnit's `OpType()` yourself.**

| Helper | Use |
|---|---|
| `GetOnlyChildOfType` | Strict downstream walk; child must be the parent's only consumer and a `SingleNode`. |
| `GetChildNodeUnitAllowQdq` | Like above but skips through Q/DQ wrappers to the next math op. |
| `GetParentOfType` / `GetParentOfInput` / `GetParentOfInputByName` | Upstream walks (no single-consumer check). |
| `GetOnlyChildOfOutput` | Downstream walk keyed on a specific output IODef (no `SingleNode` check). |
| `GetReduceAxes`, `GetInitializerDataAsInt64` | Read constant attrs/initializers. |

### Representative examples (copy these)

- **Trivial 2-node — `DQQFusion`** ([dq_q_fusion.cc:31-65](../onnxruntime/core/providers/qnn/builder/qnn_node_group/dq_q_fusion.cc#L31-L65)): guard DQ → `GetOnlyChildOfType({QUANTIZE_LINEAR})` → verify scale/zp via `IsDQQConversion` → `ValidateOnQnn` → construct.
- **Fixed-shape chain — `ChannelShuffleFusion`** ([channel_shuffle_fusion.cc:79-124](../onnxruntime/core/providers/qnn/builder/qnn_node_group/channel_shuffle_fusion.cc#L79-L124)): chains `GetOnlyChildOfType` five times (`Transpose→Reshape→Transpose→Reshape→Transpose`), then heavy numeric validation including a canceling-perm-pair check.
- **Multi-variant + QDQ-aware — `GeluFusion`** ([gelu_fusion.cc:315-484](../onnxruntime/core/providers/qnn/builder/qnn_node_group/gelu_fusion.cc#L315-L484)): walks up/down to discriminate three GELU variants; canonicalizes QDQ-duplicated tensor names; rejects mixed standalone-Q/DQ topology; keeps **separate validation vs. creation IODefs**.

## 4. Emitting QNN Nodes in `AddToModelBuilder`

**Fusions build QNN nodes directly on the `QnnModelWrapper` — they do NOT delegate back to op-builders** (`GetOpBuilder` is called only by `QnnNodeUnitWrapper`). Emission idiom (from [dq_q_fusion.cc:89-126](../onnxruntime/core/providers/qnn/builder/qnn_node_group/dq_q_fusion.cc#L89-L126)):

```cpp
QnnTensorWrapper input_tensor, output_tensor;
RETURN_IF_ERROR(qmw.MakeTensorWrapper(input_def, input_tensor));
RETURN_IF_ERROR(qmw.MakeTensorWrapper(output_def, output_tensor));
RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(input_tensor)), "...");
RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(output_tensor)), "...");
RETURN_IF_NOT(qmw.CreateQnnNode(node_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CONVERT,
                                {input_def.name}, {output_def.name}, /*params*/{}, validate), "...");
```

Supporting machinery:
- **Tensors** from `OrtNodeUnitIODef` via `qmw.MakeTensorWrapper` + `qmw.AddTensorWrapper`. Guard re-adds with `IsQnnTensorWrapperExist` when a tensor may already exist (GELU: [gelu_fusion.cc:541-546](../onnxruntime/core/providers/qnn/builder/qnn_node_group/gelu_fusion.cc#L541-L546)).
- **Params** via `QnnParamWrapper` (scalar) or tensor params + `qmw.AddParamWrapper`, collecting `param_tensor_names` for `CreateQnnNode` (ChannelShuffle axis/num_groups: [channel_shuffle_fusion.cc:165-213](../onnxruntime/core/providers/qnn/builder/qnn_node_group/channel_shuffle_fusion.cc#L165-L213)).
- **Node naming**: `utils::UniqueNameGenerator().New(node_unit)`. This generator is **stateful/global** (a process-wide counter, [qnn_utils.h:132-143](../onnxruntime/core/providers/qnn/builder/qnn_utils.h#L132)). Safe for one-shot node names; but if a *decomposing* fusion mints intermediate **tensor** names that must be referenced later, see the cross-pass footgun in [qnn_op_builder_guidelines.md](qnn_op_builder_guidelines.md) §12.5 — use a deterministic rename, not the global generator, for names that must be stable across the two passes.
- **Root input / final output**: collapse N NodeUnits into one QNN op by reading the *first* NodeUnit's relevant input and the *last* NodeUnit's output. Quant-heavy fusions instead emit several QNN ops and inject DQ/Transpose for weights (`DQMatMulIntegerFusion` / `DQConvIntegerFusion`).

## 5. Validation Pattern (`IsSupported` Dry-Run)

The universal idiom: **one `CreateOrValidateOnQnn` body, two macro entry points**, so support-checking and emission can never diverge ([gelu_fusion.cc:26-35](../onnxruntime/core/providers/qnn/builder/qnn_node_group/gelu_fusion.cc#L26-L35)):

```cpp
#define ValidateOnQnn(...) CreateOrValidateOnQnn((...), /*validate=*/true)
#define CreateOnQnn(...)   CreateOrValidateOnQnn((...), /*validate=*/false)
```

- The `validate` branch calls `qmw.ValidateQnnNode(...)` (constructs the QNN op, asks the backend, discards). The `!validate` branch calls `AddTensorWrapper` + `CreateQnnNode`.
- Note the difference from op-builders: an op builder threads a single `do_op_validation` flag *into* `CreateQnnNode`, which internally forwards to `ValidateQnnNode` when true ([qnn_model_wrapper.cc:455,503](../onnxruntime/core/providers/qnn/builder/qnn_model_wrapper.cc#L455)). Fusions instead branch explicitly — `ValidateQnnNode` in the validate branch, `CreateQnnNode(..., /*validate=*/false)` in the create branch. Same backend dry-run, two encodings; **don't pass `validate=true` to `CreateQnnNode` from a fusion.**
- `IsSupported` → `ValidateOnQnn(...)`; `AddToModelBuilder` → `CreateOnQnn(...)`.
- **`TryFusion` calls `ValidateOnQnn` as its last gate** and returns `nullptr` if it fails ([dq_q_fusion.cc:59-62](../onnxruntime/core/providers/qnn/builder/qnn_node_group/dq_q_fusion.cc#L59-L62)) — this catches backend-specific rejections (dtype, rank) **before** the NodeUnits get claimed.
- **Graceful failure = return `nullptr` / non-OK `Ort::Status`.** Never throw, never log ERROR — the dispatcher falls through to the next factory or to `QnnNodeUnitWrapper`.
- Exception precedent: `ChannelShuffleFusion` deliberately skips the validate API (backend inconsistency) and only emits in the `!validate` branch ([channel_shuffle_fusion.cc:221-233](../onnxruntime/core/providers/qnn/builder/qnn_node_group/channel_shuffle_fusion.cc#L221-L233)).

### Backend awareness (HTP / CPU / GPU)

`IsSupported`/`TryFusion` receive no backend enum directly, but `QnnModelWrapper::GetQnnBackendType()` exposes it and fusions routinely gate on it:
- LPBQ MatMul/Gemm require NPU (`IsNpuBackend(...)`, [lpbqmatmul_fusion.cc:39](../onnxruntime/core/providers/qnn/builder/qnn_node_group/lpbqmatmul_fusion.cc#L39), [lpbqgemm_fusion.cc:42](../onnxruntime/core/providers/qnn/builder/qnn_node_group/lpbqgemm_fusion.cc#L42));
- `ReshapeGemmFusionGroup` bails on GPU ([reshape_gemm_fusion.cc:160](../onnxruntime/core/providers/qnn/builder/qnn_node_group/reshape_gemm_fusion.cc#L160));
- `SpaceToDepthFusion` branches on backend ([spacetodepth_fusion.cc:400](../onnxruntime/core/providers/qnn/builder/qnn_node_group/spacetodepth_fusion.cc#L400)).

If your fused QNN op is only available on some backends, gate in **`TryFusion`** (so the pattern falls back cleanly to single-op groups) rather than only in `IsSupported`.

### What happens if `AddToModelBuilder` fails in Phase 2

If `AddToModelBuilder` returns non-OK during Compile, the error is logged at ERROR and returned immediately, **aborting the entire graph composition** — there is no fallback at this stage ([qnn_model.cc:296-303](../onnxruntime/core/providers/qnn/builder/qnn_model.cc#L296-L303)). This is exactly why `TryFusion` must run `ValidateOnQnn` before claiming: anything that passes Phase-1 `IsSupported` but fails here kills the whole partition's Compile.

## 6. Naming, File Layout, CMake

- **One `.h` + `.cc` per fusion** in `builder/qnn_node_group/`, snake_case named after the pattern (`gelu_fusion`, `channel_shuffle_fusion`). Shared helpers live in `utils.{h,cc}` and `dq_integer_op_fusion_utils.{h,cc}`.
- **Copyright header**: Microsoft for MS-authored files; Qualcomm files carry `// Copyright (c) Qualcomm Technologies, Inc.... SPDX-License-Identifier: MIT`. Preserve the existing one when editing.
- **Class conventions**: `ORT_DISALLOW_COPY_AND_ASSIGNMENT(Name);` in the public section; a doc-comment ASCII diagram of the matched pattern above the class (every header has one); `Type()` returning a literal or `static constexpr std::string_view kType`; pattern helpers in an anonymous namespace in the `.cc`.
- **CMake: nothing to do.** [onnxruntime_providers_qnn.cmake:5-10](../cmake/onnxruntime_providers_qnn.cmake#L5-L10) uses `GLOB_RECURSE ... CONFIGURE_DEPENDS "*.h" "*.cc"`; new files are picked up on the next configure.

## 7. Existing Fusions (precedent inventory)

| Starting op | Fusion | Pattern → QNN result |
|---|---|---|
| `DequantizeLinear` | `DQQFusion` | Lone `DQ→Q` (equal scale type, differing zp) → QNN Convert |
| `DynamicQuantizeLinear` | `DqlDqFusion` | `DQL→DQ` fake-quant round-trip → identity Transpose |
| `MatMulInteger` | `DQMatMulIntegerFusion` | `DQL→MatMulInteger→Cast→Mul→Mul→[Add]` → float QNN MatMul |
| `ConvInteger` | `DQConvIntegerFusion` | `DQL→ConvInteger→Cast→Mul→Mul→[Add]` → float QNN Conv2d |
| `Gather` | `GatherTransposeReshapeFusion` | `Gather(axis=4)→Transpose→Reshape` rank-6→rank-4 → QNN Gather |
| `HardSigmoid` | `HardSigmoidMulFusion` | `x * HardSigmoid(x)` → QNN HardSwish |
| `MatMul` | `LowPowerBlockQuantizedMatMulFusion` | `DQ→Q→MatMul(+DQ,DQ,Q)` LPBQ → QNN MatMul w/ LPBQ |
| `Gemm` | `LowPowerBlockQuantizedGemmFusion`, then `ReshapeGemmFusionGroup::TryFusion4/3/2` | LPBQ FC; `[Reshape→]Gemm[→Reshape]` → QNN FullyConnected |
| `Mul` | `ScaleSoftmaxFusion` | `Softmax(Mul(x, scalar))` → QNN Softmax(beta=scale) |
| `Cast` | `CastLoneQFusion` | Orphaned `Cast→Q` |
| `Erf` | `GeluFusion` | 3 GELU ONNX expansions/variants → QNN Gelu |
| `ReduceMean` | `LayerNormFusion` | Full LayerNorm expansion → QNN LayerNorm |
| `Einsum` | `ReshapeEinsumReshapeNodeGroup` | `Reshape→Einsum→Reshape` (6D) → Reshape+DepthToSpace+Transpose |
| `Reshape` | `SpaceToDepthFusion`, then `Rank6ToRank5Fusion` | SpaceToDepth lowering; rank-6 `Reshape→Transpose→Reshape` → rank-5 |
| `Transpose` | `ChannelShuffleFusion`, then `TransposeReshapeTransposeFusion` | ChannelShuffle (5-node); `Transpose→Reshape→Transpose` → Reshape |
| *(dynamic)* | `UDOQDQFusion` | `DQ→UDO→Q` → QNN custom op, via `registerUDO` |

## 8. Common Pitfalls (fusion-specific)

1. **Double-claiming nodes.** Always reach neighbors through the §3 helpers (they check `qnn_node_group_map`); never store raw `OrtNode*` edges you found yourself. For patterns where siblings share an upstream node (N MatMulIntegers sharing one DQL), later siblings must detect the existing claim and skip re-claiming it.
2. **Wrong target NodeUnit.** `GetTargetNodeUnit` determines the group's slot in the topological sort — a group is appended to the sorted list when the NodeUnit walk reaches its target ([qnn_node_group.cc:234-236](../onnxruntime/core/providers/qnn/builder/qnn_node_group/qnn_node_group.cc#L234-L236)). (It does **not** control whether/where emission happens — every group is emitted unconditionally.) Picking a target upstream of where inputs actually converge can place the group before a producer it depends on.
3. **Forgetting the QDQ-start gate.** If your trigger op is normally a QDQ group, add it to the exception list at [qnn_node_group.cc:141-147](../onnxruntime/core/providers/qnn/builder/qnn_node_group/qnn_node_group.cc#L141-L147), or `TryFusion` is never called.
4. **Mixed QDQ/float topology.** A pattern partially wrapped in standalone Q/DQ silently changes numerics — copy GELU's `WarnAndFailOnStandaloneQdq` guard and bail. Also handle exporter-duplicated tensor names (`/duplicated`) when comparing the root input.
5. **Skipping the validate gate in `TryFusion`.** Claiming NodeUnits without running `ValidateOnQnn` before construction can strand a pattern the backend can't compile: marked supported in Phase 1, fails in Phase 2, aborting `Compile`. The gate may live textually in `TryFusion` or in the shared body called from it — but it must run before the group is returned. Validating *only* in `IsSupported` (e.g. `reshape_gemm_fusion`, `reshape_einsum_reshape`) is non-conforming: it works today only because Phase-1 `IsSupported` happens to run before the partition commits.
6. **Ordering among fusions sharing a start op.** More specific patterns must precede general ones — first non-null wins.
7. **Validation/creation IODef drift.** When the pattern is sandwiched in DQ/Q, validate against the *outer* (quantized) tensors but emit against the *inner* (float) tensors — keep two IODef sets like GELU.
8. **Graph-output / multi-consumer leaks.** Don't hand-roll edge walks; the helpers reject nodes that produce graph outputs or have >1 consumer. Fusing a node whose intermediate output is consumed elsewhere corrupts the graph.
9. **Constructor throwing.** Some constructors `ORT_CXX_API_THROW` on wrong NodeUnit count. `TryFusion` must fail soft (`nullptr`), so only construct with the exact expected count.

## Files Worth Reading Before You Start

- Interface + lifecycle: [qnn_node_group.h](../onnxruntime/core/providers/qnn/builder/qnn_node_group/qnn_node_group.h)
- Dispatcher, registry, double-claim logic: [qnn_node_group.cc](../onnxruntime/core/providers/qnn/builder/qnn_node_group/qnn_node_group.cc)
- Traversal helpers: [utils.h](../onnxruntime/core/providers/qnn/builder/qnn_node_group/utils.h) / [utils.cc](../onnxruntime/core/providers/qnn/builder/qnn_node_group/utils.cc)
- Simplest end-to-end template: [dq_q_fusion.cc](../onnxruntime/core/providers/qnn/builder/qnn_node_group/dq_q_fusion.cc)
- Multi-variant + QDQ-aware template: [gelu_fusion.cc](../onnxruntime/core/providers/qnn/builder/qnn_node_group/gelu_fusion.cc)
- Shape-math + params template: [channel_shuffle_fusion.cc](../onnxruntime/core/providers/qnn/builder/qnn_node_group/channel_shuffle_fusion.cc)
