// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include "QnnBackend.h"
#include "QnnInterface.h"
#include "QnnOpDef.h"
#include "QnnTypes.h"

#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "test/providers/qnn/unit/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

// Context for constructing a QnnModelWrapper in function-level unit tests.
//
//   QnnModelWrapperTestContext ctx;
//   ctx.input_info.indices  = {{"input0", 0}};   // declare graph inputs
//   ctx.output_info.indices = {{"output0", 0}};  // declare graph outputs
//   auto wrapper = ctx.CreateWrapper(settings);
//
// Inherits OrtApiStubContext so tests keep using ctx.stub_ort_api directly
// (e.g. ctx.stub_ort_api.GetTensorData = MyStub) and pick up the initializer-
// query stubs + MakeApiPtrs() invariant guard for free.
struct QnnModelWrapperTestContext : public OrtApiStubContext {
  // Stable lvalues passed to QnnModelWrapper / QnnBackendManager. Both store
  // pointers (or reference-holding views) to these members, so they must outlive
  // any wrapper returned by CreateWrapper().
  //
  // null_logger_: cached severity FATAL via MakeNullLogger() — see qnn_unit_test_utils.h.
  //   Declared before backend_manager because QnnBackendManager keeps a pointer to it.
  // fake_graph_sentinel_: an int used as a stable address; stubs receive this
  //   pointer but never dereference it (same pattern as g_type_info_sentinel etc.).
  Ort::Logger null_logger_{MakeNullLogger()};
  int fake_graph_sentinel_{};

  // QnnModelWrapper now reads the QNN interface, backend handles, and backend type
  // through a QnnBackendManager rather than taking them as constructor arguments.
  // StubBackendManager owns one with no backend library loaded and hands back
  // mutable references, so the members below keep the names (and the stubbing
  // ergonomics) tests used before the refactor: ctx.qnn_interface.graphAddNode = ...
  StubBackendManager backend_manager{MakeApiPtrs(), null_logger_};

  QNN_INTERFACE_VER_TYPE& qnn_interface = backend_manager.QnnInterface();
  Qnn_BackendHandle_t& backend_handle = backend_manager.BackendHandle();

  // No validator interface in unit tests — validator_backend_handle stays null so
  // ValidateQnnNode routes through qnn_interface / backend_handle. Tests that want
  // the validator branch assign both.
  QNN_INTERFACE_VER_TYPE& qnn_validator_interface = backend_manager.ValidatorInterface();
  Qnn_BackendHandle_t& validator_backend_handle = backend_manager.ValidatorBackendHandle();

  qnn::GraphInputOutputInfo input_info;
  qnn::GraphInputOutputInfo output_info;

  std::unique_ptr<qnn::QnnModelWrapper> CreateWrapper(
      const qnn::ModelSettings& settings,
      qnn::QnnBackendType backend_type = qnn::QnnBackendType::HTP) {
    backend_manager.BackendType() = backend_type;
    ApiPtrs api_ptrs = MakeApiPtrs();
    const OrtGraph& fake_graph = *reinterpret_cast<const OrtGraph*>(&fake_graph_sentinel_);
    return std::make_unique<qnn::QnnModelWrapper>(
        fake_graph,
        api_ptrs,
        null_logger_,
        *backend_manager.Get(),
        input_info,
        output_info,
        settings);
  }
};

// Construct a wrapper whose tensor_name_overrides pointer is non-null.
// QnnModelWrapperTestContext::CreateWrapper only exposes the default nullptr;
// build it directly here to reach the non-null branches.
std::unique_ptr<qnn::QnnModelWrapper> MakeWrapperWithOverrides(
    QnnModelWrapperTestContext& ctx,
    const qnn::ModelSettings& settings,
    std::unordered_map<std::string, std::string>* overrides) {
  ApiPtrs api_ptrs = ctx.MakeApiPtrs();
  const OrtGraph& fake_graph = *reinterpret_cast<const OrtGraph*>(&ctx.fake_graph_sentinel_);
  ctx.backend_manager.BackendType() = qnn::QnnBackendType::HTP;
  return std::make_unique<qnn::QnnModelWrapper>(
      fake_graph,
      api_ptrs,
      ctx.null_logger_,
      *ctx.backend_manager.Get(),
      ctx.input_info,
      ctx.output_info,
      settings,
      overrides);
}
}  // namespace

// Verifies IsQnnTensorWrapperExist returns false before adding and true after.
TEST(QnnUnit_ModelWrapperTest, IsQnnTensorWrapperExist_TrueAfterAdd_FalseBeforeAdd) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  EXPECT_FALSE(wrapper->IsQnnTensorWrapperExist("t0"));

  qnn::QnnTensorWrapper tensor("t0", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                               qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(tensor)));

  EXPECT_TRUE(wrapper->IsQnnTensorWrapperExist("t0"));
  EXPECT_FALSE(wrapper->IsQnnTensorWrapperExist("t1"));
}

// Verifies AddParamWrapper adds a scalar param successfully (returns true).
TEST(QnnUnit_ModelWrapperTest, AddParamWrapper_ScalarParam_AddsSuccessfully) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  Qnn_Scalar_t scalar{};
  scalar.dataType = QNN_DATATYPE_UINT_32;
  scalar.uint32Value = 42u;
  qnn::QnnParamWrapper param(0, "node0", "stride", scalar);

  EXPECT_TRUE(wrapper->AddParamWrapper(std::move(param)));
}

// Verifies that adding a duplicate param returns true without re-inserting.
TEST(QnnUnit_ModelWrapperTest, AddParamWrapper_DuplicateParam_ReturnsTrueWithoutOverwrite) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  Qnn_Scalar_t scalar1{};
  scalar1.dataType = QNN_DATATYPE_UINT_32;
  scalar1.uint32Value = 1u;
  qnn::QnnParamWrapper param1(0, "node0", "k", scalar1);

  Qnn_Scalar_t scalar2{};
  scalar2.dataType = QNN_DATATYPE_UINT_32;
  scalar2.uint32Value = 2u;
  qnn::QnnParamWrapper param2(0, "node0", "k", scalar2);

  ASSERT_TRUE(wrapper->AddParamWrapper(std::move(param1)));
  EXPECT_TRUE(wrapper->AddParamWrapper(std::move(param2)));  // duplicate — still succeeds
}

// Verifies SetTensorNameOverride is a no-op when tensor_name_overrides_ is null
// (the default wrapper constructed by QnnModelWrapperTestContext).
TEST(QnnUnit_ModelWrapperTest, SetTensorNameOverride_NullOverrideMap_IsNoOp) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  // Should not crash — tensor_name_overrides_ is nullptr, so call is a no-op.
  EXPECT_NO_THROW(wrapper->SetTensorNameOverride("internal_name", "original_name"));
}

// Verifies GetOnnxShape returns false for a nullopt (dynamic shape).
TEST(QnnUnit_ModelWrapperTest, GetOnnxShape_NullOpt_ReturnsFalse) {
  std::vector<uint32_t> shape;
  EXPECT_FALSE(qnn::QnnModelWrapper::GetOnnxShape(std::nullopt, shape));
  EXPECT_TRUE(shape.empty());
}

// Verifies GetOnnxShape produces {1} for a scalar (empty dim list).
TEST(QnnUnit_ModelWrapperTest, GetOnnxShape_Scalar_ReturnsOneElement) {
  std::vector<uint32_t> shape;
  EXPECT_TRUE(qnn::QnnModelWrapper::GetOnnxShape(std::vector<int64_t>{}, shape));
  ASSERT_EQ(shape.size(), 1u);
  EXPECT_EQ(shape[0], 1u);
}

// Verifies GetOnnxShape converts positive dims correctly.
TEST(QnnUnit_ModelWrapperTest, GetOnnxShape_ValidDims_ConvertsCorrectly) {
  std::vector<uint32_t> shape;
  EXPECT_TRUE(qnn::QnnModelWrapper::GetOnnxShape(std::vector<int64_t>{1, 3, 224, 224}, shape));
  ASSERT_EQ(shape.size(), 4u);
  EXPECT_EQ(shape[0], 1u);
  EXPECT_EQ(shape[1], 3u);
  EXPECT_EQ(shape[2], 224u);
  EXPECT_EQ(shape[3], 224u);
}

// Verifies GetOnnxShape returns false when any dim is negative (dynamic).
TEST(QnnUnit_ModelWrapperTest, GetOnnxShape_NegativeDim_ReturnsFalse) {
  std::vector<uint32_t> shape;
  EXPECT_FALSE(qnn::QnnModelWrapper::GetOnnxShape(std::vector<int64_t>{1, -1, 224}, shape));
}

// ── ComposeQnnGraph ───────────────────────────────────────────────────────

// ComposeQnnGraph returns false immediately when no ops have been added.
TEST(QnnUnit_ModelWrapperTest, ComposeQnnGraph_EmptyOpList_ReturnsFalse) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  EXPECT_FALSE(wrapper->ComposeQnnGraph());
}

// ── GetGraphInputTensorWrappers / GetGraphOutputTensorWrappers ────────────

// When input_info.names is empty, GetGraphInputTensorWrappers returns an empty vector.
TEST(QnnUnit_ModelWrapperTest, GetGraphInputTensorWrappers_EmptyNames_ReturnsEmpty) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  std::vector<qnn::QnnTensorWrapper> result = wrapper->GetGraphInputTensorWrappers();
  EXPECT_TRUE(result.empty());
}

// A tensor registered in input_info.names and added to the wrapper must be
// returned (and moved out of the internal map) by GetGraphInputTensorWrappers.
TEST(QnnUnit_ModelWrapperTest, GetGraphInputTensorWrappers_ExistingTensor_ReturnsMoved) {
  QnnModelWrapperTestContext ctx;
  ctx.input_info.names.push_back("t_in");
  ctx.input_info.indices["t_in"] = 0;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  qnn::QnnTensorWrapper tensor("t_in", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                               qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(tensor)));

  std::vector<qnn::QnnTensorWrapper> result = wrapper->GetGraphInputTensorWrappers();
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].GetName(), "t_in");
  // The tensor was moved out — it should no longer exist in the wrapper.
  EXPECT_FALSE(wrapper->IsQnnTensorWrapperExist("t_in"));
}

// Same for outputs.
TEST(QnnUnit_ModelWrapperTest, GetGraphOutputTensorWrappers_ExistingTensor_ReturnsMoved) {
  QnnModelWrapperTestContext ctx;
  ctx.output_info.names.push_back("t_out");
  ctx.output_info.indices["t_out"] = 0;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  qnn::QnnTensorWrapper tensor("t_out", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32,
                               qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(tensor)));

  std::vector<qnn::QnnTensorWrapper> result = wrapper->GetGraphOutputTensorWrappers();
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].GetName(), "t_out");
  EXPECT_FALSE(wrapper->IsQnnTensorWrapperExist("t_out"));
}

// With offload_graph_io_quantization=true and an override in the map,
// GetGraphInputTensorWrappers must apply SetResolvedTensorName on the returned wrapper.
TEST(QnnUnit_ModelWrapperTest, GetGraphInputTensorWrappers_OffloadIOQuant_ResolvesOverrideName) {
  QnnModelWrapperTestContext ctx;
  ctx.input_info.names.push_back("internal_t");
  ctx.input_info.indices["internal_t"] = 0;
  qnn::ModelSettings settings{};
  settings.offload_graph_io_quantization = true;
  std::unordered_map<std::string, std::string> overrides;
  overrides["internal_t"] = "original_t";
  auto wrapper = MakeWrapperWithOverrides(ctx, settings, &overrides);

  qnn::QnnTensorWrapper tensor("internal_t", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                               qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{1});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(tensor)));

  std::vector<qnn::QnnTensorWrapper> result = wrapper->GetGraphInputTensorWrappers();
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].GetName(), "internal_t");                // original key
  EXPECT_EQ(result[0].GetResolvedTensorName(), "original_t");  // resolved via override
}

// ── AddCastNode ───────────────────────────────────────────────────────────

// If the output tensor already exists, AddCastNode returns OK without adding a
// duplicate node.
TEST(QnnUnit_ModelWrapperTest, AddCastNode_OutputAlreadyExists_ReturnsOK) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  // Pre-add the output tensor.
  qnn::QnnTensorWrapper out("cast_out", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_UINT_8,
                            qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(out)));

  Ort::Status status = wrapper->AddCastNode(
      "cast0", "in0", "cast_out",
      QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_UINT_8,
      qnn::QnnQuantParamsWrapper(), {4u},
      /*do_op_validation=*/false);
  EXPECT_TRUE(status.IsOK());
}

// When the output tensor is new, AddCastNode must add the tensor and the node.
TEST(QnnUnit_ModelWrapperTest, AddCastNode_NewOutput_AddsNodeAndTensor) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  Ort::Status status = wrapper->AddCastNode(
      "cast0", "in0", "cast_out",
      QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_UINT_8,
      qnn::QnnQuantParamsWrapper(), {4u},
      /*do_op_validation=*/false);
  EXPECT_TRUE(status.IsOK());
  EXPECT_TRUE(wrapper->IsQnnTensorWrapperExist("cast_out"));
}

// ── AddReshapeNode ────────────────────────────────────────────────────────

// Normal reshape with default (non-per-channel) quant params and do_op_validation=false
// must add both input and output tensors and return OK.
TEST(QnnUnit_ModelWrapperTest, AddReshapeNode_Normal_NoOpValidation_AddsNodes) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  Ort::Status status = wrapper->AddReshapeNode(
      "reshape_in", "reshape_out",
      {1u, 12u}, {12u},
      QNN_DATATYPE_FLOAT_32,
      qnn::QnnQuantParamsWrapper(),
      /*do_op_validation=*/false);
  EXPECT_TRUE(status.IsOK());
  EXPECT_TRUE(wrapper->IsQnnTensorWrapperExist("reshape_in"));
  EXPECT_TRUE(wrapper->IsQnnTensorWrapperExist("reshape_out"));
}

// The single-quant-param overload must return an error for per-channel quant.
TEST(QnnUnit_ModelWrapperTest, AddReshapeNode_PerChannelQuant_ReturnsError) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  std::vector<float> scales{1.0f, 2.0f};
  std::vector<int32_t> offsets{0, 0};
  qnn::QnnQuantParamsWrapper per_channel_quant = qnn::QnnQuantParamsWrapper::PerChannel(
      gsl::make_span(scales.data(), scales.size()),
      gsl::make_span(offsets.data(), offsets.size()),
      /*axis=*/0);

  Ort::Status status = wrapper->AddReshapeNode(
      "in", "out", {2u}, {2u},
      QNN_DATATYPE_UINT_8, per_channel_quant,
      /*do_op_validation=*/false);
  EXPECT_FALSE(status.IsOK());
}

// ── AddTransposeNode ──────────────────────────────────────────────────────

// With is_for_input=true, AddTransposeNode adds input tensor + perm param + output tensor.
TEST(QnnUnit_ModelWrapperTest, AddTransposeNode_Normal_IsForInputTrue_AddsAll) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  Ort::Status status = wrapper->AddTransposeNode(
      0, "t_in", "t_out",
      {1u, 3u, 4u}, {0u, 2u, 1u}, {1u, 4u, 3u},
      QNN_DATATYPE_FLOAT_32, qnn::QnnQuantParamsWrapper(),
      /*do_op_validation=*/false,
      /*is_for_input=*/true, /*is_for_output=*/false);
  EXPECT_TRUE(status.IsOK());
  EXPECT_TRUE(wrapper->IsQnnTensorWrapperExist("t_in"));
  EXPECT_TRUE(wrapper->IsQnnTensorWrapperExist("t_out"));
}

// With is_for_input=false and the input tensor already present (as if it was
// added by the previous node), AddTransposeNode must leave it unchanged and only
// add the perm param + output tensor.
TEST(QnnUnit_ModelWrapperTest, AddTransposeNode_Normal_IsForInputFalse_SkipsInputTensor) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  // Pre-add "t_in" to simulate it being the output of an upstream node.
  qnn::QnnTensorWrapper existing_in("t_in", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                                    qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{1u, 3u, 4u});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(existing_in)));

  Ort::Status status = wrapper->AddTransposeNode(
      0, "t_in", "t_out",
      {1u, 3u, 4u}, {0u, 2u, 1u}, {1u, 4u, 3u},
      QNN_DATATYPE_FLOAT_32, qnn::QnnQuantParamsWrapper(),
      /*do_op_validation=*/false,
      /*is_for_input=*/false, /*is_for_output=*/false);
  EXPECT_TRUE(status.IsOK());
  // t_in already existed — it must still be there and not have been replaced.
  EXPECT_TRUE(wrapper->IsQnnTensorWrapperExist("t_in"));
  EXPECT_TRUE(wrapper->IsQnnTensorWrapperExist("t_out"));
}

// Per-channel quant param must be rejected.
TEST(QnnUnit_ModelWrapperTest, AddTransposeNode_PerChannelQuant_ReturnsError) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  std::vector<float> scales{1.0f, 2.0f};
  std::vector<int32_t> offsets{0, 0};
  qnn::QnnQuantParamsWrapper per_channel_quant = qnn::QnnQuantParamsWrapper::PerChannel(
      gsl::make_span(scales.data(), scales.size()),
      gsl::make_span(offsets.data(), offsets.size()),
      /*axis=*/0);

  Ort::Status status = wrapper->AddTransposeNode(
      0, "t_in", "t_out",
      {1u, 3u}, {1u, 0u}, {3u, 1u},
      QNN_DATATYPE_FLOAT_32, per_channel_quant,
      /*do_op_validation=*/false);
  EXPECT_FALSE(status.IsOK());
}

// ── BF16 conversion path (via CreateQnnNode with do_op_validation=true) ──

namespace {
Qnn_ErrorHandle_t StubBackendValidateOpConfig(Qnn_BackendHandle_t, Qnn_OpConfig_t) {
  return QNN_BACKEND_NO_ERROR;
}
}  // namespace

// When htp_bf16_enable=true and do_op_validation=true, CreateQnnNode must:
//   1. Temporarily convert FP32 tensors to BF16 (ApplyBF16ConversionForValidation)
//   2. Call backendValidateOpConfig via the stub
//   3. Restore FP32 on exit (via BF16ConversionGuard / RestoreFP32AfterValidation)
TEST(QnnUnit_ModelWrapperTest, CreateQnnNode_BF16Enabled_OpValidation_AppliesAndRestoresConversion) {
  QnnModelWrapperTestContext ctx;
  ctx.qnn_interface.backendValidateOpConfig = StubBackendValidateOpConfig;
  qnn::ModelSettings settings{};
  settings.htp_bf16_enable = true;
  auto wrapper = ctx.CreateWrapper(settings, qnn::QnnBackendType::HTP);

  // Add two FP32 NATIVE tensors (not graph I/O).
  qnn::QnnTensorWrapper in("in0", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                           qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  qnn::QnnTensorWrapper out("out0", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                            qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(in)));
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(out)));

  // Validation path with BF16 enabled.
  bool ok = wrapper->CreateQnnNode("node0",
                                   QNN_OP_PACKAGE_NAME_QTI_AISW,
                                   QNN_OP_CAST,
                                   {"in0"}, {"out0"}, {},
                                   /*do_op_validation=*/true);
  EXPECT_TRUE(ok);

  // BF16ConversionGuard must have restored FP32 after CreateQnnNode returns.
  EXPECT_EQ(wrapper->GetQnnTensorWrapper("in0").GetTensorDataType(), QNN_DATATYPE_FLOAT_32);
  EXPECT_EQ(wrapper->GetQnnTensorWrapper("out0").GetTensorDataType(), QNN_DATATYPE_FLOAT_32);
}

// ── AddTensorWrapper extra paths ──────────────────────────────────────────

// An empty tensor name should be rejected immediately.
TEST(QnnUnit_ModelWrapperTest, AddTensorWrapper_EmptyName_ReturnsFalse) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  qnn::QnnTensorWrapper tensor("", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                               qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  EXPECT_FALSE(wrapper->AddTensorWrapper(std::move(tensor)));
}

// Adding a tensor with an already-registered name should return true (idempotent)
// and leave the original entry unchanged.
TEST(QnnUnit_ModelWrapperTest, AddTensorWrapper_DuplicateTensor_ReturnsTrueKeepsOriginal) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  qnn::QnnTensorWrapper t1("dup", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                           qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(t1)));
  ASSERT_TRUE(wrapper->IsQnnTensorWrapperExist("dup"));

  // Second add: same name, different dtype — should still succeed without overwrite.
  qnn::QnnTensorWrapper t2("dup", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_UINT_8,
                           qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{8});
  EXPECT_TRUE(wrapper->AddTensorWrapper(std::move(t2)));

  // Original FLOAT_32 entry must be preserved.
  EXPECT_EQ(wrapper->GetQnnTensorWrapper("dup").GetTensorDataType(), QNN_DATATYPE_FLOAT_32);
}

// When htp_shared_memory is enabled, a tensor that is a graph input must get
// QNN_TENSORMEMTYPE_MEMHANDLE assigned by SetTensorMemTypeFromSettings.
TEST(QnnUnit_ModelWrapperTest, AddTensorWrapper_HtpSharedMemory_GraphInput_SetsMemHandle) {
  QnnModelWrapperTestContext ctx;
  ctx.input_info.indices["t_in"] = 0;
  qnn::ModelSettings settings{};
  settings.htp_shared_memory = true;
  auto wrapper = ctx.CreateWrapper(settings);

  qnn::QnnTensorWrapper tensor("t_in", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                               qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{1});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(tensor)));

  EXPECT_EQ(qnn::GetQnnTensorMemType(wrapper->GetQnnTensorWrapper("t_in").GetQnnTensor()),
            QNN_TENSORMEMTYPE_MEMHANDLE);
}

// With htp_shared_memory enabled but the tensor is NOT a graph I/O, mem type stays RAW.
TEST(QnnUnit_ModelWrapperTest, AddTensorWrapper_HtpSharedMemory_NonGraphIO_KeepsRaw) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  settings.htp_shared_memory = true;
  auto wrapper = ctx.CreateWrapper(settings);

  qnn::QnnTensorWrapper tensor("t_nat", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                               qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{1});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(tensor)));

  EXPECT_EQ(qnn::GetQnnTensorMemType(wrapper->GetQnnTensorWrapper("t_nat").GetQnnTensor()),
            QNN_TENSORMEMTYPE_RAW);
}

// ── GetQnnTensorWrapper ───────────────────────────────────────────────────

// A tensor that was added must be retrievable by name with matching metadata.
TEST(QnnUnit_ModelWrapperTest, GetQnnTensorWrapper_ExistingTensor_ReturnsRef) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  qnn::QnnTensorWrapper tensor("t0", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_UINT_8,
                               qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{2, 2});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(tensor)));

  const qnn::QnnTensorWrapper& ref = wrapper->GetQnnTensorWrapper("t0");
  EXPECT_EQ(ref.GetName(), "t0");
  EXPECT_EQ(ref.GetTensorDataType(), QNN_DATATYPE_UINT_8);
}

// Requesting a tensor that was never added must throw.
TEST(QnnUnit_ModelWrapperTest, GetQnnTensorWrapper_NonExistingTensor_Throws) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  EXPECT_THROW(wrapper->GetQnnTensorWrapper("does_not_exist"), Ort::Exception);
}

// ── IsGraphInput / IsGraphOutput ─────────────────────────────────────────

TEST(QnnUnit_ModelWrapperTest, IsGraphInput_TrueForRegisteredName) {
  QnnModelWrapperTestContext ctx;
  ctx.input_info.indices["graph_in"] = 0;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  EXPECT_TRUE(wrapper->IsGraphInput("graph_in"));
  EXPECT_FALSE(wrapper->IsGraphInput("other"));
}

TEST(QnnUnit_ModelWrapperTest, IsGraphOutput_TrueForRegisteredName) {
  QnnModelWrapperTestContext ctx;
  ctx.output_info.indices["graph_out"] = 0;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  EXPECT_TRUE(wrapper->IsGraphOutput("graph_out"));
  EXPECT_FALSE(wrapper->IsGraphOutput("other"));
}

// ── Getters ───────────────────────────────────────────────────────────────

TEST(QnnUnit_ModelWrapperTest, GetModelSettings_ReturnsConstructedSettings) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  settings.htp_bf16_enable = true;
  settings.offload_graph_io_quantization = true;
  auto wrapper = ctx.CreateWrapper(settings);

  const qnn::ModelSettings& got = wrapper->GetModelSettings();
  EXPECT_TRUE(got.htp_bf16_enable);
  EXPECT_TRUE(got.offload_graph_io_quantization);
  EXPECT_FALSE(got.htp_shared_memory);
}

TEST(QnnUnit_ModelWrapperTest, GetQnnBackendType_ReturnsCPU) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings, qnn::QnnBackendType::CPU);

  EXPECT_EQ(wrapper->GetQnnBackendType(), qnn::QnnBackendType::CPU);
}

// ── SetTensorNameOverride (non-null map paths) ────────────────────────────

// A valid (internal, original) pair must be inserted into the map.
TEST(QnnUnit_ModelWrapperTest, SetTensorNameOverride_NonNullMap_InsertsEntry) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  std::unordered_map<std::string, std::string> overrides;
  auto wrapper = MakeWrapperWithOverrides(ctx, settings, &overrides);

  wrapper->SetTensorNameOverride("internal_name", "original_name");
  ASSERT_EQ(overrides.size(), 1u);
  EXPECT_EQ(overrides.at("internal_name"), "original_name");
}

// Calling SetTensorNameOverride twice with the same key must keep the first value.
TEST(QnnUnit_ModelWrapperTest, SetTensorNameOverride_NonNullMap_IgnoresDuplicate) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  std::unordered_map<std::string, std::string> overrides;
  auto wrapper = MakeWrapperWithOverrides(ctx, settings, &overrides);

  wrapper->SetTensorNameOverride("key", "first");
  wrapper->SetTensorNameOverride("key", "second");  // duplicate — should be ignored

  ASSERT_EQ(overrides.size(), 1u);
  EXPECT_EQ(overrides.at("key"), "first");
}

// An empty internal name must not insert any entry.
TEST(QnnUnit_ModelWrapperTest, SetTensorNameOverride_EmptyInternal_IsNoOp) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  std::unordered_map<std::string, std::string> overrides;
  auto wrapper = MakeWrapperWithOverrides(ctx, settings, &overrides);

  wrapper->SetTensorNameOverride("", "original");
  EXPECT_TRUE(overrides.empty());
}

// An empty original name must not insert any entry.
TEST(QnnUnit_ModelWrapperTest, SetTensorNameOverride_EmptyOriginal_IsNoOp) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  std::unordered_map<std::string, std::string> overrides;
  auto wrapper = MakeWrapperWithOverrides(ctx, settings, &overrides);

  wrapper->SetTensorNameOverride("internal", "");
  EXPECT_TRUE(overrides.empty());
}

// ── CreateQnnGraph ────────────────────────────────────────────────────────

// Passing a null context handle must return false before any SDK call.
TEST(QnnUnit_ModelWrapperTest, CreateQnnGraph_NullContext_ReturnsFalse) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  EXPECT_FALSE(wrapper->CreateQnnGraph(nullptr, "my_graph"));
}

// ── CreateQnnNode ─────────────────────────────────────────────────────────

// With do_op_validation=false, CreateQnnNode simply stores the op descriptor
// and returns true — no QNN SDK function is called.
TEST(QnnUnit_ModelWrapperTest, CreateQnnNode_NoValidation_ReturnsTrue) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  bool ok = wrapper->CreateQnnNode("relu0",
                                   QNN_OP_PACKAGE_NAME_QTI_AISW,
                                   QNN_OP_CAST,
                                   {"input0"},
                                   {"output0"},
                                   {},
                                   /*do_op_validation=*/false);
  EXPECT_TRUE(ok);
}

// ── ComposeQnnGraph full path via tensorCreateGraphTensor + graphAddNode stubs ──

namespace {
Qnn_ErrorHandle_t StubTensorCreateSuccess(Qnn_GraphHandle_t, Qnn_Tensor_t*) {
  return QNN_TENSOR_NO_ERROR;
}
Qnn_ErrorHandle_t StubGraphAddNode(Qnn_GraphHandle_t, Qnn_OpConfig_t) {
  return QNN_GRAPH_NO_ERROR;
}

std::vector<std::string> captured_qnn_node_names;

Qnn_ErrorHandle_t StubGraphAddNodeCaptureNames(Qnn_GraphHandle_t, Qnn_OpConfig_t op_config) {
  captured_qnn_node_names.emplace_back(op_config.v1.name);
  return QNN_GRAPH_NO_ERROR;
}
}  // namespace

// ComposeQnnGraph succeeds when tensors are registered and stubs are in place.
// Covers: ComposeQnnGraph (main path), CreateQnnInputOutputTensors (do_op_validation=false),
//         CreateQnnGraphTensor, CreateQnnGraphOp.
TEST(QnnUnit_ModelWrapperTest, ComposeQnnGraph_NativeTensors_Succeeds) {
  QnnModelWrapperTestContext ctx;
  ctx.qnn_interface.tensorCreateGraphTensor = StubTensorCreateSuccess;
  ctx.qnn_interface.graphAddNode = StubGraphAddNode;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  // No graph I/O names — RegisterGraphInputOutputInOrder is a no-op loop.
  qnn::QnnTensorWrapper in("inp", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                           qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  qnn::QnnTensorWrapper out("out", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                            qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(in)));
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(out)));

  ASSERT_TRUE(wrapper->CreateQnnNode("n0", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CAST,
                                     {"inp"}, {"out"}, {}, /*do_op_validation=*/false));

  EXPECT_TRUE(wrapper->ComposeQnnGraph());
}

TEST(QnnUnit_ModelWrapperTest, ComposeQnnGraph_DuplicateNodeNames_AreRenamed) {
  QnnModelWrapperTestContext ctx;
  ctx.qnn_interface.tensorCreateGraphTensor = StubTensorCreateSuccess;
  ctx.qnn_interface.graphAddNode = StubGraphAddNodeCaptureNames;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  qnn::QnnTensorWrapper input("input", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                              qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  qnn::QnnTensorWrapper intermediate("intermediate", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                                     qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  qnn::QnnTensorWrapper intermediate2("intermediate2", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                                      qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  qnn::QnnTensorWrapper output("output", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                               qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(input)));
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(intermediate)));
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(intermediate2)));
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(output)));
  ASSERT_TRUE(wrapper->CreateQnnNode("duplicate", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CAST,
                                     {"input"}, {"intermediate"}, {}, false));
  ASSERT_TRUE(wrapper->CreateQnnNode("duplicate", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CAST,
                                     {"intermediate"}, {"intermediate2"}, {}, false));
  ASSERT_TRUE(wrapper->CreateQnnNode("duplicate_2", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CAST,
                                     {"intermediate2"}, {"output"}, {}, false));

  captured_qnn_node_names.clear();
  ASSERT_TRUE(wrapper->ComposeQnnGraph());
  ASSERT_EQ(captured_qnn_node_names.size(), 3u);
  EXPECT_EQ(captured_qnn_node_names[0], "duplicate");
  EXPECT_EQ(captured_qnn_node_names[1], "duplicate_2");
  EXPECT_EQ(captured_qnn_node_names[2], "duplicate_2_2");
}

// With graph I/O names set, RegisterGraphInputOutputInOrder registers APP_WRITE / APP_READ
// tensors before processing ops.
// Covers: RegisterGraphInputOutputInOrder (non-empty path).
TEST(QnnUnit_ModelWrapperTest, RegisterGraphInputOutputInOrder_AppTypeIOTensors_RegisteredBeforeOps) {
  QnnModelWrapperTestContext ctx;
  ctx.input_info.names.push_back("inp");
  ctx.input_info.indices["inp"] = 0;
  ctx.output_info.names.push_back("out");
  ctx.output_info.indices["out"] = 0;
  ctx.qnn_interface.tensorCreateGraphTensor = StubTensorCreateSuccess;
  ctx.qnn_interface.graphAddNode = StubGraphAddNode;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  qnn::QnnTensorWrapper in("inp", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                           qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  qnn::QnnTensorWrapper out("out", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32,
                            qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(in)));
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(out)));

  ASSERT_TRUE(wrapper->CreateQnnNode("n0", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CAST,
                                     {"inp"}, {"out"}, {}, false));

  EXPECT_TRUE(wrapper->ComposeQnnGraph());
}

// A scalar qnn::QnnParamWrapper can be passed through ComposeQnnGraph.
// Covers: CreateQnnParamTensors (scalar-param branch — no QNN SDK call).
TEST(QnnUnit_ModelWrapperTest, ComposeQnnGraph_WithScalarParam_Succeeds) {
  QnnModelWrapperTestContext ctx;
  ctx.qnn_interface.tensorCreateGraphTensor = StubTensorCreateSuccess;
  ctx.qnn_interface.graphAddNode = StubGraphAddNode;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  qnn::QnnTensorWrapper in("inp", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                           qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  qnn::QnnTensorWrapper out("out", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                            qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(in)));
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(out)));

  Qnn_Scalar_t scalar{};
  scalar.dataType = QNN_DATATYPE_UINT_32;
  scalar.uint32Value = 1u;
  qnn::QnnParamWrapper param(0, "n0", "stride", scalar);
  std::string param_key = param.GetParamTensorName();  // "n0_0_stride"
  ASSERT_TRUE(wrapper->AddParamWrapper(std::move(param)));

  ASSERT_TRUE(wrapper->CreateQnnNode("n0", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CAST,
                                     {"inp"}, {"out"}, {param_key}, false));

  EXPECT_TRUE(wrapper->ComposeQnnGraph());
}

// With htp_bf16_enable=true, ComposeQnnGraph invokes ProcessBF16Conversions which
// inserts FP32→BF16 cast ops for graph inputs and BF16→FP32 cast ops for graph outputs.
// All inputs are graph inputs so IsConstantInput is short-circuited (safe with null OrtApi).
// Covers: ProcessBF16Conversions, ProcessBF16InputConversion, ProcessBF16OutputConversion,
//         CreateBF16CastTensor.
TEST(QnnUnit_ModelWrapperTest, ComposeQnnGraph_BF16Enabled_ProcessesConversions) {
  QnnModelWrapperTestContext ctx;
  ctx.input_info.names.push_back("inp");
  ctx.input_info.indices["inp"] = 0;
  ctx.output_info.names.push_back("out");
  ctx.output_info.indices["out"] = 0;
  ctx.qnn_interface.tensorCreateGraphTensor = StubTensorCreateSuccess;
  ctx.qnn_interface.graphAddNode = StubGraphAddNode;
  qnn::ModelSettings settings{};
  settings.htp_bf16_enable = true;
  auto wrapper = ctx.CreateWrapper(settings, qnn::QnnBackendType::HTP);

  // FP32 graph I/O tensors — BF16 conversion will insert Cast ops around the main op.
  qnn::QnnTensorWrapper in("inp", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                           qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  qnn::QnnTensorWrapper out("out", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32,
                            qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(in)));
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(out)));

  ASSERT_TRUE(wrapper->CreateQnnNode("n0", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CAST,
                                     {"inp"}, {"out"}, {}, false));

  EXPECT_TRUE(wrapper->ComposeQnnGraph());
}

// ── CreateQnnGraph extra paths, validation, error paths, Graph_GetNumInitializers stub ──

namespace {
// Stub for graphCreate: sets graph_ to a non-null sentinel and returns success.
static int g_graph_sentinel = 1;
Qnn_ErrorHandle_t StubGraphCreate(Qnn_ContextHandle_t, const char*,
                                  const QnnGraph_Config_t**, Qnn_GraphHandle_t* graph_out) {
  *graph_out = reinterpret_cast<Qnn_GraphHandle_t>(&g_graph_sentinel);
  return QNN_GRAPH_NO_ERROR;
}

// Stub for Graph_GetNumInitializers: returns 0 initializers (no initializers in the graph).
// This makes IsConstantInput() return false safely without any real OrtApi context.
OrtStatus* StubGetNumInitializersZero(const OrtGraph*, size_t* count) noexcept {
  *count = 0;
  return nullptr;  // nullptr = success (no ORT error)
}
// Stub for Graph_GetInitializers: no-op for empty initializer list.
OrtStatus* StubGetInitializersEmpty(const OrtGraph*, const OrtValueInfo**, size_t) noexcept {
  return nullptr;
}
}  // namespace

// A non-null context with an empty graph name must return false (line ~31-33).
TEST(QnnUnit_ModelWrapperTest, CreateQnnGraph_EmptyName_ReturnsFalse) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  auto fake_ctx = reinterpret_cast<Qnn_ContextHandle_t>(static_cast<uintptr_t>(1));
  EXPECT_FALSE(wrapper->CreateQnnGraph(fake_ctx, ""));
}

// First call succeeds (graphCreate stub sets graph_ non-null); second call returns false
// because graph_name_ is already set (line ~22-25).
// Also covers the successful create path (lines ~36-49).
TEST(QnnUnit_ModelWrapperTest, CreateQnnGraph_AlreadyInitialized_ReturnsFalse) {
  QnnModelWrapperTestContext ctx;
  ctx.qnn_interface.graphCreate = StubGraphCreate;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  auto fake_ctx = reinterpret_cast<Qnn_ContextHandle_t>(static_cast<uintptr_t>(1));
  EXPECT_TRUE(wrapper->CreateQnnGraph(fake_ctx, "my_graph"));
  EXPECT_FALSE(wrapper->CreateQnnGraph(fake_ctx, "my_graph"));
}

// CreateQnnNode with do_op_validation=true and BF16 disabled calls backendValidateOpConfig.
// Covers the else-branch (no BF16 conversion) in CreateQnnNode (line ~464-466).
TEST(QnnUnit_ModelWrapperTest, CreateQnnNode_WithValidation_NoBF16_Succeeds) {
  QnnModelWrapperTestContext ctx;
  ctx.qnn_interface.backendValidateOpConfig = StubBackendValidateOpConfig;
  qnn::ModelSettings settings{};
  // htp_bf16_enable defaults to false — exercises the non-BF16 validation path.
  auto wrapper = ctx.CreateWrapper(settings, qnn::QnnBackendType::HTP);

  qnn::QnnTensorWrapper in("in0", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                           qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  qnn::QnnTensorWrapper out("out0", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                            qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(in)));
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(out)));

  bool ok = wrapper->CreateQnnNode("n0", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CAST,
                                   {"in0"}, {"out0"}, {}, /*do_op_validation=*/true);
  EXPECT_TRUE(ok);
}

// If an op references an input tensor not in model_tensors_map_,
// CreateQnnInputOutputTensors returns false → ComposeQnnGraph returns false (line ~173-174).
TEST(QnnUnit_ModelWrapperTest, ComposeQnnGraph_MissingInputTensor_ReturnsFalse) {
  QnnModelWrapperTestContext ctx;
  ctx.qnn_interface.tensorCreateGraphTensor = StubTensorCreateSuccess;
  ctx.qnn_interface.graphAddNode = StubGraphAddNode;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  // Add only the output tensor; "nonexistent_in" is never added to the map.
  qnn::QnnTensorWrapper out("out", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                            qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(out)));
  ASSERT_TRUE(wrapper->CreateQnnNode("n0", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CAST,
                                     {"nonexistent_in"}, {"out"}, {}, false));

  EXPECT_FALSE(wrapper->ComposeQnnGraph());
}

// If an op references a param that was never added via AddParamWrapper,
// CreateQnnParamTensors returns false → ComposeQnnGraph returns false (line ~206-207).
TEST(QnnUnit_ModelWrapperTest, ComposeQnnGraph_MissingParam_ReturnsFalse) {
  QnnModelWrapperTestContext ctx;
  ctx.qnn_interface.tensorCreateGraphTensor = StubTensorCreateSuccess;
  ctx.qnn_interface.graphAddNode = StubGraphAddNode;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  qnn::QnnTensorWrapper in("inp", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                           qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  qnn::QnnTensorWrapper out("out", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                            qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(in)));
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(out)));
  // Reference a param that was never added to model_params_map_.
  ASSERT_TRUE(wrapper->CreateQnnNode("n0", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CAST,
                                     {"inp"}, {"out"}, {"missing_param"}, false));

  EXPECT_FALSE(wrapper->ComposeQnnGraph());
}

// With BF16 enabled and a NATIVE FP32 intermediate input (not a graph input or constant),
// ProcessBF16InputConversion converts the tensor's dtype in-place (line ~300-303).
// IsConstantInput is safe here because Graph_GetNumInitializers stub returns 0 initializers.
TEST(QnnUnit_ModelWrapperTest, ComposeQnnGraph_BF16_NativeIntermediateInput_ConvertedInPlace) {
  QnnModelWrapperTestContext ctx;
  // Register "out" as graph output (APP_READ).
  ctx.output_info.names.push_back("out");
  ctx.output_info.indices["out"] = 0;
  ctx.qnn_interface.tensorCreateGraphTensor = StubTensorCreateSuccess;
  ctx.qnn_interface.graphAddNode = StubGraphAddNode;
  // Stub Graph_GetNumInitializers to return 0 — makes IsConstantInput safely return false.
  ctx.stub_ort_api.Graph_GetNumInitializers = StubGetNumInitializersZero;
  // Also stub Graph_GetInitializers (called even when num=0) to avoid null-pointer SEGFAULT.
  ctx.stub_ort_api.Graph_GetInitializers = StubGetInitializersEmpty;
  qnn::ModelSettings settings{};
  settings.htp_bf16_enable = true;
  auto wrapper = ctx.CreateWrapper(settings, qnn::QnnBackendType::HTP);

  // "mid" is a NATIVE FP32 tensor that is NOT a graph input or output.
  // ProcessBF16InputConversion hits the NATIVE && FP32 branch and converts it in-place.
  qnn::QnnTensorWrapper mid("mid", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                            qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  // "out" is a graph output (FP32, APP_READ).
  qnn::QnnTensorWrapper out("out", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32,
                            qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(mid)));
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(out)));

  ASSERT_TRUE(wrapper->CreateQnnNode("n0", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CAST,
                                     {"mid"}, {"out"}, {}, false));

  EXPECT_TRUE(wrapper->ComposeQnnGraph());
}

// ── ProcessBF16OutputConversion NATIVE path, RegisterGraphInputOutputInOrder skips,
//           ProcessBF16InputConversion reuse + STATIC branch ──────────────────────────────

// ProcessBF16OutputConversion: NATIVE FP32 output NOT in graph_outputs is converted to BF16
// in-place (line ~364-368).  Input is a graph input so IsConstantInput is short-circuited.
TEST(QnnUnit_ModelWrapperTest, ComposeQnnGraph_BF16_NativeFP32Output_ConvertedInPlace) {
  QnnModelWrapperTestContext ctx;
  // "inp" is a graph input → IsGraphInput short-circuits IsConstantInput (no OrtApi needed).
  ctx.input_info.names.push_back("inp");
  ctx.input_info.indices["inp"] = 0;
  ctx.qnn_interface.tensorCreateGraphTensor = StubTensorCreateSuccess;
  ctx.qnn_interface.graphAddNode = StubGraphAddNode;
  qnn::ModelSettings settings{};
  settings.htp_bf16_enable = true;
  auto wrapper = ctx.CreateWrapper(settings, qnn::QnnBackendType::HTP);

  // "inp": APP_WRITE FP32 graph input → ProcessBF16InputConversion inserts FP32→BF16 cast.
  qnn::QnnTensorWrapper in("inp", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                           qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  // "mid": NATIVE FP32, NOT a graph output → ProcessBF16OutputConversion converts dtype in-place.
  qnn::QnnTensorWrapper mid("mid", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                            qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(in)));
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(mid)));

  ASSERT_TRUE(wrapper->CreateQnnNode("n0", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CAST,
                                     {"inp"}, {"mid"}, {}, false));

  EXPECT_TRUE(wrapper->ComposeQnnGraph());
}

// RegisterGraphInputOutputInOrder: a tensor declared as a graph input but registered as
// NATIVE (APP_WRITE expected) is skipped at the type check (line ~595 continue).
// ComposeQnnGraph still succeeds because the tensor is created in the op loop.
TEST(QnnUnit_ModelWrapperTest, RegisterGraphInputOutputInOrder_WrongType_SkipsPreRegistration) {
  QnnModelWrapperTestContext ctx;
  // "inp" declared as graph input, but the tensor is NATIVE — type mismatch → skipped.
  ctx.input_info.names.push_back("inp");
  ctx.input_info.indices["inp"] = 0;
  ctx.qnn_interface.tensorCreateGraphTensor = StubTensorCreateSuccess;
  ctx.qnn_interface.graphAddNode = StubGraphAddNode;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  qnn::QnnTensorWrapper in("inp", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                           qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  qnn::QnnTensorWrapper out("out", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                            qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(in)));
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(out)));
  ASSERT_TRUE(wrapper->CreateQnnNode("n0", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CAST,
                                     {"inp"}, {"out"}, {}, false));

  // Even though "inp" was skipped in pre-registration, op-loop creates it → succeeds.
  EXPECT_TRUE(wrapper->ComposeQnnGraph());
}

// RegisterGraphInputOutputInOrder: after the first ComposeQnnGraph call, APP_WRITE/APP_READ
// tensors are in tensor_created_map_.  Second call hits "already in map" (line ~598 continue).
TEST(QnnUnit_ModelWrapperTest, RegisterGraphInputOutputInOrder_AlreadyCreated_Skips) {
  QnnModelWrapperTestContext ctx;
  ctx.input_info.names.push_back("inp");
  ctx.input_info.indices["inp"] = 0;
  ctx.output_info.names.push_back("out");
  ctx.output_info.indices["out"] = 0;
  ctx.qnn_interface.tensorCreateGraphTensor = StubTensorCreateSuccess;
  ctx.qnn_interface.graphAddNode = StubGraphAddNode;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  qnn::QnnTensorWrapper in("inp", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                           qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  qnn::QnnTensorWrapper out("out", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32,
                            qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(in)));
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(out)));
  ASSERT_TRUE(wrapper->CreateQnnNode("n0", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CAST,
                                     {"inp"}, {"out"}, {}, false));

  // First call: tensors are created and entered into tensor_created_map_.
  EXPECT_TRUE(wrapper->ComposeQnnGraph());
  // Second call: RegisterGraphInputOutputInOrder finds them in map → continue (line ~598).
  EXPECT_TRUE(wrapper->ComposeQnnGraph());
}

// ProcessBF16InputConversion: on the second ComposeQnnGraph call, the BF16 cast output tensor
// ("inp_bf16_intermediate") already exists in model_tensors_map_.
// The "if (!IsQnnTensorWrapperExist(cast_output_name))" check (line ~282) evaluates to false,
// skipping tensor and op creation — the existing name is reused directly.
TEST(QnnUnit_ModelWrapperTest, ComposeQnnGraph_BF16_CalledTwice_CastTensorReused) {
  QnnModelWrapperTestContext ctx;
  ctx.input_info.names.push_back("inp");
  ctx.input_info.indices["inp"] = 0;
  ctx.output_info.names.push_back("out");
  ctx.output_info.indices["out"] = 0;
  ctx.qnn_interface.tensorCreateGraphTensor = StubTensorCreateSuccess;
  ctx.qnn_interface.graphAddNode = StubGraphAddNode;
  qnn::ModelSettings settings{};
  settings.htp_bf16_enable = true;
  auto wrapper = ctx.CreateWrapper(settings, qnn::QnnBackendType::HTP);

  qnn::QnnTensorWrapper in("inp", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                           qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  qnn::QnnTensorWrapper out("out", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32,
                            qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(in)));
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(out)));
  ASSERT_TRUE(wrapper->CreateQnnNode("n0", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CAST,
                                     {"inp"}, {"out"}, {}, false));

  // First call: "inp_bf16_intermediate" is created and added to model_tensors_map_.
  EXPECT_TRUE(wrapper->ComposeQnnGraph());
  // Second call: IsQnnTensorWrapperExist("inp_bf16_intermediate") = true → skip creation (line ~282).
  EXPECT_TRUE(wrapper->ComposeQnnGraph());
}

// ProcessBF16InputConversion: STATIC FP32 tensor that is NOT a constant initializer
// (IsConstantInput returns false via Graph_GetNumInitializers stub) hits the STATIC
// non-constant branch (line ~304-323) and inserts a FP32→BF16 cast op.
TEST(QnnUnit_ModelWrapperTest, ComposeQnnGraph_BF16_StaticNonConstantInput_AddsCastOp) {
  QnnModelWrapperTestContext ctx;
  // Output is a graph output (APP_READ FP32).
  ctx.output_info.names.push_back("out");
  ctx.output_info.indices["out"] = 0;
  ctx.qnn_interface.tensorCreateGraphTensor = StubTensorCreateSuccess;
  ctx.qnn_interface.graphAddNode = StubGraphAddNode;
  // IsConstantInput safely returns false via Graph_GetNumInitializers stub (0 initializers).
  ctx.stub_ort_api.Graph_GetNumInitializers = StubGetNumInitializersZero;
  ctx.stub_ort_api.Graph_GetInitializers = StubGetInitializersEmpty;
  qnn::ModelSettings settings{};
  settings.htp_bf16_enable = true;
  auto wrapper = ctx.CreateWrapper(settings, qnn::QnnBackendType::HTP);

  // "inp": STATIC FP32, not a graph input or constant → hits STATIC non-constant branch.
  // Provide a client buffer matching the tensor size (4 floats = 16 bytes) to pass the
  // size-consistency check in CreateTensorInQnnGraph.
  std::vector<uint8_t> buf(16, 0);
  qnn::QnnTensorWrapper in("inp", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_32,
                           qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4}, std::move(buf));
  // "out": APP_READ FP32 graph output → ProcessBF16OutputConversion inserts BF16→FP32 cast.
  qnn::QnnTensorWrapper out("out", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32,
                            qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(in)));
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(out)));
  ASSERT_TRUE(wrapper->CreateQnnNode("n0", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CAST,
                                     {"inp"}, {"out"}, {}, false));

  EXPECT_TRUE(wrapper->ComposeQnnGraph());
}

// ── graphAddNode failure, RegisterGraphInputOutputInOrder failure and offload-io-quant,
//           build_json_qnn_graph path, ProcessBF16Conversions failure, missing output tensor ────

namespace {
// Any non-zero Qnn_ErrorHandle_t value signals failure.
Qnn_ErrorHandle_t StubTensorCreateFail(Qnn_GraphHandle_t, Qnn_Tensor_t*) {
  return QNN_TENSOR_ERROR_INVALID_HANDLE;
}
Qnn_ErrorHandle_t StubGraphAddNodeFail(Qnn_GraphHandle_t, Qnn_OpConfig_t) {
  return QNN_TENSOR_ERROR_INVALID_HANDLE;
}
}  // namespace

// ComposeQnnGraph: graphAddNode stub returns error → CreateQnnGraphOp fails → returns false.
// Covers lines ~674-677 (rt==false log + return false in the op loop).
TEST(QnnUnit_ModelWrapperTest, ComposeQnnGraph_GraphAddNodeFails_ReturnsFalse) {
  QnnModelWrapperTestContext ctx;
  // tensorCreateGraphTensor must succeed so input/output tensors are created successfully.
  ctx.qnn_interface.tensorCreateGraphTensor = StubTensorCreateSuccess;
  // graphAddNode forced to fail — triggers the error path in CreateQnnGraphOp.
  ctx.qnn_interface.graphAddNode = StubGraphAddNodeFail;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  qnn::QnnTensorWrapper in("inp", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                           qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  qnn::QnnTensorWrapper out("out", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32,
                            qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(in)));
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(out)));
  ASSERT_TRUE(wrapper->CreateQnnNode("n0", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CAST,
                                     {"inp"}, {"out"}, {}, false));

  EXPECT_FALSE(wrapper->ComposeQnnGraph());
}

// RegisterGraphInputOutputInOrder: tensorCreateGraphTensor fails for an APP_WRITE graph input →
// RegisterGraphInputOutputInOrder returns false at line ~609-612 →
// ComposeQnnGraph returns false at line ~629.
TEST(QnnUnit_ModelWrapperTest, RegisterGraphInputOutputInOrder_TensorCreateFails_ReturnsFalse) {
  QnnModelWrapperTestContext ctx;
  ctx.input_info.names.push_back("inp");
  ctx.input_info.indices["inp"] = 0;
  // Forced failure: CreateQnnGraphTensor → CreateTensorInQnnGraph → tensorCreateGraphTensor → fail.
  ctx.qnn_interface.tensorCreateGraphTensor = StubTensorCreateFail;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  // "inp" must be in model_tensors_map_ as APP_WRITE so all pre-checks pass before the SDK call.
  qnn::QnnTensorWrapper in("inp", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                           qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(in)));
  // Add at least one op so qnn_op_property_list_ is non-empty (passes the empty-list check).
  ASSERT_TRUE(wrapper->CreateQnnNode("n0", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CAST,
                                     {"inp"}, {"out"}, {}, false));

  EXPECT_FALSE(wrapper->ComposeQnnGraph());
}

// ComposeQnnGraph with build_json_qnn_graph=true: each completed op is added to json_qnn_graph_.
// Covers line ~681 (json_qnn_graph_.AddOp inside the main op loop).
TEST(QnnUnit_ModelWrapperTest, ComposeQnnGraph_BuildJson_Succeeds) {
  QnnModelWrapperTestContext ctx;
  ctx.qnn_interface.tensorCreateGraphTensor = StubTensorCreateSuccess;
  ctx.qnn_interface.graphAddNode = StubGraphAddNode;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  qnn::QnnTensorWrapper in("inp", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                           qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  qnn::QnnTensorWrapper out("out", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32,
                            qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(in)));
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(out)));
  ASSERT_TRUE(wrapper->CreateQnnNode("n0", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CAST,
                                     {"inp"}, {"out"}, {}, false));

  // build_json_qnn_graph=true → triggers the json_qnn_graph_.AddOp branch (line ~681).
  EXPECT_TRUE(wrapper->ComposeQnnGraph(/*build_json_qnn_graph=*/true));
}

// ProcessBF16Conversions failure: BF16 enabled, but an op input is absent from model_tensors_map_.
// ProcessBF16InputConversion finds the missing tensor at line ~268 and returns false.
// ProcessBF16Conversions returns false (lines ~529-532) → ComposeQnnGraph returns false (~637-638).
// No OrtApi stubs needed: IsConstantInput is never reached (fail happens before line 276).
TEST(QnnUnit_ModelWrapperTest, ComposeQnnGraph_BF16_ProcessBF16Conversions_InputNotInMap_ReturnsFalse) {
  QnnModelWrapperTestContext ctx;
  // Empty input/output info → RegisterGraphInputOutputInOrder is a no-op → always returns true.
  qnn::ModelSettings settings{};
  settings.htp_bf16_enable = true;
  auto wrapper = ctx.CreateWrapper(settings, qnn::QnnBackendType::HTP);

  // Add "out" but intentionally omit "missing_inp" to trigger the BF16 input-not-found failure.
  qnn::QnnTensorWrapper out("out", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32,
                            qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(out)));
  ASSERT_TRUE(wrapper->CreateQnnNode("n0", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CAST,
                                     {"missing_inp"}, {"out"}, {}, false));

  EXPECT_FALSE(wrapper->ComposeQnnGraph());
}

// RegisterGraphInputOutputInOrder with offload_graph_io_quantization=true and a matching override:
// hits lines ~602-604 (SetResolvedTensorName inside the offload branch of the pre-register loop).
TEST(QnnUnit_ModelWrapperTest, RegisterGraphInputOutputInOrder_OffloadIoQuant_SetsResolvedName) {
  QnnModelWrapperTestContext ctx;
  ctx.input_info.names.push_back("inp");
  ctx.input_info.indices["inp"] = 0;
  ctx.qnn_interface.tensorCreateGraphTensor = StubTensorCreateSuccess;
  ctx.qnn_interface.graphAddNode = StubGraphAddNode;
  qnn::ModelSettings settings{};
  settings.offload_graph_io_quantization = true;
  // Non-null overrides with an entry for "inp" → GetTensorNameOverride("inp") returns non-null.
  std::unordered_map<std::string, std::string> overrides{{"inp", "inp_orig"}};
  auto wrapper = MakeWrapperWithOverrides(ctx, settings, &overrides);

  qnn::QnnTensorWrapper in("inp", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                           qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  qnn::QnnTensorWrapper out("out", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32,
                            qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(in)));
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(out)));
  ASSERT_TRUE(wrapper->CreateQnnNode("n0", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CAST,
                                     {"inp"}, {"out"}, {}, false));

  EXPECT_TRUE(wrapper->ComposeQnnGraph());
}

// ComposeQnnGraph: output tensor not in model_tensors_map_ →
// CreateQnnInputOutputTensors for outputs returns false at line ~172 →
// line ~653 (return false in output-tensor error path) is executed.
TEST(QnnUnit_ModelWrapperTest, ComposeQnnGraph_MissingOutputTensor_ReturnsFalse) {
  QnnModelWrapperTestContext ctx;
  ctx.qnn_interface.tensorCreateGraphTensor = StubTensorCreateSuccess;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  // Add "inp" but NOT "out_missing" — CreateQnnInputOutputTensors for outputs will fail.
  qnn::QnnTensorWrapper in("inp", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                           qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(in)));
  ASSERT_TRUE(wrapper->CreateQnnNode("n0", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CAST,
                                     {"inp"}, {"out_missing"}, {}, false));

  EXPECT_FALSE(wrapper->ComposeQnnGraph());
}

// ── graphRetrieve fallback, input/param tensor create failures in main loop,
//            BF16 else branches, ApplyBF16ConversionForValidation failure ──

namespace {
// graphCreate: fails without setting graph_ → triggers the graphRetrieve fallback (line ~38).
Qnn_ErrorHandle_t StubGraphCreateFail(Qnn_ContextHandle_t, const char*,
                                      const QnnGraph_Config_t**, Qnn_GraphHandle_t*) {
  return QNN_TENSOR_ERROR_INVALID_HANDLE;  // any non-zero Qnn_ErrorHandle_t
}
// graphRetrieve: fails → lines ~39-41 hit → CreateQnnGraph returns false.
Qnn_ErrorHandle_t StubGraphRetrieveFail(Qnn_ContextHandle_t, const char*, Qnn_GraphHandle_t*) {
  return QNN_TENSOR_ERROR_INVALID_HANDLE;
}
// graphRetrieve: succeeds → lines ~38-39 hit → CreateQnnGraph returns true.
// Reuses g_graph_sentinel from the same anonymous namespace (same translation unit).
Qnn_ErrorHandle_t StubGraphRetrieveSuccess(Qnn_ContextHandle_t, const char*,
                                           Qnn_GraphHandle_t* graph_out) {
  *graph_out = reinterpret_cast<Qnn_GraphHandle_t>(&g_graph_sentinel);
  return QNN_GRAPH_NO_ERROR;
}
}  // namespace

// graphCreate fails AND graphRetrieve also fails → lines ~39-41 are hit → return false.
TEST(QnnUnit_ModelWrapperTest, CreateQnnGraph_GraphCreateFails_GraphRetrieveFails_ReturnsFalse) {
  QnnModelWrapperTestContext ctx;
  ctx.qnn_interface.graphCreate = StubGraphCreateFail;
  ctx.qnn_interface.graphRetrieve = StubGraphRetrieveFail;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  auto fake_ctx = reinterpret_cast<Qnn_ContextHandle_t>(static_cast<uintptr_t>(1));
  EXPECT_FALSE(wrapper->CreateQnnGraph(fake_ctx, "my_graph"));
}

// graphCreate fails but graphRetrieve succeeds → lines ~38-39, ~47-49 are hit → return true.
TEST(QnnUnit_ModelWrapperTest, CreateQnnGraph_GraphCreateFails_GraphRetrieveSucceeds_ReturnsTrue) {
  QnnModelWrapperTestContext ctx;
  ctx.qnn_interface.graphCreate = StubGraphCreateFail;
  ctx.qnn_interface.graphRetrieve = StubGraphRetrieveSuccess;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  auto fake_ctx = reinterpret_cast<Qnn_ContextHandle_t>(static_cast<uintptr_t>(1));
  EXPECT_TRUE(wrapper->CreateQnnGraph(fake_ctx, "my_graph"));
}

// Tensor IS in model_tensors_map_ but tensorCreateGraphTensor fails during the main compose loop
// (i.e., not in the pre-registration loop — empty input/output info keeps tensor_created_map_ empty).
// Covers CreateQnnInputOutputTensors failure path at lines ~188-189.
TEST(QnnUnit_ModelWrapperTest, ComposeQnnGraph_InputTensorCreate_InMainLoop_Fails_ReturnsFalse) {
  QnnModelWrapperTestContext ctx;
  // Empty input/output info → RegisterGraphInputOutputInOrder is a no-op (no pre-registration).
  ctx.qnn_interface.tensorCreateGraphTensor = StubTensorCreateFail;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  qnn::QnnTensorWrapper in("inp", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                           qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  qnn::QnnTensorWrapper out("out", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32,
                            qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(in)));
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(out)));
  ASSERT_TRUE(wrapper->CreateQnnNode("n0", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CAST,
                                     {"inp"}, {"out"}, {}, false));
  EXPECT_FALSE(wrapper->ComposeQnnGraph());
}

// Param IS in model_params_map_ but tensorCreateGraphTensor fails (TENSOR-type param).
// Op input/output lists are empty → CreateQnnInputOutputTensors is a trivial no-op.
// Covers CreateQnnParamTensors failure path at lines ~215-216.
TEST(QnnUnit_ModelWrapperTest, ComposeQnnGraph_ParamTensorCreate_InMainLoop_Fails_ReturnsFalse) {
  QnnModelWrapperTestContext ctx;
  ctx.qnn_interface.tensorCreateGraphTensor = StubTensorCreateFail;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  // TENSOR-type param: CreateQnnGraphParam → CreateTensorInQnnGraph → tensorCreateGraphTensor.
  // Data size must match: 1 element × sizeof(uint32_t) = 4 bytes for QNN_DATATYPE_UINT_32.
  std::vector<uint32_t> shape{1};
  std::vector<uint8_t> data(sizeof(uint32_t), 0);
  qnn::QnnParamWrapper param(0, "n0", "stride", QNN_DATATYPE_UINT_32, std::move(shape), std::move(data));
  std::string pname = param.GetParamTensorName();  // "n0_0_stride"
  ASSERT_TRUE(wrapper->AddParamWrapper(std::move(param)));

  // Op with no input/output tensors so CreateQnnInputOutputTensors is a no-op.
  ASSERT_TRUE(wrapper->CreateQnnNode("n0", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CAST,
                                     {}, {}, {pname}, false));
  EXPECT_FALSE(wrapper->ComposeQnnGraph());
}

// ProcessBF16InputConversion: NATIVE BF16 input registered as a graph input (so IsGraphInput()
// short-circuits IsConstantInput), but dtype ≠ FP32 → all three BF16 conditions fail →
// else branch (line ~323) is hit: input name passed through unchanged.
// ProcessBF16OutputConversion: NATIVE BF16 output, not a graph output, dtype ≠ FP32 →
// else branch (line ~369) is also hit.
TEST(QnnUnit_ModelWrapperTest, ComposeQnnGraph_BF16_NativeBF16Input_ElseBranch_Succeeds) {
  QnnModelWrapperTestContext ctx;
  // Register "inp" as graph input so IsGraphInput() short-circuits IsConstantInput (safe w/ null OrtApi).
  ctx.input_info.names.push_back("inp");
  ctx.input_info.indices["inp"] = 0;
  ctx.qnn_interface.tensorCreateGraphTensor = StubTensorCreateSuccess;
  ctx.qnn_interface.graphAddNode = StubGraphAddNode;
  qnn::ModelSettings settings{};
  settings.htp_bf16_enable = true;
  auto wrapper = ctx.CreateWrapper(settings, qnn::QnnBackendType::HTP);

  // NATIVE BF16 tensors: is_graph_input_or_init=true for "inp" but dtype≠FP32 →
  // none of the first three BF16 conditions match → else branch at line ~323.
  qnn::QnnTensorWrapper in("inp", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_BFLOAT_16,
                           qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  qnn::QnnTensorWrapper out("out", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_BFLOAT_16,
                            qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(in)));
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(out)));
  ASSERT_TRUE(wrapper->CreateQnnNode("n0", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CAST,
                                     {"inp"}, {"out"}, {}, false));
  EXPECT_TRUE(wrapper->ComposeQnnGraph());
}

// CreateQnnNode with do_op_validation=true, BF16 enabled, input absent from model_tensors_map_.
// ApplyBF16ConversionForValidation fails (line ~383-387) → lines ~457-460 → return false.
// No SDK stubs needed: we fail before reaching backendValidateOpConfig.
TEST(QnnUnit_ModelWrapperTest, CreateQnnNode_BF16_ApplyConversionFails_InputNotInMap_ReturnsFalse) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  settings.htp_bf16_enable = true;
  auto wrapper = ctx.CreateWrapper(settings, qnn::QnnBackendType::HTP);

  // "missing_inp" is NOT in model_tensors_map_ → ApplyBF16ConversionForValidation returns false.
  bool ok = wrapper->CreateQnnNode("n0", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CAST,
                                   {"missing_inp"}, {"out"}, {}, /*do_op_validation=*/true);
  EXPECT_FALSE(ok);
}

// CreateQnnNode with do_op_validation=true: input OK but output absent from model_tensors_map_.
// CreateQnnInputOutputTensors for the output returns false (line ~172-174) → line ~473 → false.
// No SDK stubs needed: do_op_validation=true skips tensorCreateGraphTensor calls.
TEST(QnnUnit_ModelWrapperTest, CreateQnnNode_Validation_OutputNotInMap_ReturnsFalse) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  qnn::QnnTensorWrapper in("inp", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                           qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(in)));

  // "out_missing" is NOT in model_tensors_map_ → second CreateQnnInputOutputTensors fails → line ~473.
  bool ok = wrapper->CreateQnnNode("n0", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CAST,
                                   {"inp"}, {"out_missing"}, {}, /*do_op_validation=*/true);
  EXPECT_FALSE(ok);
}

// ── ValidateQnnNode with real QNN HTP backend ─────────────────────
//
// These tests dlopen libQnnHtp.so and create a real Qnn_BackendHandle_t so that
// QnnGraphOpValidation / backendValidateOpConfig exercises the actual SDK path.
// On Linux x86-64 (the unit-test host), HTP supports validation but not graph
// execution — these tests intentionally exercise only the validation path.
//
// libQnnHtp.so is part of the QAIRT SDK that is required to build the EP, so
// it must be present at test time. A missing library indicates a CI/environment
// configuration error and fails the test (ASSERT_TRUE) rather than skipping.

// ValidateQnnNode succeeds for a valid Relu op on the HTP backend.
// Covers ValidateQnnNode → QnnGraphOpValidation → backendValidateOpConfig (success path).
// Uses UFIXED_POINT_8 with per-tensor quant params — the representative HTP production path.
TEST(QnnUnit_ModelWrapperTest, ValidateQnnNode_HtpBackend_Relu_Succeeds) {
  QnnRealHtpBackendContext backend;
  ASSERT_TRUE(backend.IsValid()) << "libQnnHtp.so not available — QAIRT SDK must be installed in CI";

  QnnModelWrapperTestContext ctx;
  ctx.qnn_interface = backend.qnn_interface;
  ctx.backend_handle = backend.backend_handle;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings, qnn::QnnBackendType::HTP);

  // HTP validator requires quantized tensors for Relu — float32 is rejected at validation.
  qnn::QnnQuantParamsWrapper quant = qnn::QnnQuantParamsWrapper::PerTensor(/*scale=*/1.0f / 255.0f, /*offset=*/0);
  qnn::QnnTensorWrapper input_tw("relu_in", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_UFIXED_POINT_8,
                                 quant.Copy(), std::vector<uint32_t>{1, 4});
  qnn::QnnTensorWrapper output_tw("relu_out", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_UFIXED_POINT_8,
                                  quant.Copy(), std::vector<uint32_t>{1, 4});

  std::vector<Qnn_Tensor_t> inputs = {input_tw.GetQnnTensor()};
  std::vector<Qnn_Tensor_t> outputs = {output_tw.GetQnnTensor()};

  auto status = wrapper->ValidateQnnNode("relu_node",
                                         QNN_OP_PACKAGE_NAME_QTI_AISW,
                                         QNN_OP_RELU,
                                         std::move(inputs),
                                         std::move(outputs),
                                         {});
  EXPECT_TRUE(status.IsOK()) << status.GetErrorMessage();
}

// ValidateQnnNode fails for an unrecognised op type on the HTP backend.
// Covers ValidateQnnNode → QnnGraphOpValidation → backendValidateOpConfig (failure path).
TEST(QnnUnit_ModelWrapperTest, ValidateQnnNode_HtpBackend_InvalidOpType_Fails) {
  QnnRealHtpBackendContext backend;
  ASSERT_TRUE(backend.IsValid()) << "libQnnHtp.so not available — QAIRT SDK must be installed in CI";

  QnnModelWrapperTestContext ctx;
  ctx.qnn_interface = backend.qnn_interface;
  ctx.backend_handle = backend.backend_handle;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings, qnn::QnnBackendType::HTP);

  qnn::QnnQuantParamsWrapper quant = qnn::QnnQuantParamsWrapper::PerTensor(/*scale=*/1.0f / 255.0f, /*offset=*/0);
  qnn::QnnTensorWrapper input_tw("in", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_UFIXED_POINT_8,
                                 quant.Copy(), std::vector<uint32_t>{1, 4});
  qnn::QnnTensorWrapper output_tw("out", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_UFIXED_POINT_8,
                                  quant.Copy(), std::vector<uint32_t>{1, 4});

  std::vector<Qnn_Tensor_t> inputs = {input_tw.GetQnnTensor()};
  std::vector<Qnn_Tensor_t> outputs = {output_tw.GetQnnTensor()};

  auto status = wrapper->ValidateQnnNode("bad_node",
                                         QNN_OP_PACKAGE_NAME_QTI_AISW,
                                         "NonExistentOp_QnnUT",
                                         std::move(inputs),
                                         std::move(outputs),
                                         {});
  EXPECT_FALSE(status.IsOK());
}

// ── IsExternalOverrideTarget, IsPerChannelQuantized (no-quant path) ──

// IsExternalOverrideTarget: null override map → always returns false.
TEST(QnnUnit_ModelWrapperTest, IsExternalOverrideTarget_NullMap_ReturnsFalse) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);  // tensor_name_overrides_ = nullptr

  EXPECT_FALSE(wrapper->IsExternalOverrideTarget("any_name"));
}

// IsExternalOverrideTarget: the external name IS a target of an existing override.
TEST(QnnUnit_ModelWrapperTest, IsExternalOverrideTarget_TargetExists_ReturnsTrue) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  std::unordered_map<std::string, std::string> overrides{{"internal_a", "external_a"}};
  auto wrapper = MakeWrapperWithOverrides(ctx, settings, &overrides);

  EXPECT_TRUE(wrapper->IsExternalOverrideTarget("external_a"));
}

// IsExternalOverrideTarget: the external name is NOT a target of any override.
TEST(QnnUnit_ModelWrapperTest, IsExternalOverrideTarget_TargetNotPresent_ReturnsFalse) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  std::unordered_map<std::string, std::string> overrides{{"internal_a", "external_a"}};
  auto wrapper = MakeWrapperWithOverrides(ctx, settings, &overrides);

  EXPECT_FALSE(wrapper->IsExternalOverrideTarget("not_a_target"));
}

// IsPerChannelQuantized: io_def with no quant_param short-circuits to is_per_channel=false.
// No OrtApi calls are made — covers the early-return branch in IsPerChannelQuantized.
TEST(QnnUnit_ModelWrapperTest, IsPerChannelQuantized_NoQuantParam_ReturnsFalse) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  OrtNodeUnitIODef io_def;
  io_def.name = "t";
  io_def.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  io_def.shape = std::vector<int64_t>{4};
  io_def.quant_param = std::nullopt;  // no quant param → immediate false

  bool is_per_channel = true;  // should be overwritten to false
  int64_t axis = 99;
  Ort::Status status = wrapper->IsPerChannelQuantized(io_def, is_per_channel, axis);

  EXPECT_TRUE(status.IsOK());
  EXPECT_FALSE(is_per_channel);
}

// ── IsConstantInput true path, GetConstantTensor, GetTensorType STATIC ──

namespace {
// Fake OrtValueInfo sentinel — its address is used as an opaque pointer in stub arrays.
static int g_init_vi_sentinel = 0;

// Name returned by StubGetVIName.  Set before each test that uses it.
static const char* g_init_vi_name = "";

OrtStatus* StubGetNumInit1(const OrtGraph*, size_t* n) noexcept {
  *n = 1;
  return nullptr;
}
OrtStatus* StubGetInits1(const OrtGraph*, const OrtValueInfo** arr, size_t) noexcept {
  arr[0] = reinterpret_cast<const OrtValueInfo*>(&g_init_vi_sentinel);
  return nullptr;
}
OrtStatus* StubGetVIName(const OrtValueInfo*, const char** nm) noexcept {
  *nm = g_init_vi_name;
  return nullptr;
}
OrtStatus* StubVIIsConstantTrue(const OrtValueInfo*, bool* b) noexcept {
  *b = true;
  return nullptr;
}
OrtStatus* StubVIIsConstantFalse(const OrtValueInfo*, bool* b) noexcept {
  *b = false;
  return nullptr;
}
}  // namespace

// IsConstantInput: initializer found AND ValueInfo_IsConstantInitializer returns true → true.
// Covers FindInitializer "found" path + IsConstantInput true branch.
TEST(QnnUnit_ModelWrapperTest, IsConstantInput_FoundAndIsConstant_ReturnsTrue) {
  QnnModelWrapperTestContext ctx;
  ctx.stub_ort_api.Graph_GetNumInitializers = StubGetNumInit1;
  ctx.stub_ort_api.Graph_GetInitializers = StubGetInits1;
  ctx.stub_ort_api.GetValueInfoName = StubGetVIName;
  ctx.stub_ort_api.ValueInfo_IsConstantInitializer = StubVIIsConstantTrue;
  g_init_vi_name = "my_weight";

  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  EXPECT_TRUE(wrapper->IsConstantInput("my_weight"));
}

// IsConstantInput: initializer found but ValueInfo_IsConstantInitializer returns false → false.
// Covers the "found but not constant" branch in IsConstantInput.
TEST(QnnUnit_ModelWrapperTest, IsConstantInput_FoundButNotConstant_ReturnsFalse) {
  QnnModelWrapperTestContext ctx;
  ctx.stub_ort_api.Graph_GetNumInitializers = StubGetNumInit1;
  ctx.stub_ort_api.Graph_GetInitializers = StubGetInits1;
  ctx.stub_ort_api.GetValueInfoName = StubGetVIName;
  ctx.stub_ort_api.ValueInfo_IsConstantInitializer = StubVIIsConstantFalse;
  g_init_vi_name = "my_weight";

  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  EXPECT_FALSE(wrapper->IsConstantInput("my_weight"));
}

// GetConstantTensor: initializer found AND is constant → returns non-null pointer.
// Covers the success path of GetConstantTensor.
TEST(QnnUnit_ModelWrapperTest, GetConstantTensor_FoundAndIsConstant_ReturnsNonNull) {
  QnnModelWrapperTestContext ctx;
  ctx.stub_ort_api.Graph_GetNumInitializers = StubGetNumInit1;
  ctx.stub_ort_api.Graph_GetInitializers = StubGetInits1;
  ctx.stub_ort_api.GetValueInfoName = StubGetVIName;
  ctx.stub_ort_api.ValueInfo_IsConstantInitializer = StubVIIsConstantTrue;
  g_init_vi_name = "const_w";

  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  EXPECT_NE(wrapper->GetConstantTensor("const_w"), nullptr);
}

// GetConstantTensor: tensor not in initializers (0 initializers) → returns null.
// Covers the "not found" path in GetConstantTensor.
TEST(QnnUnit_ModelWrapperTest, GetConstantTensor_NotFound_ReturnsNull) {
  QnnModelWrapperTestContext ctx;
  ctx.stub_ort_api.Graph_GetNumInitializers = StubGetNumInitializersZero;
  ctx.stub_ort_api.Graph_GetInitializers = StubGetInitializersEmpty;

  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  EXPECT_EQ(wrapper->GetConstantTensor("nonexistent"), nullptr);
}

// GetTensorType: tensor is a constant initializer → QNN_TENSOR_TYPE_STATIC.
// Covers the IsConstantInput-true branch of GetTensorType.
TEST(QnnUnit_ModelWrapperTest, GetTensorType_ConstantInput_ReturnsStatic) {
  QnnModelWrapperTestContext ctx;
  ctx.stub_ort_api.Graph_GetNumInitializers = StubGetNumInit1;
  ctx.stub_ort_api.Graph_GetInitializers = StubGetInits1;
  ctx.stub_ort_api.GetValueInfoName = StubGetVIName;
  ctx.stub_ort_api.ValueInfo_IsConstantInitializer = StubVIIsConstantTrue;
  g_init_vi_name = "const_w";

  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  EXPECT_EQ(wrapper->GetTensorType("const_w"), QNN_TENSOR_TYPE_STATIC);
}

// ── GetTensorInfo, MakeTensorWrapper(TensorInfo), MakeTensorWrapper(OrtNodeUnitIODef),
//             AddNoopReshapeNode ──────────────────────────────────────────────────────────────

// GetTensorInfo: FP32 tensor with no quant_param, not an initializer.
// Covers: quant_param.Init early-exit → GetQnnDataType → GetOnnxShape → IsConstantInput=false.
TEST(QnnUnit_ModelWrapperTest, GetTensorInfo_NoQuantParam_Float_NotInitializer_Succeeds) {
  QnnModelWrapperTestContext ctx;
  ctx.stub_ort_api.Graph_GetNumInitializers = StubGetNumInitializersZero;
  ctx.stub_ort_api.Graph_GetInitializers = StubGetInitializersEmpty;

  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  OrtNodeUnitIODef io_def;
  io_def.name = "t";
  io_def.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  io_def.shape = std::vector<int64_t>{1, 3, 4};
  io_def.quant_param = std::nullopt;

  qnn::TensorInfo info{};
  Ort::Status status = wrapper->GetTensorInfo(io_def, info);

  EXPECT_TRUE(status.IsOK());
  EXPECT_EQ(info.qnn_data_type, QNN_DATATYPE_FLOAT_32);
  ASSERT_EQ(info.shape.size(), 3u);
  EXPECT_EQ(info.shape[0], 1u);
  EXPECT_EQ(info.shape[1], 3u);
  EXPECT_EQ(info.shape[2], 4u);
  EXPECT_FALSE(info.is_initializer);
  EXPECT_EQ(info.initializer_tensor, nullptr);
}

// GetTensorInfo: tensor IS a constant initializer — covers is_initializer=true branch
// and GetConstantTensor call inside GetTensorInfo.
TEST(QnnUnit_ModelWrapperTest, GetTensorInfo_IsConstantInitializer_SetsInitializerFields) {
  QnnModelWrapperTestContext ctx;
  ctx.stub_ort_api.Graph_GetNumInitializers = StubGetNumInit1;
  ctx.stub_ort_api.Graph_GetInitializers = StubGetInits1;
  ctx.stub_ort_api.GetValueInfoName = StubGetVIName;
  ctx.stub_ort_api.ValueInfo_IsConstantInitializer = StubVIIsConstantTrue;
  g_init_vi_name = "const_w";

  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  OrtNodeUnitIODef io_def;
  io_def.name = "const_w";
  io_def.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  io_def.shape = std::vector<int64_t>{4};
  io_def.quant_param = std::nullopt;

  qnn::TensorInfo info{};
  Ort::Status status = wrapper->GetTensorInfo(io_def, info);

  EXPECT_TRUE(status.IsOK());
  EXPECT_TRUE(info.is_initializer);
  EXPECT_NE(info.initializer_tensor, nullptr);
}

// MakeTensorWrapper(TensorInfo, name, wrapper): non-initializer, graph output → APP_READ.
// GetTensorType calls IsConstantInput which needs GetNumInitializers=0 stub.
TEST(QnnUnit_ModelWrapperTest, MakeTensorWrapper_TensorInfo_NonInit_GraphOutput_ReturnsAppRead) {
  QnnModelWrapperTestContext ctx;
  ctx.stub_ort_api.Graph_GetNumInitializers = StubGetNumInitializersZero;
  ctx.stub_ort_api.Graph_GetInitializers = StubGetInitializersEmpty;
  ctx.output_info.indices["out_tensor"] = 0;

  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  qnn::TensorInfo tinfo{};
  tinfo.qnn_data_type = QNN_DATATYPE_FLOAT_32;
  tinfo.shape = {1u, 4u};
  tinfo.is_initializer = false;
  tinfo.initializer_tensor = nullptr;

  qnn::QnnTensorWrapper tw;
  Ort::Status status = wrapper->MakeTensorWrapper(tinfo, "out_tensor", tw);

  EXPECT_TRUE(status.IsOK());
  EXPECT_EQ(tw.GetName(), "out_tensor");
  EXPECT_EQ(tw.GetTensorDataType(), QNN_DATATYPE_FLOAT_32);
  EXPECT_EQ(tw.GetTensorType(), QNN_TENSOR_TYPE_APP_READ);
}

// MakeTensorWrapper(TensorInfo, name, wrapper): non-initializer, not graph I/O → NATIVE.
TEST(QnnUnit_ModelWrapperTest, MakeTensorWrapper_TensorInfo_NonInit_Native_ReturnsNative) {
  QnnModelWrapperTestContext ctx;
  ctx.stub_ort_api.Graph_GetNumInitializers = StubGetNumInitializersZero;
  ctx.stub_ort_api.Graph_GetInitializers = StubGetInitializersEmpty;

  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  qnn::TensorInfo tinfo{};
  tinfo.qnn_data_type = QNN_DATATYPE_UINT_8;
  tinfo.shape = {4u};
  tinfo.is_initializer = false;
  tinfo.initializer_tensor = nullptr;

  qnn::QnnTensorWrapper tw;
  Ort::Status status = wrapper->MakeTensorWrapper(tinfo, "mid_tensor", tw);

  EXPECT_TRUE(status.IsOK());
  EXPECT_EQ(tw.GetName(), "mid_tensor");
  EXPECT_EQ(tw.GetTensorDataType(), QNN_DATATYPE_UINT_8);
  EXPECT_EQ(tw.GetTensorType(), QNN_TENSOR_TYPE_NATIVE);
}

// MakeTensorWrapper(OrtNodeUnitIODef, wrapper): non-initializer, graph input → APP_WRITE.
// Covers the first overload — goes through GetTensorInfo then constructs qnn::QnnTensorWrapper.
TEST(QnnUnit_ModelWrapperTest, MakeTensorWrapper_IODef_GraphInput_ReturnsAppWrite) {
  QnnModelWrapperTestContext ctx;
  ctx.stub_ort_api.Graph_GetNumInitializers = StubGetNumInitializersZero;
  ctx.stub_ort_api.Graph_GetInitializers = StubGetInitializersEmpty;
  ctx.input_info.indices["inp_t"] = 0;

  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  OrtNodeUnitIODef io_def;
  io_def.name = "inp_t";
  io_def.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  io_def.shape = std::vector<int64_t>{1, 4};
  io_def.quant_param = std::nullopt;

  qnn::QnnTensorWrapper tw;
  Ort::Status status = wrapper->MakeTensorWrapper(io_def, tw);

  EXPECT_TRUE(status.IsOK());
  EXPECT_EQ(tw.GetName(), "inp_t");
  EXPECT_EQ(tw.GetTensorDataType(), QNN_DATATYPE_FLOAT_32);
  EXPECT_EQ(tw.GetTensorType(), QNN_TENSOR_TYPE_APP_WRITE);
}

// AddNoopReshapeNode: input not in model_tensors_map_ → RETURN_IF fires → error.
TEST(QnnUnit_ModelWrapperTest, AddNoopReshapeNode_InputNotInMap_ReturnsError) {
  QnnModelWrapperTestContext ctx;
  ctx.stub_ort_api.Graph_GetNumInitializers = StubGetNumInitializersZero;
  ctx.stub_ort_api.Graph_GetInitializers = StubGetInitializersEmpty;

  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  OrtNodeUnitIODef output;
  output.name = "out";
  output.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  output.shape = std::vector<int64_t>{4};
  output.quant_param = std::nullopt;

  // "missing_in" is not in model_tensors_map_ → error returned.
  Ort::Status status = wrapper->AddNoopReshapeNode("noop0", "missing_in", output, false);
  EXPECT_FALSE(status.IsOK());
}

// AddNoopReshapeNode: success path — input present, shapes match, do_op_validation=false.
// Covers the full AddNoopReshapeNode path: map lookup → MakeTensorWrapper → shape check →
// AddTensorWrapper → CreateQnnNode.
TEST(QnnUnit_ModelWrapperTest, AddNoopReshapeNode_MatchingShapes_Succeeds) {
  QnnModelWrapperTestContext ctx;
  ctx.stub_ort_api.Graph_GetNumInitializers = StubGetNumInitializersZero;
  ctx.stub_ort_api.Graph_GetInitializers = StubGetInitializersEmpty;

  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  // Pre-add the input tensor (shape {1, 4}).
  qnn::QnnTensorWrapper in("in_t", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                           qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{1u, 4u});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(in)));

  OrtNodeUnitIODef output;
  output.name = "out_t";
  output.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  output.shape = std::vector<int64_t>{1, 4};  // same shape as input
  output.quant_param = std::nullopt;

  Ort::Status status = wrapper->AddNoopReshapeNode("noop0", "in_t", output, false);
  EXPECT_TRUE(status.IsOK());
  EXPECT_TRUE(wrapper->IsQnnTensorWrapperExist("out_t"));
}

// AddNoopReshapeNode: input and output have different shapes → shape-mismatch check fires → error.
TEST(QnnUnit_ModelWrapperTest, AddNoopReshapeNode_ShapeMismatch_ReturnsError) {
  QnnModelWrapperTestContext ctx;
  ctx.stub_ort_api.Graph_GetNumInitializers = StubGetNumInitializersZero;
  ctx.stub_ort_api.Graph_GetInitializers = StubGetInitializersEmpty;

  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  // Input shape = {4} (rank 1), output shape = {1, 4} (rank 2) → mismatch.
  qnn::QnnTensorWrapper in("in_t", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                           qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4u});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(in)));

  OrtNodeUnitIODef output;
  output.name = "out_t";
  output.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  output.shape = std::vector<int64_t>{1, 4};  // different rank → dims won't match
  output.quant_param = std::nullopt;

  Ort::Status status = wrapper->AddNoopReshapeNode("noop0", "in_t", output, false);
  EXPECT_FALSE(status.IsOK());
}

// ── UnpackInitializerData / UnpackZeroPoints ─────────────────────────
// Covers two previously-uncovered functions that require OrtApi tensor-data stubs.

namespace {
// Sentinel lvalues used as non-null opaque-token pointers in unit-test stubs
// (e.g., reinterpret_cast<const OrtValueInfo*>(&g_type_info_sentinel)).
// All stubs receiving these pointers MUST NOT dereference them — they only
// check for non-null or pass them through to another stub unchanged.
static int g_type_info_sentinel = 0;
static int g_shape_info_sentinel = 0;
static int g_initializer_value_sentinel = 0;

// Per-test configuration set before each unpack/shape test.
static ONNXTensorElementDataType g_element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
static std::vector<int64_t> g_tensor_dims;
static const void* g_tensor_raw_data = nullptr;

OrtStatus* StubGraphGetModelPathEmpty(const OrtGraph*,
                                      const ORTCHAR_T** model_path) noexcept {
  static const ORTCHAR_T empty_path[] = "";
  *model_path = empty_path;
  return nullptr;
}
OrtStatus* StubValueInfoGetExternalNull(const OrtValueInfo*,
                                        OrtExternalInitializerInfo** info) noexcept {
  *info = nullptr;
  return nullptr;
}
OrtStatus* StubGetValueInfoTypeInfo(const OrtValueInfo*,
                                    const OrtTypeInfo** type_info) noexcept {
  *type_info = reinterpret_cast<const OrtTypeInfo*>(&g_type_info_sentinel);
  return nullptr;
}
OrtStatus* StubCastTypeInfoToTensorInfo(const OrtTypeInfo*,
                                        const OrtTensorTypeAndShapeInfo** info) noexcept {
  *info = reinterpret_cast<const OrtTensorTypeAndShapeInfo*>(&g_shape_info_sentinel);
  return nullptr;
}
OrtStatus* StubGetTensorElementType(const OrtTensorTypeAndShapeInfo*,
                                    ONNXTensorElementDataType* out) noexcept {
  *out = g_element_type;
  return nullptr;
}
OrtStatus* StubGetDimensionsCount(const OrtTensorTypeAndShapeInfo*,
                                  size_t* out) noexcept {
  *out = g_tensor_dims.size();
  return nullptr;
}
OrtStatus* StubGetDimensions(const OrtTensorTypeAndShapeInfo*,
                             int64_t* dim_values, size_t count) noexcept {
  for (size_t i = 0; i < count && i < g_tensor_dims.size(); ++i) {
    dim_values[i] = g_tensor_dims[i];
  }
  return nullptr;
}
OrtStatus* StubValueInfoGetInitializerValue(const OrtValueInfo*,
                                            const OrtValue** value) noexcept {
  *value = reinterpret_cast<const OrtValue*>(&g_initializer_value_sentinel);
  return nullptr;
}
OrtStatus* StubGetTensorData(const OrtValue*, const void** out) noexcept {
  *out = g_tensor_raw_data;
  return nullptr;
}

// Installs all unpack stubs on ctx.
void SetupUnpackStubs(QnnModelWrapperTestContext& ctx) {
  ctx.stub_ort_api.Graph_GetModelPath = StubGraphGetModelPathEmpty;
  ctx.stub_ort_api.ValueInfo_GetExternalInitializerInfo = StubValueInfoGetExternalNull;
  ctx.stub_ort_api.GetValueInfoTypeInfo = StubGetValueInfoTypeInfo;
  ctx.stub_ort_api.CastTypeInfoToTensorInfo = StubCastTypeInfoToTensorInfo;
  ctx.stub_ort_api.GetTensorElementType = StubGetTensorElementType;
  ctx.stub_ort_api.GetDimensionsCount = StubGetDimensionsCount;
  ctx.stub_ort_api.GetDimensions = StubGetDimensions;
  ctx.stub_ort_api.ValueInfo_GetInitializerValue = StubValueInfoGetInitializerValue;
  ctx.stub_ort_api.GetTensorData = StubGetTensorData;
}
}  // namespace

// UnpackInitializerData: basic UINT8 tensor — raw bytes are copied as-is.
TEST(QnnUnit_ModelWrapperTest, UnpackInitializerData_UINT8_ReturnsRawBytes) {
  static const uint8_t kData[] = {10, 20, 30, 40};
  g_element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  g_tensor_dims = {4};
  g_tensor_raw_data = kData;

  QnnModelWrapperTestContext ctx;
  SetupUnpackStubs(ctx);
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  auto fake_vi = reinterpret_cast<const OrtValueInfo*>(&g_type_info_sentinel);
  std::vector<uint8_t> result;
  Ort::Status s = wrapper->UnpackInitializerData(fake_vi, result, /*unpack_4bit=*/false);
  ASSERT_TRUE(s.IsOK());
  ASSERT_EQ(result.size(), 4u);
  EXPECT_EQ(result[0], 10u);
  EXPECT_EQ(result[3], 40u);
}

// UnpackInitializerData: INT8 scalar — 1-byte tensor.
TEST(QnnUnit_ModelWrapperTest, UnpackInitializerData_INT8_ReturnsRawBytes) {
  static const int8_t kData[] = {-5};
  g_element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
  g_tensor_dims = {1};
  g_tensor_raw_data = kData;

  QnnModelWrapperTestContext ctx;
  SetupUnpackStubs(ctx);
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  auto fake_vi = reinterpret_cast<const OrtValueInfo*>(&g_type_info_sentinel);
  std::vector<uint8_t> result;
  Ort::Status s = wrapper->UnpackInitializerData(fake_vi, result, /*unpack_4bit=*/false);
  ASSERT_TRUE(s.IsOK());
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(static_cast<int8_t>(result[0]), -5);
}

// UnpackZeroPoints: null input pointer → RETURN_IF fires → error.
TEST(QnnUnit_ModelWrapperTest, UnpackZeroPoints_NullInput_ReturnsError) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  std::vector<int32_t> zps;
  ONNXTensorElementDataType dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  Ort::Status s = wrapper->UnpackZeroPoints(nullptr, zps, dt);
  EXPECT_FALSE(s.IsOK());
}

// UnpackZeroPoints: UINT8 — zero-points are negated (QNN uses -offset).
TEST(QnnUnit_ModelWrapperTest, UnpackZeroPoints_UINT8_NegatesValues) {
  static const uint8_t kData[] = {128, 0, 255, 1};
  g_element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  g_tensor_dims = {4};
  g_tensor_raw_data = kData;

  QnnModelWrapperTestContext ctx;
  SetupUnpackStubs(ctx);
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  auto fake_vi = reinterpret_cast<const OrtValueInfo*>(&g_type_info_sentinel);
  std::vector<int32_t> zps;
  ONNXTensorElementDataType dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  Ort::Status s = wrapper->UnpackZeroPoints(fake_vi, zps, dt);
  ASSERT_TRUE(s.IsOK());
  ASSERT_EQ(zps.size(), 4u);
  EXPECT_EQ(zps[0], -128);
  EXPECT_EQ(zps[1], 0);
  EXPECT_EQ(zps[2], -255);
  EXPECT_EQ(zps[3], -1);
  EXPECT_EQ(dt, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
}

// UnpackZeroPoints: INT8 — signed zero-points are negated.
TEST(QnnUnit_ModelWrapperTest, UnpackZeroPoints_INT8_NegatesValues) {
  static const int8_t kData[] = {-10, 5};
  g_element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
  g_tensor_dims = {2};
  g_tensor_raw_data = kData;

  QnnModelWrapperTestContext ctx;
  SetupUnpackStubs(ctx);
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  auto fake_vi = reinterpret_cast<const OrtValueInfo*>(&g_type_info_sentinel);
  std::vector<int32_t> zps;
  ONNXTensorElementDataType dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  Ort::Status s = wrapper->UnpackZeroPoints(fake_vi, zps, dt);
  ASSERT_TRUE(s.IsOK());
  ASSERT_EQ(zps.size(), 2u);
  EXPECT_EQ(zps[0], 10);
  EXPECT_EQ(zps[1], -5);
}

// UnpackZeroPoints: UINT16 — covers uint16_t lambda instantiation.
TEST(QnnUnit_ModelWrapperTest, UnpackZeroPoints_UINT16_NegatesValues) {
  static const uint16_t kData[] = {1000, 2000};
  g_element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
  g_tensor_dims = {2};
  g_tensor_raw_data = kData;

  QnnModelWrapperTestContext ctx;
  SetupUnpackStubs(ctx);
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  auto fake_vi = reinterpret_cast<const OrtValueInfo*>(&g_type_info_sentinel);
  std::vector<int32_t> zps;
  ONNXTensorElementDataType dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  Ort::Status s = wrapper->UnpackZeroPoints(fake_vi, zps, dt);
  ASSERT_TRUE(s.IsOK());
  ASSERT_EQ(zps.size(), 2u);
  EXPECT_EQ(zps[0], -1000);
  EXPECT_EQ(zps[1], -2000);
}

// UnpackZeroPoints: INT16 — covers int16_t lambda instantiation.
TEST(QnnUnit_ModelWrapperTest, UnpackZeroPoints_INT16_NegatesValues) {
  static const int16_t kData[] = {-300, 400};
  g_element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
  g_tensor_dims = {2};
  g_tensor_raw_data = kData;

  QnnModelWrapperTestContext ctx;
  SetupUnpackStubs(ctx);
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  auto fake_vi = reinterpret_cast<const OrtValueInfo*>(&g_type_info_sentinel);
  std::vector<int32_t> zps;
  ONNXTensorElementDataType dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  Ort::Status s = wrapper->UnpackZeroPoints(fake_vi, zps, dt);
  ASSERT_TRUE(s.IsOK());
  ASSERT_EQ(zps.size(), 2u);
  EXPECT_EQ(zps[0], 300);
  EXPECT_EQ(zps[1], -400);
}

// UnpackZeroPoints: INT32 — covers int32_t lambda instantiation.
TEST(QnnUnit_ModelWrapperTest, UnpackZeroPoints_INT32_NegatesValues) {
  static const int32_t kData[] = {100000, -1};
  g_element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  g_tensor_dims = {2};
  g_tensor_raw_data = kData;

  QnnModelWrapperTestContext ctx;
  SetupUnpackStubs(ctx);
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  auto fake_vi = reinterpret_cast<const OrtValueInfo*>(&g_type_info_sentinel);
  std::vector<int32_t> zps;
  ONNXTensorElementDataType dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  Ort::Status s = wrapper->UnpackZeroPoints(fake_vi, zps, dt);
  ASSERT_TRUE(s.IsOK());
  ASSERT_EQ(zps.size(), 2u);
  EXPECT_EQ(zps[0], -100000);
  EXPECT_EQ(zps[1], 1);
}

// UnpackZeroPoints: UINT32 — covers uint32_t lambda instantiation.
TEST(QnnUnit_ModelWrapperTest, UnpackZeroPoints_UINT32_NegatesValues) {
  static const uint32_t kData[] = {7u};
  g_element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
  g_tensor_dims = {1};
  g_tensor_raw_data = kData;

  QnnModelWrapperTestContext ctx;
  SetupUnpackStubs(ctx);
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  auto fake_vi = reinterpret_cast<const OrtValueInfo*>(&g_type_info_sentinel);
  std::vector<int32_t> zps;
  ONNXTensorElementDataType dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  Ort::Status s = wrapper->UnpackZeroPoints(fake_vi, zps, dt);
  ASSERT_TRUE(s.IsOK());
  ASSERT_EQ(zps.size(), 1u);
  EXPECT_EQ(zps[0], -7);
}

// UnpackZeroPoints: unsupported type (FLOAT) → hits default case → error.
TEST(QnnUnit_ModelWrapperTest, UnpackZeroPoints_UnsupportedType_ReturnsError) {
  static const float kData[] = {1.0f};
  g_element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  g_tensor_dims = {1};
  g_tensor_raw_data = kData;

  QnnModelWrapperTestContext ctx;
  SetupUnpackStubs(ctx);
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  auto fake_vi = reinterpret_cast<const OrtValueInfo*>(&g_type_info_sentinel);
  std::vector<int32_t> zps;
  ONNXTensorElementDataType dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  Ort::Status s = wrapper->UnpackZeroPoints(fake_vi, zps, dt);
  EXPECT_FALSE(s.IsOK());
}

// ── IsPerChannelQuantized ───────────────────────────────────────────────────
// GetInitializerShape (called internally) uses GetValueInfoTypeInfo,
// CastTypeInfoToTensorInfo, GetDimensionsCount, GetDimensions — all already
// provided by the unpack stubs above.  Set g_tensor_dims before each test.

namespace {
// Installs only the 4 stubs required by utils::GetInitializerShape.
void SetupShapeStubs(QnnModelWrapperTestContext& ctx) {
  ctx.stub_ort_api.GetValueInfoTypeInfo = StubGetValueInfoTypeInfo;
  ctx.stub_ort_api.CastTypeInfoToTensorInfo = StubCastTypeInfoToTensorInfo;
  ctx.stub_ort_api.GetDimensionsCount = StubGetDimensionsCount;
  ctx.stub_ort_api.GetDimensions = StubGetDimensions;
}
}  // namespace

// IsPerChannelQuantized: scale pointer is null → RETURN_IF fires → error.
TEST(QnnUnit_ModelWrapperTest, IsPerChannelQuantized_NullScale_ReturnsError) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  OrtNodeUnitIODef io_def;
  io_def.name = "t";
  io_def.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  io_def.quant_param = OrtNodeUnitIODef::QuantParam{/*scale=*/nullptr};

  bool is_per_channel = false;
  int64_t axis = 0;
  Ort::Status s = wrapper->IsPerChannelQuantized(io_def, is_per_channel, axis);
  EXPECT_FALSE(s.IsOK());
}

// IsPerChannelQuantized: scalar scale (0-dim tensor) → per-tensor.
TEST(QnnUnit_ModelWrapperTest, IsPerChannelQuantized_ScalarScale_PerTensor) {
  g_tensor_dims = {};  // 0-dim scalar

  QnnModelWrapperTestContext ctx;
  SetupShapeStubs(ctx);
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  OrtNodeUnitIODef io_def;
  io_def.name = "t";
  io_def.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  auto fake_scale = reinterpret_cast<const OrtValueInfo*>(&g_type_info_sentinel);
  io_def.quant_param = OrtNodeUnitIODef::QuantParam{fake_scale};

  bool is_per_channel = true;
  int64_t axis = -1;
  Ort::Status s = wrapper->IsPerChannelQuantized(io_def, is_per_channel, axis);
  ASSERT_TRUE(s.IsOK());
  EXPECT_FALSE(is_per_channel);
}

// IsPerChannelQuantized: 1-element vector scale → per-tensor.
TEST(QnnUnit_ModelWrapperTest, IsPerChannelQuantized_OneElemVectorScale_PerTensor) {
  g_tensor_dims = {1};  // 1-element vector

  QnnModelWrapperTestContext ctx;
  SetupShapeStubs(ctx);
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  OrtNodeUnitIODef io_def;
  io_def.name = "t";
  io_def.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  auto fake_scale = reinterpret_cast<const OrtValueInfo*>(&g_type_info_sentinel);
  io_def.quant_param = OrtNodeUnitIODef::QuantParam{fake_scale};

  bool is_per_channel = true;
  int64_t axis = -1;
  Ort::Status s = wrapper->IsPerChannelQuantized(io_def, is_per_channel, axis);
  ASSERT_TRUE(s.IsOK());
  EXPECT_FALSE(is_per_channel);
}

// IsPerChannelQuantized: multi-element scale, no axis → per-channel, default axis = 1.
TEST(QnnUnit_ModelWrapperTest, IsPerChannelQuantized_MultiElemScale_DefaultAxis) {
  g_tensor_dims = {4};  // 4-element → per-channel

  QnnModelWrapperTestContext ctx;
  SetupShapeStubs(ctx);
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  OrtNodeUnitIODef io_def;
  io_def.name = "t";
  io_def.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  auto fake_scale = reinterpret_cast<const OrtValueInfo*>(&g_type_info_sentinel);
  io_def.quant_param = OrtNodeUnitIODef::QuantParam{fake_scale};  // axis = nullopt

  bool is_per_channel = false;
  int64_t axis = 0;
  Ort::Status s = wrapper->IsPerChannelQuantized(io_def, is_per_channel, axis);
  ASSERT_TRUE(s.IsOK());
  EXPECT_TRUE(is_per_channel);
  EXPECT_EQ(axis, 1);  // value_or(1)
}

// IsPerChannelQuantized: multi-element scale, explicit positive axis → preserved.
TEST(QnnUnit_ModelWrapperTest, IsPerChannelQuantized_MultiElemScale_PositiveAxis) {
  g_tensor_dims = {4};

  QnnModelWrapperTestContext ctx;
  SetupShapeStubs(ctx);
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  OrtNodeUnitIODef io_def;
  io_def.name = "t";
  io_def.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  auto fake_scale = reinterpret_cast<const OrtValueInfo*>(&g_type_info_sentinel);
  io_def.quant_param = OrtNodeUnitIODef::QuantParam{fake_scale, nullptr, /*axis=*/2};

  bool is_per_channel = false;
  int64_t axis = 0;
  Ort::Status s = wrapper->IsPerChannelQuantized(io_def, is_per_channel, axis);
  ASSERT_TRUE(s.IsOK());
  EXPECT_TRUE(is_per_channel);
  EXPECT_EQ(axis, 2);
}

// IsPerChannelQuantized: negative axis → normalized to rank + axis.
// scale=[4], axis=-1, tensor shape=[4,4,4] (rank 3) → axis = 3 + (-1) = 2.
TEST(QnnUnit_ModelWrapperTest, IsPerChannelQuantized_NegativeAxis_NormalizedToPositive) {
  g_tensor_dims = {4};

  QnnModelWrapperTestContext ctx;
  SetupShapeStubs(ctx);
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  OrtNodeUnitIODef io_def;
  io_def.name = "t";
  io_def.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  io_def.shape = std::vector<int64_t>{4, 4, 4};  // rank 3
  auto fake_scale = reinterpret_cast<const OrtValueInfo*>(&g_type_info_sentinel);
  io_def.quant_param = OrtNodeUnitIODef::QuantParam{fake_scale, nullptr, /*axis=*/-1};

  bool is_per_channel = false;
  int64_t axis = 0;
  Ort::Status s = wrapper->IsPerChannelQuantized(io_def, is_per_channel, axis);
  ASSERT_TRUE(s.IsOK());
  EXPECT_TRUE(is_per_channel);
  EXPECT_EQ(axis, 2);  // -1 + rank(3) = 2
}

// IsPerChannelQuantized: negative axis but shape is nullopt
// → GetOnnxShape returns false → RETURN_IF_NOT fires → error.
TEST(QnnUnit_ModelWrapperTest, IsPerChannelQuantized_NegativeAxis_NoShape_ReturnsError) {
  g_tensor_dims = {4};

  QnnModelWrapperTestContext ctx;
  SetupShapeStubs(ctx);
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  OrtNodeUnitIODef io_def;
  io_def.name = "t";
  io_def.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  io_def.shape = std::nullopt;  // no shape → GetOnnxShape returns false
  auto fake_scale = reinterpret_cast<const OrtValueInfo*>(&g_type_info_sentinel);
  io_def.quant_param = OrtNodeUnitIODef::QuantParam{fake_scale, nullptr, /*axis=*/-1};

  bool is_per_channel = false;
  int64_t axis = 0;
  Ort::Status s = wrapper->IsPerChannelQuantized(io_def, is_per_channel, axis);
  EXPECT_FALSE(s.IsOK());
}

// =============================================================================
// AddTensorWrapper — htp_shared_memory permutations
// =============================================================================

// Verifies that when htp_shared_memory is disabled (default), the mem type of a
// graph input tensor remains QNN_TENSORMEMTYPE_RAW.
TEST(QnnUnit_ModelWrapperTest, AddTensorWrapper_SharedMemoryDisabled_GraphInput_MemTypeIsRaw) {
  QnnModelWrapperTestContext ctx;
  ctx.input_info.indices = {{"input0", 0}};

  qnn::ModelSettings settings{};
  settings.htp_shared_memory = false;
  auto wrapper = ctx.CreateWrapper(settings);

  qnn::QnnTensorWrapper tensor("input0", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                               qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{1, 3, 224, 224});

  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(tensor)));

  const auto& stored = wrapper->GetQnnTensorWrapper("input0");
  EXPECT_EQ(qnn::GetQnnTensorMemType(stored.GetQnnTensor()), QNN_TENSORMEMTYPE_RAW);
}

// Verifies that when htp_shared_memory is enabled, a graph input tensor
// gets mem type set to QNN_TENSORMEMTYPE_MEMHANDLE.
TEST(QnnUnit_ModelWrapperTest, AddTensorWrapper_SharedMemoryEnabled_GraphInput_MemTypeIsMemHandle) {
  QnnModelWrapperTestContext ctx;
  ctx.input_info.indices = {{"input0", 0}};

  qnn::ModelSettings settings{};
  settings.htp_shared_memory = true;
  auto wrapper = ctx.CreateWrapper(settings);

  qnn::QnnTensorWrapper tensor("input0", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                               qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{1, 3, 224, 224});

  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(tensor)));

  const auto& stored = wrapper->GetQnnTensorWrapper("input0");
  EXPECT_EQ(qnn::GetQnnTensorMemType(stored.GetQnnTensor()), QNN_TENSORMEMTYPE_MEMHANDLE);
}

// Verifies that when htp_shared_memory is enabled, a graph output tensor
// gets mem type set to QNN_TENSORMEMTYPE_MEMHANDLE.
TEST(QnnUnit_ModelWrapperTest, AddTensorWrapper_SharedMemoryEnabled_GraphOutput_MemTypeIsMemHandle) {
  QnnModelWrapperTestContext ctx;
  ctx.output_info.indices = {{"output0", 0}};

  qnn::ModelSettings settings{};
  settings.htp_shared_memory = true;
  auto wrapper = ctx.CreateWrapper(settings);

  qnn::QnnTensorWrapper tensor("output0", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32,
                               qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{1, 1000});

  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(tensor)));

  const auto& stored = wrapper->GetQnnTensorWrapper("output0");
  EXPECT_EQ(qnn::GetQnnTensorMemType(stored.GetQnnTensor()), QNN_TENSORMEMTYPE_MEMHANDLE);
}

// Verifies that when htp_shared_memory is enabled, an intermediate (native) tensor
// that is neither a graph input nor output retains QNN_TENSORMEMTYPE_RAW.
TEST(QnnUnit_ModelWrapperTest, AddTensorWrapper_SharedMemoryEnabled_IntermediateTensor_MemTypeIsRaw) {
  QnnModelWrapperTestContext ctx;
  // "intermediate0" is NOT in input_info or output_info.

  qnn::ModelSettings settings{};
  settings.htp_shared_memory = true;
  auto wrapper = ctx.CreateWrapper(settings);

  qnn::QnnTensorWrapper tensor("intermediate0", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                               qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{1, 256});

  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(tensor)));

  const auto& stored = wrapper->GetQnnTensorWrapper("intermediate0");
  EXPECT_EQ(qnn::GetQnnTensorMemType(stored.GetQnnTensor()), QNN_TENSORMEMTYPE_RAW);
}

// Verifies that when htp_shared_memory is disabled, a graph output tensor
// retains QNN_TENSORMEMTYPE_RAW.
TEST(QnnUnit_ModelWrapperTest, AddTensorWrapper_SharedMemoryDisabled_GraphOutput_MemTypeIsRaw) {
  QnnModelWrapperTestContext ctx;
  ctx.output_info.indices = {{"output0", 0}};

  qnn::ModelSettings settings{};
  settings.htp_shared_memory = false;
  auto wrapper = ctx.CreateWrapper(settings);

  qnn::QnnTensorWrapper tensor("output0", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32,
                               qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{1, 1000});

  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(tensor)));

  const auto& stored = wrapper->GetQnnTensorWrapper("output0");
  EXPECT_EQ(qnn::GetQnnTensorMemType(stored.GetQnnTensor()), QNN_TENSORMEMTYPE_RAW);
}

// Verifies that both graph input and output tensors get MEMHANDLE when
// htp_shared_memory is enabled, within the same wrapper instance.
TEST(QnnUnit_ModelWrapperTest, AddTensorWrapper_SharedMemoryEnabled_BothInputAndOutput_MemTypeIsMemHandle) {
  QnnModelWrapperTestContext ctx;
  ctx.input_info.indices = {{"input0", 0}};
  ctx.output_info.indices = {{"output0", 0}};

  qnn::ModelSettings settings{};
  settings.htp_shared_memory = true;
  auto wrapper = ctx.CreateWrapper(settings);

  qnn::QnnTensorWrapper input_tensor("input0", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                                     qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{1, 3, 224, 224});
  qnn::QnnTensorWrapper output_tensor("output0", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32,
                                      qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{1, 1000});

  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(input_tensor)));
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(output_tensor)));

  const auto& stored_input = wrapper->GetQnnTensorWrapper("input0");
  EXPECT_EQ(qnn::GetQnnTensorMemType(stored_input.GetQnnTensor()), QNN_TENSORMEMTYPE_MEMHANDLE);

  const auto& stored_output = wrapper->GetQnnTensorWrapper("output0");
  EXPECT_EQ(qnn::GetQnnTensorMemType(stored_output.GetQnnTensor()), QNN_TENSORMEMTYPE_MEMHANDLE);
}

// Verifies that adding a duplicate tensor (same name) returns true
// and does not overwrite the existing entry.
TEST(QnnUnit_ModelWrapperTest, AddTensorWrapper_DuplicateTensor_ReturnsTrueWithoutOverwrite) {
  QnnModelWrapperTestContext ctx;
  ctx.input_info.indices = {{"input0", 0}};

  qnn::ModelSettings settings{};
  settings.htp_shared_memory = false;
  auto wrapper = ctx.CreateWrapper(settings);

  qnn::QnnTensorWrapper tensor1("input0", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                                qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{1, 3, 224, 224});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(tensor1)));

  // Attempt to add another tensor with the same name
  qnn::QnnTensorWrapper tensor2("input0", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16,
                                qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{1, 3, 112, 112});
  EXPECT_TRUE(wrapper->AddTensorWrapper(std::move(tensor2)));

  // Should still have the original data type
  const auto& stored = wrapper->GetQnnTensorWrapper("input0");
  EXPECT_EQ(stored.GetTensorDataType(), QNN_DATATYPE_FLOAT_32);
}

// Verifies that adding a tensor with an empty name returns false.
TEST(QnnUnit_ModelWrapperTest, AddTensorWrapper_SharedMemoryDisabled_EmptyName_ReturnsFalse) {
  QnnModelWrapperTestContext ctx;

  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  qnn::QnnTensorWrapper tensor("", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                               qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{1, 256});

  EXPECT_FALSE(wrapper->AddTensorWrapper(std::move(tensor)));
}

// =============================================================================
// FoldedConstant tracking — regression coverage for per-channel constant DQ
// feeding Conv weight that previously leaked the DQ output as a graph input.
// =============================================================================

TEST(QnnUnit_ModelWrapperTest, FoldedConstant_DefaultIsFalse) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  EXPECT_FALSE(wrapper->IsFoldedConstant("not_marked"));
  EXPECT_FALSE(wrapper->IsEffectivelyConstantInput("not_marked"));
}

TEST(QnnUnit_ModelWrapperTest, FoldedConstant_MarkMakesTensorFolded) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  wrapper->MarkTensorAsFoldedConstant("weight_dq");

  EXPECT_TRUE(wrapper->IsFoldedConstant("weight_dq"));
  EXPECT_TRUE(wrapper->IsEffectivelyConstantInput("weight_dq"));
  EXPECT_FALSE(wrapper->IsFoldedConstant("other_tensor"));
  EXPECT_FALSE(wrapper->IsEffectivelyConstantInput("other_tensor"));
}

TEST(QnnUnit_ModelWrapperTest, FoldedConstant_MarkIsIdempotent) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  wrapper->MarkTensorAsFoldedConstant("weight_dq");
  wrapper->MarkTensorAsFoldedConstant("weight_dq");

  EXPECT_TRUE(wrapper->IsFoldedConstant("weight_dq"));
  EXPECT_TRUE(wrapper->IsEffectivelyConstantInput("weight_dq"));
}

TEST(QnnUnit_ModelWrapperTest, FoldedConstant_MultipleTensorsTrackedIndependently) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  wrapper->MarkTensorAsFoldedConstant("a");
  wrapper->MarkTensorAsFoldedConstant("b");

  EXPECT_TRUE(wrapper->IsFoldedConstant("a"));
  EXPECT_TRUE(wrapper->IsFoldedConstant("b"));
  EXPECT_FALSE(wrapper->IsFoldedConstant("c"));

  EXPECT_TRUE(wrapper->IsEffectivelyConstantInput("a"));
  EXPECT_TRUE(wrapper->IsEffectivelyConstantInput("b"));
  EXPECT_FALSE(wrapper->IsEffectivelyConstantInput("c"));
}

// Marking is independent of AddTensorWrapper so op builders can mark before or after.
TEST(QnnUnit_ModelWrapperTest, FoldedConstant_DoesNotRequireTensorWrapper) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  wrapper->MarkTensorAsFoldedConstant("phantom_tensor");

  EXPECT_TRUE(wrapper->IsFoldedConstant("phantom_tensor"));
  EXPECT_TRUE(wrapper->IsEffectivelyConstantInput("phantom_tensor"));
  EXPECT_FALSE(wrapper->IsQnnTensorWrapperExist("phantom_tensor"));
}

// Folded-constant outputs MUST map to STATIC (not NATIVE) so they aren't treated as runtime intermediates.
TEST(QnnUnit_ModelWrapperTest, FoldedConstant_GetTensorTypeIsStatic) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  EXPECT_EQ(wrapper->GetTensorType("unmarked"), QNN_TENSOR_TYPE_NATIVE);
  wrapper->MarkTensorAsFoldedConstant("folded");
  EXPECT_EQ(wrapper->GetTensorType("folded"), QNN_TENSOR_TYPE_STATIC);
}

// The op-builder query path: an op builder holding a QnnModelWrapper must observe
// the HTP arch the backend manager resolved, so it can pick a translation per arch.
TEST(QnnUnit_ModelWrapperTest, GetQnnBackendManager_HtpArch_ReturnsBackendManagerArch) {
  QnnModelWrapperTestContext ctx;
  ctx.backend_manager.HtpArch() = QNN_HTP_DEVICE_ARCH_V79;

  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  EXPECT_EQ(wrapper->GetHtpArch(), QNN_HTP_DEVICE_ARCH_V79);
}

// An arch set after the wrapper was constructed is still visible — the wrapper does
// not cache it. Matches production order: the wrapper is built per graph, while the
// manager's arch is resolved once during backend setup.
TEST(QnnUnit_ModelWrapperTest, GetQnnBackendManager_HtpArch_ReflectsLaterChange) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  ASSERT_EQ(wrapper->GetHtpArch(), QNN_HTP_DEVICE_ARCH_NONE);

  ctx.backend_manager.HtpArch() = QNN_HTP_DEVICE_ARCH_V75;
  EXPECT_EQ(wrapper->GetHtpArch(), QNN_HTP_DEVICE_ARCH_V75);
}

// GetQnnBackendType() is a live read from the manager, not a construction-time copy:
// build the wrapper as CPU, flip the manager to HTP, and the wrapper follows.
TEST(QnnUnit_ModelWrapperTest, GetQnnBackendType_FollowsBackendManager_AfterConstruction) {
  QnnModelWrapperTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings, qnn::QnnBackendType::CPU);

  ASSERT_EQ(wrapper->GetQnnBackendType(), qnn::QnnBackendType::CPU);

  ctx.backend_manager.BackendType() = qnn::QnnBackendType::HTP;
  EXPECT_EQ(wrapper->GetQnnBackendType(), qnn::QnnBackendType::HTP);
}

// GetTensorInfo() resolves the ONNX→QNN type map through the manager's backend type.
// The GPU map in qnn_utils.cc CreateMap() is the only one with an INT4 entry, so an
// unquantized INT4 tensor succeeds on GPU...
TEST(QnnUnit_ModelWrapperTest, GetTensorInfo_GpuBackend_Int4_UsesGpuTypeMap) {
  QnnModelWrapperTestContext ctx;
  ctx.stub_ort_api.Graph_GetNumInitializers = StubGetNumInitializersZero;
  ctx.stub_ort_api.Graph_GetInitializers = StubGetInitializersEmpty;

  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings, qnn::QnnBackendType::GPU);

  OrtNodeUnitIODef io_def;
  io_def.name = "t_int4";
  io_def.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4;
  io_def.shape = std::vector<int64_t>{8};
  io_def.quant_param = std::nullopt;

  qnn::TensorInfo info{};
  Ort::Status status = wrapper->GetTensorInfo(io_def, info);

  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  EXPECT_EQ(info.qnn_data_type, QNN_DATATYPE_SFIXED_POINT_4);
}

// ...and fails on HTP, which uses the base map. Together with the test above this
// proves the backend type reaching GetQnnDataType() comes from the manager.
TEST(QnnUnit_ModelWrapperTest, GetTensorInfo_HtpBackend_Int4_Fails) {
  QnnModelWrapperTestContext ctx;
  ctx.stub_ort_api.Graph_GetNumInitializers = StubGetNumInitializersZero;
  ctx.stub_ort_api.Graph_GetInitializers = StubGetInitializersEmpty;

  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings, qnn::QnnBackendType::HTP);

  OrtNodeUnitIODef io_def;
  io_def.name = "t_int4";
  io_def.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4;
  io_def.shape = std::vector<int64_t>{8};
  io_def.quant_param = std::nullopt;

  qnn::TensorInfo info{};
  Ort::Status status = wrapper->GetTensorInfo(io_def, info);

  EXPECT_FALSE(status.IsOK());
}

namespace {
// Records the data type of the first input tensor the validator was handed, so a
// test can tell whether the BF16 conversion ran before validation. Reset before use.
Qnn_DataType_t g_validated_input_data_type = QNN_DATATYPE_UNDEFINED;

Qnn_ErrorHandle_t StubBackendValidateOpConfigCaptureDataType(Qnn_BackendHandle_t, Qnn_OpConfig_t op_config) {
  if (op_config.version == QNN_OPCONFIG_VERSION_1 &&
      op_config.v1.numOfInputs > 0 &&
      op_config.v1.inputTensors != nullptr) {
    g_validated_input_data_type = qnn::GetQnnTensorDataType(op_config.v1.inputTensors[0]);
  }
  return QNN_BACKEND_NO_ERROR;
}

// Adds one FP32 NATIVE input and one FP32 NATIVE output, the shape the BF16
// validation path expects (neither tensor is graph I/O).
void AddFp32NativeInputOutput(qnn::QnnModelWrapper& wrapper) {
  qnn::QnnTensorWrapper in("in0", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                           qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  qnn::QnnTensorWrapper out("out0", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                            qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{4});
  ASSERT_TRUE(wrapper.AddTensorWrapper(std::move(in)));
  ASSERT_TRUE(wrapper.AddTensorWrapper(std::move(out)));
}
}  // namespace

// IsBF16ConversionEnabled() reads the backend type from the manager. On HTP with
// htp_bf16_enable=true the validator must see BF16 inputs...
TEST(QnnUnit_ModelWrapperTest, CreateQnnNode_BF16Enabled_HtpBackend_ValidatorSeesBf16) {
  QnnModelWrapperTestContext ctx;
  ctx.qnn_interface.backendValidateOpConfig = StubBackendValidateOpConfigCaptureDataType;
  g_validated_input_data_type = QNN_DATATYPE_UNDEFINED;

  qnn::ModelSettings settings{};
  settings.htp_bf16_enable = true;
  auto wrapper = ctx.CreateWrapper(settings, qnn::QnnBackendType::HTP);
  AddFp32NativeInputOutput(*wrapper);

  EXPECT_TRUE(wrapper->CreateQnnNode("node0",
                                     QNN_OP_PACKAGE_NAME_QTI_AISW,
                                     QNN_OP_CAST,
                                     {"in0"}, {"out0"}, {},
                                     /*do_op_validation=*/true));

  EXPECT_EQ(g_validated_input_data_type, QNN_DATATYPE_BFLOAT_16);
  // BF16ConversionGuard restores FP32 once CreateQnnNode returns.
  EXPECT_EQ(wrapper->GetQnnTensorWrapper("in0").GetTensorDataType(), QNN_DATATYPE_FLOAT_32);
}

// ...while the same settings on CPU must skip the conversion entirely, because the
// manager — not a constructor argument — is what reports the backend type.
TEST(QnnUnit_ModelWrapperTest, CreateQnnNode_BF16Enabled_CpuBackend_ValidatorSeesFp32) {
  QnnModelWrapperTestContext ctx;
  ctx.qnn_interface.backendValidateOpConfig = StubBackendValidateOpConfigCaptureDataType;
  g_validated_input_data_type = QNN_DATATYPE_UNDEFINED;

  qnn::ModelSettings settings{};
  settings.htp_bf16_enable = true;
  auto wrapper = ctx.CreateWrapper(settings, qnn::QnnBackendType::CPU);
  AddFp32NativeInputOutput(*wrapper);

  EXPECT_TRUE(wrapper->CreateQnnNode("node0",
                                     QNN_OP_PACKAGE_NAME_QTI_AISW,
                                     QNN_OP_CAST,
                                     {"in0"}, {"out0"}, {},
                                     /*do_op_validation=*/true));

  EXPECT_EQ(g_validated_input_data_type, QNN_DATATYPE_FLOAT_32);
  EXPECT_EQ(wrapper->GetQnnTensorWrapper("in0").GetTensorDataType(), QNN_DATATYPE_FLOAT_32);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
