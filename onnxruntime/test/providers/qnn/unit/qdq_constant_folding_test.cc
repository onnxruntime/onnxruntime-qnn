// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Function-level unit tests for qdq_constant_folding.cc.
//
// GetEffectivelyConstantTensorBytes() hands initializer bytes to callers that read them as
// int8_t / uint8_t. QnnModelWrapper::UnpackInitializerData() expands a sub-byte initializer to
// one byte per element with the unused high bits masked off, which QNN needs (it reads only the
// low bits via `bitwidth`) but which reads as a positive number in an int8_t decode -- so an INT4
// -1 would come back as 15. These tests pin the sign extension that prevents that.
//
// Initializer access is mocked through mock_init_registry, so no real ORT graph is needed.

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "core/providers/qnn/builder/opbuilder/qdq_constant_folding.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/ort_api.h"

#include "test/providers/qnn/unit/mock_init_registry.h"
#include "test/providers/qnn/unit/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

// QnnModelWrapper over a fake graph whose initializers come from g_mock_init_reg, which is what
// makes IsConstantInput() / GetConstantTensor() / UnpackInitializerData() resolve by name.
struct MockInitWrapperFixture {
  OrtApiStubContext ctx;
  // The Ort::ConstValueInfo / ConstTypeInfo wrappers that utils::GetOnnxTensorElemDataType() goes
  // through dispatch on the GLOBAL Ort::GetApi(), not on api_ptrs_, so without this the mock
  // OrtValueInfo* would be handed to the real ORT runtime and SIGSEGV.
  OrtGlobalApiOverride global_api_override{&ctx.stub_ort_api};
  Ort::Logger null_logger_{MakeNullLogger()};
  int fake_graph_sentinel_{};
  // QnnModelWrapper reads the QNN interface / handles / backend type through a
  // QnnBackendManager. None of the tests here touch the QNN interface — they only
  // need a manager reporting HTP — so it is left unstubbed.
  //
  // Declared after null_logger_ (the manager keeps a pointer to it) and safe to
  // build before the ctor body reseeds ctx.stub_ort_api, because ApiPtrs holds a
  // reference to that table rather than a copy of it.
  StubBackendManager backend_manager{ctx.MakeApiPtrs(), null_logger_};
  qnn::GraphInputOutputInfo input_info;
  qnn::GraphInputOutputInfo output_info;
  std::unique_ptr<qnn::QnnModelWrapper> wrapper;

  MockInitWrapperFixture() {
    g_mock_init_reg.clear();
    // Seed from the real OrtApi so everything these paths touch beyond the mocked initializer
    // queries still works -- notably CreateStatus / ReleaseStatus, which MAKE_EP_FAIL() needs on
    // the not-found path and which a zero-initialised table would leave null.
    ctx.stub_ort_api = *OrtGetApiBase()->GetApi(ORT_API_VERSION);
    SetupMockInitRegistryStubs(ctx);
    ApiPtrs api_ptrs = ctx.MakeApiPtrs();
    const OrtGraph& fake_graph = *reinterpret_cast<const OrtGraph*>(&fake_graph_sentinel_);
    backend_manager.BackendType() = qnn::QnnBackendType::HTP;
    wrapper = std::make_unique<qnn::QnnModelWrapper>(
        fake_graph, api_ptrs, null_logger_,
        *backend_manager.Get(),
        input_info, output_info,
        qnn::ModelSettings{});
  }
};

std::vector<int8_t> AsInt8(const std::vector<uint8_t>& bytes) {
  std::vector<int8_t> out(bytes.size());
  std::memcpy(out.data(), bytes.data(), bytes.size());
  return out;
}

// Minimal DQ input def: the fold decision reads only the input's name, the registered
// initializer's element type, and the scale's shape (a >1-element scale is per-channel).
OrtNodeUnitIODef QuantizedInputDef(const std::string& name, const OrtValueInfo* scale) {
  OrtNodeUnitIODef io_def;
  io_def.name = name;
  io_def.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  io_def.quant_param = OrtNodeUnitIODef::QuantParam{scale};
  return io_def;
}

bool CanSubstitute(qnn::QnnModelWrapper& wrapper, const std::string& name, const OrtValueInfo* scale) {
  bool can_substitute = false;
  EXPECT_TRUE(qnn::CanSubstituteRuntimeDequantize(wrapper, QuantizedInputDef(name, scale),
                                                  can_substitute)
                  .IsOK());
  return can_substitute;
}

}  // namespace

TEST(QnnUnit_QdqConstantFoldingTest, ConstantBytes_Int4_AreSignExtended) {
  MockInitWrapperFixture fx;
  // Every representable INT4 value, so both nibble positions and the whole negative half are hit.
  const std::vector<int8_t> values{-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};
  g_mock_init_reg.AddTensorInt4As8bit("w_int4", {16}, values);

  std::vector<uint8_t> bytes;
  ASSERT_TRUE(qnn::GetEffectivelyConstantTensorBytes(*fx.wrapper, "w_int4", bytes).IsOK());
  ASSERT_EQ(bytes.size(), values.size());  // one byte per element, not packed nibbles
  // Without the sign extension, -8..-1 would come back as 8..15.
  EXPECT_EQ(AsInt8(bytes), values);
}

TEST(QnnUnit_QdqConstantFoldingTest, ConstantBytes_Uint4_AreUnchanged) {
  MockInitWrapperFixture fx;
  const std::vector<uint8_t> values{0, 1, 7, 8, 14, 15};
  g_mock_init_reg.AddTensorUint4As8bit("w_uint4", {6}, values);

  std::vector<uint8_t> bytes;
  ASSERT_TRUE(qnn::GetEffectivelyConstantTensorBytes(*fx.wrapper, "w_uint4", bytes).IsOK());
  EXPECT_EQ(bytes, values);
}

TEST(QnnUnit_QdqConstantFoldingTest, ConstantBytes_Uint8_AreUnchanged) {
  MockInitWrapperFixture fx;
  const std::vector<uint8_t> values{0, 1, 127, 128, 255};
  g_mock_init_reg.AddTensorUint8("w_uint8", {5}, values);

  std::vector<uint8_t> bytes;
  ASSERT_TRUE(qnn::GetEffectivelyConstantTensorBytes(*fx.wrapper, "w_uint8", bytes).IsOK());
  EXPECT_EQ(bytes, values);
}

TEST(QnnUnit_QdqConstantFoldingTest, ConstantBytes_UnknownTensor_Fails) {
  MockInitWrapperFixture fx;
  std::vector<uint8_t> bytes;
  EXPECT_FALSE(qnn::GetEffectivelyConstantTensorBytes(*fx.wrapper, "missing", bytes).IsOK());
}

// A per-channel or INT4/INT2 constant must fold at any size: QNN has no standalone
// per-channel Dequantize, and declining would also stop constant-ness from reaching the
// next Q/DQ hop, which is what keeps chained weights STATIC instead of graph inputs.
TEST(QnnUnit_QdqConstantFoldingTest, RuntimeDequantizeSubstitute_Refused_MustFold) {
  MockInitWrapperFixture fx;
  g_mock_init_reg.AddTensorUint8("w_u8", {4}, {0, 1, 2, 3});
  g_mock_init_reg.AddTensorInt4As8bit("w_i4", {4}, {-8, -1, 1, 7});
  const OrtValueInfo* per_tensor_scale = g_mock_init_reg.AddScalarFloat("scale", 0.1f);
  const OrtValueInfo* per_channel_scale = g_mock_init_reg.AddTensorFloat("scales", {4},
                                                                         {0.1f, 0.2f, 0.3f, 0.4f});

  EXPECT_FALSE(CanSubstitute(*fx.wrapper, "w_u8", per_channel_scale));
  EXPECT_FALSE(CanSubstitute(*fx.wrapper, "w_i4", per_tensor_scale));
}

// Everything else dequantizes identically at runtime, so the size budget may decline it.
// UINT4 is included deliberately: only the signed sub-byte types carry the high-bit mask
// hazard that the fold path has to undo.
TEST(QnnUnit_QdqConstantFoldingTest, RuntimeDequantizeSubstitute_Allowed) {
  MockInitWrapperFixture fx;
  g_mock_init_reg.AddTensorUint8("w_u8", {4}, {0, 1, 2, 3});
  g_mock_init_reg.AddTensorUint4As8bit("w_u4", {4}, {0, 1, 14, 15});
  const OrtValueInfo* per_tensor_scale = g_mock_init_reg.AddScalarFloat("scale", 0.1f);

  EXPECT_TRUE(CanSubstitute(*fx.wrapper, "w_u8", per_tensor_scale));
  EXPECT_TRUE(CanSubstitute(*fx.wrapper, "w_u4", per_tensor_scale));
  // An unregistered name is a previously-folded intermediate: plain bytes, no mask hazard.
  EXPECT_TRUE(CanSubstitute(*fx.wrapper, "folded_intermediate", per_tensor_scale));
}

// Size boundary for the inputs the checks above admit. Exact by construction: the budget
// is compared in elements so an adversarial shape cannot wrap past it.
TEST(QnnUnit_QdqConstantFoldingTest, SkipPredicate_SmallFolds) {
  EXPECT_FALSE(qnn::ShouldSkipConstantDQFold(6));           // bias-sized tensors fold
  EXPECT_FALSE(qnn::ShouldSkipConstantDQFold(256 * 1024));  // exactly at budget: fold
}

TEST(QnnUnit_QdqConstantFoldingTest, SkipPredicate_LargeSkips) {
  EXPECT_TRUE(qnn::ShouldSkipConstantDQFold(256 * 1024 + 1));  // just past budget
  EXPECT_TRUE(qnn::ShouldSkipConstantDQFold(1536 * 6144));     // psx0 FC dims: 36 MB as FP32
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
