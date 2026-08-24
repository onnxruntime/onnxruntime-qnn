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
  QNN_INTERFACE_VER_TYPE qnn_interface = QNN_INTERFACE_VER_TYPE_INIT;
  Qnn_BackendHandle_t backend_handle = nullptr;
  QNN_INTERFACE_VER_TYPE qnn_validator_interface = QNN_INTERFACE_VER_TYPE_INIT;
  Qnn_BackendHandle_t validator_backend_handle = nullptr;
  Ort::Logger null_logger_{MakeNullLogger()};
  int fake_graph_sentinel_{};
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
    wrapper = std::make_unique<qnn::QnnModelWrapper>(
        fake_graph, api_ptrs, null_logger_,
        qnn_interface, backend_handle,
        qnn_validator_interface, validator_backend_handle,
        input_info, output_info,
        qnn::QnnBackendType::HTP, qnn::ModelSettings{});
  }
};

std::vector<int8_t> AsInt8(const std::vector<uint8_t>& bytes) {
  std::vector<int8_t> out(bytes.size());
  std::memcpy(out.data(), bytes.data(), bytes.size());
  return out;
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

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
