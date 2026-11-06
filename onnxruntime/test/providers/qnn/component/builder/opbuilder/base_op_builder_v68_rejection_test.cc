// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Component-tier tests verifying the blanket rejection of FP32/FP16 tensors on
// HTP v68 at GetCapability time.  Two suites, no live QNN library required:
//
//   QnnBaseOpBuilderV68   — BaseOpBuilder::IsOpSupported rejects FP32/FP16 on v68
//                           via ProcessDataTypes; passes on v69 (guard stays silent).
//
//   QnnFusionV68Rejection — RejectIfHtpV68Float (the utility called by all
//                           float-supporting fusions) rejects FP32/FP16 on v68,
//                           lets INT8 through, and stays silent on v69.
//
// Both suites use a stub-backed QnnModelWrapper with HtpArch set at construction
// time.  The v68 guard fires before any QNN API call, so no backend session is
// needed.

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "HTP/QnnHtpDevice.h"

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "test/providers/qnn/infra/backend_contexts.h"
#include "test/providers/qnn/infra/mock_node_unit.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

std::unique_ptr<qnn::QnnModelWrapper> MakeV68HtpWrapper(OpBuilderTestContext& ctx) {
  ctx.backend_manager.HtpArch() = QNN_HTP_DEVICE_ARCH_V68;
  qnn::ModelSettings settings{};
  return ctx.CreateWrapper(settings, qnn::QnnBackendType::HTP);
}

}  // namespace

// =============================================================================
// Suite 1: BaseOpBuilder — single-node ops
// =============================================================================

// FP32 Relu must be rejected on HTP v68 via IsOpSupported.
TEST(QnnBaseOpBuilderV68, Relu_FP32_Rejected) {
  OpBuilderTestContext ctx;
  auto wrapper = MakeV68HtpWrapper(ctx);
  ASSERT_NE(wrapper, nullptr);

  auto input = MakeMockIODef("data", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{1, 4});
  auto output = MakeMockIODef("result", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{1, 4});
  auto node_unit = MakeMockNodeUnit("Relu", {input}, {output});

  const qnn::IOpBuilder* builder = qnn::GetOpBuilder("Relu");
  ASSERT_NE(builder, nullptr);

  Ort::Status status = builder->IsOpSupported(*wrapper, node_unit, ctx.ort_logger);
  EXPECT_FALSE(status.IsOK()) << "FP32 Relu must be rejected on HTP v68";
}

// FP16 Add must be rejected on HTP v68 via IsOpSupported.
TEST(QnnBaseOpBuilderV68, Add_FP16_Rejected) {
  OpBuilderTestContext ctx;
  auto wrapper = MakeV68HtpWrapper(ctx);
  ASSERT_NE(wrapper, nullptr);

  auto a = MakeMockIODef("a", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, std::vector<int64_t>{1, 4});
  auto b = MakeMockIODef("b", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, std::vector<int64_t>{1, 4});
  auto c = MakeMockIODef("c", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, std::vector<int64_t>{1, 4});
  auto node_unit = MakeMockNodeUnit("Add", {a, b}, {c});

  const qnn::IOpBuilder* builder = qnn::GetOpBuilder("Add");
  ASSERT_NE(builder, nullptr);

  Ort::Status status = builder->IsOpSupported(*wrapper, node_unit, ctx.ort_logger);
  EXPECT_FALSE(status.IsOK()) << "FP16 Add must be rejected on HTP v68";
}

// On V69 the v68 guard must not fire.
TEST(QnnBaseOpBuilderV68, Relu_FP32_NotRejectedOnV69) {
  OpBuilderTestContext ctx;
  ctx.backend_manager.HtpArch() = QNN_HTP_DEVICE_ARCH_V69;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings, qnn::QnnBackendType::HTP);
  ASSERT_NE(wrapper, nullptr);

  auto input = MakeMockIODef("data", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{1, 4});
  auto output = MakeMockIODef("result", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{1, 4});
  auto node_unit = MakeMockNodeUnit("Relu", {input}, {output});

  const qnn::IOpBuilder* builder = qnn::GetOpBuilder("Relu");
  ASSERT_NE(builder, nullptr);

  Ort::Status status = builder->IsOpSupported(*wrapper, node_unit, ctx.ort_logger);
  if (!status.IsOK()) {
    const std::string msg = status.GetErrorMessage();
    EXPECT_EQ(msg.find("v68"), std::string::npos)
        << "v68 guard must not fire on V69; got: " << msg;
  }
}

// =============================================================================
// Suite 2: Fusion node groups — RejectIfHtpV68Float utility
// =============================================================================

// FP32 io_def must be rejected on v68 (covers all float fusions).
TEST(QnnFusionV68Rejection, FP32_Rejected) {
  OpBuilderTestContext ctx;
  auto wrapper = MakeV68HtpWrapper(ctx);
  ASSERT_NE(wrapper, nullptr);

  auto io_def = MakeMockIODef("x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{1, 128});

  Ort::Status status = qnn::RejectIfHtpV68Float(*wrapper, io_def);
  EXPECT_FALSE(status.IsOK()) << "FP32 io_def must be rejected on HTP v68";
}

// FP16 io_def must also be rejected on v68.
TEST(QnnFusionV68Rejection, FP16_Rejected) {
  OpBuilderTestContext ctx;
  auto wrapper = MakeV68HtpWrapper(ctx);
  ASSERT_NE(wrapper, nullptr);

  auto io_def = MakeMockIODef("x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, std::vector<int64_t>{1, 128});

  Ort::Status status = qnn::RejectIfHtpV68Float(*wrapper, io_def);
  EXPECT_FALSE(status.IsOK()) << "FP16 io_def must be rejected on HTP v68";
}

// INT8 io_def must pass through the guard unchanged.
TEST(QnnFusionV68Rejection, INT8_NotRejected) {
  OpBuilderTestContext ctx;
  auto wrapper = MakeV68HtpWrapper(ctx);
  ASSERT_NE(wrapper, nullptr);

  auto io_def = MakeMockIODef("x", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, std::vector<int64_t>{1, 128});

  Ort::Status status = qnn::RejectIfHtpV68Float(*wrapper, io_def);
  EXPECT_TRUE(status.IsOK()) << "INT8 io_def must not be rejected on HTP v68";
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
