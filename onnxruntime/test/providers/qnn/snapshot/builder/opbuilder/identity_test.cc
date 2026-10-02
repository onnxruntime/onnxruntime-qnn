// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Op-builder snapshot tests for IdentityOpBuilder.

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"
#include "test/providers/qnn/infra/specs/builder/opbuilder/identity_specs.h"
#include "test/providers/qnn/snapshot/snapshot.h"

using namespace onnxruntime;
using namespace onnxruntime::qnn;

namespace onnxruntime {
namespace test {

namespace {

void RunPlainIdentitySnapshot(const IdentitySpec& spec) {
  const IOpBuilder* builder = GetOpBuilder("Identity");
  ASSERT_NE(builder, nullptr);

  const ONNXTensorElementDataType dtype =
      spec.kind == IdentityDataKind::Float ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                                           : ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;

  g_mock_init_reg.clear();
  OpBuilderTestContext ctx;
  SetupMockInitRegistryStubs(ctx);
  QnnRealHtpBackendManagerContext htp;
  if (!htp.IsValid()) GTEST_SKIP() << "libQnnHtp.so not available";
  std::unique_ptr<QnnModelWrapper> wrapper =
      MakeSnapshotWrapperHtpJson(ctx, htp, {"data"}, {"output"});
  ASSERT_NE(wrapper, nullptr) << "Failed to initialize QNN graph for snapshot test";

  auto data = MakeMockIODef("data", dtype, spec.shape);
  auto output = MakeMockIODef("output", dtype, spec.shape);
  auto node_unit = MakeMockNodeUnit("Identity", {data}, {output}, "identity_node");

  auto status = builder->AddToModelBuilder(*wrapper, node_unit, ctx.ort_logger, false);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  ASSERT_TRUE(wrapper->ComposeQnnGraph(/*build_json_qnn_graph=*/true)) << "ComposeQnnGraph failed";
  AssertSnapshotJson(*wrapper, spec.name);
}

void RunQDQIdentitySnapshot(const IdentitySpec& spec) {
  const IOpBuilder* builder = GetOpBuilder("Identity");
  ASSERT_NE(builder, nullptr);

  g_mock_init_reg.clear();
  const OrtValueInfo* input_scale = g_mock_init_reg.AddScalarFloat("input_scale", spec.input_scale);
  const OrtValueInfo* input_zp = g_mock_init_reg.AddScalarUint8("input_zp", spec.zero_point);
  const OrtValueInfo* output_scale = g_mock_init_reg.AddScalarFloat("output_scale", spec.output_scale);
  const OrtValueInfo* output_zp = g_mock_init_reg.AddScalarUint8("output_zp", spec.zero_point);

  OpBuilderTestContext ctx;
  SetupMockInitRegistryStubs(ctx);
  QnnRealHtpBackendManagerContext htp;
  if (!htp.IsValid()) GTEST_SKIP() << "libQnnHtp.so not available";
  std::unique_ptr<QnnModelWrapper> wrapper =
      MakeSnapshotWrapperHtpJson(ctx, htp, {"data"}, {"output"});
  ASSERT_NE(wrapper, nullptr) << "Failed to initialize QNN graph for snapshot test";

  auto data = MakeMockQDQIODef("data", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, spec.shape,
                                input_scale, input_zp);
  auto output = MakeMockQDQIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, spec.shape,
                                  output_scale, output_zp);
  auto node_unit = MakeMockQDQNodeUnit("Identity", {data}, {output}, "identity_node");

  auto status = builder->AddToModelBuilder(*wrapper, node_unit, ctx.ort_logger, false);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  ASSERT_TRUE(wrapper->ComposeQnnGraph(/*build_json_qnn_graph=*/true)) << "ComposeQnnGraph failed";
  AssertSnapshotJson(*wrapper, spec.name);
}

void RunIdentitySnapshot(const IdentitySpec& spec) {
  if (spec.kind == IdentityDataKind::QDQUint8) {
    RunQDQIdentitySnapshot(spec);
  } else {
    RunPlainIdentitySnapshot(spec);
  }
}

}  // namespace

class QnnSnapshot_Identity_OpBuilderTest : public ::testing::TestWithParam<IdentitySpec> {};

TEST_P(QnnSnapshot_Identity_OpBuilderTest, Case) {
  RunIdentitySnapshot(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    , QnnSnapshot_Identity_OpBuilderTest,
    ::testing::ValuesIn(kIdentitySpecs),
    [](const ::testing::TestParamInfo<IdentitySpec>& i) { return std::string(i.param.name); });

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
