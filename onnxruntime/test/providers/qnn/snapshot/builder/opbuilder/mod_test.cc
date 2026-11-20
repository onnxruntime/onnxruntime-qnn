// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"
#include "test/providers/qnn/infra/specs/builder/opbuilder/mod_specs.h"
#include "test/providers/qnn/snapshot/snapshot.h"

using namespace onnxruntime;
using namespace onnxruntime::qnn;

namespace onnxruntime::test {
namespace {

void RunModSnapshot(const ModSpec& spec) {
  const IOpBuilder* builder = GetOpBuilder("Mod");
  ASSERT_NE(builder, nullptr);

  g_mock_init_reg.clear();
  if (spec.divisor_is_initializer) {
    g_mock_init_reg.AddTensorInt64("divisor", {1, 5}, spec.divisor);
  }

  OpBuilderTestContext ctx;
  SetupMockInitRegistryStubs(ctx);
  QnnRealHtpBackendManagerContext htp;
  if (!htp.IsValid()) GTEST_SKIP() << "libQnnHtp.so not available";

  std::vector<std::string> graph_inputs{"dividend"};
  if (!spec.divisor_is_initializer) {
    graph_inputs.push_back("divisor");
  }
  auto wrapper = MakeSnapshotWrapperHtpJson(ctx, htp, graph_inputs, {"output"});
  ASSERT_NE(wrapper, nullptr);

  auto dividend = MakeMockIODef("dividend", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                                std::vector<int64_t>{1, 4, 5});
  auto divisor = MakeMockIODef("divisor", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                               std::vector<int64_t>{1, 5});
  auto output = MakeMockIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                              std::vector<int64_t>{1, 4, 5});
  auto node = MakeMockNodeUnit("Mod", {dividend, divisor}, {output}, "mod_node", "", 13);

  const auto status = builder->AddToModelBuilder(*wrapper, node, ctx.ort_logger, false);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  ASSERT_TRUE(wrapper->ComposeQnnGraph(true)) << "ComposeQnnGraph failed";
  AssertSnapshotJson(*wrapper, spec.name);
}

}  // namespace

class QnnSnapshot_Mod_OpBuilderTest : public ::testing::TestWithParam<ModSpec> {};

TEST_P(QnnSnapshot_Mod_OpBuilderTest, Case) {
  RunModSnapshot(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    , QnnSnapshot_Mod_OpBuilderTest, ::testing::ValuesIn(kModSpecs),
    [](const ::testing::TestParamInfo<ModSpec>& info) { return std::string(info.param.name); });

}  // namespace onnxruntime::test

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
