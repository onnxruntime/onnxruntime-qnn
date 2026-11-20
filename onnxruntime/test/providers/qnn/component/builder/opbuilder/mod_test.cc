// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

using namespace onnxruntime;
using namespace onnxruntime::qnn;

namespace onnxruntime::test {

TEST(QnnUnit_Mod_ComponentTest, RejectsFmodAttribute) {
  const IOpBuilder* builder = GetOpBuilder("Mod");
  ASSERT_NE(builder, nullptr);

  OpBuilderTestContext ctx;
  ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings, QnnBackendType::HTP);

  auto dividend = MakeMockIODef("dividend", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                std::vector<int64_t>{1, 4, 5});
  auto divisor = MakeMockIODef("divisor", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                               std::vector<int64_t>{1, 5});
  auto output = MakeMockIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                              std::vector<int64_t>{1, 4, 5});
  auto node = MakeMockNodeUnit(
      "Mod", {dividend, divisor}, {output}, "mod_fmod", "", 13, 0,
      {FakeOpAttr::MakeInt64("fmod", 1)});

  const auto status = builder->AddToModelBuilder(*wrapper, node, ctx.ort_logger, false);
  EXPECT_FALSE(status.IsOK());
  EXPECT_NE(std::string(status.GetErrorMessage()).find("fmod == 0"), std::string::npos);
}

}  // namespace onnxruntime::test

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
