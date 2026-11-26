// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstring>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {

// ONNX permits one input, but the current Mean builder rejects it before node emission.
// This proves the provider's rejection boundary, not a backend limitation.
TEST(QnnUnit_Mean_ComponentTest, SingleInputUnsupported) {
  const auto* builder = qnn::GetOpBuilder("Mean");
  ASSERT_NE(builder, nullptr);
  OpBuilderTestContext ctx;
  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);
  auto input = MakeMockIODef("input", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                             std::vector<int64_t>{4});
  auto output = MakeMockIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                              std::vector<int64_t>{4});
  auto node = MakeMockNodeUnit("Mean", {input}, {output}, "mean_single", "", 13);
  const auto status = builder->AddToModelBuilder(*wrapper, node, ctx.ort_logger, true);
  EXPECT_FALSE(status.IsOK());
  EXPECT_NE(std::string(status.GetErrorMessage()).find("at least two inputs"), std::string::npos);
  EXPECT_EQ(wrapper->GetModelTensorsMap().count("output"), 0u);
}

// Read the actual builder-created static tensor, not a guessed JSON data field.
TEST(QnnUnit_Mean_ComponentTest, StaticDivisorMatchesInputCount) {
  const auto* builder = qnn::GetOpBuilder("Mean");
  ASSERT_NE(builder, nullptr);
  for (size_t count : {size_t{2}, size_t{4}}) {
    SCOPED_TRACE(count);
    OpBuilderTestContext ctx;
    qnn::ModelSettings settings{};
    auto wrapper = ctx.CreateWrapper(settings);
    std::vector<OrtNodeUnitIODef> inputs;
    for (size_t i = 0; i < count; ++i) {
      inputs.push_back(MakeMockIODef("input_" + std::to_string(i),
                                     ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                     std::vector<int64_t>{4}));
    }
    auto output = MakeMockIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                std::vector<int64_t>{4});
    auto node = MakeMockNodeUnit("Mean", inputs, {output}, "mean_divisor", "", 13);
    const auto status = builder->AddToModelBuilder(*wrapper, node, ctx.ort_logger, false);
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
    size_t static_count = 0;
    for (const auto& [name, tensor_wrapper] : wrapper->GetModelTensorsMap()) {
      const auto& tensor = tensor_wrapper.GetQnnTensor();
      if (qnn::GetQnnTensorType(tensor) != QNN_TENSOR_TYPE_STATIC) continue;
      ++static_count;
      EXPECT_EQ(qnn::GetQnnTensorDataType(tensor), QNN_DATATYPE_FLOAT_32);
      ASSERT_EQ(qnn::GetQnnTensorRank(tensor), 1u);
      EXPECT_EQ(qnn::GetQnnTensorDims(tensor)[0], 1u);
      const auto& buffer = qnn::GetQnnTensorClientBuf(tensor);
      ASSERT_EQ(buffer.dataSize, sizeof(float));
      ASSERT_NE(buffer.data, nullptr);
      float divisor = 0.0f;
      std::memcpy(&divisor, buffer.data, sizeof(divisor));
      EXPECT_FLOAT_EQ(divisor, static_cast<float>(count));
    }
    EXPECT_EQ(static_count, 1u);
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
