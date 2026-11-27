// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "test/providers/qnn/infra/specs/builder/opbuilder/mean_specs.h"
#include "test/providers/qnn/qnn_test_utils.h"

namespace onnxruntime {
namespace test {

inline std::vector<TestInputDef<float>> MakeMeanInputs(const MeanSpec& spec) {
  std::vector<TestInputDef<float>> inputs;
  for (const auto& input : spec.inputs) {
    inputs.emplace_back(input.shape, false,
                        input.data.empty()
                            ? GetFloatDataInRange(input.range_min, input.range_max,
                                                  SizeFromDimension(input.shape, 0))
                            : input.data);
  }
  return inputs;
}

inline GetTestModelFn BuildMeanF32(const MeanSpec& spec) {
  return BuildOpTestCase<float>("Mean_node", "Mean", MakeMeanInputs(spec), {}, {});
}

inline GetTestQDQModelFn<uint8_t> BuildMeanQDQ(const MeanSpec& spec) {
  // Preserve independent input-range parameters and CPU-output-derived QDQ.
  return BuildQDQOpTestCase<uint8_t>("Mean_node", "Mean", MakeMeanInputs(spec), {}, {});
}

inline ProviderOptions MeanProviderOptions(const MeanSpec& spec) {
  ProviderOptions options{{"backend_type", "htp"}};
  if (spec.qdq) options["offload_graph_io_quantization"] = "0";
  return options;
}

inline GetTestModelFn BuildMeanSessionModel(const MeanSpec& spec) {
  if (!spec.qdq) return BuildMeanF32(spec);
  auto build_f32 = BuildMeanF32(spec);
  auto build_qdq = BuildMeanQDQ(spec);
  return [build_f32, build_qdq, opset = spec.opset](ModelTestBuilder& builder) {
    std::unique_ptr<ModelAndBuilder> model;
    CreateModelInMemory(model, build_f32, opset);
    std::vector<Ort::Value> outputs;
    InferenceModelCPU(model->model_data, "mean_snapshot_f32", model->builder.feeds_, outputs);
    ASSERT_EQ(outputs.size(), 1u);
    const auto info = outputs[0].GetTensorTypeAndShapeInfo();
    ASSERT_EQ(info.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    std::vector<QuantParams<uint8_t>> output_qparams{
        GetDataQuantParams<uint8_t>(
            gsl::span<const float>(outputs[0].GetTensorData<float>(), info.GetElementCount()))};
    build_qdq(builder, output_qparams);
  };
}

}  // namespace test
}  // namespace onnxruntime
